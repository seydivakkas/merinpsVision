"""Unit tests for M5 drift services.

Covers:
- DriftMonitorService.run_window(): persists DriftWindow, correct trend_status
- DriftMonitorService.run_window(): stable series -> STABLE pattern
- DriftMonitorService.run_window(): dramatic drop -> non-STABLE trend/pattern
- IncidentService.evaluate(): signal count, evidence dict
- IncidentService.open(): below minimum -> None (no DB write)
- IncidentService.open(): at minimum -> IncidentRecord persisted
- IncidentService.open(): priority mapping (2 signals -> P2_REVIEW, 5 -> P0_BLOCKED)
- IncidentService.list_open(): returns only unresolved incidents
- Round-trip: run_window -> open -> list_open
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from weavevision.domain.enums import DriftPattern, IncidentPriority, TrendStatus
from weavevision.evaluation.incident_triage import TriageEvidence
from weavevision.persistence.database import Database
from weavevision.services.drift_monitor_service import DriftMonitorService
from weavevision.services.incident_service import (
    IncidentService,
    _signal_count_to_priority,
)
from weavevision.settings import load_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tmp_db() -> Database:
    tmp = tempfile.mktemp(suffix=".sqlite3")
    db = Database(Path(tmp))
    db.migrate()
    return db


def _services() -> tuple[DriftMonitorService, IncidentService]:
    settings = load_settings()
    db = _tmp_db()
    return DriftMonitorService(settings, db), IncidentService(settings, db)


# ---------------------------------------------------------------------------
# DriftMonitorService
# ---------------------------------------------------------------------------


class TestDriftMonitorService:
    def test_stable_series_returns_drift_window(self) -> None:
        """Stable series persists a DriftWindow with STABLE status."""
        monitor, _ = _services()
        values = np.full(10, 0.85)
        window = monitor.run_window(
            "image_ap50",
            values,
            baseline_mean=0.85,
            baseline_std=0.01,
            model_id="model_abc",
        )
        assert window.trend_status == TrendStatus.STABLE
        assert window.drift_pattern == DriftPattern.STABLE
        assert window.metric_value == pytest.approx(0.85)

    def test_sudden_drop_non_stable_status(self) -> None:
        """Sudden drop produces a non-STABLE trend status."""
        monitor, _ = _services()
        stable = np.full(10, 0.90)
        drop = np.full(5, 0.60)
        values = np.concatenate([stable, drop])
        window = monitor.run_window(
            "image_ap50",
            values,
            baseline_mean=0.90,
            baseline_std=0.02,
            model_id="model_abc",
        )
        assert window.trend_status != TrendStatus.STABLE

    def test_window_id_unique_per_call(self) -> None:
        """Each run_window() call generates a unique window_id."""
        monitor, _ = _services()
        values = np.full(5, 0.85)
        w1 = monitor.run_window("ap50", values, 0.85, 0.01, model_id="m")
        w2 = monitor.run_window("ap50", values, 0.85, 0.01, model_id="m")
        assert w1.window_id != w2.window_id

    def test_window_persisted_in_db(self) -> None:
        """run_window() result appears in DriftWindowRepository.list_recent()."""
        settings = load_settings()
        db = _tmp_db()
        monitor = DriftMonitorService(settings, db)

        from weavevision.persistence.repositories import DriftWindowRepository

        repo = DriftWindowRepository(db)
        values = np.full(3, 0.85)
        window = monitor.run_window("ap50", values, 0.85, 0.01, model_id="m")
        rows = repo.list_recent()
        assert any(r["window_id"] == window.window_id for r in rows)

    def test_optional_fields_propagated(self) -> None:
        """threshold_id and source_manifest_sha256 are stored on the window."""
        monitor, _ = _services()
        values = np.full(3, 0.82)
        window = monitor.run_window(
            "ap50",
            values,
            0.85,
            0.01,
            model_id="model_abc",
            threshold_id="thr_001",
            source_manifest_sha256="abc123",
        )
        assert window.threshold_id == "thr_001"
        assert window.source_manifest_sha256 == "abc123"

    def test_empty_values_raises(self) -> None:
        """Forwarded ValueError from trend_monitor for empty input."""
        monitor, _ = _services()
        with pytest.raises(ValueError):
            monitor.run_window("ap50", np.array([]), 0.85, 0.01, model_id="m")

    def test_baseline_std_zero_raises(self) -> None:
        """Forwarded ValueError from trend_monitor for std=0."""
        monitor, _ = _services()
        with pytest.raises(ValueError):
            monitor.run_window("ap50", np.array([0.85]), 0.85, 0.0, model_id="m")

    def test_policy_exposed(self) -> None:
        """DriftMonitorService.policy property returns DriftAlertPolicy."""
        monitor, _ = _services()
        assert monitor.policy.minimum_signals == 2  # from app.yaml


# ---------------------------------------------------------------------------
# IncidentService.evaluate()
# ---------------------------------------------------------------------------


class TestIncidentServiceEvaluate:
    def test_evaluate_zero_signals(self) -> None:
        monitor, incident = _services()
        window = monitor.run_window("ap50", np.full(3, 0.85), 0.85, 0.01, model_id="m")
        ev = TriageEvidence()
        decision = incident.evaluate(window, ev)
        assert decision.confirming_signal_count == 0
        assert decision.incident_id == "PENDING"

    def test_evaluate_two_signals(self) -> None:
        monitor, incident = _services()
        window = monitor.run_window("ap50", np.full(3, 0.85), 0.85, 0.01, model_id="m")
        ev = TriageEvidence(ewma_alert=True, cusum_alert=True)
        decision = incident.evaluate(window, ev)
        assert decision.confirming_signal_count == 2

    def test_evaluate_evidence_dict_populated(self) -> None:
        """All 6 fields appear in the evidence dict."""
        monitor, incident = _services()
        window = monitor.run_window("ap50", np.full(3, 0.85), 0.85, 0.01, model_id="m")
        ev = TriageEvidence(psi_high=True)
        decision = incident.evaluate(window, ev)
        assert "ewma_alert" in decision.evidence
        assert "psi_high" in decision.evidence
        assert decision.evidence["psi_high"] is True
        assert decision.evidence["ewma_alert"] is False

    def test_evaluate_does_not_persist(self) -> None:
        """evaluate() must not write any incident to the database."""
        monitor, incident = _services()
        window = monitor.run_window("ap50", np.full(3, 0.85), 0.85, 0.01, model_id="m")
        ev = TriageEvidence(ewma_alert=True, cusum_alert=True)
        incident.evaluate(window, ev)
        assert incident.list_open() == []


# ---------------------------------------------------------------------------
# IncidentService.open()
# ---------------------------------------------------------------------------


class TestIncidentServiceOpen:
    def test_below_minimum_returns_none(self) -> None:
        """1 signal < minimum=2 -> open() returns None."""
        monitor, incident = _services()
        window = monitor.run_window("ap50", np.full(3, 0.85), 0.85, 0.01, model_id="m")
        ev = TriageEvidence(ewma_alert=True)  # 1 signal
        result = incident.open(window, ev)
        assert result is None

    def test_minimum_met_returns_incident(self) -> None:
        """2 signals == minimum=2 -> IncidentRecord returned."""
        monitor, incident = _services()
        window = monitor.run_window("ap50", np.full(3, 0.85), 0.85, 0.01, model_id="m")
        ev = TriageEvidence(ewma_alert=True, cusum_alert=True)
        result = incident.open(window, ev)
        assert result is not None
        assert result.priority == IncidentPriority.P2_REVIEW

    def test_incident_persisted(self) -> None:
        """Opened incident appears in list_open()."""
        monitor, incident = _services()
        window = monitor.run_window("ap50", np.full(3, 0.85), 0.85, 0.01, model_id="m")
        ev = TriageEvidence(ewma_alert=True, cusum_alert=True)
        result = incident.open(window, ev)
        assert result is not None
        open_list = incident.list_open()
        assert any(r["incident_id"] == result.incident_id for r in open_list)

    def test_below_minimum_no_db_write(self) -> None:
        """Insufficient signals -> nothing written to DB."""
        monitor, incident = _services()
        window = monitor.run_window("ap50", np.full(3, 0.85), 0.85, 0.01, model_id="m")
        incident.open(window, TriageEvidence(ewma_alert=True))
        assert incident.list_open() == []

    def test_root_cause_stored(self) -> None:
        """root_cause is propagated to the IncidentRecord."""
        monitor, incident = _services()
        window = monitor.run_window("ap50", np.full(3, 0.85), 0.85, 0.01, model_id="m")
        ev = TriageEvidence(ewma_alert=True, cusum_alert=True, psi_high=True)
        result = incident.open(window, ev, root_cause="texture_shift")
        assert result is not None
        assert result.root_cause == "texture_shift"

    def test_drift_pattern_inherited_from_window(self) -> None:
        """IncidentRecord.drift_pattern matches the source window."""
        monitor, incident = _services()
        window = monitor.run_window("ap50", np.full(3, 0.85), 0.85, 0.01, model_id="m")
        ev = TriageEvidence(ewma_alert=True, cusum_alert=True)
        result = incident.open(window, ev)
        assert result is not None
        assert result.drift_pattern == window.drift_pattern


# ---------------------------------------------------------------------------
# _signal_count_to_priority mapping
# ---------------------------------------------------------------------------


class TestSignalCountToPriority:
    def test_1_signal_info(self) -> None:
        assert _signal_count_to_priority(1) == IncidentPriority.INFO

    def test_2_signals_p2(self) -> None:
        assert _signal_count_to_priority(2) == IncidentPriority.P2_REVIEW

    def test_3_signals_p1(self) -> None:
        assert _signal_count_to_priority(3) == IncidentPriority.P1_INCIDENT

    def test_4_signals_p1(self) -> None:
        assert _signal_count_to_priority(4) == IncidentPriority.P1_INCIDENT

    def test_5_signals_p0(self) -> None:
        assert _signal_count_to_priority(5) == IncidentPriority.P0_BLOCKED

    def test_6_signals_p0(self) -> None:
        assert _signal_count_to_priority(6) == IncidentPriority.P0_BLOCKED


# ---------------------------------------------------------------------------
# Round-trip integration
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_window_to_incident_full_pipeline(self) -> None:
        """Full pipeline: run_window -> open -> list_open round-trip."""
        settings = load_settings()
        db = _tmp_db()
        monitor = DriftMonitorService(settings, db)
        incident_svc = IncidentService(settings, db)

        # Step 1: compute and persist window
        stable = np.full(10, 0.90)
        drop = np.full(5, 0.60)
        values = np.concatenate([stable, drop])
        window = monitor.run_window(
            "image_ap50",
            values,
            baseline_mean=0.90,
            baseline_std=0.02,
            model_id="model_xyz",
        )

        # Step 2: build evidence and open incident
        ev = TriageEvidence(ewma_alert=True, cusum_alert=True, psi_high=True)
        result = incident_svc.open(window, ev, root_cause="production_drift")

        assert result is not None
        assert result.model_id == "model_xyz"
        assert result.priority in {
            IncidentPriority.P2_REVIEW,
            IncidentPriority.P1_INCIDENT,
            IncidentPriority.P0_BLOCKED,
        }

        # Step 3: verify DB
        open_list = incident_svc.list_open()
        assert len(open_list) >= 1
        ids = [r["incident_id"] for r in open_list]
        assert result.incident_id in ids
