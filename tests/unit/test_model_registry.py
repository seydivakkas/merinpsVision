"""Unit tests for M7: ModelRegistryService and CanaryService.

Covers:
ModelRegistryService:
- register: model lands as CANDIDATE
- promote: valid transition CANDIDATE -> VALIDATED
- promote: invalid transition raises ValueError
- promote: RETIRED is terminal -> ValueError
- promote: REJECTED is terminal -> ValueError
- demote: transitions to RETIRED
- demote: RETIRED -> ValueError (already terminal)
- get_status: returns correct current status
- list_models: all / filtered by status
- rollback: sets to_model ACTIVE_BENCHMARK, from_model RETIRED
- rollback: persists RollbackEvent
- list_rollbacks: returns rollback audit records

CanaryService:
- evaluate: disagreement_rate <= max and recall_delta >= min -> PASSED
- evaluate: disagreement_rate > max -> FAILED
- evaluate: recall_delta < min -> FAILED
- evaluate: both conditions fail -> FAILED
- evaluate: boundary (exactly at threshold) -> PASSED
- evaluate: sample_count < 1 -> ValueError
- evaluate: disagreement_rate > 1 -> ValueError
- evaluate: record persisted after evaluate()
- list_canaries: all records
- list_canaries: filtered by champion_model_id
- Canary thresholds load from real app.yaml
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from weavevision.domain.enums import CanaryStatus, ModelStatus, RollbackReason
from weavevision.domain.errors import DatabaseError
from weavevision.persistence.database import Database
from weavevision.services.canary_service import CanaryService
from weavevision.services.model_registry_service import ModelRegistryService
from weavevision.settings import load_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tmp_db() -> Database:
    db = Database(Path(tempfile.mktemp(suffix=".sqlite3")))
    db.migrate()
    return db


def _registry() -> ModelRegistryService:
    settings = load_settings()
    return ModelRegistryService(settings, _tmp_db())


def _canary() -> CanaryService:
    settings = load_settings()
    return CanaryService(settings, _tmp_db())


def _register(svc: ModelRegistryService, model_id: str = "m001") -> None:
    svc.register(
        model_id,
        algorithm="patchcore",
        artifact_path="/models/m001.ckpt",
        artifact_sha256="abc" * 21 + "a",
    )


# ---------------------------------------------------------------------------
# ModelRegistryService
# ---------------------------------------------------------------------------


class TestModelRegistryRegister:
    def test_register_creates_candidate(self) -> None:
        svc = _registry()
        _register(svc, "m001")
        assert svc.get_status("m001") == ModelStatus.CANDIDATE

    def test_register_idempotent(self) -> None:
        """INSERT OR IGNORE: second register call is silently ignored."""
        svc = _registry()
        _register(svc, "m001")
        _register(svc, "m001")  # second call -- no error
        assert svc.get_status("m001") == ModelStatus.CANDIDATE

    def test_unknown_model_raises(self) -> None:
        svc = _registry()
        with pytest.raises(DatabaseError):
            svc.get_status("nonexistent")


class TestModelRegistryPromote:
    def test_candidate_to_validated(self) -> None:
        svc = _registry()
        _register(svc, "m001")
        result = svc.promote("m001", ModelStatus.VALIDATED)
        assert result == ModelStatus.VALIDATED
        assert svc.get_status("m001") == ModelStatus.VALIDATED

    def test_validated_to_active_benchmark(self) -> None:
        svc = _registry()
        _register(svc, "m001")
        svc.promote("m001", ModelStatus.VALIDATED)
        svc.promote("m001", ModelStatus.ACTIVE_BENCHMARK)
        assert svc.get_status("m001") == ModelStatus.ACTIVE_BENCHMARK

    def test_active_benchmark_to_pilot(self) -> None:
        svc = _registry()
        _register(svc, "m001")
        svc.promote("m001", ModelStatus.VALIDATED)
        svc.promote("m001", ModelStatus.ACTIVE_BENCHMARK)
        svc.promote("m001", ModelStatus.ACTIVE_COMPANY_PILOT)
        assert svc.get_status("m001") == ModelStatus.ACTIVE_COMPANY_PILOT

    def test_invalid_skip_transition_raises(self) -> None:
        """CANDIDATE -> ACTIVE_BENCHMARK is not a valid one-step transition."""
        svc = _registry()
        _register(svc, "m001")
        with pytest.raises(ValueError, match="not allowed"):
            svc.promote("m001", ModelStatus.ACTIVE_BENCHMARK)

    def test_candidate_to_rejected(self) -> None:
        """CANDIDATE -> REJECTED is allowed (failed eval)."""
        svc = _registry()
        _register(svc, "m001")
        svc.promote("m001", ModelStatus.REJECTED)
        assert svc.get_status("m001") == ModelStatus.REJECTED

    def test_retired_is_terminal(self) -> None:
        svc = _registry()
        _register(svc, "m001")
        svc.demote("m001")  # -> RETIRED
        with pytest.raises(ValueError):
            svc.promote("m001", ModelStatus.VALIDATED)

    def test_rejected_is_terminal(self) -> None:
        svc = _registry()
        _register(svc, "m001")
        svc.promote("m001", ModelStatus.REJECTED)
        with pytest.raises(ValueError):
            svc.promote("m001", ModelStatus.VALIDATED)


class TestModelRegistryDemote:
    def test_demote_candidate(self) -> None:
        svc = _registry()
        _register(svc, "m001")
        result = svc.demote("m001")
        assert result == ModelStatus.RETIRED
        assert svc.get_status("m001") == ModelStatus.RETIRED

    def test_demote_already_retired_raises(self) -> None:
        svc = _registry()
        _register(svc, "m001")
        svc.demote("m001")
        with pytest.raises(ValueError):
            svc.demote("m001")


class TestModelRegistryList:
    def test_list_all_models(self) -> None:
        svc = _registry()
        _register(svc, "m001")
        _register(svc, "m002")
        assert len(svc.list_models()) == 2

    def test_list_filtered_by_status(self) -> None:
        svc = _registry()
        _register(svc, "m001")
        _register(svc, "m002")
        svc.promote("m001", ModelStatus.VALIDATED)
        candidates = svc.list_models(status=ModelStatus.CANDIDATE)
        assert len(candidates) == 1
        assert candidates[0]["model_id"] == "m002"

    def test_list_empty(self) -> None:
        svc = _registry()
        assert svc.list_models() == []


class TestModelRegistryRollback:
    def test_rollback_transitions_statuses(self) -> None:
        settings = load_settings()
        db = _tmp_db()
        svc = ModelRegistryService(settings, db)
        _register(svc, "champion")
        _register(svc, "rollback_target")
        # Promote champion to ACTIVE_BENCHMARK
        svc.promote("champion", ModelStatus.VALIDATED)
        svc.promote("champion", ModelStatus.ACTIVE_BENCHMARK)
        # rollback_target stays CANDIDATE -> rollback sets it to ACTIVE_BENCHMARK directly
        svc.rollback(
            "champion",
            "rollback_target",
            RollbackReason.RECALL_DROP,
            triggered_by="autobot",
        )
        assert svc.get_status("champion") == ModelStatus.RETIRED
        assert svc.get_status("rollback_target") == ModelStatus.ACTIVE_BENCHMARK

    def test_rollback_persists_event(self) -> None:
        settings = load_settings()
        db = _tmp_db()
        svc = ModelRegistryService(settings, db)
        _register(svc, "champion")
        _register(svc, "rollback_target")
        event = svc.rollback(
            "champion",
            "rollback_target",
            RollbackReason.FP_SPIKE,
            triggered_by="qa_system",
            incident_id="inc_001",
        )
        rollbacks = svc.list_rollbacks()
        assert len(rollbacks) == 1
        assert rollbacks[0]["rollback_id"] == event.rollback_id
        assert rollbacks[0]["reason"] == "FP_SPIKE"
        assert rollbacks[0]["incident_id"] == "inc_001"

    def test_rollback_event_has_correct_fields(self) -> None:
        settings = load_settings()
        db = _tmp_db()
        svc = ModelRegistryService(settings, db)
        _register(svc, "old_model")
        _register(svc, "stable_model")
        event = svc.rollback(
            "old_model",
            "stable_model",
            RollbackReason.DRIFT_WORSENING,
            triggered_by="drift_monitor",
        )
        assert event.from_model_id == "old_model"
        assert event.to_model_id == "stable_model"
        assert event.reason == RollbackReason.DRIFT_WORSENING
        assert event.incident_id is None


# ---------------------------------------------------------------------------
# CanaryService
# ---------------------------------------------------------------------------


class TestCanaryServiceEvaluate:
    def test_both_conditions_pass(self) -> None:
        """disagreement_rate <= 0.05 and recall_delta >= -0.02 -> PASSED."""
        svc = _canary()
        result = svc.evaluate(
            "champ",
            "challenger",
            sample_count=100,
            disagreement_rate=0.03,
            critical_recall_delta=0.01,
        )
        assert result.status == CanaryStatus.PASSED

    def test_disagreement_rate_too_high(self) -> None:
        """disagreement_rate > 0.05 -> FAILED."""
        svc = _canary()
        result = svc.evaluate(
            "champ",
            "challenger",
            sample_count=100,
            disagreement_rate=0.10,
            critical_recall_delta=0.0,
        )
        assert result.status == CanaryStatus.FAILED

    def test_recall_delta_too_low(self) -> None:
        """critical_recall_delta < -0.02 -> FAILED."""
        svc = _canary()
        result = svc.evaluate(
            "champ",
            "challenger",
            sample_count=100,
            disagreement_rate=0.01,
            critical_recall_delta=-0.05,
        )
        assert result.status == CanaryStatus.FAILED

    def test_both_conditions_fail(self) -> None:
        svc = _canary()
        result = svc.evaluate(
            "champ",
            "challenger",
            sample_count=50,
            disagreement_rate=0.20,
            critical_recall_delta=-0.10,
        )
        assert result.status == CanaryStatus.FAILED

    def test_boundary_exactly_at_threshold_passes(self) -> None:
        """Exactly at threshold -> PASSED (boundary inclusive)."""
        svc = _canary()
        result = svc.evaluate(
            "champ",
            "challenger",
            sample_count=200,
            disagreement_rate=0.05,  # == max
            critical_recall_delta=-0.02,  # == min
        )
        assert result.status == CanaryStatus.PASSED

    def test_sample_count_zero_raises(self) -> None:
        svc = _canary()
        with pytest.raises(ValueError, match="sample_count"):
            svc.evaluate(
                "c",
                "d",
                sample_count=0,
                disagreement_rate=0.01,
                critical_recall_delta=0.0,
            )

    def test_disagreement_rate_above_1_raises(self) -> None:
        svc = _canary()
        with pytest.raises(ValueError, match="disagreement_rate"):
            svc.evaluate(
                "c",
                "d",
                sample_count=10,
                disagreement_rate=1.5,
                critical_recall_delta=0.0,
            )

    def test_evaluate_persists_record(self) -> None:
        svc = _canary()
        result = svc.evaluate(
            "champ",
            "chal",
            sample_count=100,
            disagreement_rate=0.02,
            critical_recall_delta=0.0,
        )
        rows = svc.list_canaries()
        assert any(r["canary_id"] == result.canary_id for r in rows)

    def test_list_canaries_filtered(self) -> None:
        svc = _canary()
        svc.evaluate(
            "champ_A", "chal_X", sample_count=10, disagreement_rate=0.01, critical_recall_delta=0.0
        )
        svc.evaluate(
            "champ_B", "chal_Y", sample_count=10, disagreement_rate=0.01, critical_recall_delta=0.0
        )
        rows = svc.list_canaries(champion_model_id="champ_A")
        assert len(rows) == 1
        assert rows[0]["champion_model_id"] == "champ_A"

    def test_canary_thresholds_from_yaml(self) -> None:
        """Verify thresholds round-trip from real app.yaml."""
        settings = load_settings()
        assert settings.drift.canary_max_disagreement_rate == pytest.approx(0.05)
        assert settings.drift.canary_min_recall_delta == pytest.approx(-0.02)
