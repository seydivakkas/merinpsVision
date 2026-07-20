"""Unit tests for evaluation/incident_triage.py and alert_policy.py (M3).

Covers:
- TriageEvidence construction (all False, all True, partial)
- count_confirming_signals: 0, 1, 2, 6
- requires_retraining_request: boundary at default minimum_signals=2
- requires_retraining_request: minimum_signals < 1 -> ValueError
- DriftAlertPolicy wrapper methods (psi, drop, needs_retraining, count)
- build_drift_alert_policy: loads thresholds from Settings
"""

from __future__ import annotations

import pytest

from weavevision.evaluation.alert_policy import (
    DriftAlertPolicy,
    build_drift_alert_policy,
)
from weavevision.evaluation.incident_triage import (
    TriageEvidence,
    count_confirming_signals,
    requires_retraining_request,
)
from weavevision.settings import load_settings

# ---------------------------------------------------------------------------
# TriageEvidence
# ---------------------------------------------------------------------------


class TestTriageEvidence:
    def test_all_false_default(self) -> None:
        ev = TriageEvidence()
        assert not any(
            (
                ev.ewma_alert,
                ev.cusum_alert,
                ev.psi_high,
                ev.bbsd_significant,
                ev.uae_above_p99,
                ev.labeled_metric_confirmed,
            )
        )

    def test_all_true(self) -> None:
        ev = TriageEvidence(
            ewma_alert=True,
            cusum_alert=True,
            psi_high=True,
            bbsd_significant=True,
            uae_above_p99=True,
            labeled_metric_confirmed=True,
        )
        assert all(
            (
                ev.ewma_alert,
                ev.cusum_alert,
                ev.psi_high,
                ev.bbsd_significant,
                ev.uae_above_p99,
                ev.labeled_metric_confirmed,
            )
        )

    def test_partial_construction(self) -> None:
        ev = TriageEvidence(ewma_alert=True, psi_high=True)
        assert ev.ewma_alert is True
        assert ev.psi_high is True
        assert ev.cusum_alert is False

    def test_frozen_immutable(self) -> None:
        """TriageEvidence must be immutable (frozen=True)."""
        ev = TriageEvidence(ewma_alert=True)
        with pytest.raises((AttributeError, TypeError)):
            ev.ewma_alert = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# count_confirming_signals
# ---------------------------------------------------------------------------


class TestCountConfirmingSignals:
    def test_zero_signals(self) -> None:
        assert count_confirming_signals(TriageEvidence()) == 0

    def test_one_signal(self) -> None:
        assert count_confirming_signals(TriageEvidence(ewma_alert=True)) == 1

    def test_two_signals(self) -> None:
        ev = TriageEvidence(ewma_alert=True, cusum_alert=True)
        assert count_confirming_signals(ev) == 2

    def test_three_signals(self) -> None:
        ev = TriageEvidence(ewma_alert=True, psi_high=True, uae_above_p99=True)
        assert count_confirming_signals(ev) == 3

    def test_six_signals(self) -> None:
        ev = TriageEvidence(
            ewma_alert=True,
            cusum_alert=True,
            psi_high=True,
            bbsd_significant=True,
            uae_above_p99=True,
            labeled_metric_confirmed=True,
        )
        assert count_confirming_signals(ev) == 6

    def test_count_uses_all_fields(self) -> None:
        """Verifies each field independently contributes +1."""
        for field in (
            "ewma_alert",
            "cusum_alert",
            "psi_high",
            "bbsd_significant",
            "uae_above_p99",
            "labeled_metric_confirmed",
        ):
            ev = TriageEvidence(**{field: True})
            assert count_confirming_signals(ev) == 1, f"Field {field!r} not counted"


# ---------------------------------------------------------------------------
# requires_retraining_request
# ---------------------------------------------------------------------------


class TestRequiresRetrainingRequest:
    def test_zero_signals_no_retraining(self) -> None:
        assert not requires_retraining_request(TriageEvidence())

    def test_one_signal_below_default_minimum(self) -> None:
        ev = TriageEvidence(ewma_alert=True)
        assert not requires_retraining_request(ev)  # default minimum=2

    def test_two_signals_meets_default_minimum(self) -> None:
        ev = TriageEvidence(ewma_alert=True, cusum_alert=True)
        assert requires_retraining_request(ev)

    def test_six_signals_always_requires(self) -> None:
        ev = TriageEvidence(
            ewma_alert=True,
            cusum_alert=True,
            psi_high=True,
            bbsd_significant=True,
            uae_above_p99=True,
            labeled_metric_confirmed=True,
        )
        assert requires_retraining_request(ev)

    def test_custom_minimum_1(self) -> None:
        ev = TriageEvidence(ewma_alert=True)
        assert requires_retraining_request(ev, minimum_signals=1)

    def test_custom_minimum_3_not_met(self) -> None:
        ev = TriageEvidence(ewma_alert=True, cusum_alert=True)
        assert not requires_retraining_request(ev, minimum_signals=3)

    def test_custom_minimum_3_met(self) -> None:
        ev = TriageEvidence(ewma_alert=True, cusum_alert=True, psi_high=True)
        assert requires_retraining_request(ev, minimum_signals=3)

    def test_minimum_signals_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="minimum_signals"):
            requires_retraining_request(TriageEvidence(), minimum_signals=0)

    def test_minimum_signals_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="minimum_signals"):
            requires_retraining_request(TriageEvidence(), minimum_signals=-1)

    def test_exact_boundary_one_below(self) -> None:
        """One signal below minimum -> False."""
        ev = TriageEvidence(ewma_alert=True, cusum_alert=True)  # 2 signals
        assert not requires_retraining_request(ev, minimum_signals=3)

    def test_exact_boundary_meets(self) -> None:
        """Exactly at minimum -> True."""
        ev = TriageEvidence(ewma_alert=True, cusum_alert=True)
        assert requires_retraining_request(ev, minimum_signals=2)


# ---------------------------------------------------------------------------
# DriftAlertPolicy wrappers
# ---------------------------------------------------------------------------


class TestDriftAlertPolicy:
    """Tests for DriftAlertPolicy convenience wrappers."""

    def _policy(self) -> DriftAlertPolicy:
        return DriftAlertPolicy(
            minimum_signals=2,
            ewma_lambda=0.25,
            ewma_limit_sigma=3.0,
            cusum_k_sigma=0.25,
            cusum_h_sigma=4.0,
            psi_medium_threshold=0.10,
            psi_high_threshold=0.25,
            sudden_drop_review_pp=2.0,
            sudden_drop_incident_pp=5.0,
            sudden_drop_block_pp=10.0,
        )

    def test_psi_severity_low(self) -> None:
        assert self._policy().psi_severity_for(0.05) == "LOW"

    def test_psi_severity_medium(self) -> None:
        assert self._policy().psi_severity_for(0.15) == "MEDIUM"

    def test_psi_severity_high(self) -> None:
        assert self._policy().psi_severity_for(0.30) == "HIGH"

    def test_classify_drop_stable(self) -> None:
        assert self._policy().classify_drop(10.0, 10.5) == "STABLE"

    def test_classify_drop_review(self) -> None:
        assert self._policy().classify_drop(8.0, 10.0) == "P2_REVIEW"

    def test_classify_drop_incident(self) -> None:
        assert self._policy().classify_drop(5.0, 10.0) == "P1_INCIDENT"

    def test_classify_drop_blocked(self) -> None:
        assert self._policy().classify_drop(-1.0, 10.0) == "P0_BLOCKED"

    def test_needs_retraining_false(self) -> None:
        ev = TriageEvidence(ewma_alert=True)  # 1 < minimum=2
        assert not self._policy().needs_retraining(ev)

    def test_needs_retraining_true(self) -> None:
        ev = TriageEvidence(ewma_alert=True, cusum_alert=True)
        assert self._policy().needs_retraining(ev)

    def test_signal_count_delegation(self) -> None:
        ev = TriageEvidence(ewma_alert=True, psi_high=True, uae_above_p99=True)
        assert self._policy().signal_count(ev) == 3

    def test_frozen_immutable(self) -> None:
        policy = self._policy()
        with pytest.raises((AttributeError, TypeError)):
            policy.minimum_signals = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_drift_alert_policy: round-trip through real YAML
# ---------------------------------------------------------------------------


class TestBuildDriftAlertPolicy:
    """Integration: build policy from the real configs/app.yaml."""

    def test_loads_from_real_config(self) -> None:
        """build_drift_alert_policy reads correct thresholds from YAML."""
        settings = load_settings()
        policy = build_drift_alert_policy(settings)

        # Values must match app.yaml drift section defaults
        assert policy.minimum_signals == 2
        assert policy.ewma_lambda == pytest.approx(0.25)
        assert policy.cusum_h_sigma == pytest.approx(4.0)
        assert policy.psi_medium_threshold == pytest.approx(0.10)
        assert policy.psi_high_threshold == pytest.approx(0.25)
        assert policy.sudden_drop_block_pp == pytest.approx(10.0)

    def test_policy_is_functional_after_build(self) -> None:
        """Policy methods work correctly after loading from YAML."""
        settings = load_settings()
        policy = build_drift_alert_policy(settings)

        assert policy.psi_severity_for(0.05) == "LOW"
        assert policy.classify_drop(0.0, 10.0) == "P0_BLOCKED"
        ev = TriageEvidence(ewma_alert=True, cusum_alert=True)
        assert policy.needs_retraining(ev) is True
