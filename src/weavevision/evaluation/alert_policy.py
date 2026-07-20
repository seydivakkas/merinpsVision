"""Alert policy helpers that read thresholds from the loaded Settings.

This module is a thin adapter: it translates ``DriftPolicyConfig`` values
into calls to the pure-logic functions in ``trend_monitor``,
``psi``, and ``incident_triage``.  It must remain free of GPU, I/O, and
database operations.

Typical usage::

    from weavevision.settings import load_settings
    from weavevision.evaluation.alert_policy import build_drift_alert_policy

    policy = build_drift_alert_policy(load_settings())
    signals_needed = policy.minimum_signals          # from config
    psi_severity = policy.psi_severity_for(0.18)     # "MEDIUM"
"""

from __future__ import annotations

from dataclasses import dataclass

from weavevision.evaluation.incident_triage import (
    TriageEvidence,
    count_confirming_signals,
    requires_retraining_request,
)
from weavevision.evaluation.psi import psi_severity
from weavevision.evaluation.trend_monitor import classify_sudden_drop
from weavevision.settings import DriftPolicyConfig, Settings


@dataclass(frozen=True)
class DriftAlertPolicy:
    """Config-bound alert policy for a single Settings instance.

    All thresholds are resolved at construction time from the YAML config
    and are immutable thereafter.  Call-sites must not inline threshold
    values -- always route through this object.

    Attributes:
        minimum_signals: Minimum confirming signals for a retraining request.
        ewma_lambda: EWMA smoothing constant.
        ewma_limit_sigma: EWMA lower-limit sigma multiplier.
        cusum_k_sigma: CUSUM allowance sigma multiplier.
        cusum_h_sigma: CUSUM decision-level sigma multiplier.
        psi_medium_threshold: PSI LOW/MEDIUM boundary.
        psi_high_threshold: PSI MEDIUM/HIGH boundary.
        sudden_drop_review_pp: P2_REVIEW sudden-drop threshold.
        sudden_drop_incident_pp: P1_INCIDENT sudden-drop threshold.
        sudden_drop_block_pp: P0_BLOCKED sudden-drop threshold.
    """

    minimum_signals: int
    ewma_lambda: float
    ewma_limit_sigma: float
    cusum_k_sigma: float
    cusum_h_sigma: float
    psi_medium_threshold: float
    psi_high_threshold: float
    sudden_drop_review_pp: float
    sudden_drop_incident_pp: float
    sudden_drop_block_pp: float

    # ------------------------------------------------------------------
    # Convenience wrappers -- keep all threshold references in one place
    # ------------------------------------------------------------------

    def psi_severity_for(self, psi_value: float) -> str:
        """Return ``'LOW'``, ``'MEDIUM'`` or ``'HIGH'`` for *psi_value*.

        Delegates to ``evaluation.psi.psi_severity`` with config thresholds.
        """
        return psi_severity(
            psi_value,
            medium_threshold=self.psi_medium_threshold,
            high_threshold=self.psi_high_threshold,
        )

    def classify_drop(self, current: float, prior: float) -> str:
        """Classify sudden drop using config thresholds.

        Returns ``'P0_BLOCKED'``, ``'P1_INCIDENT'``, ``'P2_REVIEW'``
        or ``'STABLE'``.
        """
        return classify_sudden_drop(
            current,
            prior,
            review_drop_pp=self.sudden_drop_review_pp,
            incident_drop_pp=self.sudden_drop_incident_pp,
            block_drop_pp=self.sudden_drop_block_pp,
        )

    def needs_retraining(self, evidence: TriageEvidence) -> bool:
        """Return True if evidence satisfies the configured minimum signals."""
        return requires_retraining_request(evidence, minimum_signals=self.minimum_signals)

    def signal_count(self, evidence: TriageEvidence) -> int:
        """Return the number of True confirming signals in *evidence*."""
        return count_confirming_signals(evidence)


def build_drift_alert_policy(settings: Settings) -> DriftAlertPolicy:
    """Construct a ``DriftAlertPolicy`` from validated ``Settings``.

    Args:
        settings: Loaded and validated application settings.

    Returns:
        An immutable ``DriftAlertPolicy`` backed by the YAML configuration.
    """
    cfg: DriftPolicyConfig = settings.drift
    return DriftAlertPolicy(
        minimum_signals=cfg.retraining_min_confirming_signals,
        ewma_lambda=cfg.ewma_lambda,
        ewma_limit_sigma=cfg.ewma_limit_sigma,
        cusum_k_sigma=cfg.cusum_k_sigma,
        cusum_h_sigma=cfg.cusum_h_sigma,
        psi_medium_threshold=cfg.psi_medium_threshold,
        psi_high_threshold=cfg.psi_high_threshold,
        sudden_drop_review_pp=cfg.sudden_drop_review_pp,
        sudden_drop_incident_pp=cfg.sudden_drop_incident_pp,
        sudden_drop_block_pp=cfg.sudden_drop_block_pp,
    )
