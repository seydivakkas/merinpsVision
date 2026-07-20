"""Incident triage: 2-of-N confirming signal rule for drift incidents.

All computation is pure Python/dataclass -- no I/O, no GPU, no external deps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TriageEvidence:
    """Snapshot of the six independent confirming signals for a drift incident.

    Each field represents one independent evidence source.  A signal is True
    only when the corresponding monitor has passed its own decision threshold.

    Attributes:
        ewma_alert: EWMA downward-drift control limit breached.
        cusum_alert: CUSUM downward-drift decision level breached.
        psi_high: PSI >= high_threshold (typically 0.25).
        bbsd_significant: BBSD/MMD two-sample test is significant.
        uae_above_p99: UAE reconstruction error above 99th-percentile baseline.
        labeled_metric_confirmed: Ground-truth evaluation on labeled data
            confirms metric degradation.
    """

    ewma_alert: bool = False
    cusum_alert: bool = False
    psi_high: bool = False
    bbsd_significant: bool = False
    uae_above_p99: bool = False
    labeled_metric_confirmed: bool = False


def count_confirming_signals(evidence: TriageEvidence) -> int:
    """Return the number of True signals in *evidence*.

    Args:
        evidence: Populated ``TriageEvidence`` instance.

    Returns:
        Integer in [0, 6].
    """
    return sum(
        (
            evidence.ewma_alert,
            evidence.cusum_alert,
            evidence.psi_high,
            evidence.bbsd_significant,
            evidence.uae_above_p99,
            evidence.labeled_metric_confirmed,
        )
    )


def requires_retraining_request(
    evidence: TriageEvidence,
    *,
    minimum_signals: int = 2,
) -> bool:
    """Return True if enough independent signals confirm drift.

    A retraining request must NOT be opened until at least
    *minimum_signals* independent evidence sources corroborate the drift.
    This prevents false positives from noisy individual monitors.

    Args:
        evidence: Populated ``TriageEvidence`` instance.
        minimum_signals: Minimum number of True signals required.
            Read from ``configs/app.yaml -> drift.retraining_min_confirming_signals``;
            do NOT hardcode this value in call sites.

    Returns:
        True if ``count_confirming_signals(evidence) >= minimum_signals``.

    Raises:
        ValueError: If ``minimum_signals`` < 1.
    """
    if minimum_signals < 1:
        raise ValueError("minimum_signals must be >= 1.")
    return count_confirming_signals(evidence) >= minimum_signals
