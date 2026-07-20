"""Incident lifecycle service for drift governance.

Owns the decision boundary between a confirmed drift window and an
audit-grade incident record.  Delegates triage logic to
``evaluation.incident_triage`` and persistence to ``IncidentRepository``.

No GPU, no file I/O beyond the database write.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from weavevision.domain.enums import IncidentPriority
from weavevision.domain.schemas import DriftWindow, IncidentRecord, TriageDecision
from weavevision.evaluation.incident_triage import (
    TriageEvidence,
    count_confirming_signals,
    requires_retraining_request,
)
from weavevision.persistence.database import Database
from weavevision.persistence.repositories import IncidentRepository
from weavevision.settings import Settings


class IncidentService:
    """Open, query, and triage drift incidents.

    Args:
        settings: Loaded application settings.
        database: Initialised ``Database`` instance.
    """

    def __init__(self, settings: Settings, database: Database) -> None:
        self._settings = settings
        self._repo = IncidentRepository(database)
        self._min_signals: int = settings.drift.retraining_min_confirming_signals

    def evaluate(
        self,
        window: DriftWindow,
        evidence: TriageEvidence,
        *,
        reviewer: str | None = None,
    ) -> TriageDecision:
        """Evaluate evidence signals and return a triage decision record.

        Does NOT persist the decision.  Callers may inspect ``TriageDecision``
        and decide whether to call ``open()`` based on the outcome.

        Args:
            window: The drift window that triggered the evaluation.
            evidence: Populated ``TriageEvidence`` from upstream monitors.
            reviewer: Optional reviewer identifier for the audit trail.

        Returns:
            A populated ``TriageDecision`` schema instance.
        """
        signal_count = count_confirming_signals(evidence)
        evidence_dict = {
            "ewma_alert": evidence.ewma_alert,
            "cusum_alert": evidence.cusum_alert,
            "psi_high": evidence.psi_high,
            "bbsd_significant": evidence.bbsd_significant,
            "uae_above_p99": evidence.uae_above_p99,
            "labeled_metric_confirmed": evidence.labeled_metric_confirmed,
        }
        return TriageDecision(
            decision_id=f"dec_{uuid4().hex[:12]}",
            incident_id="PENDING",
            confirming_signal_count=signal_count,
            evidence=evidence_dict,
            reviewer=reviewer,
            created_at=datetime.now(UTC),
        )

    def open(
        self,
        window: DriftWindow,
        evidence: TriageEvidence,
        *,
        root_cause: str | None = None,
    ) -> IncidentRecord | None:
        """Open an incident if evidence meets the configured minimum signals.

        Returns ``None`` without persisting anything when the signal count is
        below the ``retraining_min_confirming_signals`` threshold from
        ``configs/app.yaml``.

        Args:
            window: The triggering drift window (must already be persisted).
            evidence: Populated ``TriageEvidence``.
            root_cause: Optional root-cause annotation.

        Returns:
            Persisted ``IncidentRecord`` or ``None`` if threshold not met.
        """
        if not requires_retraining_request(evidence, minimum_signals=self._min_signals):
            return None

        signal_count = count_confirming_signals(evidence)
        priority = _signal_count_to_priority(signal_count)

        incident = IncidentRecord(
            incident_id=f"inc_{uuid4().hex[:12]}",
            priority=priority,
            drift_pattern=window.drift_pattern,
            root_cause=root_cause,
            affected_window_id=window.window_id,
            model_id=window.model_id,
            threshold_id=window.threshold_id,
            created_at=datetime.now(UTC),
        )
        self._repo.save(incident)
        return incident

    def list_open(self) -> list[dict[str, object]]:
        """Return all unresolved incidents from the database.

        Returns:
            List of raw row dictionaries ordered by ``created_at DESC``.
        """
        return self._repo.list_open()


def _signal_count_to_priority(signal_count: int) -> IncidentPriority:
    """Map a confirming signal count to an ``IncidentPriority``.

    Mapping (initial policy -- adjust in ``configs/app.yaml`` when calibrated):
        1   -> INFO
        2   -> P2_REVIEW
        3-4 -> P1_INCIDENT
        5-6 -> P0_BLOCKED

    Args:
        signal_count: Number of True confirming signals (0-6).

    Returns:
        Appropriate ``IncidentPriority``.
    """
    if signal_count >= 5:
        return IncidentPriority.P0_BLOCKED
    if signal_count >= 3:
        return IncidentPriority.P1_INCIDENT
    if signal_count >= 2:
        return IncidentPriority.P2_REVIEW
    return IncidentPriority.INFO
