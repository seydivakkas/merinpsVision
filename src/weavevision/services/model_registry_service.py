"""Model registry state machine and rollback audit service.

Valid state transitions:
    CANDIDATE -> VALIDATED
    VALIDATED -> ACTIVE_BENCHMARK
    ACTIVE_BENCHMARK -> ACTIVE_COMPANY_PILOT
    Any -> RETIRED  (graceful deprecation)
    Any -> REJECTED (failed evaluation)

Rollback: promotes `to_model_id` to ACTIVE_BENCHMARK and demotes
`from_model_id` to RETIRED, writing a ``RollbackEvent`` audit record.

No GPU, no file I/O beyond the database writes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from weavevision.domain.enums import ModelStatus, RollbackReason
from weavevision.domain.errors import DatabaseError
from weavevision.domain.schemas import RollbackEvent
from weavevision.persistence.database import Database
from weavevision.settings import Settings

# Valid one-step state transitions (from -> set of allowed to-states)
_ALLOWED_TRANSITIONS: dict[ModelStatus, set[ModelStatus]] = {
    ModelStatus.CANDIDATE: {
        ModelStatus.VALIDATED,
        ModelStatus.REJECTED,
        ModelStatus.RETIRED,
    },
    ModelStatus.VALIDATED: {
        ModelStatus.ACTIVE_BENCHMARK,
        ModelStatus.RETIRED,
        ModelStatus.REJECTED,
    },
    ModelStatus.ACTIVE_BENCHMARK: {
        ModelStatus.ACTIVE_COMPANY_PILOT,
        ModelStatus.RETIRED,
        ModelStatus.REJECTED,
    },
    ModelStatus.ACTIVE_COMPANY_PILOT: {ModelStatus.RETIRED, ModelStatus.REJECTED},
    ModelStatus.RETIRED: set(),
    ModelStatus.REJECTED: set(),
}


class ModelRegistryService:
    """Manage model lifecycle state transitions and rollback events.

    Args:
        settings: Loaded application settings (unused today, kept for DI
            consistency and future threshold reads).
        database: Initialised ``Database`` instance.
    """

    def __init__(self, settings: Settings, database: Database) -> None:
        self._settings = settings
        self._db = database

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        model_id: str,
        algorithm: str,
        artifact_path: str,
        artifact_sha256: str,
        *,
        metrics_path: str | None = None,
    ) -> None:
        """Register a new model as CANDIDATE in the registry.

        Args:
            model_id: Unique model identifier.
            algorithm: Model family (e.g. ``'patchcore'``).
            artifact_path: Absolute path to the model weights file.
            artifact_sha256: SHA-256 of the artifact file.
            metrics_path: Optional path to the evaluation metrics JSON.
        """
        with self._db.connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO models (
                    model_id, algorithm, status, artifact_path,
                    artifact_sha256, metrics_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    model_id,
                    algorithm,
                    ModelStatus.CANDIDATE.value,
                    artifact_path,
                    artifact_sha256,
                    metrics_path,
                    datetime.now(UTC).isoformat(),
                ),
            )

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def promote(
        self,
        model_id: str,
        to_status: ModelStatus,
        *,
        reason: str | None = None,
    ) -> ModelStatus:
        """Transition *model_id* to *to_status*.

        Args:
            model_id: Target model identifier.
            to_status: Desired new status (must be a valid one-step transition).
            reason: Optional human-readable reason logged in the DB.

        Returns:
            The new ``ModelStatus`` after the transition.

        Raises:
            ValueError: If the transition is not permitted.
            DatabaseError: If *model_id* does not exist in the registry.
        """
        current = self._get_status(model_id)
        allowed = _ALLOWED_TRANSITIONS.get(current, set())
        if to_status not in allowed:
            raise ValueError(
                f"Transition {current.value} -> {to_status.value} is not allowed. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        self._set_status(model_id, to_status)
        return to_status

    def demote(self, model_id: str) -> ModelStatus:
        """Retire *model_id* by transitioning it to RETIRED.

        Any non-terminal status can be retired.

        Args:
            model_id: Model to retire.

        Returns:
            ``ModelStatus.RETIRED``.

        Raises:
            ValueError: If already in a terminal state (RETIRED or REJECTED).
            DatabaseError: If *model_id* does not exist.
        """
        return self.promote(model_id, ModelStatus.RETIRED)

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(
        self,
        from_model_id: str,
        to_model_id: str,
        reason: RollbackReason,
        triggered_by: str,
        *,
        incident_id: str | None = None,
    ) -> RollbackEvent:
        """Roll back production to *to_model_id* and audit the event.

        Promotes *to_model_id* to ACTIVE_BENCHMARK and demotes
        *from_model_id* to RETIRED.  Writes a ``RollbackEvent`` record.

        Args:
            from_model_id: Current model being replaced.
            to_model_id: Model to restore as ACTIVE_BENCHMARK.
            reason: Enum reason for the rollback.
            triggered_by: Identifier of the user or automation that triggered.
            incident_id: Optional linked incident ID for audit trail.

        Returns:
            The persisted ``RollbackEvent``.

        Raises:
            ValueError: If either model does not exist or transitions fail.
        """
        # Promote rollback target to ACTIVE_BENCHMARK (allow from any state)
        self._set_status(to_model_id, ModelStatus.ACTIVE_BENCHMARK)
        # Retire the failing model
        self._set_status(from_model_id, ModelStatus.RETIRED)

        event = RollbackEvent(
            rollback_id=f"rb_{uuid4().hex[:12]}",
            from_model_id=from_model_id,
            to_model_id=to_model_id,
            reason=reason,
            triggered_by=triggered_by,
            incident_id=incident_id,
            created_at=datetime.now(UTC),
        )
        self._persist_rollback(event)
        return event

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_status(self, model_id: str) -> ModelStatus:
        """Return the current ``ModelStatus`` for *model_id*.

        Raises:
            DatabaseError: If *model_id* is not found.
        """
        return self._get_status(model_id)

    def list_models(self, *, status: ModelStatus | None = None) -> list[dict[str, object]]:
        """Return all models, optionally filtered by status.

        Args:
            status: If provided, only models with this status are returned.

        Returns:
            List of raw row dicts ordered by ``created_at DESC``.
        """
        with self._db.connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM models WHERE status = ? ORDER BY created_at DESC",
                    (status.value,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def list_rollbacks(self) -> list[dict[str, object]]:
        """Return all rollback events ordered by ``created_at DESC``."""
        with self._db.connect() as conn:
            rows = conn.execute("SELECT * FROM rollback_events ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_status(self, model_id: str) -> ModelStatus:
        with self._db.connect() as conn:
            row = conn.execute(
                "SELECT status FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
        if row is None:
            raise DatabaseError(f"model_id '{model_id}' not found in registry")
        return ModelStatus(row["status"])

    def _set_status(self, model_id: str, status: ModelStatus) -> None:
        with self._db.connect() as conn:
            conn.execute(
                "UPDATE models SET status = ? WHERE model_id = ?",
                (status.value, model_id),
            )

    def _persist_rollback(self, event: RollbackEvent) -> None:
        with self._db.connect() as conn:
            conn.execute(
                """INSERT INTO rollback_events (
                    rollback_id, from_model_id, to_model_id, reason,
                    triggered_by, incident_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.rollback_id,
                    event.from_model_id,
                    event.to_model_id,
                    event.reason.value,
                    event.triggered_by,
                    event.incident_id,
                    event.created_at.isoformat(),
                ),
            )
