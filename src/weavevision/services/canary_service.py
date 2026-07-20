"""Canary evaluation service: champion-vs-challenger comparison.

Reads decision thresholds from ``configs/app.yaml -> drift``
(canary_max_disagreement_rate, canary_min_recall_delta).
No GPU, no file I/O beyond the database write.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from weavevision.domain.enums import CanaryStatus
from weavevision.domain.schemas import CanaryEvaluation
from weavevision.persistence.database import Database
from weavevision.settings import Settings


class CanaryService:
    """Run and persist champion-vs-challenger canary comparisons.

    Decision rule (initial policy from ``configs/app.yaml -> drift``):
        PASSED  when:
            disagreement_rate <= canary_max_disagreement_rate
            AND critical_recall_delta >= canary_min_recall_delta
        FAILED  otherwise.

    Args:
        settings: Loaded application settings.
        database: Initialised ``Database`` instance (migrate() called).
    """

    def __init__(self, settings: Settings, database: Database) -> None:
        self._settings = settings
        self._db = database
        self._max_disagreement: float = settings.drift.canary_max_disagreement_rate
        self._min_recall_delta: float = settings.drift.canary_min_recall_delta

    def evaluate(
        self,
        champion_model_id: str,
        challenger_model_id: str,
        *,
        sample_count: int,
        disagreement_rate: float,
        critical_recall_delta: float,
        latency_p95_ms: float = 0.0,
    ) -> CanaryEvaluation:
        """Evaluate challenger performance and persist a canary record.

        Args:
            champion_model_id: Registry ID of the current production model.
            challenger_model_id: Registry ID of the model under evaluation.
            sample_count: Number of images processed in the parallel run.
            disagreement_rate: Fraction of images where decisions differed.
            critical_recall_delta: recall(challenger) - recall(champion).
                Negative means challenger has worse recall.
            latency_p95_ms: 95th-percentile latency for the challenger (ms).

        Returns:
            Persisted ``CanaryEvaluation`` instance with PASSED/FAILED status.

        Raises:
            ValueError: If ``sample_count < 1`` or rates outside [0, 1].
        """
        if sample_count < 1:
            raise ValueError("sample_count must be >= 1")
        if not 0.0 <= disagreement_rate <= 1.0:
            raise ValueError("disagreement_rate must be in [0, 1]")

        passed = (
            disagreement_rate <= self._max_disagreement
            and critical_recall_delta >= self._min_recall_delta
        )
        status = CanaryStatus.PASSED if passed else CanaryStatus.FAILED

        canary = CanaryEvaluation(
            canary_id=f"can_{uuid4().hex[:12]}",
            champion_model_id=champion_model_id,
            challenger_model_id=challenger_model_id,
            sample_count=sample_count,
            disagreement_rate=disagreement_rate,
            critical_recall_delta=critical_recall_delta,
            latency_p95_ms=latency_p95_ms,
            status=status,
            created_at=datetime.now(UTC),
        )
        self._persist(canary)
        return canary

    def list_canaries(
        self,
        *,
        champion_model_id: str | None = None,
    ) -> list[dict[str, object]]:
        """Return canary evaluation records, optionally filtered by champion.

        Args:
            champion_model_id: If provided, only return canaries for this
                champion. Otherwise return all records.

        Returns:
            List of raw row dictionaries ordered by ``created_at DESC``.
        """
        with self._db.connect() as conn:
            if champion_model_id:
                rows = conn.execute(
                    "SELECT * FROM canary_runs WHERE champion_model_id = ?"
                    " ORDER BY created_at DESC",
                    (champion_model_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM canary_runs ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def _persist(self, canary: CanaryEvaluation) -> None:
        with self._db.connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO canary_runs (
                    canary_id, champion_model_id, challenger_model_id,
                    sample_count, disagreement_rate, critical_recall_delta,
                    latency_p95_ms, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    canary.canary_id,
                    canary.champion_model_id,
                    canary.challenger_model_id,
                    canary.sample_count,
                    canary.disagreement_rate,
                    canary.critical_recall_delta,
                    canary.latency_p95_ms,
                    canary.status.value,
                    canary.created_at.isoformat(),
                ),
            )
