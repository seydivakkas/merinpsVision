"""Typed repositories for analyses and expert feedback."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from weavevision.domain.enums import FeedbackVerdict
from weavevision.domain.schemas import AnalysisResult
from weavevision.persistence.database import Database


class AnalysisRepository:
    """Persist and retrieve audit-ready analysis summaries."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def save(self, result: AnalysisResult, result_json_path: Path) -> None:
        """Insert or replace one analysis summary."""
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO analyses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.analysis_id,
                    result.created_at.isoformat(),
                    result.source.filename,
                    result.source.sha256,
                    result.prediction.decision.value,
                    result.prediction.review_priority.value,
                    result.prediction.raw_anomaly_score,
                    result.prediction.normalized_anomaly_score,
                    result.prediction.anomaly_area_ratio,
                    result.prediction.region_count,
                    result.model.model_id if result.model else None,
                    result.threshold.threshold_id if result.threshold else None,
                    result.quality_gate.status.value,
                    result.timing_ms.total,
                    str(result_json_path),
                ),
            )

    def list_recent(self, limit: int = 100) -> list[dict[str, object]]:
        """Return recent analyses as plain dictionaries."""
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]


class FeedbackRepository:
    """Persist quality expert feedback linked to an analysis."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def save(
        self,
        analysis_id: str,
        reviewer: str,
        verdict: FeedbackVerdict,
        comment: str | None = None,
        defect_type: str | None = None,
        corrected_mask_path: Path | None = None,
    ) -> str:
        """Persist feedback and return its generated identifier."""
        feedback_id = f"fb_{uuid4().hex}"
        with self.database.connect() as connection:
            connection.execute(
                "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    feedback_id,
                    analysis_id,
                    datetime.now(UTC).isoformat(),
                    reviewer,
                    verdict.value,
                    defect_type,
                    comment,
                    str(corrected_mask_path) if corrected_mask_path else None,
                ),
            )
        return feedback_id
