"""Typed repositories for analyses and expert feedback."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from weavevision.domain.enums import FeedbackVerdict
from weavevision.domain.schemas import AnalysisResult, DriftWindow, IncidentRecord
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


# ---------------------------------------------------------------------------
# Drift lifecycle repositories (M4)
# ---------------------------------------------------------------------------


class DriftWindowRepository:
    """Persist and retrieve drift monitoring window records."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def save(self, window: DriftWindow) -> None:
        """Insert or replace one drift window audit record.

        Args:
            window: Populated ``DriftWindow`` schema instance.
        """
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO drift_windows (
                    window_id, model_id, threshold_id, metric_name,
                    window_start, window_end, metric_value, ewma_value,
                    cusum_value, psi_value, bbsd_mmd, uae_p95_error,
                    trend_status, drift_pattern, source_manifest_sha256,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    window.window_id,
                    window.model_id,
                    window.threshold_id,
                    window.metric_name,
                    window.window_start.isoformat(),
                    window.window_end.isoformat(),
                    window.metric_value,
                    window.ewma_value,
                    window.cusum_value,
                    window.psi_value,
                    window.bbsd_mmd,
                    window.uae_p95_error,
                    window.trend_status.value,
                    window.drift_pattern.value,
                    window.source_manifest_sha256,
                    window.created_at.isoformat(),
                ),
            )

    def list_recent(self, limit: int = 100) -> list[dict[str, object]]:
        """Return recent drift windows ordered by creation time."""
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM drift_windows ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]


class IncidentRepository:
    """Persist and retrieve drift incident records."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def save(self, incident: IncidentRecord) -> None:
        """Insert or replace one drift incident audit record.

        Args:
            incident: Populated ``IncidentRecord`` schema instance.
        """
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO drift_incidents (
                    incident_id, priority, drift_pattern, root_cause,
                    affected_window_id, model_id, threshold_id,
                    action_taken, resolved_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    incident.incident_id,
                    incident.priority.value,
                    incident.drift_pattern.value,
                    incident.root_cause,
                    incident.affected_window_id,
                    incident.model_id,
                    incident.threshold_id,
                    incident.action_taken,
                    incident.resolved_at.isoformat() if incident.resolved_at else None,
                    incident.created_at.isoformat(),
                ),
            )

    def list_open(self) -> list[dict[str, object]]:
        """Return all unresolved incidents (resolved_at IS NULL)."""
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM drift_incidents WHERE resolved_at IS NULL ORDER BY created_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]
