"""SQLite connection management and idempotent schema migration."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from weavevision.domain.errors import DatabaseError

SCHEMA_VERSION = 2


class Database:
    """Small SQLite unit-of-work boundary for local audit records."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a transactional SQLite connection.

        Yields:
            Configured connection with foreign keys and WAL enabled.

        Raises:
            DatabaseError: If opening or committing the transaction fails.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            yield connection
            connection.commit()
        except sqlite3.Error as exc:
            if connection is not None:
                connection.rollback()
            raise DatabaseError(f"SQLite operation failed: {exc}") from exc
        finally:
            if connection is not None:
                connection.close()

    def migrate(self) -> None:
        """Create or upgrade the idempotent local schema."""
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analyses (
                    analysis_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    source_filename TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    review_priority TEXT NOT NULL,
                    raw_score REAL NOT NULL,
                    normalized_score REAL,
                    anomaly_area_ratio REAL NOT NULL,
                    region_count INTEGER NOT NULL,
                    model_id TEXT,
                    threshold_id TEXT,
                    quality_status TEXT NOT NULL,
                    total_latency_ms REAL NOT NULL,
                    result_json_path TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    defect_type_optional TEXT,
                    comment TEXT,
                    corrected_mask_path_optional TEXT,
                    FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id)
                );
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    algorithm TEXT NOT NULL,
                    status TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    artifact_sha256 TEXT NOT NULL,
                    metrics_path TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS thresholds (
                    threshold_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    image_threshold REAL NOT NULL,
                    pixel_threshold REAL NOT NULL,
                    method TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                -- ---------------------------------------------------------
                -- Drift lifecycle tables (M4, schema_version=2)
                -- All CREATE TABLE statements are idempotent.
                -- ---------------------------------------------------------
                CREATE TABLE IF NOT EXISTS drift_windows (
                    window_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    threshold_id TEXT,
                    metric_name TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    metric_value REAL,
                    ewma_value REAL,
                    cusum_value REAL,
                    psi_value REAL,
                    bbsd_mmd REAL,
                    uae_p95_error REAL,
                    trend_status TEXT NOT NULL,
                    drift_pattern TEXT NOT NULL,
                    source_manifest_sha256 TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS drift_incidents (
                    incident_id TEXT PRIMARY KEY,
                    priority TEXT NOT NULL,
                    drift_pattern TEXT NOT NULL,
                    root_cause TEXT,
                    affected_window_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    threshold_id TEXT,
                    action_taken TEXT,
                    resolved_at TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (affected_window_id)
                        REFERENCES drift_windows(window_id)
                );
                CREATE TABLE IF NOT EXISTS labeling_queue (
                    item_id TEXT PRIMARY KEY,
                    image_sha256 TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    priority_bucket TEXT NOT NULL
                        CHECK (priority_bucket IN ('P0','P1','P2','P3')),
                    selection_reason TEXT NOT NULL,
                    drift_score REAL,
                    uncertainty_score REAL,
                    assigned_reviewer TEXT,
                    verdict TEXT,
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS canary_runs (
                    canary_id TEXT PRIMARY KEY,
                    champion_model_id TEXT NOT NULL,
                    challenger_model_id TEXT NOT NULL,
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    disagreement_rate REAL NOT NULL DEFAULT 0.0,
                    critical_recall_delta REAL NOT NULL DEFAULT 0.0,
                    latency_p95_ms REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rollback_events (
                    rollback_id TEXT PRIMARY KEY,
                    from_model_id TEXT NOT NULL,
                    to_model_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    triggered_by TEXT NOT NULL,
                    incident_id TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def health(self) -> bool:
        """Return whether the database can migrate and answer a query."""
        self.migrate()
        with self.connect() as connection:
            row = connection.execute("SELECT 1 AS ok").fetchone()
            return bool(row and row["ok"] == 1)
