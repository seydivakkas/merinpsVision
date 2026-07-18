"""SQLite connection management and idempotent schema migration."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from weavevision.domain.errors import DatabaseError

SCHEMA_VERSION = 1


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
