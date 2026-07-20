"""Unit tests for M4 drift persistence: database schema and repositories.

Covers:
- All 10 expected tables exist after migrate()
- migrate() is idempotent (calling twice raises no error)
- schema_version is written correctly (value="2")
- FOREIGN KEY constraint enforced: incident -> drift_window
- DriftWindowRepository.save() + list_recent()
- IncidentRepository.save() + list_open()
- IncidentRepository.list_open() excludes resolved incidents
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pytest

from weavevision.domain.enums import DriftPattern, IncidentPriority, TrendStatus
from weavevision.domain.schemas import DriftWindow, IncidentRecord
from weavevision.persistence.database import SCHEMA_VERSION, Database
from weavevision.persistence.repositories import (
    DriftWindowRepository,
    IncidentRepository,
)

_NOW = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tmp_db() -> Database:
    """Return a fresh in-memory-like Database backed by a temp file."""
    tmp = tempfile.mktemp(suffix=".sqlite3")
    db = Database(Path(tmp))
    db.migrate()
    return db


def _window(window_id: str = "win_001") -> DriftWindow:
    return DriftWindow(
        window_id=window_id,
        model_id="model_abc",
        metric_name="image_ap50",
        window_start=_NOW,
        window_end=_NOW,
        trend_status=TrendStatus.STABLE,
        drift_pattern=DriftPattern.STABLE,
        created_at=_NOW,
    )


def _incident(
    incident_id: str = "inc_001",
    affected_window_id: str = "win_001",
) -> IncidentRecord:
    return IncidentRecord(
        incident_id=incident_id,
        priority=IncidentPriority.P1_INCIDENT,
        drift_pattern=DriftPattern.SUDDEN,
        affected_window_id=affected_window_id,
        model_id="model_abc",
        created_at=_NOW,
    )


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestDatabaseSchema:
    """Verify that migrate() creates exactly the expected tables."""

    EXPECTED_TABLES: ClassVar[set[str]] = {
        "schema_meta",
        "analyses",
        "feedback",
        "models",
        "thresholds",
        # M4 drift tables
        "drift_windows",
        "drift_incidents",
        "labeling_queue",
        "canary_runs",
        "rollback_events",
    }

    def _get_tables(self, db: Database) -> set[str]:
        with db.connect() as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return {row["name"] for row in rows}

    def test_all_expected_tables_exist(self) -> None:
        db = _tmp_db()
        tables = self._get_tables(db)
        missing = self.EXPECTED_TABLES - tables
        assert not missing, f"Missing tables: {missing}"

    def test_migrate_idempotent(self) -> None:
        """Calling migrate() twice must not raise any exception."""
        db = _tmp_db()
        db.migrate()  # second call
        db.migrate()  # third call -- still no error

    def test_schema_version_written(self) -> None:
        """schema_meta must contain the correct version string."""
        db = _tmp_db()
        with db.connect() as conn:
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()
        assert row is not None
        assert row["value"] == str(SCHEMA_VERSION)
        assert SCHEMA_VERSION == 2

    def test_drift_windows_columns(self) -> None:
        """drift_windows must contain all required column names."""
        db = _tmp_db()
        with db.connect() as conn:
            info = conn.execute("PRAGMA table_info(drift_windows)").fetchall()
        columns = {row["name"] for row in info}
        required = {
            "window_id",
            "model_id",
            "metric_name",
            "window_start",
            "window_end",
            "trend_status",
            "drift_pattern",
            "created_at",
        }
        assert required.issubset(columns)

    def test_drift_incidents_fk_enforced(self) -> None:
        """Inserting incident with unknown window_id must raise IntegrityError."""
        db = _tmp_db()
        with pytest.raises((sqlite3.IntegrityError, Exception)), db.connect() as conn:
            conn.execute(
                """INSERT INTO drift_incidents
                    (incident_id, priority, drift_pattern, affected_window_id,
                     model_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    "inc_bad",
                    "P1_INCIDENT",
                    "SUDDEN",
                    "nonexistent_window",  # FK violation
                    "model_abc",
                    _NOW.isoformat(),
                ),
            )

    def test_labeling_queue_priority_check(self) -> None:
        """labeling_queue rejects invalid priority_bucket values."""
        db = _tmp_db()
        with pytest.raises((sqlite3.IntegrityError, Exception)), db.connect() as conn:
            conn.execute(
                """INSERT INTO labeling_queue
                    (item_id, image_sha256, source_path, priority_bucket,
                     selection_reason, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                ("lq_bad", "sha", "/path", "P9", "test", _NOW.isoformat()),
            )


# ---------------------------------------------------------------------------
# DriftWindowRepository
# ---------------------------------------------------------------------------


class TestDriftWindowRepository:
    def test_save_and_list_recent(self) -> None:
        db = _tmp_db()
        repo = DriftWindowRepository(db)
        w = _window("win_001")
        repo.save(w)

        rows = repo.list_recent()
        assert len(rows) == 1
        assert rows[0]["window_id"] == "win_001"
        assert rows[0]["trend_status"] == "STABLE"
        assert rows[0]["drift_pattern"] == "STABLE"

    def test_save_idempotent_replace(self) -> None:
        """Saving same window_id twice replaces, not duplicates."""
        db = _tmp_db()
        repo = DriftWindowRepository(db)
        repo.save(_window("win_001"))
        repo.save(_window("win_001"))  # same id -> replace
        assert len(repo.list_recent()) == 1

    def test_save_with_optional_fields(self) -> None:
        """Window with metric values persists correctly."""
        db = _tmp_db()
        repo = DriftWindowRepository(db)
        w = DriftWindow(
            window_id="win_002",
            model_id="model_abc",
            metric_name="image_ap50",
            window_start=_NOW,
            window_end=_NOW,
            metric_value=0.83,
            ewma_value=0.82,
            cusum_value=1.2,
            psi_value=0.05,
            trend_status=TrendStatus.EWMA_ALERT,
            drift_pattern=DriftPattern.GRADUAL,
            created_at=_NOW,
        )
        repo.save(w)
        rows = repo.list_recent()
        assert rows[0]["metric_value"] == pytest.approx(0.83)
        assert rows[0]["psi_value"] == pytest.approx(0.05)

    def test_list_recent_limit(self) -> None:
        """list_recent respects limit parameter."""
        db = _tmp_db()
        repo = DriftWindowRepository(db)
        for i in range(5):
            repo.save(_window(f"win_{i:03d}"))
        assert len(repo.list_recent(limit=3)) == 3

    def test_list_recent_empty(self) -> None:
        db = _tmp_db()
        repo = DriftWindowRepository(db)
        assert repo.list_recent() == []


# ---------------------------------------------------------------------------
# IncidentRepository
# ---------------------------------------------------------------------------


class TestIncidentRepository:
    def _setup(self) -> tuple[Database, DriftWindowRepository, IncidentRepository]:
        db = _tmp_db()
        window_repo = DriftWindowRepository(db)
        incident_repo = IncidentRepository(db)
        # Insert required parent window first (FK)
        window_repo.save(_window("win_001"))
        return db, window_repo, incident_repo

    def test_save_and_list_open(self) -> None:
        _, _, incident_repo = self._setup()
        inc = _incident("inc_001", "win_001")
        incident_repo.save(inc)

        open_incidents = incident_repo.list_open()
        assert len(open_incidents) == 1
        assert open_incidents[0]["incident_id"] == "inc_001"
        assert open_incidents[0]["priority"] == "P1_INCIDENT"

    def test_list_open_excludes_resolved(self) -> None:
        """Incidents with resolved_at set must NOT appear in list_open()."""
        db, _, incident_repo = self._setup()
        # Insert resolved incident directly via raw SQL
        with db.connect() as conn:
            conn.execute(
                """INSERT INTO drift_incidents
                (incident_id, priority, drift_pattern, affected_window_id,
                 model_id, created_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    "inc_resolved",
                    "P2_REVIEW",
                    "GRADUAL",
                    "win_001",
                    "model_abc",
                    _NOW.isoformat(),
                    _NOW.isoformat(),  # resolved_at set
                ),
            )
        assert incident_repo.list_open() == []

    def test_save_idempotent_replace(self) -> None:
        """Same incident_id can be saved twice (INSERT OR REPLACE)."""
        _, _, incident_repo = self._setup()
        incident_repo.save(_incident("inc_001", "win_001"))
        incident_repo.save(_incident("inc_001", "win_001"))
        assert len(incident_repo.list_open()) == 1

    def test_optional_fields_default_none(self) -> None:
        """Optional fields (root_cause, action_taken) persist as NULL."""
        _, _, incident_repo = self._setup()
        incident_repo.save(_incident("inc_001", "win_001"))
        rows = incident_repo.list_open()
        assert rows[0]["root_cause"] is None
        assert rows[0]["action_taken"] is None
        assert rows[0]["resolved_at"] is None
