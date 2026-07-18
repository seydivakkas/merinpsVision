"""SQLite idempotent migration tests."""

from pathlib import Path

from weavevision.persistence.database import Database


def test_database_migration_is_idempotent(tmp_path: Path) -> None:
    database = Database(tmp_path / "audit.sqlite3")
    database.migrate()
    database.migrate()
    assert database.health()
