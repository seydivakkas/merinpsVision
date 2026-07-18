"""Quality expert feedback application service."""

from __future__ import annotations

from pathlib import Path

from weavevision.domain.enums import FeedbackVerdict
from weavevision.persistence.database import Database
from weavevision.persistence.repositories import FeedbackRepository


class FeedbackService:
    """Persist review feedback without altering original analysis evidence."""

    def __init__(self, database_path: Path) -> None:
        database = Database(database_path)
        database.migrate()
        self.repository = FeedbackRepository(database)

    def submit(
        self,
        analysis_id: str,
        reviewer: str,
        verdict: FeedbackVerdict,
        comment: str | None = None,
    ) -> str:
        """Store a reviewer verdict and return the feedback identifier."""
        return self.repository.save(analysis_id, reviewer, verdict, comment)
