"""Batch discovery, safe ZIP extraction, and partial-failure analysis."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from weavevision.data.transforms import SUPPORTED_SUFFIXES
from weavevision.domain.errors import WeaveVisionError
from weavevision.domain.schemas import BatchItemFailure, BatchResult
from weavevision.services.analysis_service import AnalysisService
from weavevision.services.report_service import ReportService


class BatchService:
    """Analyze folders or ZIP archives while isolating individual item failures."""

    def __init__(self, analysis: AnalysisService) -> None:
        self.analysis = analysis
        self.reports = ReportService(analysis.settings)

    def analyze(
        self,
        source: Path,
        output_root: Path | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        """Analyze a folder or safe ZIP with duplicate and partial-failure handling."""
        temporary: tempfile.TemporaryDirectory[str] | None = None
        try:
            root = source
            if source.suffix.casefold() == ".zip":
                temporary = tempfile.TemporaryDirectory(prefix="weavevision_batch_")
                root = Path(temporary.name)
                safe_extract_zip(source, root)
            paths = sorted(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
            )
            results = []
            failures = []
            seen_hashes: set[str] = set()
            for index, path in enumerate(paths, start=1):
                try:
                    result = self.analysis.analyze(path, output_root)
                    if result.source.sha256 in seen_hashes:
                        failures.append(
                            BatchItemFailure(
                                filename=path.name,
                                error_code="WV_DUPLICATE_INPUT",
                                message="duplicate input image skipped",
                            )
                        )
                    else:
                        seen_hashes.add(result.source.sha256)
                        results.append(result)
                except WeaveVisionError as exc:
                    failures.append(
                        BatchItemFailure(
                            filename=path.name, error_code=exc.code, message=exc.message
                        )
                    )
                if progress:
                    progress(index, len(paths))
            batch = BatchResult(
                batch_id=f"batch_{uuid4().hex}",
                created_at=datetime.now(UTC),
                results=results,
                failures=failures,
            )
            destination = (
                output_root or self.analysis.settings.resolved_artifacts_root() / "reports"
            )
            self.reports.write_batch(batch, destination)
            return batch
        finally:
            if temporary is not None:
                temporary.cleanup()


def safe_extract_zip(archive: Path, destination: Path, max_uncompressed_mb: int = 500) -> None:
    """Extract a ZIP after traversal, member count, ratio, and size checks."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        if len(members) > 10_000:
            raise ValueError("ZIP contains too many members")
        total = sum(item.file_size for item in members)
        if total > max_uncompressed_mb * 1024 * 1024:
            raise ValueError("ZIP uncompressed size exceeds configured limit")
        for member in members:
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError("ZIP path traversal rejected")
            if member.compress_size and member.file_size / member.compress_size > 200:
                raise ValueError("ZIP suspicious compression ratio rejected")
        for member in members:
            if member.is_dir():
                continue
            target = (destination / member.filename).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(member) as source_handle, target.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)
