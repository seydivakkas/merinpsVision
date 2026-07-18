"""Canonical JSON report output."""

from __future__ import annotations

import os
from pathlib import Path

from weavevision.domain.schemas import AnalysisResult, BatchResult


def write_json_report(result: AnalysisResult | BatchResult, destination: Path) -> Path:
    """Atomically write a validated result as UTF-8 JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    return destination
