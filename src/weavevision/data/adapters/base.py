"""Base contract and shared helpers for dataset verification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from weavevision.domain.schemas import DatasetManifest


class DatasetAdapter(ABC):
    """Verify an external dataset and produce a canonical manifest."""

    def __init__(self, root: Path, manifest_path: Path) -> None:
        self.root = root.resolve()
        self.manifest_path = manifest_path.resolve()

    @abstractmethod
    def verify(self) -> DatasetManifest:
        """Verify dataset structure, content, license metadata, and split identity."""
