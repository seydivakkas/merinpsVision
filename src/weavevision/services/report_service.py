"""Report artifact orchestration for analysis and batch results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from weavevision.domain.schemas import AnalysisResult, ArtifactPaths, BatchResult
from weavevision.reporting.csv_report import write_csv_report
from weavevision.reporting.html_report import write_html_report
from weavevision.reporting.json_report import write_json_report
from weavevision.settings import Settings


class ReportService:
    """Persist required visual and structured reports without inventing narrative."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def write_analysis(
        self,
        result: AnalysisResult,
        output_root: Path,
        *,
        heatmap: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        overlay: np.ndarray | None = None,
    ) -> AnalysisResult:
        """Write analysis visuals plus JSON, CSV, and HTML and return updated paths."""
        folder = output_root / result.analysis_id
        folder.mkdir(parents=True, exist_ok=True)
        heatmap_path = self._save_image(folder / "heatmap.png", heatmap)
        mask_path = self._save_image(folder / "mask.png", mask)
        overlay_path = self._save_image(folder / "overlay.png", overlay)
        json_path = folder / "analysis.json"
        html_path = folder / "report.html"
        paths = ArtifactPaths(
            overlay_path=overlay_path,
            mask_path=mask_path,
            heatmap_path=heatmap_path,
            json_path=json_path,
            html_path=html_path,
        )
        updated = result.model_copy(update={"artifacts": paths})
        write_json_report(updated, json_path)
        write_csv_report(updated, folder / "analysis.csv")
        write_html_report(updated, html_path, self._template_root())
        return updated

    def write_batch(self, result: BatchResult, output_root: Path) -> dict[str, Path]:
        """Write aggregate batch JSON, CSV, and HTML artifacts."""
        folder = output_root / result.batch_id
        folder.mkdir(parents=True, exist_ok=True)
        paths = {
            "json": write_json_report(result, folder / "batch.json"),
            "csv": write_csv_report(result, folder / "batch.csv"),
            "html": write_html_report(result, folder / "report.html", self._template_root()),
        }
        return paths

    @staticmethod
    def _save_image(path: Path, value: np.ndarray | None) -> Path | None:
        if value is None:
            return None
        Image.fromarray(value).save(path)
        return path

    def _template_root(self) -> Path:
        return self.settings.project_root / "src" / "weavevision" / "reporting" / "templates"
