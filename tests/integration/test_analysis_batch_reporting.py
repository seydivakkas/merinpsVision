"""End-to-end service, report, persistence, and partial batch tests."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from weavevision.domain.schemas import ModelIdentity, ThresholdArtifact
from weavevision.services.analysis_service import AnalysisService
from weavevision.services.batch_service import BatchService
from weavevision.settings import load_settings


class ColorResidualPredictor:
    """Test-only deterministic residual detector with real image-dependent output."""

    @property
    def model_id(self) -> str:
        return "model_fixture_residual"

    def predict_array(self, image_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        red = image_rgb[:, :, 0].astype(np.float32)
        smooth = cv2.GaussianBlur(red, (15, 15), 0)
        anomaly_map = np.abs(red - smooth) / 255.0
        return float(np.quantile(anomaly_map, 0.995)), anomaly_map

    def export(self, destination: Path) -> Path:
        destination.write_text("test-only residual parameters", encoding="utf-8")
        return destination


def service(tmp_path: Path) -> AnalysisService:
    """Construct an isolated analysis service with a locked validation threshold."""
    settings = load_settings().model_copy(
        update={
            "paths": load_settings().paths.model_copy(
                update={
                    "artifacts_root": tmp_path / "artifacts",
                    "database": tmp_path / "artifacts" / "audit.sqlite3",
                }
            )
        }
    )
    threshold = ThresholdArtifact(
        threshold_id="thr_fixture",
        model_id="model_fixture_residual",
        dataset_manifest_sha256="a" * 64,
        image_threshold=0.08,
        pixel_threshold=0.08,
        method="fixture_validation",
        target_normal_fpr=0.05,
        calibration_split="validation",
        created_at=datetime.now(UTC),
        status="LOCKED",
    )
    identity = ModelIdentity(
        model_id="model_fixture_residual",
        model_name="test_residual",
        model_artifact_sha256="b" * 64,
        config_sha256="c" * 64,
    )
    return AnalysisService(settings, ColorResidualPredictor(), identity, threshold)


def test_single_analysis_writes_all_reports(tmp_path: Path, textile_image: Path) -> None:
    result = service(tmp_path).analyze(textile_image)
    assert result.artifacts.json_path and result.artifacts.json_path.is_file()
    assert result.artifacts.html_path and result.artifacts.html_path.is_file()
    assert result.artifacts.overlay_path and result.artifacts.overlay_path.is_file()


def test_batch_isolates_corrupt_image(tmp_path: Path, textile_image: Path) -> None:
    folder = tmp_path / "batch"
    folder.mkdir()
    shutil.copy2(textile_image, folder / "valid.png")
    (folder / "corrupt.png").write_bytes(b"not an image")
    result = BatchService(service(tmp_path)).analyze(folder)
    assert len(result.results) == 1
    assert len(result.failures) == 1
    assert result.failures[0].error_code == "WV_IMAGE_INVALID"
