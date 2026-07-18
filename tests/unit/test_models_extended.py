"""Model adapter helpers, factory, active composition, and plots tests."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from weavevision.domain.enums import ModelStatus
from weavevision.domain.errors import ConfigError, ModelNotReadyError
from weavevision.domain.schemas import ModelManifest, ThresholdArtifact
from weavevision.evaluation.plots import save_score_distribution
from weavevision.models.anomalib_adapter import (
    AnomalibAdapter,
    _flatten_predictions,
    _prediction_values,
)
from weavevision.models.export import sha256_artifact
from weavevision.models.factory import create_anomalib_model
from weavevision.models.registry import ModelRegistry
from weavevision.services.factory import load_active_analysis_service
from weavevision.settings import load_settings


def settings_at(tmp_path: Path):  # type: ignore[no-untyped-def]
    settings = load_settings()
    return settings.model_copy(
        update={
            "paths": settings.paths.model_copy(
                update={
                    "data_root": tmp_path / "data",
                    "artifacts_root": tmp_path / "artifacts",
                    "database": tmp_path / "artifacts" / "db.sqlite3",
                }
            )
        }
    )


def test_prediction_helpers_flatten_batches() -> None:
    batch = SimpleNamespace(
        pred_score=np.array([0.2, 0.8]),
        anomaly_map=np.array([np.zeros((2, 2)), np.ones((2, 2))]),
    )
    values = _flatten_predictions([batch])
    assert len(values) == 2
    assert values[1][0] == pytest.approx(0.8)
    score, anomaly_map = _prediction_values(batch)
    assert score == pytest.approx(0.2)
    assert anomaly_map.shape == (2, 2, 2)
    assert _flatten_predictions(None) == []
    with pytest.raises(ValueError, match="empty"):
        _prediction_values(SimpleNamespace(pred_score=np.array([]), anomaly_map=np.array([])))


def test_adapter_requires_artifact_and_can_use_loaded_inferencer(tmp_path: Path) -> None:
    adapter = AnomalibAdapter("model", "patchcore", {"name": "patchcore"}, device="cpu")
    with pytest.raises(ModelNotReadyError):
        adapter.predict_array(np.zeros((16, 16, 3), dtype=np.uint8))
    with pytest.raises(ModelNotReadyError):
        adapter.export(tmp_path / "export.xml")
    artifact = tmp_path / "artifact.xml"
    artifact.write_text("placeholder", encoding="utf-8")
    adapter.artifact_path = artifact
    adapter._inferencer = SimpleNamespace(
        predict=lambda _image: SimpleNamespace(
            pred_score=np.array([0.4]), anomaly_map=np.ones((1, 4, 4))
        )
    )
    score, anomaly_map = adapter.predict_array(np.zeros((4, 4, 3), dtype=np.uint8))
    assert score == pytest.approx(0.4)
    assert anomaly_map.shape == (4, 4)
    copied = adapter.export(tmp_path / "copied.xml")
    assert copied.read_text() == "placeholder"


def test_model_factory_allowlist_and_arguments() -> None:
    patchcore = create_anomalib_model(
        "patchcore",
        {
            "name": "patchcore",
            "backbone": "resnet18",
            "layers": ["layer2"],
            "pre_trained": False,
        },
    )
    assert patchcore.__class__.__name__ == "Patchcore"
    efficient = create_anomalib_model(
        "efficient_ad", {"name": "efficient_ad", "model_size": "small"}
    )
    assert efficient.__class__.__name__ == "EfficientAd"
    with pytest.raises(ConfigError, match="unsupported model"):
        create_anomalib_model("unknown", {"name": "unknown"})
    with pytest.raises(ConfigError, match="arguments"):
        create_anomalib_model("patchcore", {"name": "patchcore", "unsafe": True})


def test_active_service_composition_and_plot(tmp_path: Path) -> None:
    settings = settings_at(tmp_path)
    artifact = tmp_path / "model.xml"
    artifact.write_text("xml", encoding="utf-8")
    artifact.with_suffix(".bin").write_bytes(b"weights")
    threshold = ThresholdArtifact(
        threshold_id="thr_active",
        model_id="model_active",
        dataset_manifest_sha256="a" * 64,
        image_threshold=0.5,
        pixel_threshold=0.5,
        method="validation_f1",
        calibration_split="validation",
        created_at=datetime.now(UTC),
        status="LOCKED",
    )
    threshold_path = settings.resolved_artifacts_root() / "experiments" / "run" / "thresholds.json"
    threshold_path.parent.mkdir(parents=True)
    threshold_path.write_text(threshold.model_dump_json(), encoding="utf-8")
    registry = ModelRegistry(settings.resolved_artifacts_root() / "models")
    registry.register(
        ModelManifest(
            model_id="model_active",
            status=ModelStatus.ACTIVE_BENCHMARK,
            algorithm="patchcore",
            dataset_manifest_sha256="a" * 64,
            training_run_id="run",
            config_sha256="b" * 64,
            artifact_path=artifact,
            artifact_sha256=sha256_artifact(artifact),
            threshold_id="thr_active",
            created_at=datetime.now(UTC),
        )
    )
    service = load_active_analysis_service(settings)
    assert service.threshold and service.threshold.threshold_id == "thr_active"
    plot = save_score_distribution(
        np.array([0.1, 0.2]), np.array([0.8, 0.9]), 0.5, tmp_path / "plot.png"
    )
    assert plot.is_file()
