"""Extended calibration, benchmark, robustness, and evaluation service tests."""

from pathlib import Path

import numpy as np
import pytest

from weavevision.evaluation.benchmark import benchmark_callable
from weavevision.evaluation.calibration import (
    calibrate_image_threshold,
    calibrate_pixel_threshold,
)
from weavevision.evaluation.metrics import image_metrics, recall_at_normal_fpr
from weavevision.evaluation.robustness import perturb
from weavevision.services.evaluation_service import EvaluationService


def test_anomaly_validation_produces_locked_threshold() -> None:
    artifact = calibrate_image_threshold(
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([0.7, 0.8, 0.9]),
        split="validation",
        model_id="model",
        dataset_manifest_sha256="a" * 64,
        target_normal_fpr=0.25,
    )
    assert artifact.status == "LOCKED"
    assert artifact.method == "recall_at_fpr_then_f1"


def test_calibration_rejects_invalid_scores_and_calibrates_pixels() -> None:
    with pytest.raises(ValueError, match="finite"):
        calibrate_image_threshold(
            np.array([np.nan]),
            None,
            split="validation",
            model_id="model",
            dataset_manifest_sha256="a" * 64,
        )
    normal_maps = np.array([[[0.1, 0.2], [0.2, 0.1]]])
    provisional, provisional_method = calibrate_pixel_threshold(
        normal_maps, None, None, split="validation"
    )
    assert provisional > 0
    assert "quantile" in provisional_method
    anomaly_maps = np.array([[[0.1, 0.9], [0.8, 0.1]]])
    masks = np.array([[[0, 1], [1, 0]]])
    locked, method = calibrate_pixel_threshold(normal_maps, anomaly_maps, masks, split="validation")
    assert locked > 0
    assert method == "pixel_f1"
    with pytest.raises(ValueError, match="validation-only"):
        calibrate_pixel_threshold(normal_maps, None, None, split="test")


def test_recall_at_fpr_and_single_class_metric() -> None:
    result = recall_at_normal_fpr(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert result["recall"] == 1.0
    single = image_metrics(np.array([0, 0]), np.array([0.1, 0.2]), 0.5)
    assert single["image_auroc"] is None


def test_benchmark_callable_measures_operation() -> None:
    counter = {"value": 0}

    def operation() -> None:
        counter["value"] += 1

    result = benchmark_callable(operation, warmup_runs=2, measured_runs=3)
    assert counter["value"] == 5
    assert result["latency_ms"]["p95"] >= 0
    with pytest.raises(ValueError):
        benchmark_callable(operation, measured_runs=0)


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("brightness", 0.8),
        ("contrast", 1.2),
        ("gaussian_noise", 5.0),
        ("blur", 3.0),
        ("rotation", 5.0),
        ("downsample", 0.5),
        ("jpeg", 70.0),
        ("occlusion", 0.2),
    ],
)
def test_robustness_perturbations_preserve_shape(kind: str, value: float) -> None:
    image = np.random.default_rng(4).integers(0, 255, (64, 96, 3), dtype=np.uint8)
    assert perturb(image, kind, value).shape == image.shape


def test_robustness_rejects_invalid_parameters() -> None:
    image = np.ones((64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="odd"):
        perturb(image, "blur", 4)
    with pytest.raises(ValueError, match="fraction"):
        perturb(image, "occlusion", 2)
    with pytest.raises(ValueError, match="unsupported"):
        perturb(image, "unknown", 1)


def test_evaluation_service_persists_locked_metrics(tmp_path: Path) -> None:
    destination = tmp_path / "metrics.json"
    result = EvaluationService().evaluate_arrays(
        np.array([0, 1]),
        np.array([0.1, 0.9]),
        0.5,
        destination,
        masks=np.array([[[0, 0]], [[0, 1]]]),
        maps=np.array([[[0.1, 0.2]], [[0.2, 0.9]]]),
        pixel_threshold=0.5,
    )
    assert destination.is_file()
    assert result["false_negative"] == 0
