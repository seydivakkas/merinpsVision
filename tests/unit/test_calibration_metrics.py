"""Validation isolation and metric edge case tests."""

import numpy as np
import pytest

from weavevision.evaluation.calibration import calibrate_image_threshold
from weavevision.evaluation.metrics import image_metrics, pixel_metrics


def test_calibration_rejects_test_split() -> None:
    with pytest.raises(ValueError, match="validation"):
        calibrate_image_threshold(
            np.array([0.1, 0.2]),
            np.array([0.8]),
            split="test",
            model_id="model",
            dataset_manifest_sha256="a" * 64,
        )


def test_normal_only_threshold_is_provisional() -> None:
    artifact = calibrate_image_threshold(
        np.array([0.1, 0.2, 0.3]),
        None,
        split="validation",
        model_id="model",
        dataset_manifest_sha256="a" * 64,
    )
    assert artifact.status == "PROVISIONAL_NORMAL_ONLY"


def test_image_and_pixel_metrics_use_locked_threshold() -> None:
    metrics = image_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 0
    pixel = pixel_metrics(np.array([[0, 0], [1, 1]]), np.array([[0.1, 0.2], [0.8, 0.9]]), 0.5)
    assert pixel["pixel_iou"] == 1.0
