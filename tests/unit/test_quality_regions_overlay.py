"""Quality gate, postprocessing region, and overlay tests."""

import numpy as np

from weavevision.domain.enums import Decision, QualityGateStatus
from weavevision.inference.overlay import render_artifacts
from weavevision.inference.postprocess import postprocess_prediction
from weavevision.inference.quality_gate import evaluate_quality


def test_tiny_image_abstains() -> None:
    result = evaluate_quality(np.ones((16, 16, 3), dtype=np.uint8) * 100)
    assert result.status is QualityGateStatus.ABSTAIN
    assert "image_too_small" in result.reasons


def test_region_and_overlay_preserve_dimensions() -> None:
    image = np.random.default_rng(42).integers(0, 255, (96, 128, 3), dtype=np.uint8)
    quality = evaluate_quality(image)
    anomaly_map = np.zeros((96, 128), dtype=np.float32)
    anomaly_map[20:40, 30:60] = 0.9
    prediction, mask, regions = postprocess_prediction(0.8, anomaly_map, quality, 0.5, 0.5)
    heatmap, overlay = render_artifacts(image, anomaly_map, mask)
    assert prediction.decision is Decision.ANOMALY
    assert len(regions) == 1
    assert heatmap.shape == image.shape == overlay.shape
