"""Sealed-test metric and evidence artifact generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from weavevision.evaluation.metrics import image_metrics, pixel_metrics


class EvaluationService:
    """Compute metrics at already locked thresholds; never calibrate on test data."""

    def evaluate_arrays(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        image_threshold: float,
        destination: Path,
        *,
        masks: np.ndarray | None = None,
        maps: np.ndarray | None = None,
        pixel_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Compute and persist image and optional pixel metrics from sealed arrays."""
        metrics = image_metrics(labels, scores, image_threshold)
        if masks is not None and maps is not None and pixel_threshold is not None:
            metrics.update(pixel_metrics(masks, maps, pixel_threshold))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics
