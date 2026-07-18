"""Validation-only image and pixel threshold calibration."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
from sklearn.metrics import f1_score

from weavevision.domain.schemas import ThresholdArtifact


def calibrate_image_threshold(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray | None,
    *,
    split: str,
    model_id: str,
    dataset_manifest_sha256: str,
    target_normal_fpr: float = 0.05,
) -> ThresholdArtifact:
    """Calibrate an image threshold from validation evidence only.

    Args:
        normal_scores: Validation normal anomaly scores.
        anomaly_scores: Optional validation anomaly scores.
        split: Must be exactly ``validation``.
        model_id: Immutable model identity.
        dataset_manifest_sha256: Calibration dataset identity.
        target_normal_fpr: Maximum intended validation normal false-positive rate.

    Returns:
        Locked or explicitly provisional threshold artifact.

    Raises:
        ValueError: If sealed test data or empty normal scores are supplied.
    """
    if split != "validation":
        raise ValueError("threshold calibration is allowed only on the validation split")
    normal = np.asarray(normal_scores, dtype=np.float64).reshape(-1)
    if normal.size == 0 or not np.all(np.isfinite(normal)):
        raise ValueError("finite validation normal scores are required")
    if anomaly_scores is None or np.asarray(anomaly_scores).size == 0:
        threshold = float(np.quantile(normal, 0.995))
        method = "normal_quantile_0.995"
        status = "PROVISIONAL_NORMAL_ONLY"
    else:
        anomaly = np.asarray(anomaly_scores, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(anomaly)):
            raise ValueError("anomaly validation scores must be finite")
        fpr_floor = float(np.quantile(normal, 1.0 - target_normal_fpr, method="higher"))
        candidates = np.unique(np.concatenate((normal, anomaly, np.array([fpr_floor]))))
        labels = np.concatenate(
            (np.zeros(normal.size, dtype=int), np.ones(anomaly.size, dtype=int))
        )
        scores = np.concatenate((normal, anomaly))
        eligible = [value for value in candidates if np.mean(normal >= value) <= target_normal_fpr]
        if not eligible:
            eligible = [float(np.nextafter(normal.max(), np.inf))]
        threshold = float(
            max(eligible, key=lambda value: f1_score(labels, scores >= value, zero_division=0))
        )
        method = "recall_at_fpr_then_f1"
        status = "LOCKED"
    return ThresholdArtifact(
        threshold_id=f"thr_{uuid4().hex}",
        model_id=model_id,
        dataset_manifest_sha256=dataset_manifest_sha256,
        image_threshold=threshold,
        pixel_threshold=threshold,
        method=method,
        target_normal_fpr=target_normal_fpr,
        created_at=datetime.now(UTC),
        status=status,
    )


def calibrate_pixel_threshold(
    normal_maps: np.ndarray,
    anomaly_maps: np.ndarray | None,
    anomaly_masks: np.ndarray | None,
    *,
    split: str,
) -> tuple[float, str]:
    """Calibrate a pixel threshold from validation maps and optional masks."""
    if split != "validation":
        raise ValueError("pixel threshold calibration is validation-only")
    normal = np.asarray(normal_maps, dtype=np.float64).reshape(-1)
    if normal.size == 0:
        raise ValueError("normal validation maps are required")
    if anomaly_maps is None or anomaly_masks is None:
        return float(np.quantile(normal, 0.999)), "normal_pixel_quantile_0.999"
    scores = np.concatenate((normal, np.asarray(anomaly_maps, dtype=np.float64).reshape(-1)))
    labels = np.concatenate(
        (np.zeros(normal.size, dtype=int), np.asarray(anomaly_masks).astype(bool).reshape(-1))
    )
    candidates = np.quantile(scores, np.linspace(0.8, 0.9999, 256))
    threshold = max(
        candidates,
        key=lambda value: f1_score(labels, scores >= value, zero_division=0),
    )
    return float(threshold), "pixel_f1"
