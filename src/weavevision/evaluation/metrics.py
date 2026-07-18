"""Image and pixel anomaly metrics at locked thresholds."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def image_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute required image-level metrics and raw confusion counts."""
    truth = np.asarray(labels, dtype=int).reshape(-1)
    values = np.asarray(scores, dtype=float).reshape(-1)
    if truth.size != values.size or truth.size == 0:
        raise ValueError("labels and scores must be non-empty and aligned")
    predicted = (values >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average="binary", zero_division=0
    )
    return {
        "image_auroc": _safe_binary_metric(roc_auc_score, truth, values),
        "image_average_precision": _safe_binary_metric(average_precision_score, truth, values),
        "f1_at_locked_threshold": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "threshold": float(threshold),
    }


def pixel_metrics(masks: np.ndarray, maps: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute pixel AUROC/AP plus IoU and Dice at a locked pixel threshold."""
    truth = np.asarray(masks).astype(bool).reshape(-1)
    values = np.asarray(maps, dtype=float).reshape(-1)
    if truth.size != values.size or truth.size == 0:
        raise ValueError("pixel masks and maps must be non-empty and aligned")
    predicted = values >= threshold
    intersection = int(np.logical_and(truth, predicted).sum())
    union = int(np.logical_or(truth, predicted).sum())
    denominator = int(truth.sum() + predicted.sum())
    return {
        "pixel_auroc": _safe_binary_metric(roc_auc_score, truth.astype(int), values),
        "pixel_average_precision": _safe_binary_metric(
            average_precision_score, truth.astype(int), values
        ),
        "pixel_iou": float(intersection / union) if union else 1.0,
        "pixel_dice": float(2 * intersection / denominator) if denominator else 1.0,
        "pixel_threshold": float(threshold),
    }


def recall_at_normal_fpr(
    labels: np.ndarray, scores: np.ndarray, target_normal_fpr: float
) -> dict[str, float]:
    """Measure anomaly recall at a threshold set by the supplied normal scores."""
    truth = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    normal = values[truth == 0]
    anomaly = values[truth == 1]
    if normal.size == 0 or anomaly.size == 0:
        raise ValueError("both normal and anomaly samples are required")
    threshold = float(np.quantile(normal, 1.0 - target_normal_fpr, method="higher"))
    return {"recall": float(np.mean(anomaly >= threshold)), "threshold": threshold}


def _safe_binary_metric(function: Any, labels: np.ndarray, scores: np.ndarray) -> float | None:
    if np.unique(labels).size < 2:
        return None
    return float(function(labels, scores))
