"""Thresholding, region extraction, and operational review priority."""

from __future__ import annotations

import cv2
import numpy as np

from weavevision.domain.enums import Decision, QualityGateStatus, ReviewPriority
from weavevision.domain.schemas import PredictionResult, QualityGateResult, RegionResult
from weavevision.inference.regions import extract_regions


def postprocess_prediction(
    raw_score: float,
    anomaly_map: np.ndarray,
    quality: QualityGateResult,
    image_threshold: float,
    pixel_threshold: float,
    min_component_area_pixels: int = 16,
) -> tuple[PredictionResult, np.ndarray, list[RegionResult]]:
    """Apply locked thresholds and return structured framework-agnostic results."""
    if quality.status is QualityGateStatus.ABSTAIN:
        prediction = PredictionResult(
            decision=Decision.ABSTAIN,
            raw_anomaly_score=raw_score,
            normalized_anomaly_score=None,
            review_priority=ReviewPriority.ABSTAIN,
            anomaly_area_ratio=0.0,
            region_count=0,
        )
        return prediction, np.zeros_like(anomaly_map, dtype=np.uint8), []
    smoothed = cv2.GaussianBlur(anomaly_map.astype(np.float32), (3, 3), 0)
    mask = (smoothed >= pixel_threshold).astype(np.uint8) * 255
    regions = extract_regions(mask, smoothed, min_component_area_pixels)
    filtered_mask = np.zeros_like(mask)
    for region in regions:
        contour = np.asarray(region.contour, dtype=np.int32)
        if contour.size:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
    area_ratio = float(np.mean(filtered_mask > 0))
    score_ratio = raw_score / image_threshold if image_threshold > 0 else float("inf")
    if quality.status is QualityGateStatus.REVIEW:
        decision = Decision.REVIEW
    else:
        decision = Decision.ANOMALY if raw_score >= image_threshold else Decision.NORMAL
    if decision is Decision.NORMAL:
        priority = ReviewPriority.P0
    elif score_ratio >= 2.0 or area_ratio >= 0.1:
        priority = ReviewPriority.P3
    elif score_ratio >= 1.25 or area_ratio >= 0.03:
        priority = ReviewPriority.P2
    else:
        priority = ReviewPriority.P1
    prediction = PredictionResult(
        decision=decision,
        raw_anomaly_score=raw_score,
        normalized_anomaly_score=None,
        review_priority=priority,
        anomaly_area_ratio=area_ratio,
        region_count=len(regions),
    )
    return prediction, filtered_mask, regions
