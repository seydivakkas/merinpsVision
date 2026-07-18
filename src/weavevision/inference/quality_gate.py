"""Pure image quality measurements and conservative gate decisions."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from weavevision.domain.enums import QualityGateStatus
from weavevision.domain.schemas import QualityGateResult


@dataclass(frozen=True, slots=True)
class QualityGateConfig:
    """Versionable conservative starting thresholds."""

    min_width: int = 64
    min_height: int = 64
    min_aspect_ratio: float = 0.1
    max_aspect_ratio: float = 10.0
    blur_review_variance: float = 20.0
    dark_review_mean: float = 12.0
    bright_review_mean: float = 243.0
    minimum_std: float = 2.0


def evaluate_quality(
    image_rgb: np.ndarray, config: QualityGateConfig | None = None
) -> QualityGateResult:
    """Measure an RGB image and return PASS, REVIEW, or ABSTAIN."""
    settings = config or QualityGateConfig()
    reasons: list[str] = []
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        return QualityGateResult(
            status=QualityGateStatus.ABSTAIN,
            reasons=["unsupported_channel_or_bit_depth"],
            metrics={"shape": str(image_rgb.shape), "dtype": str(image_rgb.dtype)},
        )
    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
    standard_deviation = float(gray.std())
    aspect = width / height
    status = QualityGateStatus.PASS
    if width < settings.min_width or height < settings.min_height:
        status = QualityGateStatus.ABSTAIN
        reasons.append("image_too_small")
    if not settings.min_aspect_ratio <= aspect <= settings.max_aspect_ratio:
        status = QualityGateStatus.ABSTAIN
        reasons.append("extreme_aspect_ratio")
    if standard_deviation < settings.minimum_std:
        status = QualityGateStatus.ABSTAIN
        reasons.append("near_uniform_image")
    if status is not QualityGateStatus.ABSTAIN and blur < settings.blur_review_variance:
        status = QualityGateStatus.REVIEW
        reasons.append("possible_blur")
    if status is not QualityGateStatus.ABSTAIN and mean < settings.dark_review_mean:
        status = QualityGateStatus.REVIEW
        reasons.append("possible_underexposure")
    if status is not QualityGateStatus.ABSTAIN and mean > settings.bright_review_mean:
        status = QualityGateStatus.REVIEW
        reasons.append("possible_overexposure")
    return QualityGateResult(
        status=status,
        reasons=reasons,
        metrics={
            "width": width,
            "height": height,
            "aspect_ratio": aspect,
            "blur_laplacian_variance": blur,
            "luminance_mean": mean,
            "luminance_std": standard_deviation,
        },
    )
