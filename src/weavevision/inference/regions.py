"""Connected component extraction in original image coordinates."""

from __future__ import annotations

import cv2
import numpy as np

from weavevision.domain.schemas import RegionResult


def extract_regions(
    mask: np.ndarray, anomaly_map: np.ndarray, min_area_pixels: int
) -> list[RegionResult]:
    """Extract filtered connected regions and score statistics."""
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    image_area = mask.shape[0] * mask.shape[1]
    regions: list[RegionResult] = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < min_area_pixels:
            continue
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        width = int(stats[component, cv2.CC_STAT_WIDTH])
        height = int(stats[component, cv2.CC_STAT_HEIGHT])
        component_mask = (labels == component).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = (
            max(contours, key=cv2.contourArea).reshape(-1, 2) if contours else np.empty((0, 2))
        )
        values = anomaly_map[labels == component]
        regions.append(
            RegionResult(
                region_id=len(regions) + 1,
                bbox_xyxy=(x, y, x + width, y + height),
                area_pixels=area,
                area_ratio=area / image_area,
                mean_anomaly_score=float(values.mean()),
                max_anomaly_score=float(values.max()),
                centroid_xy=(float(centroids[component, 0]), float(centroids[component, 1])),
                contour=[(int(point[0]), int(point[1])) for point in contour],
            )
        )
    return regions
