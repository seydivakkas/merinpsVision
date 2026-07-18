"""Colorblind-readable heatmap, mask, and contour rendering."""

from __future__ import annotations

import cv2
import numpy as np


def render_artifacts(
    image_rgb: np.ndarray, anomaly_map: np.ndarray, mask: np.ndarray, alpha: float = 0.45
) -> tuple[np.ndarray, np.ndarray]:
    """Render normalized Viridis heatmap and an image-preserving contour overlay."""
    minimum = float(anomaly_map.min())
    maximum = float(anomaly_map.max())
    normalized = (
        np.zeros_like(anomaly_map, dtype=np.uint8)
        if maximum <= minimum
        else np.clip((anomaly_map - minimum) * 255 / (maximum - minimum), 0, 255).astype(np.uint8)
    )
    heatmap_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (230, 85, 13), thickness=2)
    return heatmap_rgb, overlay
