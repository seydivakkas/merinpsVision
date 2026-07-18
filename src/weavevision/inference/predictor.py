"""Device-independent direct and tiled model prediction orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from weavevision.data.tiling import create_tile_coordinates, extract_tiles, merge_tile_maps
from weavevision.domain.protocols import AnomalyPredictor


@dataclass(frozen=True, slots=True)
class RawPrediction:
    """Raw model output and measured stage timings."""

    score: float
    anomaly_map: np.ndarray
    preprocess_ms: float
    inference_ms: float


def predict_image(
    predictor: AnomalyPredictor,
    image_rgb: np.ndarray,
    *,
    tiled: bool,
    tile_size: tuple[int, int] = (512, 512),
    overlap: float = 0.25,
) -> RawPrediction:
    """Run direct or weighted tiled prediction and restore original geometry."""
    started = time.perf_counter()
    height, width = image_rgb.shape[:2]
    coordinates = create_tile_coordinates((height, width), tile_size, overlap) if tiled else []
    tiles = extract_tiles(image_rgb, coordinates) if coordinates else [image_rgb]
    preprocess_ms = (time.perf_counter() - started) * 1000
    inference_started = time.perf_counter()
    scores: list[float] = []
    maps: list[np.ndarray] = []
    for tile in tiles:
        score, anomaly_map = predictor.predict_array(tile)
        scores.append(float(score))
        if anomaly_map.shape != tile.shape[:2]:
            anomaly_map = cv2.resize(
                anomaly_map.astype(np.float32),
                (tile.shape[1], tile.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        maps.append(anomaly_map.astype(np.float32))
    inference_ms = (time.perf_counter() - inference_started) * 1000
    merged = merge_tile_maps(maps, coordinates, (height, width)) if coordinates else maps[0]
    return RawPrediction(max(scores), merged, preprocess_ms, inference_ms)
