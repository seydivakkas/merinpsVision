"""Aspect-preserving tiled inference coordinates and weighted map merging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TileCoordinate:
    """One tile location in original image coordinates."""

    tile_id: int
    x0: int
    y0: int
    x1: int
    y1: int


def create_tile_coordinates(
    image_shape: tuple[int, int],
    tile_size: tuple[int, int],
    overlap: float,
) -> list[TileCoordinate]:
    """Create full-coverage tile coordinates including anchored border tiles."""
    height, width = image_shape
    tile_height, tile_width = tile_size
    if min(height, width, tile_height, tile_width) <= 0:
        raise ValueError("image and tile dimensions must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    actual_height = min(tile_height, height)
    actual_width = min(tile_width, width)
    stride_y = max(1, round(actual_height * (1.0 - overlap)))
    stride_x = max(1, round(actual_width * (1.0 - overlap)))
    y_starts = _axis_starts(height, actual_height, stride_y)
    x_starts = _axis_starts(width, actual_width, stride_x)
    return [
        TileCoordinate(index, x, y, x + actual_width, y + actual_height)
        for index, (y, x) in enumerate((y, x) for y in y_starts for x in x_starts)
    ]


def extract_tiles(image: np.ndarray, coordinates: list[TileCoordinate]) -> list[np.ndarray]:
    """Extract tiles while preserving channel order and coordinate identity."""
    return [image[item.y0 : item.y1, item.x0 : item.x1].copy() for item in coordinates]


def merge_tile_maps(
    tile_maps: list[np.ndarray],
    coordinates: list[TileCoordinate],
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Merge tile anomaly maps with a positive center-weighted overlap window."""
    if len(tile_maps) != len(coordinates) or not tile_maps:
        raise ValueError("tile maps and coordinates must be non-empty and aligned")
    accumulator = np.zeros(output_shape, dtype=np.float64)
    weights = np.zeros(output_shape, dtype=np.float64)
    for tile_map, coordinate in zip(tile_maps, coordinates, strict=True):
        expected = (coordinate.y1 - coordinate.y0, coordinate.x1 - coordinate.x0)
        if tile_map.shape != expected:
            raise ValueError(f"tile map shape {tile_map.shape} does not match {expected}")
        window = _weight_window(expected)
        ys = slice(coordinate.y0, coordinate.y1)
        xs = slice(coordinate.x0, coordinate.x1)
        accumulator[ys, xs] += tile_map.astype(np.float64) * window
        weights[ys, xs] += window
    if np.any(weights == 0):
        raise ValueError("tile coordinates do not fully cover the output")
    return (accumulator / weights).astype(np.float32)


def _axis_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    anchored = length - tile
    if starts[-1] != anchored:
        starts.append(anchored)
    return starts


def _weight_window(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y = np.hanning(height) if height > 2 else np.ones(height)
    x = np.hanning(width) if width > 2 else np.ones(width)
    return np.maximum(np.outer(y, x), 1e-3)
