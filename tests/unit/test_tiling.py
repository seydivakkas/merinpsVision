"""Tile coordinate and weighted merge tests."""

import numpy as np

from weavevision.data.tiling import create_tile_coordinates, extract_tiles, merge_tile_maps


def test_tiles_cover_borders_and_round_trip_constant_map() -> None:
    coordinates = create_tile_coordinates((130, 150), (64, 64), 0.25)
    assert coordinates[0].x0 == 0 and coordinates[0].y0 == 0
    assert max(item.x1 for item in coordinates) == 150
    assert max(item.y1 for item in coordinates) == 130
    image = np.ones((130, 150, 3), dtype=np.uint8) * 17
    tiles = extract_tiles(image, coordinates)
    maps = [np.ones(tile.shape[:2], dtype=np.float32) * 0.7 for tile in tiles]
    merged = merge_tile_maps(maps, coordinates, image.shape[:2])
    np.testing.assert_allclose(merged, 0.7, atol=1e-6)
