"""Generate deterministic textile fixtures for pipeline tests, never benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def texture(index: int, size: int = 128) -> np.ndarray:
    """Create a deterministic woven RGB texture."""
    y, x = np.mgrid[:size, :size]
    base = 112 + 32 * np.sin((x + index * 3) / 6.0) + 24 * np.cos((y - index) / 9.0)
    weave = 10 * ((x % 4 == 0) | (y % 4 == 0))
    rgb = np.stack((base + weave, base * 0.82 + weave, base * 0.58), axis=-1)
    noise = np.random.default_rng(index).normal(0, 2.0, rgb.shape)
    return np.clip(rgb + noise, 0, 255).astype(np.uint8)


def add_anomaly(image: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Insert one controlled rectangular weave break and return its exact mask."""
    output = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    x0 = 18 + (index * 17) % 60
    y0 = 24 + (index * 13) % 54
    width = 16 + index % 4 * 3
    height = 10 + index % 3 * 4
    output[y0 : y0 + height, x0 : x0 + width] = np.array([235, 38, 50], dtype=np.uint8)
    mask[y0 : y0 + height, x0 : x0 + width] = 255
    return output, mask


def generate(root: Path) -> None:
    """Create an MVTec-like fixture tree with 20 images and four masks."""
    folders = {
        "train": root / "carpet" / "train" / "good",
        "test_good": root / "carpet" / "test" / "good",
        "test_defect": root / "carpet" / "test" / "weave_break",
        "masks": root / "carpet" / "ground_truth" / "weave_break",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    for index in range(12):
        Image.fromarray(texture(index)).save(folders["train"] / f"train_{index:03d}.png")
    for index in range(4):
        Image.fromarray(texture(100 + index)).save(folders["test_good"] / f"good_{index:03d}.png")
    for index in range(4):
        image, mask = add_anomaly(texture(200 + index), index)
        Image.fromarray(image).save(folders["test_defect"] / f"defect_{index:03d}.png")
        Image.fromarray(mask).save(folders["masks"] / f"defect_{index:03d}_mask.png")


def main() -> None:
    """Parse the target root and generate deterministic fixtures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/fixtures"))
    args = parser.parse_args()
    generate(args.root)


if __name__ == "__main__":
    main()
