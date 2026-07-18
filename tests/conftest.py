"""Shared deterministic test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def textile_image(tmp_path: Path) -> Path:
    """Create one deterministic valid textile-like image."""
    y, x = np.mgrid[:128, :128]
    base = 110 + 30 * np.sin(x / 5) + 20 * np.cos(y / 7)
    rgb = np.stack((base, base * 0.8, base * 0.6), axis=-1)
    path = tmp_path / "textile.png"
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)
    return path
