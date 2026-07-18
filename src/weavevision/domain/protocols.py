"""Runtime-independent interfaces for model prediction and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class AnomalyPredictor(Protocol):
    """Protocol implemented by deployable anomaly model adapters."""

    @property
    def model_id(self) -> str:
        """Return immutable model identifier."""
        ...

    def predict_array(self, image_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        """Predict raw image score and full-resolution anomaly map."""
        ...

    def export(self, destination: Path) -> Path:
        """Export the model artifact and return its path."""
        ...
