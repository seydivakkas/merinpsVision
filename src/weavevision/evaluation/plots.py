"""Evidence plots generated from real experiment arrays."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_score_distribution(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
    destination: Path,
) -> Path:
    """Save a score distribution plot with the locked threshold."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(normal_scores, bins=30, alpha=0.7, label="normal")
    axis.hist(anomaly_scores, bins=30, alpha=0.7, label="anomaly")
    axis.axvline(threshold, color="black", linestyle="--", label="locked threshold")
    axis.set_xlabel("Anomaly score (not probability)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination
