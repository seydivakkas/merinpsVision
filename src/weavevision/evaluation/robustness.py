"""Deterministic image perturbations for robustness evaluation."""

from __future__ import annotations

import cv2
import numpy as np


def perturb(image: np.ndarray, kind: str, value: float, seed: int = 42) -> np.ndarray:
    """Apply one configured perturbation without changing source evidence."""
    if kind == "brightness":
        return np.clip(image.astype(float) * value, 0, 255).astype(np.uint8)
    if kind == "contrast":
        centered = (image.astype(float) - 127.5) * value + 127.5
        return np.clip(centered, 0, 255).astype(np.uint8)
    if kind == "gaussian_noise":
        noise = np.random.default_rng(seed).normal(0, value, image.shape)
        return np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
    if kind == "blur":
        kernel = int(value)
        if kernel % 2 == 0 or kernel < 1:
            raise ValueError("blur kernel must be a positive odd integer")
        return cv2.GaussianBlur(image, (kernel, kernel), 0)
    if kind == "rotation":
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), value, 1.0)
        return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    if kind == "downsample":
        height, width = image.shape[:2]
        small = cv2.resize(image, None, fx=value, fy=value, interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    if kind == "jpeg":
        success, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(value)]
        )
        if not success:
            raise ValueError("JPEG perturbation encoding failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("JPEG perturbation decoding failed")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if kind == "occlusion":
        fraction = float(value)
        if not 0.0 < fraction < 1.0:
            raise ValueError("occlusion fraction must be in (0, 1)")
        output = image.copy()
        height, width = image.shape[:2]
        block_height = max(1, int(height * fraction))
        block_width = max(1, int(width * fraction))
        y0 = (height - block_height) // 2
        x0 = (width - block_width) // 2
        output[y0 : y0 + block_height, x0 : x0 + block_width] = np.median(
            image.reshape(-1, 3), axis=0
        ).astype(np.uint8)
        return output
    raise ValueError(f"unsupported perturbation: {kind}")
