"""Safe image decoding with EXIF correction and explicit RGB conversion."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from weavevision.domain.errors import ImageValidationError, UnsupportedFormatError
from weavevision.domain.schemas import SourceImageMetadata

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image_rgb(path: Path) -> tuple[np.ndarray, SourceImageMetadata]:
    """Safely decode, orient, and convert an image to uint8 RGB.

    Raises:
        UnsupportedFormatError: If the suffix is unsupported.
        ImageValidationError: If decoding or controlled conversion fails.
    """
    if path.suffix.casefold() not in SUPPORTED_SUFFIXES:
        raise UnsupportedFormatError(f"unsupported image format: {path.suffix}")
    try:
        with Image.open(path) as opened:
            opened.verify()
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            if image.mode.startswith("I;16"):
                array16 = np.asarray(image, dtype=np.uint16)
                low, high = np.percentile(array16, (0.5, 99.5))
                if high <= low:
                    converted = np.zeros_like(array16, dtype=np.uint8)
                else:
                    converted = np.clip((array16 - low) * 255 / (high - low), 0, 255).astype(
                        np.uint8
                    )
                image = Image.fromarray(converted, mode="L")
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ImageValidationError(f"unable to decode image {path.name}: {exc}") from exc
    metadata = SourceImageMetadata(
        filename=path.name,
        sha256=sha256_file(path),
        width=array.shape[1],
        height=array.shape[0],
        mode="RGB",
    )
    return array, metadata
