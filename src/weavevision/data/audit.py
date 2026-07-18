"""Exact duplicate and source-identity leakage auditing."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from weavevision.domain.errors import DataLeakageError
from weavevision.domain.schemas import DatasetManifest


def audit_manifest_leakage(manifest: DatasetManifest) -> dict[str, Any]:
    """Audit duplicate hashes and source groups across dataset splits.

    Args:
        manifest: Verified manifest candidate.

    Returns:
        Structured audit with collisions and a PASS/FAIL status.
    """
    by_hash: dict[str, set[str]] = defaultdict(set)
    by_source: dict[str, set[str]] = defaultdict(set)
    by_normalized_name: dict[str, set[str]] = defaultdict(set)
    for item in manifest.files:
        by_hash[item.sha256].add(item.split)
        by_source[item.source_image_id].add(item.split)
        normalized = Path(item.relative_path).stem.casefold().replace("-", "").replace("_", "")
        by_normalized_name[normalized].add(item.split)
    exact = {key: sorted(value) for key, value in by_hash.items() if len(value) > 1}
    sources = {key: sorted(value) for key, value in by_source.items() if len(value) > 1}
    names = {key: sorted(value) for key, value in by_normalized_name.items() if len(value) > 1}
    return {
        "status": "FAIL" if exact or sources else "PASS",
        "dataset_id": manifest.dataset_id,
        "manifest_sha256": manifest.manifest_sha256,
        "exact_hash_cross_split": exact,
        "source_identity_cross_split": sources,
        "normalized_name_cross_split": names,
    }


def require_no_leakage(manifest: DatasetManifest) -> dict[str, Any]:
    """Return audit evidence or raise when critical leakage exists.

    Raises:
        DataLeakageError: If exact hashes or source identities cross splits.
    """
    report = audit_manifest_leakage(manifest)
    if report["status"] != "PASS":
        raise DataLeakageError("dataset contains duplicate or source-identity cross-split leakage")
    return report


def average_hash(path: Path, hash_size: int = 8) -> str:
    """Compute a compact perceptual average hash for diagnostic near-duplicate review."""
    with Image.open(path) as image:
        gray = image.convert("L").resize((hash_size, hash_size))
        pixels = list(gray.getdata())
    mean = sum(pixels) / len(pixels)
    bits = "".join("1" if value >= mean else "0" for value in pixels)
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"
