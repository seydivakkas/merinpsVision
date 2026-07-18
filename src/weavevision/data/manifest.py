"""Canonical dataset manifest hashing and atomic persistence."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import orjson

from weavevision.domain.schemas import DatasetManifest


def canonical_json_bytes(manifest: DatasetManifest) -> bytes:
    """Serialize a manifest canonically while excluding its self-referential hash.

    Args:
        manifest: Validated dataset manifest.

    Returns:
        UTF-8 JSON bytes with sorted keys and no insignificant whitespace.
    """
    payload = manifest.model_dump(mode="json")
    payload["manifest_sha256"] = ""
    return orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)


def compute_manifest_sha256(manifest: DatasetManifest) -> str:
    """Compute the canonical SHA-256 identity of a dataset manifest."""
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def finalize_manifest(manifest: DatasetManifest) -> DatasetManifest:
    """Return a copy containing its canonical SHA-256 identity."""
    return manifest.model_copy(update={"manifest_sha256": compute_manifest_sha256(manifest)})


def write_manifest_atomic(manifest: DatasetManifest, destination: Path) -> Path:
    """Validate the hash and atomically write a formatted manifest.

    Args:
        manifest: Manifest to persist.
        destination: Target JSON path.

    Returns:
        Destination path.

    Raises:
        ValueError: If the embedded manifest hash is missing or invalid.
        OSError: If persistence fails.
    """
    expected = compute_manifest_sha256(manifest)
    if manifest.manifest_sha256 != expected:
        raise ValueError("manifest_sha256 does not match canonical content")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(
        orjson.dumps(manifest.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
    )
    os.replace(temporary, destination)
    return destination


def read_manifest(path: Path) -> DatasetManifest:
    """Read a manifest and verify its canonical identity."""
    manifest = DatasetManifest.model_validate_json(path.read_bytes())
    if manifest.manifest_sha256 != compute_manifest_sha256(manifest):
        raise ValueError(f"manifest integrity check failed: {path}")
    return manifest
