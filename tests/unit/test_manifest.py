"""Canonical manifest integrity tests."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from weavevision.data.manifest import (
    compute_manifest_sha256,
    finalize_manifest,
    read_manifest,
    write_manifest_atomic,
)
from weavevision.domain.enums import DatasetVerificationStatus
from weavevision.domain.schemas import (
    DatasetCounts,
    DatasetManifest,
    DatasetSource,
    SplitPolicy,
)


def manifest() -> DatasetManifest:
    """Return a minimal valid manifest candidate."""
    return DatasetManifest(
        dataset_id="fixture",
        source=DatasetSource(
            name="fixture",
            category="carpet",
            license="local",
            commercial_use=True,
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            source_url="local:fixture",
        ),
        counts=DatasetCounts(
            images_total=0,
            train_normal=0,
            validation_normal=0,
            validation_anomaly=0,
            test_normal=0,
            test_anomaly=0,
            masks=0,
        ),
        split_policy=SplitPolicy(method="group", seed=42, group_key="source_image_id"),
        files=[],
        verification_status=DatasetVerificationStatus.VERIFIED,
    )


def test_manifest_hash_is_canonical_and_round_trips(tmp_path: Path) -> None:
    value = finalize_manifest(manifest())
    destination = write_manifest_atomic(value, tmp_path / "manifest.json")
    loaded = read_manifest(destination)
    assert loaded.manifest_sha256 == value.manifest_sha256
    assert compute_manifest_sha256(loaded) == value.manifest_sha256


def test_manifest_rejects_tampered_hash(tmp_path: Path) -> None:
    value = manifest().model_copy(update={"manifest_sha256": "0" * 64})
    with pytest.raises(ValueError, match="does not match"):
        write_manifest_atomic(value, tmp_path / "manifest.json")


def test_split_policy_rejects_test_calibration() -> None:
    with pytest.raises(ValueError, match="sealed test"):
        SplitPolicy(method="invalid", seed=42, group_key="source", test_used_for_calibration=True)
