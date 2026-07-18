"""Group split and leakage audit tests."""

from datetime import UTC, datetime

import pytest

from weavevision.data.audit import audit_manifest_leakage, require_no_leakage
from weavevision.data.split import deterministic_group_split
from weavevision.domain.enums import DatasetVerificationStatus
from weavevision.domain.errors import DataLeakageError
from weavevision.domain.schemas import (
    DatasetCounts,
    DatasetFile,
    DatasetManifest,
    DatasetSource,
    SplitPolicy,
)


def test_group_split_keeps_parent_items_together() -> None:
    items = list(range(8))
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    train, validation = deterministic_group_split(items, groups, 0.25, 42)
    for left, right in ((0, 1), (2, 3), (4, 5), (6, 7)):
        assert (left in train and right in train) or (left in validation and right in validation)
    assert deterministic_group_split(items, groups, 0.25, 42) == (train, validation)


def test_audit_rejects_exact_duplicate_across_splits() -> None:
    files = [
        DatasetFile(
            relative_path="train/a.png",
            sha256="a" * 64,
            width=64,
            height=64,
            label="normal",
            split="train",
            source_image_id="source-a",
        ),
        DatasetFile(
            relative_path="test/a.png",
            sha256="a" * 64,
            width=64,
            height=64,
            label="normal",
            split="test",
            source_image_id="source-b",
        ),
    ]
    value = DatasetManifest(
        dataset_id="leaky",
        source=DatasetSource(
            name="fixture",
            category="test",
            license="local",
            commercial_use=True,
            retrieved_at=datetime.now(UTC),
            source_url="local:test",
        ),
        counts=DatasetCounts(
            images_total=2,
            train_normal=1,
            validation_normal=0,
            validation_anomaly=0,
            test_normal=1,
            test_anomaly=0,
            masks=0,
        ),
        split_policy=SplitPolicy(method="group", seed=42, group_key="source_image_id"),
        files=files,
        verification_status=DatasetVerificationStatus.VERIFIED,
    )
    assert audit_manifest_leakage(value)["status"] == "FAIL"
    with pytest.raises(DataLeakageError):
        require_no_leakage(value)
