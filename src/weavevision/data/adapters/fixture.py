"""Programmatic fixture adapter isolated from benchmark claims."""

from __future__ import annotations

from datetime import UTC, datetime

from weavevision.data.adapters.mvtec_ad import MVTecADAdapter
from weavevision.data.manifest import finalize_manifest, write_manifest_atomic
from weavevision.domain.schemas import DatasetManifest, DatasetSource, SplitPolicy

FIXTURE_RETRIEVED_AT = datetime(2026, 1, 1, tzinfo=UTC)


class FixtureAdapter(MVTecADAdapter):
    """Verify deterministic synthetic fixtures using the strict MVTec-like layout."""

    def verify(self) -> DatasetManifest:
        """Verify fixtures and replace external-dataset provenance with test-only provenance."""
        base = super().verify()
        fixture = finalize_manifest(
            base.model_copy(
                update={
                    "dataset_id": "weavevision_fixture",
                    "source": DatasetSource(
                        name="Programmatic WeaveVision Fixture",
                        category="synthetic_textile",
                        license="Project test fixture",
                        commercial_use=True,
                        retrieved_at=FIXTURE_RETRIEVED_AT,
                        source_url="local:scripts/generate_fixtures.py",
                    ),
                    "split_policy": SplitPolicy(
                        method="programmatic train/test; normal-only provisional calibration",
                        seed=42,
                        group_key="source_image_id",
                    ),
                    "manifest_sha256": "",
                }
            )
        )
        write_manifest_atomic(fixture, self.manifest_path)
        return fixture
