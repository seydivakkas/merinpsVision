"""MVTec AD category verifier with mask pairing and canonical manifests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from weavevision.data.adapters.base import DatasetAdapter
from weavevision.data.audit import require_no_leakage
from weavevision.data.manifest import finalize_manifest, write_manifest_atomic
from weavevision.data.transforms import SUPPORTED_SUFFIXES, sha256_file
from weavevision.domain.enums import DatasetVerificationStatus
from weavevision.domain.errors import DatasetNotFoundError, DatasetStructureError
from weavevision.domain.schemas import (
    DatasetCounts,
    DatasetFile,
    DatasetManifest,
    DatasetSource,
    SplitPolicy,
)


class MVTecADAdapter(DatasetAdapter):
    """Verify one official MVTec AD category without modifying source files."""

    def __init__(self, root: Path, manifest_path: Path, category: str = "carpet") -> None:
        super().__init__(root, manifest_path)
        self.category = category

    def verify(self) -> DatasetManifest:
        """Validate layout, decode images, pair masks, and write a manifest.

        Raises:
            DatasetNotFoundError: If the category directory is missing.
            DatasetStructureError: If required folders or mask pairs are invalid.
        """
        category_root = self.root / self.category
        train_good = category_root / "train" / "good"
        test_root = category_root / "test"
        ground_truth = category_root / "ground_truth"
        if not category_root.is_dir():
            raise DatasetNotFoundError(f"MVTec category not found: {category_root}")
        if not train_good.is_dir() or not test_root.is_dir():
            raise DatasetStructureError("MVTec category requires train/good and test directories")
        files: list[DatasetFile] = []
        for path in self._images(train_good):
            files.append(self._item(category_root, path, "train", "normal", None, None))
        for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            label = "normal" if defect_dir.name == "good" else "anomaly"
            for path in self._images(defect_dir):
                mask: Path | None = None
                if label == "anomaly":
                    candidates = [
                        ground_truth / defect_dir.name / f"{path.stem}_mask{path.suffix}",
                        ground_truth / defect_dir.name / f"{path.stem}_mask.png",
                    ]
                    mask = next(
                        (candidate for candidate in candidates if candidate.is_file()), None
                    )
                    if mask is None:
                        raise DatasetStructureError(f"missing ground-truth mask for {path}")
                files.append(self._item(category_root, path, "test", label, defect_dir.name, mask))
        if not files:
            raise DatasetStructureError("MVTec category contains no supported images")
        manifest = finalize_manifest(
            DatasetManifest(
                dataset_id=f"mvtec_ad_{self.category}",
                source=DatasetSource(
                    name="MVTec Anomaly Detection Dataset",
                    category=self.category,
                    license="Official MVTec AD terms; non-commercial restrictions apply",
                    commercial_use=False,
                    retrieved_at=datetime.now(UTC),
                    source_url="https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
                ),
                counts=self._counts(files),
                split_policy=SplitPolicy(
                    method="official_train_test; validation pending train group split",
                    seed=42,
                    group_key="source_image_id",
                ),
                files=files,
                verification_status=DatasetVerificationStatus.VERIFIED,
            )
        )
        require_no_leakage(manifest)
        write_manifest_atomic(manifest, self.manifest_path)
        return manifest

    @staticmethod
    def _images(root: Path) -> list[Path]:
        return sorted(
            path for path in root.rglob("*") if path.suffix.casefold() in SUPPORTED_SUFFIXES
        )

    @staticmethod
    def _item(
        category_root: Path,
        path: Path,
        split: str,
        label: str,
        defect_type: str | None,
        mask: Path | None,
    ) -> DatasetFile:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        return DatasetFile(
            relative_path=path.relative_to(category_root).as_posix(),
            sha256=sha256_file(path),
            width=width,
            height=height,
            label=label,
            split=split,
            defect_type=defect_type,
            mask_path=mask.relative_to(category_root).as_posix() if mask else None,
            source_image_id=path.stem,
        )

    @staticmethod
    def _counts(files: list[DatasetFile]) -> DatasetCounts:
        return DatasetCounts(
            images_total=len(files),
            train_normal=sum(item.split == "train" and item.label == "normal" for item in files),
            validation_normal=sum(
                item.split == "validation" and item.label == "normal" for item in files
            ),
            validation_anomaly=sum(
                item.split == "validation" and item.label == "anomaly" for item in files
            ),
            test_normal=sum(item.split == "test" and item.label == "normal" for item in files),
            test_anomaly=sum(item.split == "test" and item.label == "anomaly" for item in files),
            masks=sum(item.mask_path is not None for item in files),
        )
