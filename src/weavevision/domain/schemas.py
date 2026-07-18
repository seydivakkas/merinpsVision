"""Pydantic contracts for datasets, models, predictions, and reports."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from weavevision.domain.enums import (
    DatasetVerificationStatus,
    Decision,
    ExperimentStatus,
    ModelStatus,
    QualityGateStatus,
    ReviewPriority,
)


class ContractModel(BaseModel):
    """Base model with strict input and JSON-safe path handling."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class SourceImageMetadata(ContractModel):
    """Immutable source image identity and geometry."""

    filename: str
    sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mode: str


class QualityGateResult(ContractModel):
    """Quality gate decision, reasons, and measured input properties."""

    status: QualityGateStatus
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)


class RegionResult(ContractModel):
    """Connected anomalous image region in original coordinates."""

    region_id: int = Field(ge=1)
    bbox_xyxy: tuple[int, int, int, int]
    area_pixels: int = Field(ge=0)
    area_ratio: float = Field(ge=0.0, le=1.0)
    mean_anomaly_score: float
    max_anomaly_score: float
    centroid_xy: tuple[float, float]
    contour: list[tuple[int, int]] = Field(default_factory=list)


class PredictionResult(ContractModel):
    """Framework-agnostic prediction values and generated arrays."""

    decision: Decision
    raw_anomaly_score: float
    normalized_anomaly_score: float | None = None
    review_priority: ReviewPriority
    anomaly_area_ratio: float = Field(ge=0.0, le=1.0)
    region_count: int = Field(ge=0)


class ArtifactPaths(ContractModel):
    """Files produced for one analysis."""

    original_path: Path | None = None
    overlay_path: Path | None = None
    mask_path: Path | None = None
    heatmap_path: Path | None = None
    json_path: Path | None = None
    html_path: Path | None = None


class TimingResult(ContractModel):
    """Measured stage timings in milliseconds."""

    quality_gate: float = Field(ge=0.0)
    preprocess: float = Field(ge=0.0)
    inference: float = Field(ge=0.0)
    postprocess: float = Field(ge=0.0)
    total: float = Field(ge=0.0)


class ModelIdentity(ContractModel):
    """Model provenance embedded in every analysis."""

    model_id: str
    model_name: str
    model_artifact_sha256: str
    config_sha256: str


class ThresholdIdentity(ContractModel):
    """Locked threshold provenance embedded in every analysis."""

    threshold_id: str
    image_threshold: float
    pixel_threshold: float


class AnalysisResult(ContractModel):
    """Audit-ready single-image analysis contract."""

    schema_version: str = "1.0.0"
    analysis_id: str
    run_id: str
    created_at: datetime
    source: SourceImageMetadata
    quality_gate: QualityGateResult
    model: ModelIdentity | None
    threshold: ThresholdIdentity | None
    prediction: PredictionResult
    regions: list[RegionResult] = Field(default_factory=list)
    artifacts: ArtifactPaths = Field(default_factory=ArtifactPaths)
    timing_ms: TimingResult
    warnings: list[str] = Field(default_factory=list)


class BatchItemFailure(ContractModel):
    """Isolated failure for a single batch item."""

    filename: str
    error_code: str
    message: str


class BatchResult(ContractModel):
    """Batch analysis aggregate with partial failures."""

    schema_version: str = "1.0.0"
    batch_id: str
    created_at: datetime
    results: list[AnalysisResult] = Field(default_factory=list)
    failures: list[BatchItemFailure] = Field(default_factory=list)


class DatasetFile(ContractModel):
    """One manifest file with identity, geometry, and split provenance."""

    relative_path: str
    sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    label: Literal["normal", "anomaly"]
    split: Literal["train", "validation", "test"]
    defect_type: str | None = None
    mask_path: str | None = None
    source_image_id: str


class DatasetSource(ContractModel):
    """Dataset source and license metadata."""

    name: str
    category: str
    license: str
    commercial_use: bool | None
    retrieved_at: datetime
    source_url: str


class DatasetCounts(ContractModel):
    """Verified dataset split counts."""

    images_total: int = Field(ge=0)
    train_normal: int = Field(ge=0)
    validation_normal: int = Field(ge=0)
    validation_anomaly: int = Field(ge=0)
    test_normal: int = Field(ge=0)
    test_anomaly: int = Field(ge=0)
    masks: int = Field(ge=0)


class SplitPolicy(ContractModel):
    """Split provenance and calibration isolation policy."""

    method: str
    seed: int
    group_key: str
    test_used_for_calibration: bool = False

    @model_validator(mode="after")
    def reject_test_calibration(self) -> SplitPolicy:
        """Reject any manifest that allows sealed test calibration."""
        if self.test_used_for_calibration:
            raise ValueError("sealed test data cannot be used for calibration")
        return self


class DatasetManifest(ContractModel):
    """Canonical dataset manifest and verification state."""

    schema_version: str = "1.0.0"
    dataset_id: str
    source: DatasetSource
    counts: DatasetCounts
    split_policy: SplitPolicy
    files: list[DatasetFile]
    verification_status: DatasetVerificationStatus
    manifest_sha256: str = ""


class ThresholdArtifact(ContractModel):
    """Immutable validation-derived image and pixel thresholds."""

    schema_version: str = "1.0.0"
    threshold_id: str
    model_id: str
    dataset_manifest_sha256: str
    image_threshold: float
    pixel_threshold: float
    method: str
    target_normal_fpr: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_split: Literal["validation"] = "validation"
    created_at: datetime
    status: Literal["LOCKED", "PROVISIONAL_NORMAL_ONLY"]


class ModelManifest(ContractModel):
    """Model registry metadata and integrity evidence."""

    schema_version: str = "1.0.0"
    model_id: str
    status: ModelStatus
    algorithm: Literal["patchcore", "efficient_ad", "padim"]
    dataset_manifest_sha256: str
    training_run_id: str
    config_sha256: str
    artifact_path: Path
    artifact_sha256: str
    threshold_id: str | None = None
    metrics_path: Path | None = None
    created_at: datetime
    promoted_at: datetime | None = None
    promotion_reason: str | None = None


class BenchmarkResult(ContractModel):
    """Evidence-backed benchmark result; absent metrics remain absent."""

    schema_version: str = "1.0.0"
    run_id: str
    status: ExperimentStatus
    model_id: str
    dataset_id: str
    metrics: dict[str, float | int | None] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    restrictions: list[str] = Field(default_factory=list)
    environment: dict[str, Any] = Field(default_factory=dict)
