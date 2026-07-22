"""Side-effect-free application configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from weavevision.domain.errors import ConfigError


class AppConfig(BaseModel):
    """User-facing application configuration."""

    model_config = ConfigDict(extra="forbid")
    name: str = "WeaveVision"
    organization_name: str | None = None
    environment: str = "local"
    language: str = "tr"
    max_upload_mb: int = Field(default=200, gt=0)
    telemetry: bool = False


class PathConfig(BaseModel):
    """Project-relative data and artifact paths."""

    model_config = ConfigDict(extra="forbid")
    data_root: Path = Path("data")
    artifacts_root: Path = Path("artifacts")
    database: Path = Path("artifacts/weavevision.sqlite3")


class RuntimeConfig(BaseModel):
    """Runtime device and determinism controls."""

    model_config = ConfigDict(extra="forbid")
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    precision: str = "auto"
    num_workers: int = Field(default=4, ge=0)
    deterministic: bool = True
    seed: int = 42


class InferenceConfig(BaseModel):
    """Image quality and tiled inference settings."""

    model_config = ConfigDict(extra="forbid")
    quality_gate_enabled: bool = True
    tiling_enabled: bool = True
    tile_size: tuple[int, int] = (512, 512)
    tile_overlap: float = Field(default=0.25, ge=0.0, lt=1.0)
    min_component_area_px: int = Field(default=16, ge=1)
    heatmap_alpha: float = Field(default=0.45, ge=0.0, le=1.0)


class ReportingConfig(BaseModel):
    """Report artifact persistence switches."""

    model_config = ConfigDict(extra="forbid")
    save_original_copy: bool = False
    save_heatmap: bool = True
    save_mask: bool = True
    save_overlay: bool = True


class DriftPolicyConfig(BaseModel):
    """Drift monitoring and alert policy thresholds.

    Loaded from ``configs/app.yaml -> drift``.  All default values are
    initial policies -- calibrate against operational-validation data only.
    """

    model_config = ConfigDict(extra="forbid")
    # EWMA/CUSUM parameters
    ewma_lambda: float = Field(default=0.25, gt=0.0, le=1.0)
    ewma_limit_sigma: float = Field(default=3.0, gt=0.0)
    cusum_k_sigma: float = Field(default=0.25, gt=0.0)
    cusum_h_sigma: float = Field(default=4.0, gt=0.0)
    # PSI thresholds
    psi_medium_threshold: float = Field(default=0.10, gt=0.0)
    psi_high_threshold: float = Field(default=0.25, gt=0.0)
    # Triage
    retraining_min_confirming_signals: int = Field(default=2, ge=1, le=6)
    # Sudden-drop thresholds (same unit as the monitored metric)
    sudden_drop_review_pp: float = Field(default=2.0, gt=0.0)
    sudden_drop_incident_pp: float = Field(default=5.0, gt=0.0)
    sudden_drop_block_pp: float = Field(default=10.0, gt=0.0)
    # Gradual drift thresholds
    gradual_window_weeks: int = Field(default=4, ge=1)
    gradual_weekly_drop_pp: float = Field(default=0.5, gt=0.0)
    gradual_min_consecutive_weeks: int = Field(default=3, ge=1)
    # Retraining governance
    retraining_min_target_images: int = Field(default=200, ge=1)
    retraining_min_labeled_validation_images: int = Field(default=100, ge=1)
    retraining_cooldown_days: int = Field(default=7, ge=0)
    # Canary evaluation thresholds (M7)
    canary_max_disagreement_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    canary_min_recall_delta: float = Field(default=-0.02)


class Settings(BaseModel):
    """Validated application settings with resolved absolute paths."""

    model_config = ConfigDict(extra="forbid")
    project_root: Path
    app: AppConfig
    paths: PathConfig
    runtime: RuntimeConfig
    inference: InferenceConfig
    reporting: ReportingConfig
    drift: DriftPolicyConfig = Field(default_factory=DriftPolicyConfig)

    def resolved_data_root(self) -> Path:
        """Return the absolute data root without creating it."""
        return self._resolve(self.paths.data_root)

    def resolved_artifacts_root(self) -> Path:
        """Return the absolute artifacts root without creating it."""
        return self._resolve(self.paths.artifacts_root)

    def resolved_database(self) -> Path:
        """Return the absolute SQLite path without opening it."""
        return self._resolve(self.paths.database)

    def resolved_database_path(self) -> Path:
        """Alias for resolved_database."""
        return self.resolved_database()

    def _resolve(self, path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest parent containing the master specification.

    Args:
        start: Directory used as the upward search origin.

    Returns:
        Absolute repository root.

    Raises:
        ConfigError: If no repository root marker can be found.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "WEAVEVISION_CURSOR_MASTER_BUILD_SPEC.md").is_file():
            return candidate
    raise ConfigError("WEAVEVISION_CURSOR_MASTER_BUILD_SPEC.md not found")


def load_settings(config_path: Path | None = None) -> Settings:
    """Load and validate YAML settings without import-time side effects.

    Args:
        config_path: Optional YAML configuration path.

    Returns:
        Fully validated settings with an explicit project root.

    Raises:
        ConfigError: If YAML cannot be read or fails schema validation.
    """
    root = find_project_root(config_path.parent if config_path else None)
    path = config_path or root / "configs" / "app.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ConfigError(f"configuration must be a mapping: {path}")
        return Settings(project_root=root, **payload)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"invalid configuration {path}: {exc}") from exc
