"""Integrity-checked application service composition."""

from __future__ import annotations

from weavevision.domain.errors import ModelNotReadyError, ThresholdNotFoundError
from weavevision.domain.schemas import ModelIdentity, ThresholdArtifact
from weavevision.models.anomalib_adapter import AnomalibAdapter
from weavevision.models.registry import ModelRegistry
from weavevision.services.analysis_service import AnalysisService
from weavevision.settings import Settings


def load_active_analysis_service(settings: Settings, *, company: bool = False) -> AnalysisService:
    """Compose analysis service from the hash-verified active model and threshold artifact."""
    registry = ModelRegistry(settings.resolved_artifacts_root() / "models")
    manifest = registry.active(company=company)
    if manifest is None:
        raise ModelNotReadyError("no active model is registered")
    if manifest.threshold_id is None:
        raise ThresholdNotFoundError("active model has no threshold identity")
    candidates = settings.resolved_artifacts_root().glob("experiments/*/thresholds*.json")
    threshold = None
    for path in candidates:
        candidate = ThresholdArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        if candidate.threshold_id == manifest.threshold_id:
            threshold = candidate
            break
    if threshold is None:
        raise ThresholdNotFoundError(f"threshold artifact not found: {manifest.threshold_id}")
    predictor = AnomalibAdapter(
        model_id=manifest.model_id,
        algorithm=manifest.algorithm,
        model_config={"name": manifest.algorithm},
        artifact_path=manifest.artifact_path,
        device=settings.runtime.device,
    )
    identity = ModelIdentity(
        model_id=manifest.model_id,
        model_name=manifest.algorithm,
        model_artifact_sha256=manifest.artifact_sha256,
        config_sha256=manifest.config_sha256,
    )
    return AnalysisService(settings, predictor, identity, threshold)
