"""Streamlit resource loading keyed by immutable model provenance."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from weavevision.domain.schemas import ModelIdentity, ThresholdArtifact
from weavevision.models.anomalib_adapter import AnomalibAdapter
from weavevision.services.analysis_service import AnalysisService
from weavevision.settings import load_settings


@st.cache_resource(show_spinner=False)
def analysis_service(
    model_id: str | None,
    artifact_sha256: str | None,
    threshold_id: str | None,
    preprocessing_hash: str | None,
    algorithm: str | None = None,
    artifact_path: str | None = None,
    config_sha256: str | None = None,
    threshold_json: str | None = None,
) -> AnalysisService:
    """Load immutable analysis resources using the required cache identity tuple."""
    del preprocessing_hash
    settings = load_settings()
    if (
        model_id is None
        or artifact_sha256 is None
        or threshold_id is None
        or algorithm is None
        or artifact_path is None
        or config_sha256 is None
        or threshold_json is None
    ):
        return AnalysisService(settings)
    threshold = ThresholdArtifact.model_validate_json(threshold_json)
    predictor = AnomalibAdapter(
        model_id=model_id,
        algorithm=algorithm,
        model_config={"name": algorithm},
        artifact_path=Path(artifact_path),
    )
    identity = ModelIdentity(
        model_id=model_id,
        model_name=algorithm,
        model_artifact_sha256=artifact_sha256,
        config_sha256=config_sha256,
    )
    return AnalysisService(settings, predictor, identity, threshold)


def clear_model_cache() -> None:
    """Invalidate cached resources after model or threshold promotion."""
    analysis_service.clear()
