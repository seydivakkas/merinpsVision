"""Single-image quality, inference, reporting, and audit orchestration."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from weavevision.data.transforms import load_image_rgb
from weavevision.domain.enums import Decision, QualityGateStatus, ReviewPriority
from weavevision.domain.errors import ModelNotReadyError, ThresholdNotFoundError
from weavevision.domain.protocols import AnomalyPredictor
from weavevision.domain.schemas import (
    AnalysisResult,
    ModelIdentity,
    PredictionResult,
    RegionResult,
    ThresholdArtifact,
    ThresholdIdentity,
    TimingResult,
)
from weavevision.inference.overlay import render_artifacts
from weavevision.inference.postprocess import postprocess_prediction
from weavevision.inference.predictor import predict_image
from weavevision.inference.quality_gate import evaluate_quality
from weavevision.persistence.database import Database
from weavevision.persistence.repositories import AnalysisRepository
from weavevision.services.report_service import ReportService
from weavevision.settings import Settings


class AnalysisService:
    """Framework-agnostic end-to-end analysis transaction boundary."""

    def __init__(
        self,
        settings: Settings,
        predictor: AnomalyPredictor | None = None,
        model: ModelIdentity | None = None,
        threshold: ThresholdArtifact | None = None,
    ) -> None:
        self.settings = settings
        self.predictor = predictor
        self.model = model
        self.threshold = threshold
        database = Database(settings.resolved_database())
        database.migrate()
        self.repository = AnalysisRepository(database)
        self.reports = ReportService(settings)

    def analyze(self, source_path: Path, output_root: Path | None = None) -> AnalysisResult:
        """Analyze one image, persist audit evidence, and return its structured result.

        Raises:
            ModelNotReadyError: If a valid input requires inference but no model is active.
            ThresholdNotFoundError: If a model lacks a calibration artifact.
        """
        total_started = time.perf_counter()
        image, metadata = load_image_rgb(source_path)
        quality_started = time.perf_counter()
        quality = evaluate_quality(image)
        quality_ms = (time.perf_counter() - quality_started) * 1000
        heatmap: np.ndarray | None = None
        overlay: np.ndarray | None = None
        mask: np.ndarray | None = None
        preprocess_ms = inference_ms = postprocess_ms = 0.0
        if quality.status is QualityGateStatus.ABSTAIN:
            prediction = PredictionResult(
                decision=Decision.ABSTAIN,
                raw_anomaly_score=0.0,
                normalized_anomaly_score=None,
                review_priority=ReviewPriority.ABSTAIN,
                anomaly_area_ratio=0.0,
                region_count=0,
            )
            regions: list[RegionResult] = []
            model = None
            threshold_identity = None
        else:
            if self.predictor is None or self.model is None:
                raise ModelNotReadyError("no active integrity-verified model is available")
            if self.threshold is None:
                raise ThresholdNotFoundError("active model has no validation threshold")
            raw = predict_image(
                self.predictor,
                image,
                tiled=self.settings.inference.tiling_enabled,
                tile_size=self.settings.inference.tile_size,
                overlap=self.settings.inference.tile_overlap,
            )
            preprocess_ms, inference_ms = raw.preprocess_ms, raw.inference_ms
            post_started = time.perf_counter()
            prediction, mask, regions = postprocess_prediction(
                raw.score,
                raw.anomaly_map,
                quality,
                self.threshold.image_threshold,
                self.threshold.pixel_threshold,
                self.settings.inference.min_component_area_px,
            )
            heatmap, overlay = render_artifacts(
                image, raw.anomaly_map, mask, self.settings.inference.heatmap_alpha
            )
            postprocess_ms = (time.perf_counter() - post_started) * 1000
            model = self.model
            threshold_identity = ThresholdIdentity(
                threshold_id=self.threshold.threshold_id,
                image_threshold=self.threshold.image_threshold,
                pixel_threshold=self.threshold.pixel_threshold,
            )
        total_ms = (time.perf_counter() - total_started) * 1000
        result = AnalysisResult(
            analysis_id=f"ana_{uuid4().hex}",
            run_id=f"run_{uuid4().hex}",
            created_at=datetime.now(UTC),
            source=metadata,
            quality_gate=quality,
            model=model,
            threshold=threshold_identity,
            prediction=prediction,
            regions=regions,
            timing_ms=TimingResult(
                quality_gate=quality_ms,
                preprocess=preprocess_ms,
                inference=inference_ms,
                postprocess=postprocess_ms,
                total=total_ms,
            ),
            warnings=list(quality.reasons),
        )
        destination = output_root or self.settings.resolved_artifacts_root() / "reports"
        result = self.reports.write_analysis(
            result, destination, heatmap=heatmap, mask=mask, overlay=overlay
        )
        if result.artifacts.json_path is None:
            raise RuntimeError("report service did not create the analysis JSON path")
        self.repository.save(result, result.artifacts.json_path)
        return result
