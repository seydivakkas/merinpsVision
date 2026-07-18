"""Analysis result JSON contract tests."""

from datetime import UTC, datetime

from weavevision.domain.enums import Decision, QualityGateStatus, ReviewPriority
from weavevision.domain.schemas import (
    AnalysisResult,
    PredictionResult,
    QualityGateResult,
    SourceImageMetadata,
    TimingResult,
)


def test_analysis_result_round_trips_json() -> None:
    value = AnalysisResult(
        analysis_id="ana_test",
        run_id="run_test",
        created_at=datetime.now(UTC),
        source=SourceImageMetadata(
            filename="sample.png", sha256="a" * 64, width=128, height=128, mode="RGB"
        ),
        quality_gate=QualityGateResult(status=QualityGateStatus.PASS),
        model=None,
        threshold=None,
        prediction=PredictionResult(
            decision=Decision.ABSTAIN,
            raw_anomaly_score=0.0,
            review_priority=ReviewPriority.ABSTAIN,
            anomaly_area_ratio=0.0,
            region_count=0,
        ),
        timing_ms=TimingResult(quality_gate=1, preprocess=0, inference=0, postprocess=0, total=1),
    )
    assert AnalysisResult.model_validate_json(value.model_dump_json()) == value
