"""Run real PatchCore fixture inference and report-contract smoke evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weavevision.domain.schemas import ModelIdentity, ModelManifest, ThresholdArtifact
from weavevision.evaluation.benchmark import benchmark_callable
from weavevision.models.anomalib_adapter import AnomalibAdapter
from weavevision.services.analysis_service import AnalysisService
from weavevision.services.batch_service import BatchService
from weavevision.settings import load_settings


def run(run_id: str) -> dict[str, object]:
    """Analyze fixture normal/anomaly images and a 20-item batch using a real checkpoint."""
    settings = load_settings()
    run_root = settings.resolved_artifacts_root() / "experiments" / run_id
    run_manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    model_manifest = ModelManifest.model_validate_json(
        Path(run_manifest["model_manifest_path"]).read_text(encoding="utf-8")
    )
    threshold = ThresholdArtifact.model_validate_json(
        (run_root / "thresholds.json").read_text(encoding="utf-8")
    )
    predictor = AnomalibAdapter(
        model_id=model_manifest.model_id,
        algorithm=model_manifest.algorithm,
        model_config={"name": model_manifest.algorithm},
        artifact_path=model_manifest.artifact_path,
        device="cpu",
    )
    identity = ModelIdentity(
        model_id=model_manifest.model_id,
        model_name=model_manifest.algorithm,
        model_artifact_sha256=model_manifest.artifact_sha256,
        config_sha256=model_manifest.config_sha256,
    )
    service = AnalysisService(settings, predictor, identity, threshold)
    fixture_root = settings.project_root / "data" / "fixtures" / "carpet"
    normal = fixture_root / "test" / "good" / "good_000.png"
    anomaly = fixture_root / "test" / "weave_break" / "defect_000.png"
    normal_result = service.analyze(normal)
    anomaly_result = service.analyze(anomaly)
    batch = BatchService(service).analyze(fixture_root)
    latency = benchmark_callable(
        lambda: predictor.predict_array(normal), warmup_runs=2, measured_runs=5
    )
    evidence = {
        "status": "PASS",
        "purpose": "PIPELINE_SMOKE_ONLY_NO_BENCHMARK_CLAIMS",
        "run_id": run_id,
        "model_id": model_manifest.model_id,
        "threshold_id": threshold.threshold_id,
        "threshold_status": threshold.status,
        "normal_analysis_id": normal_result.analysis_id,
        "normal_decision": normal_result.prediction.decision.value,
        "anomaly_analysis_id": anomaly_result.analysis_id,
        "anomaly_decision": anomaly_result.prediction.decision.value,
        "batch_completed": len(batch.results),
        "batch_failures": len(batch.failures),
        "cpu_latency": latency,
    }
    destination = run_root / "fixture_smoke.json"
    destination.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


def main() -> None:
    """Parse a training run and emit smoke evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run_id), indent=2))


if __name__ == "__main__":
    main()
