"""Generate fixture-only perturbation deltas without tuning the model or threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from weavevision.domain.schemas import ModelManifest, ThresholdArtifact
from weavevision.evaluation.robustness import perturb
from weavevision.models.anomalib_adapter import AnomalibAdapter
from weavevision.settings import load_settings


def run(run_id: str) -> dict[str, object]:
    """Measure score deltas and decision flips over configured fixture perturbations."""
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
    root = settings.project_root / "data" / "fixtures" / "carpet" / "test"
    paths = sorted((root / "good").glob("*.png")) + sorted((root / "weave_break").glob("*.png"))
    images = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
    baseline = np.asarray([predictor.predict_array(image)[0] for image in images])
    cases = {
        "brightness": [0.7, 1.3],
        "contrast": [0.7, 1.3],
        "gaussian_noise": [10.0],
        "blur": [5.0],
        "jpeg": [50.0],
        "rotation": [-5.0, 5.0],
        "downsample": [0.5],
        "occlusion": [0.2],
    }
    results = []
    for kind, values in cases.items():
        for value in values:
            scores = np.asarray(
                [predictor.predict_array(perturb(image, kind, value))[0] for image in images]
            )
            results.append(
                {
                    "kind": kind,
                    "value": value,
                    "mean_score": float(scores.mean()),
                    "mean_absolute_delta": float(np.mean(np.abs(scores - baseline))),
                    "max_absolute_delta": float(np.max(np.abs(scores - baseline))),
                    "decision_flips": int(
                        np.sum(
                            (scores >= threshold.image_threshold)
                            != (baseline >= threshold.image_threshold)
                        )
                    ),
                }
            )
    evidence: dict[str, object] = {
        "status": "PASS_WITH_RESTRICTIONS",
        "purpose": "SYNTHETIC_FIXTURE_ROBUSTNESS_NO_MODEL_SELECTION",
        "run_id": run_id,
        "model_id": model_manifest.model_id,
        "threshold_id": threshold.threshold_id,
        "sample_count": len(images),
        "baseline_mean_score": float(baseline.mean()),
        "perturbations": results,
        "ablation": {
            "status": "NOT_RUN",
            "reason": "tile/coreset selection requires a verified real benchmark protocol",
        },
    }
    destination = run_root / "robustness.json"
    destination.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


def main() -> None:
    """Parse run identifier and emit robustness evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run_id), indent=2))


if __name__ == "__main__":
    main()
