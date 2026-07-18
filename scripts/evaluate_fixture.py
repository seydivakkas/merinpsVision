"""Evaluate the sealed synthetic fixture test split without benchmark claims."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from weavevision.domain.schemas import ModelManifest, ThresholdArtifact
from weavevision.evaluation.metrics import image_metrics, pixel_metrics
from weavevision.evaluation.plots import save_score_distribution
from weavevision.models.anomalib_adapter import AnomalibAdapter
from weavevision.settings import load_settings


def evaluate(run_id: str) -> dict[str, object]:
    """Generate metrics, confusion evidence, per-image rows, plot, and failure gallery."""
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
    fixture_root = settings.project_root / "data" / "fixtures" / "carpet"
    samples = [
        *((path, 0, None) for path in sorted((fixture_root / "test" / "good").glob("*.png"))),
        *(
            (
                path,
                1,
                fixture_root / "ground_truth" / "weave_break" / f"{path.stem}_mask.png",
            )
            for path in sorted((fixture_root / "test" / "weave_break").glob("*.png"))
        ),
    ]
    labels: list[int] = []
    scores: list[float] = []
    maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    gallery = run_root / "failure_gallery"
    for path, label, mask_path in samples:
        image = np.asarray(Image.open(path).convert("RGB"))
        score, anomaly_map = predictor.predict_array(image)
        anomaly_map = cv2.resize(anomaly_map, (image.shape[1], image.shape[0]))
        mask = (
            np.asarray(Image.open(mask_path).convert("L")) > 0
            if mask_path
            else np.zeros(image.shape[:2], dtype=bool)
        )
        predicted = int(score >= threshold.image_threshold)
        outcome = {
            (0, 0): "true_negative",
            (0, 1): "false_positive",
            (1, 0): "false_negative",
            (1, 1): "true_positive",
        }[(label, predicted)]
        target = gallery / outcome / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        labels.append(label)
        scores.append(score)
        maps.append(anomaly_map)
        masks.append(mask)
        rows.append(
            {
                "filename": path.name,
                "true_label": label,
                "predicted_label": predicted,
                "raw_score": score,
                "threshold": threshold.image_threshold,
                "outcome": outcome,
                "model_id": model_manifest.model_id,
            }
        )
    labels_array = np.asarray(labels)
    scores_array = np.asarray(scores)
    metrics = image_metrics(labels_array, scores_array, threshold.image_threshold)
    metrics.update(pixel_metrics(np.asarray(masks), np.asarray(maps), threshold.pixel_threshold))
    evidence: dict[str, object] = {
        "status": "PASS_WITH_RESTRICTIONS",
        "purpose": "SYNTHETIC_FIXTURE_PIPELINE_EVIDENCE_ONLY",
        "restriction": "not an open-source benchmark or company performance result",
        "run_id": run_id,
        "model_id": model_manifest.model_id,
        "threshold_id": threshold.threshold_id,
        "threshold_status": threshold.status,
        "metrics": metrics,
    }
    (run_root / "metrics.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    with (run_root / "per_image_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (run_root / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "predicted_normal", "predicted_anomaly"])
        writer.writerow(["actual_normal", metrics["true_negative"], metrics["false_positive"]])
        writer.writerow(["actual_anomaly", metrics["false_negative"], metrics["true_positive"]])
    save_score_distribution(
        scores_array[labels_array == 0],
        scores_array[labels_array == 1],
        threshold.image_threshold,
        run_root / "plots" / "score_distribution.png",
    )
    return evidence


def main() -> None:
    """Parse run identifier and print fixture evaluation evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run_id), indent=2))


if __name__ == "__main__":
    main()
