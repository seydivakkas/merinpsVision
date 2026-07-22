"""Reproducible Anomalib training orchestration and candidate registration."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import yaml

from weavevision.data.adapters.factory import adapter_from_config
from weavevision.data.audit import require_no_leakage
from weavevision.domain.enums import ModelStatus
from weavevision.domain.schemas import ModelManifest
from weavevision.evaluation.calibration import calibrate_image_threshold
from weavevision.models.anomalib_adapter import AnomalibAdapter
from weavevision.models.export import sha256_artifact
from weavevision.models.registry import ModelRegistry
from weavevision.settings import Settings


class TrainingService:
    """Train a governed model and generate provenance artifacts before evaluation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def train(self, experiment_config: Path) -> dict[str, Any]:
        """Run dataset verification, fit, provisional calibration, and candidate artifacts."""
        experiment = _read_yaml(experiment_config)
        root = self.settings.project_root
        dataset_config = root / str(experiment["experiment"]["dataset_config"])
        model_config_path = root / str(experiment["experiment"]["model_config"])
        model_payload = _read_yaml(model_config_path)
        manifest = adapter_from_config(dataset_config).verify()
        leakage = require_no_leakage(manifest)
        algorithm = str(model_payload["model"]["name"])
        now = datetime.now(UTC)
        run_id = f"{now:%Y%m%d_%H%M%S}_{manifest.dataset_id}_{algorithm}_{_git_short_sha(root)}"
        model_id = f"model_{uuid4().hex}"
        run_root = self.settings.resolved_artifacts_root() / "experiments" / run_id
        checkpoint = run_root / "checkpoints" / "model.ckpt"
        adapter = AnomalibAdapter(
            model_id=model_id,
            algorithm=algorithm,
            model_config=dict(model_payload["model"]),
            device=str(experiment.get("runtime", {}).get("device", "auto")),
        )
        dataset_payload = _read_yaml(dataset_config)["dataset"]
        dataset_root = root / str(dataset_payload["root"])
        trainer = experiment.get("trainer", {})
        train_batch_size = int(model_payload.get("data", {}).get("train_batch_size", 2))
        eval_batch_size = int(model_payload.get("data", {}).get("eval_batch_size", 4))
        precision_value = trainer.get(
            "precision", model_payload.get("trainer", {}).get("precision")
        )
        precision = str(precision_value) if precision_value is not None else None

        while True:
            try:
                artifact, validation_predictions = adapter.fit_mvtec(
                    dataset_root,
                    str(dataset_payload.get("category", "carpet")),
                    checkpoint,
                    seed=int(experiment.get("runtime", {}).get("seed", 42)),
                    max_epochs=trainer.get("max_epochs"),
                    num_workers=0,
                    train_batch_size=train_batch_size,
                    eval_batch_size=eval_batch_size,
                    precision=precision,
                )
                break
            except RuntimeError as exc:
                err_msg = str(exc).lower()
                if "out of memory" in err_msg and train_batch_size > 1:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    train_batch_size = max(1, train_batch_size // 2)
                    eval_batch_size = max(1, eval_batch_size // 2)
                    continue
                raise
        normal_scores = np.asarray([score for score, _ in validation_predictions], dtype=float)
        if normal_scores.size == 0:
            raise RuntimeError("training produced no validation predictions for calibration")
        threshold = calibrate_image_threshold(
            normal_scores,
            None,
            split="validation",
            model_id=model_id,
            dataset_manifest_sha256=manifest.manifest_sha256,
        )
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "config.resolved.yaml").write_text(
            yaml.safe_dump(experiment, sort_keys=True), encoding="utf-8"
        )
        (run_root / "dataset_manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        (run_root / "leakage_audit.json").write_text(
            json.dumps(leakage, indent=2), encoding="utf-8"
        )
        (run_root / "thresholds.json").write_text(
            threshold.model_dump_json(indent=2), encoding="utf-8"
        )
        np.save(run_root / "validation_scores.npy", normal_scores)
        environment = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_sha": _git_short_sha(root),
        }
        (run_root / "environment.json").write_text(
            json.dumps(environment, indent=2), encoding="utf-8"
        )
        result = {
            "status": "PASS",
            "run_id": run_id,
            "model_id": model_id,
            "algorithm": algorithm,
            "artifact_path": str(artifact),
            "artifact_sha256": sha256_artifact(artifact),
            "threshold_id": threshold.threshold_id,
            "threshold_status": threshold.status,
            "dataset_manifest_sha256": manifest.manifest_sha256,
            "validation_prediction_count": len(validation_predictions),
        }
        model_manifest = ModelManifest.model_validate(
            {
                "model_id": model_id,
                "status": ModelStatus.CANDIDATE,
                "algorithm": algorithm,
                "dataset_manifest_sha256": manifest.manifest_sha256,
                "training_run_id": run_id,
                "config_sha256": config_sha256(model_payload),
                "artifact_path": artifact,
                "artifact_sha256": result["artifact_sha256"],
                "threshold_id": threshold.threshold_id,
                "metrics_path": None,
                "created_at": datetime.now(UTC),
            }
        )
        registry_path = ModelRegistry(self.settings.resolved_artifacts_root() / "models").register(
            model_manifest
        )
        result["model_manifest_path"] = str(registry_path)
        (run_root / "run_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return payload


def _git_short_sha(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "uncommitted"


def config_sha256(payload: dict[str, Any]) -> str:
    """Hash a resolved configuration with sorted JSON keys."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
