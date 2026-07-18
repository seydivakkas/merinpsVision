"""Measure in-process PatchCore CUDA forward latency without loading pickle artifacts."""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

from weavevision.settings import load_settings


def run() -> dict[str, object]:
    """Fit fixture PatchCore and benchmark direct CUDA forwards in the same trusted process."""
    if not torch.cuda.is_available():
        return {"status": "BLOCKED", "reason": "torch CUDA unavailable"}
    settings = load_settings()
    datamodule = MVTecAD(
        root=settings.project_root / "data" / "fixtures",
        category="carpet",
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
        val_split_mode="from_train",
        val_split_ratio=0.2,
        seed=42,
    )
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        pre_trained=True,
    )
    engine = Engine(
        accelerator="gpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        deterministic=True,
        default_root_dir=settings.resolved_artifacts_root() / "benchmarks" / "gpu_smoke",
    )
    engine.fit(model=model, datamodule=datamodule)
    batch = next(iter(datamodule.test_dataloader()))
    image = batch.image.cuda(non_blocking=False)
    model = model.cuda().eval()
    warmup_runs = 5
    measured_runs = 20
    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(image)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(measured_runs):
            started = time.perf_counter()
            output = model(image)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000)
    values = np.asarray(samples)
    result: dict[str, object] = {
        "status": "PASS_WITH_RESTRICTIONS",
        "purpose": "FIXTURE_GPU_HARDWARE_SMOKE_NOT_PRODUCTION_BENCHMARK",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "input_shape": list(image.shape),
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "latency_ms": {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "mean": float(values.mean()),
        },
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
        "output_has_score": hasattr(output, "pred_score"),
        "output_has_map": hasattr(output, "anomaly_map"),
    }
    destination = settings.resolved_artifacts_root() / "benchmarks" / "gpu_patchcore_smoke.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
