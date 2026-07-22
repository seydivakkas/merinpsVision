"""GPU Out-Of-Memory prevention and retry tests."""

import os
from pathlib import Path

import pytest
import yaml

from weavevision.services.training_service import TrainingService
from weavevision.settings import load_settings


@pytest.mark.smoke
@pytest.mark.gpu
def test_gpu_training_does_not_oom(tmp_path: Path) -> None:
    """Smoke test for PatchCore training on GPU to ensure no OOM.

    If the batch size is too large, the TrainingService OOM retry policy
    should catch the CUDA OutOfMemoryError, halve the batch size, and
    complete successfully.
    """
    settings = load_settings()

    # We will create a temporary config that forces CUDA.
    smoke_config_path = settings.project_root / "configs" / "experiments" / "smoke.yaml"
    smoke_payload = yaml.safe_load(smoke_config_path.read_text(encoding="utf-8"))

    smoke_payload["runtime"]["device"] = "cuda"

    temp_config = tmp_path / "smoke_gpu.yaml"
    temp_config.write_text(yaml.safe_dump(smoke_payload), encoding="utf-8")

    # Enable test databases and artifacts
    os.environ["WEAVEVISION_DATA_ROOT"] = str(tmp_path / "data")
    os.environ["WEAVEVISION_ARTIFACTS_ROOT"] = str(tmp_path / "artifacts")

    test_settings = load_settings()
    service = TrainingService(test_settings)

    try:
        result = service.train(temp_config)
        assert result["status"] == "PASS"
        assert result["algorithm"] == "patchcore"
        assert Path(result["artifact_path"]).exists()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            pytest.fail(f"GPU OOM occurred and was not mitigated: {exc}")
        else:
            raise
