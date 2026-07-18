"""Typer CLI exit-code and machine-readable output tests."""

import json
from pathlib import Path

from typer.testing import CliRunner

from weavevision.cli import app

runner = CliRunner()


def test_doctor_and_model_list_json() -> None:
    doctor = runner.invoke(app, ["doctor", "--json"])
    assert doctor.exit_code == 0
    assert json.loads(doctor.stdout)["status"] == "PASS"
    models = runner.invoke(app, ["model", "list", "--json"])
    assert models.exit_code == 0
    assert isinstance(json.loads(models.stdout), list)


def test_dataset_audit_and_blocked_benchmark() -> None:
    manifest_path = Path("data/manifests/weavevision_fixture.json")
    manifest_before = manifest_path.read_bytes()
    audit = runner.invoke(
        app, ["dataset", "audit", "--config", "configs/datasets/fixture.yaml", "--json"]
    )
    assert audit.exit_code == 0
    assert json.loads(audit.stdout)["verification_status"] == "VERIFIED"
    assert manifest_path.read_bytes() == manifest_before
    benchmark = runner.invoke(
        app, ["benchmark", "--config", "configs/experiments/robustness.yaml", "--json"]
    )
    assert benchmark.exit_code == 1
    assert json.loads(benchmark.stdout)["status"] == "BLOCKED"


def test_evaluate_missing_run_is_not_run() -> None:
    result = runner.invoke(
        app, ["evaluate", "--run-id", "missing_run", "--split", "test", "--json"]
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "NOT_RUN"


def test_infer_and_batch_report_model_not_ready(tmp_path: Path) -> None:
    image = "data/fixtures/carpet/test/good/good_000.png"
    infer = runner.invoke(app, ["infer", "--input", image, "--output", str(tmp_path), "--json"])
    assert infer.exit_code == 2
    assert json.loads(infer.stdout)["error_code"] == "WV_MODEL_NOT_READY"
    batch = runner.invoke(
        app,
        ["batch", "--input", "data/fixtures/carpet/test/good", "--output", str(tmp_path), "--json"],
    )
    assert batch.exit_code == 2
    assert json.loads(batch.stdout)["error_code"] == "WV_MODEL_NOT_READY"
