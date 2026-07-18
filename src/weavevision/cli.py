"""Typer command-line interface for WeaveVision services."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from weavevision.domain.errors import WeaveVisionError
from weavevision.logging_config import configure_logging
from weavevision.services.health_service import HealthService
from weavevision.settings import load_settings

app = typer.Typer(help="WeaveVision anomaly detection and quality analytics")
dataset_app = typer.Typer(help="Dataset governance commands")
model_app = typer.Typer(help="Model registry commands")
app.add_typer(dataset_app, name="dataset")
app.add_typer(model_app, name="model")


def _emit(payload: object, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        typer.echo(payload)


@app.command()
def doctor(
    as_json: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON")] = False,
    config: Annotated[Path | None, typer.Option(help="Application YAML path")] = None,
) -> None:
    """Validate runtime, writable paths, GPU visibility, and SQLite."""
    try:
        settings = load_settings(config)
        configure_logging(settings.resolved_artifacts_root() / "logs" / "weavevision.jsonl")
        result = HealthService(settings).collect()
        _emit(result, as_json)
        if result["status"] != "PASS":
            raise typer.Exit(code=1)
    except WeaveVisionError as exc:
        _emit({"status": "FAIL", "error_code": exc.code, "message": exc.message}, as_json)
        raise typer.Exit(code=2) from exc


@app.command()
def serve() -> None:
    """Start the Streamlit user interface."""
    from streamlit.web import cli as stcli

    settings = load_settings()
    target = settings.project_root / "src" / "weavevision" / "ui" / "app.py"
    raise SystemExit(stcli.main_run([str(target)]))


@app.command()
def train(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Train a governed experiment and register its candidate model."""
    from weavevision.services.training_service import TrainingService

    try:
        settings = load_settings()
        result = TrainingService(settings).train(config)
        _emit(result, as_json)
    except WeaveVisionError as exc:
        _emit({"status": "FAIL", "error_code": exc.code, "message": exc.message}, as_json)
        raise typer.Exit(code=2) from exc


@app.command()
def calibrate(
    run_id: Annotated[str, typer.Option(help="Existing training run identifier")],
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Recompute a provisional threshold from persisted validation scores."""
    import numpy as np

    from weavevision.domain.schemas import ThresholdArtifact
    from weavevision.evaluation.calibration import calibrate_image_threshold

    settings = load_settings()
    run_root = settings.resolved_artifacts_root() / "experiments" / run_id
    run_manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    scores = np.load(run_root / "validation_scores.npy", allow_pickle=False)
    threshold = calibrate_image_threshold(
        scores,
        None,
        split="validation",
        model_id=str(run_manifest["model_id"]),
        dataset_manifest_sha256=str(run_manifest["dataset_manifest_sha256"]),
    )
    destination = run_root / f"thresholds_{threshold.threshold_id}.json"
    destination.write_text(threshold.model_dump_json(indent=2), encoding="utf-8")
    payload = ThresholdArtifact.model_validate_json(destination.read_text()).model_dump(mode="json")
    _emit(payload, as_json)


@app.command()
def infer(
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Analyze one image through the active integrity-checked service."""
    from weavevision.services.factory import load_active_analysis_service

    try:
        result = load_active_analysis_service(load_settings()).analyze(input_path, output)
        _emit(result.model_dump(mode="json"), as_json)
    except WeaveVisionError as exc:
        _emit({"status": "FAIL", "error_code": exc.code, "message": exc.message}, as_json)
        raise typer.Exit(code=2) from exc


@app.command("batch")
def batch_command(
    input_path: Annotated[Path, typer.Option("--input", exists=True)],
    output: Annotated[Path, typer.Option()],
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Analyze a folder or ZIP with partial-failure isolation."""
    from weavevision.services.batch_service import BatchService
    from weavevision.services.factory import load_active_analysis_service

    try:
        result = BatchService(load_active_analysis_service(load_settings())).analyze(
            input_path, output
        )
        _emit(result.model_dump(mode="json"), as_json)
    except WeaveVisionError as exc:
        _emit({"status": "FAIL", "error_code": exc.code, "message": exc.message}, as_json)
        raise typer.Exit(code=2) from exc


@app.command()
def evaluate(
    run_id: Annotated[str, typer.Option()],
    split: Annotated[str, typer.Option()] = "test",
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show generated sealed-test metrics for an existing run."""
    if split != "test":
        raise typer.BadParameter("only the sealed test split is accepted")
    path = load_settings().resolved_artifacts_root() / "experiments" / run_id / "metrics.json"
    if not path.is_file():
        _emit({"status": "NOT_RUN", "metrics_path": str(path)}, as_json)
        raise typer.Exit(code=1)
    _emit(json.loads(path.read_text(encoding="utf-8")), as_json)


@app.command()
def benchmark(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Report benchmark readiness without inventing missing measurements."""
    payload = {
        "status": "BLOCKED",
        "config": str(config),
        "reason": "requires an active eligible model and verified real benchmark dataset",
    }
    _emit(payload, as_json)
    raise typer.Exit(code=1)


@dataset_app.command("audit")
def dataset_audit(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Audit a configured dataset and emit its verified manifest."""
    from weavevision.data.adapters.factory import adapter_from_config

    result = adapter_from_config(config).verify()
    _emit(result.model_dump(mode="json"), as_json)


@model_app.command("list")
def model_list(as_json: Annotated[bool, typer.Option("--json")] = False) -> None:
    """List registered model manifests."""
    from weavevision.models.registry import ModelRegistry

    settings = load_settings()
    manifests = ModelRegistry(settings.resolved_artifacts_root() / "models").list()
    _emit([manifest.model_dump(mode="json") for manifest in manifests], as_json)


@model_app.command("show")
def model_show(
    model_id: Annotated[str, typer.Option()],
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show one model manifest after artifact integrity verification."""
    from weavevision.models.registry import ModelRegistry

    settings = load_settings()
    manifest = ModelRegistry(settings.resolved_artifacts_root() / "models").get(model_id)
    _emit(manifest.model_dump(mode="json"), as_json)


@model_app.command("promote")
def model_promote(
    model_id: Annotated[str, typer.Option()],
    reason: Annotated[str, typer.Option()],
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Promote an eligible evidence-complete benchmark model."""
    from weavevision.models.registry import ModelRegistry

    settings = load_settings()
    manifest = ModelRegistry(settings.resolved_artifacts_root() / "models").promote(
        model_id, reason
    )
    _emit(manifest.model_dump(mode="json"), as_json)
