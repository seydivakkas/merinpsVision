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
drift_app = typer.Typer(help="Drift monitoring and incident commands")
app.add_typer(dataset_app, name="dataset")
app.add_typer(model_app, name="model")
app.add_typer(drift_app, name="drift")


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


# ---------------------------------------------------------------------------
# model rollback (M8)
# ---------------------------------------------------------------------------


@model_app.command("rollback")
def model_rollback(
    from_model_id: Annotated[str, typer.Argument(help="Model being replaced")],
    to_model_id: Annotated[str, typer.Argument(help="Model to restore")],
    reason: Annotated[
        str,
        typer.Option(
            help="Rollback reason: HASH_MISMATCH | RECALL_DROP | FP_SPIKE | "
            "LATENCY | DRIFT_WORSENING | SAFETY_ALARM"
        ),
    ] = "DRIFT_WORSENING",
    triggered_by: Annotated[str, typer.Option(help="User or automation ID")] = "cli",
    incident_id: Annotated[str | None, typer.Option(help="Linked incident ID")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Roll back production model and write a governance audit record.

    Promotes TO_MODEL_ID to ACTIVE_BENCHMARK and retires FROM_MODEL_ID.
    Writes a RollbackEvent to the SQLite audit trail.
    """
    from weavevision.domain.enums import RollbackReason
    from weavevision.persistence.database import Database
    from weavevision.services.model_registry_service import ModelRegistryService

    try:
        rollback_reason = RollbackReason(reason)
    except ValueError as err:
        typer.echo(
            f"Invalid reason '{reason}'. Valid values: "
            + ", ".join(r.value for r in RollbackReason),
            err=True,
        )
        raise typer.Exit(code=2) from err

    settings = load_settings()
    db = Database(settings.resolved_database())
    db.migrate()
    svc = ModelRegistryService(settings, db)
    try:
        event = svc.rollback(
            from_model_id,
            to_model_id,
            rollback_reason,
            triggered_by=triggered_by,
            incident_id=incident_id,
        )
        _emit(
            {
                "status": "OK",
                "rollback_id": event.rollback_id,
                "from_model_id": event.from_model_id,
                "to_model_id": event.to_model_id,
                "reason": event.reason.value,
            },
            as_json,
        )
    except WeaveVisionError as exc:
        _emit({"status": "FAIL", "error_code": exc.code, "message": exc.message}, as_json)
        raise typer.Exit(code=2) from exc


# ---------------------------------------------------------------------------
# drift commands (M8)
# ---------------------------------------------------------------------------


@drift_app.command("status")
def drift_status(
    model_id: Annotated[str, typer.Argument(help="Model ID to query drift windows for")],
    limit: Annotated[int, typer.Option(help="Maximum number of windows to show")] = 20,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show recent drift monitoring windows for a model.

    Lists the last LIMIT windows ordered by creation time (newest first).
    Non-STABLE trend statuses are highlighted in the table view.
    """
    from weavevision.persistence.database import Database

    settings = load_settings()
    db = Database(settings.resolved_database())
    db.migrate()

    with db.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM drift_windows WHERE model_id = ? ORDER BY created_at DESC LIMIT ?",
            (model_id, limit),
        ).fetchall()

    windows = [dict(row) for row in rows]

    if as_json:
        _emit({"model_id": model_id, "windows": windows}, as_json)
        return

    if not windows:
        typer.echo(f"No drift windows found for model '{model_id}'.")
        return

    header = (
        f"{'window_id':<16}  {'metric':<14}  {'value':>7}  "
        f"{'trend_status':<14}  {'pattern':<12}  created_at"
    )
    typer.echo(f"\nDrift status for model: {model_id} (last {limit} windows)\n")
    typer.echo(header)
    typer.echo("-" * 88)
    for w in windows:
        val = f"{w['metric_value']:.4f}" if w["metric_value"] is not None else "  N/A"
        status = str(w["trend_status"])
        color = typer.colors.RED if status != "STABLE" else typer.colors.GREEN
        status_colored = typer.style(f"{status:<14}", fg=color)
        typer.echo(
            f"{str(w['window_id'])[:14]:<16}  {w['metric_name']!s:<14}  "
            f"{val:>7}  {status_colored}  {w['drift_pattern']!s:<12}  {w['created_at']}"
        )


@drift_app.command("incidents")
def drift_incidents(
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List all open (unresolved) drift incidents.

    An incident is open when its resolved_at field is NULL in the database.
    Use this to identify pending action items for the ML operations team.
    """
    from weavevision.persistence.database import Database
    from weavevision.services.incident_service import IncidentService

    settings = load_settings()
    db = Database(settings.resolved_database())
    db.migrate()
    svc = IncidentService(settings, db)
    incidents = svc.list_open()

    if as_json:
        _emit(incidents, as_json)
        return

    if not incidents:
        typer.echo(typer.style("No open incidents.", fg=typer.colors.GREEN))
        return

    typer.echo(f"\nOpen drift incidents ({len(incidents)} total)\n")
    typer.echo(
        f"{'incident_id':<16}  {'priority':<14}  {'pattern':<12}  {'model_id':<16}  created_at"
    )
    typer.echo("-" * 82)
    for inc in incidents:
        priority = str(inc["priority"])
        color = (
            typer.colors.RED
            if priority == "P0_BLOCKED"
            else typer.colors.YELLOW
            if priority == "P1_INCIDENT"
            else typer.colors.BRIGHT_BLUE
        )
        priority_colored = typer.style(f"{priority:<14}", fg=color)
        typer.echo(
            f"{str(inc['incident_id'])[:14]:<16}  {priority_colored}  "
            f"{inc['drift_pattern']!s:<12}  {inc['model_id']!s:<16}  {inc['created_at']}"
        )


@drift_app.command("queue")
def drift_queue(
    limit: Annotated[int, typer.Option(help="Maximum items to show")] = 50,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List pending (unreviewed) active-learning labeling queue items.

    Items are sorted by priority bucket (P0 first) then creation time.
    Use this to assign labeling tasks to quality experts.
    """
    from weavevision.persistence.database import Database

    settings = load_settings()
    db = Database(settings.resolved_database())
    db.migrate()

    with db.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM labeling_queue WHERE verdict IS NULL"
            " ORDER BY priority_bucket ASC, created_at ASC LIMIT ?",
            (limit,),
        ).fetchall()

    items = [dict(row) for row in rows]

    if as_json:
        _emit(items, as_json)
        return

    if not items:
        typer.echo(typer.style("Labeling queue is empty.", fg=typer.colors.GREEN))
        return

    typer.echo(f"\nPending labeling queue ({len(items)} items)\n")
    typer.echo(f"{'item_id':<16}  {'bucket':<6}  {'drift_score':>11}  {'reason':<20}  source_path")
    typer.echo("-" * 80)
    for item in items:
        score = f"{item['drift_score']:.4f}" if item["drift_score"] is not None else "    N/A"
        bucket = str(item["priority_bucket"])
        color = (
            typer.colors.RED
            if bucket == "P0"
            else typer.colors.YELLOW
            if bucket == "P1"
            else typer.colors.CYAN
        )
        bucket_colored = typer.style(f"{bucket:<6}", fg=color)
        reason = str(item["selection_reason"])[:18]
        path_str = str(item["source_path"])
        typer.echo(
            f"{str(item['item_id'])[:14]:<16}  {bucket_colored}  "
            f"{score:>11}  {reason:<20}  {path_str}"
        )
