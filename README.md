# WeaveVision — Carpet Anomaly Detection and Quality Analytics

![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)

WeaveVision is a local-first, modular-monolith application for one-class visual anomaly
detection on carpet and textile images. It separates dataset governance, model experiments,
inference, audit persistence, reporting, and the Streamlit interface.

## Problem

The system learns from normal reference images and produces an image anomaly score and a
pixel anomaly map. Scores are not probabilities. A quality gate can return `REVIEW` or
`ABSTAIN` when input evidence is unsafe.

## Scope

The MVP covers MVTec AD carpet governance, PatchCore baseline support, EfficientAD challenger
support, validation-only threshold calibration, tiled inference, JSON/CSV/HTML reports,
SQLite audit history, expert feedback, benchmarks, and a degraded-ready UI. It does not
generate carpet designs or autonomously override quality experts.

## Architecture

The Streamlit UI and Typer CLI call application services. Services orchestrate pure domain,
data, inference, evaluation, reporting, registry, and persistence modules. Model framework
objects remain behind adapters.

## Quick start

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
uv run weavevision doctor
uv run weavevision serve
```

Linux:

```bash
bash scripts/bootstrap.sh
uv run weavevision doctor
uv run weavevision serve
```

## Dataset setup

MVTec AD requires manual license acceptance and is not committed to Git. Place its `carpet`
directory under `data/external/mvtec_ad/`, then run:

```bash
uv run python scripts/verify_mvtec.py --root data/external/mvtec_ad --category carpet
```

Fixture images are programmatically generated only for pipeline tests and never produce
benchmark claims.

## Train and evaluate

```bash
uv run weavevision train --config configs/experiments/benchmark_mvtec_carpet.yaml
uv run weavevision evaluate --run-id <run_id> --split test
```

These commands require a verified dataset manifest. The sealed test split is never used for
threshold calibration.

## UI

Run `uv run weavevision serve`. The interface remains available when there is no active model
and explicitly reports `MODEL_NOT_READY`.

## Reports

Analysis services emit JSON, CSV, HTML, heatmap, mask, and overlay artifacts under
`artifacts/reports/`. Generated artifacts are ignored by Git.

## Metrics

| Latest verified run | Model | Dataset | Metrics path | Latency path | Status |
|---|---|---|---|---|---|
| NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

This table is updated only from generated evidence artifacts.

## License

![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Bu yazılım yalnızca görüntüleme ve kişisel öğrenim amacıyla paylaşılmıştır. Yazarın açık yazılı izni olmaksızın kopyalanamaz, çoğaltılamaz, dağıtılamaz, ticari veya ticari olmayan projelerde kullanılamaz, değiştirilemez veya satılamaz.

## Limitations

- Open-source benchmarks do not establish production performance for a company.
- No company threshold is locked without controlled company pilot data and expert labels.
- Missing real datasets or model artifacts are shown as `BLOCKED` or `NOT_RUN`.
- The UI does not call anomaly models directly and cannot bypass registry integrity checks.

## Company pilot

The first authorized deployment unit is one pattern, one camera, one lighting condition, and
one locked threshold. See `docs/COMPANY_PILOT_RUNBOOK.md`.

## Reproducibility

Python is locked to 3.11 and dependencies are locked by `uv.lock`. Each run records resolved
configuration, dataset manifest hash, Git revision, environment, thresholds, and model hash.

## Security and privacy

Uploads stay local by default. ZIP traversal, decompression bombs, unsafe filenames, and
model artifact hash mismatches are rejected. Secrets, datasets, model weights, and generated
reports are excluded from Git.
