# WeaveVision Final Verdict

## Overall Status

PASS_WITH_RESTRICTIONS

## Implemented

- Python 3.11/UV modular monolith, strict configuration and domain contracts
- Dataset manifests, canonical hashes, source-group split, duplicate/leakage audit
- MVTec/normalized secondary adapters and programmatic fixture generator
- Anomalib 2.5 PatchCore training and safe OpenVINO export with XML+BIN integrity hash
- Validation-only provisional threshold calibration and sealed-array metrics
- Quality gate, ABSTAIN, tiled inference, regions, masks, heatmaps, overlays
- Single and partial-failure batch services, SQLite audit history and expert feedback
- JSON, CSV, HTML and PNG reports
- Degraded-ready Streamlit pages, CLI, CI, Docker profiles and release audit

## Verified Evidence

| Claim | Artifact | Status |
|---|---|---|
| Python 3.11 environment doctor | `artifacts/benchmarks/system_doctor.json` | PASS |
| Fixture manifest and leakage governance | `data/manifests/weavevision_fixture.json` | PASS |
| Real PatchCore fixture training/export | `artifacts/experiments/20260718_002106_weavevision_fixture_patchcore_uncommitted/run_manifest.json` | PASS |
| Single/batch/report/SQLite smoke | same run `fixture_smoke.json` and `artifacts/reports/` | PASS |
| Unit/contract/integration tests | execution log | PASS |
| Real MVTec AD carpet benchmark | dataset absent | BLOCKED |
| RTX 4070 Torch CUDA hardware/model smoke | `artifacts/benchmarks/gpu_patchcore_smoke.json` | PASS_WITH_RESTRICTIONS |
| EfficientAD same-protocol comparison | real benchmark dataset absent | BLOCKED |
| Company pilot | authorized company data absent | BLOCKED |

## Real Metrics

Generated fixture-only pipeline evidence:

- `artifacts/experiments/20260718_002106_weavevision_fixture_patchcore_uncommitted/metrics.json`
- `artifacts/experiments/20260718_002106_weavevision_fixture_patchcore_uncommitted/fixture_smoke.json`

These are not open-source benchmark or company-performance claims. The provisional threshold
produced 4/4 false positives on fixture normal test images and is not eligible for promotion.

## Open Blockers

- Official MVTec AD carpet data requires manual license acceptance/download.
- Full end-to-end GPU latency on a verified real dataset is not yet available.
- Docker image execution awaits a running Docker Desktop Linux daemon.
- No authorized company pilot data or expert labels are present.

## Restrictions

- Open-source benchmark does not establish company production performance.
- MVTec license is non-commercial/restricted and must be reviewed at source.
- Company threshold is not locked without company pilot data.
- No model is marked active; UI intentionally remains in model-not-ready state.

## Next Authorized Target

Provide the officially downloaded MVTec AD carpet dataset, then run the real PatchCore and
EfficientAD same-protocol benchmark. After that, proceed to a company-controlled single-pattern
pilot.
