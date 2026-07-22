# Architecture

WeaveVision is a local modular monolith. CLI and Streamlit are delivery adapters. Application
services orchestrate domain contracts, dataset governance, framework-hidden model adapters,
inference, evaluation, reporting, registry, and SQLite persistence. Data, model artifacts,
threshold artifacts, and experiment evidence remain separate and hash-addressed.

## Layer Map

```
UI (Streamlit pages)  ──┐
CLI (Typer commands)  ──┼──▶  services/          ──▶  evaluation/    (pure math, CPU only)
                         │     drift_monitor_service   trend_monitor  (EWMA + CUSUM)
                         │     incident_service         psi            (PSI distribution shift)
                         │     active_learning_service  incident_triage (2-of-N signal rule)
                         │     canary_service            alert_policy   (config-driven thresholds)
                         │     model_registry_service
                         │
                         └──▶  persistence/       ──▶  SQLite (WAL mode)
                                DriftWindowRepository    drift_windows
                                IncidentRepository       drift_incidents
                                LabelingQueueRepository  labeling_queue
                                CanaryRepository         canary_runs
                                RollbackRepository       rollback_events
```

## Drift Lifecycle (M1–M10)

| Milestone | Component | Role |
|-----------|-----------|------|
| M1 | `domain/enums.py`, `domain/schemas.py` | Audit-safe contracts (`ConfigDict(extra="forbid")`) |
| M2 | `evaluation/trend_monitor.py`, `evaluation/psi.py` | CUSUM + EWMA + PSI (CPU/NumPy only) |
| M3 | `evaluation/incident_triage.py`, `evaluation/alert_policy.py` | 2-of-N signal triage rule |
| M4 | `persistence/database.py`, `persistence/repositories.py` | 5 idempotent SQLite tables |
| M5 | `services/drift_monitor_service.py`, `services/incident_service.py` | Orchestration layer |
| M6 | `services/active_learning_service.py` | Greedy-coreset labeling queue |
| M7 | `services/canary_service.py`, `services/model_registry_service.py` | Canary & hash-verified rollback |
| M8 | `cli.py` | `drift status/incidents/queue`, `model rollback` |
| M9 | `ui/pages/7–10_*.py` | Streamlit dashboard pages |
| M10 | `configs/app.yaml`, docs | Policy config, architecture doc |

### RTX 4070 Laptop GPU Constraints

- **Drift math** (EWMA, CUSUM, PSI, triage) → CPU/NumPy only.
- **Embedding extraction** for active learning → GPU, immediately `.detach().cpu().numpy()`.
- **Concurrent training + canary** → blocked via `artifacts/.gpu.lock` (filelock).
- **Mixed precision** → `bfloat16` (Ada Lovelace native, no gradient scaler needed).
- **PatchCore coreset** → ≤2% subsampling to stay within 8 GB VRAM.
