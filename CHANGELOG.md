# Changelog

## Unreleased

### Added

- **Drift Lifecycle Governance (M1–M10)**: EWMA + CUSUM trend monitoring, PSI distribution shift
  detection, 2-of-N incident triage rule, greedy-coreset active learning queue, champion-vs-challenger
  canary evaluation, hash-verified model rollback with full audit trail.
  - `domain/enums.py`: `DriftPattern`, `TrendStatus`, `IncidentPriority`, `RetrainingStrategy`, `CanaryStatus`, `RollbackReason`
  - `domain/schemas.py`: `DriftWindow`, `TrendPoint`, `IncidentRecord`, `TriageDecision`, `RetrainingRequest`, `LabelingQueueItem`, `CanaryEvaluation`, `RollbackEvent`
  - `evaluation/trend_monitor.py`, `evaluation/psi.py`, `evaluation/incident_triage.py`, `evaluation/alert_policy.py`
  - `services/drift_monitor_service.py`, `services/incident_service.py`, `services/active_learning_service.py`, `services/canary_service.py`
  - `services/model_registry_service.py`: rollback() with `RollbackEvent` audit trail
  - `persistence/database.py`: 5 idempotent drift tables (drift_windows, drift_incidents, labeling_queue, canary_runs, rollback_events)
  - `cli.py`: `drift status`, `drift incidents`, `drift queue`, `model rollback` commands
  - `ui/pages/7_Domain_Shift.py`, `8_Drift_Incidents.py`, `9_Labeling_Queue.py`, `10_Canary_and_Rollback.py`
  - `configs/app.yaml`: `drift:` policy block (all thresholds config-driven, never hardcoded)
- Modular WeaveVision repository bootstrap, strict domain contracts, settings, logging,
  SQLite schema, and environment doctor.
- Governed datasets, PatchCore/EfficientAD adapters, raw-score calibration, tiled inference,
  quality gate, reports, Streamlit product, CUDA/OpenVINO deployment paths, CI and Docker files.

### Changed

- Scope is locked to carpet/textile anomaly detection and quality analytics.

### Fixed

- Nothing yet.

### Security

- Local-only defaults and repository exclusions for data, weights, reports, and secrets.
- Pickle-based model deployment loading is rejected; OpenVINO XML+BIN are integrity-hashed as
  one artifact identity.

### Deprecated

- Nothing yet.

### Removed

- Generative carpet design, GAN, SDXL, and LoRA concerns from the product scope.
