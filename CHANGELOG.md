# Changelog

## Unreleased

### Added

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
