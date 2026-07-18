# Experiment Protocol

Official splits are preserved. Validation derives only from training sources with a
deterministic group split. Calibration rejects the sealed test split. Every experiment stores
resolved config, environment, dataset manifest and hash, leakage audit, seed, thresholds,
metrics, timings, memory, overlays, and a run manifest. Missing evidence remains `NOT_RUN`.
