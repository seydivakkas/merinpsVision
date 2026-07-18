# Execution Log

## 2026-07-18 — M0 Workspace Audit

- Workspace inventory: only the 3,200-line master specification was present.
- `git status --short --branch`: failed as expected because the directory was not a repository.
- `python --version`: `Python 3.14.3` (system default; rejected for project use).
- `py -0p`: Astral CPython `3.11.15` is available.
- `uv --version`: `uv 0.11.8`.
- `nvidia-smi`: NVIDIA GeForce RTX 4070 Laptop GPU, driver 592.27, reported CUDA 13.1,
  8188 MiB VRAM.
- `docker --version`: Docker 29.2.1.
- Master specification renamed to `WEAVEVISION_CURSOR_MASTER_BUILD_SPEC.md`.
- `git init`: PASS.

## 2026-07-18 — M1 Bootstrap

- `uv sync --extra dev`: FAIL (first attempt). Hatchling correctly rejected the missing
  `README.md`; the README was then created. No dependency versions were weakened.
- `uv sync --extra dev`: PASS on second attempt; 154 packages locked and installed with
  Python 3.11.15 and Anomalib 2.5.0.
- `uv run weavevision doctor --json`: PASS. NVIDIA hardware visible, but Torch 2.8.0+cpu;
  CUDA runtime unavailable and recorded as a blocker.

## 2026-07-18 — M2 Dataset Governance

- `scripts/generate_fixtures.py`: generated 20 test/training images plus four exact masks.
- `weavevision dataset audit --config configs/datasets/fixture.yaml --json`: PASS; 20 source
  images verified, no exact/source cross-split leakage, canonical manifest written.
- Official MVTec AD carpet: BLOCKED because the licensed dataset is not present.

## 2026-07-18 — M3–M5 Model, Evaluation, Inference and Reporting

- Initial PatchCore training with Lightning `.ckpt`: PASS training, deployment inference
  rejected because `.ckpt` is not a deployment format.
- Anomalib Torch export: produced `.pt`, but inference requested `TRUST_REMOTE_CODE=1` for
  pickle loading. This was rejected under the arbitrary-pickle security rule.
- Added Anomalib OpenVINO extra and changed deployment to OpenVINO IR. XML and BIN are hashed
  together by the registry integrity function.
- `weavevision train --config configs/experiments/smoke.yaml --json`: PASS for run
  `20260718_000126_weavevision_fixture_patchcore_uncommitted`; real PatchCore model, OpenVINO
  export hash `641b67c7...2702f19`, two validation-normal predictions, threshold status
  `PROVISIONAL_NORMAL_ONLY`.
- `scripts/run_fixture_smoke.py`: PASS; one normal, one synthetic anomaly and 24 batch items;
  zero batch failures. CPU latency on five measured direct model calls: p50 83.54 ms,
  p95 89.30 ms. This is fixture-only pipeline evidence.
- `scripts/evaluate_fixture.py`: PASS_WITH_RESTRICTIONS. Image AUROC 0.50, AP 0.50,
  TP=4/TN=0/FP=4/FN=0; pixel AUROC 0.9689, pixel AP 0.1170, IoU 0.0235. These poor synthetic
  fixture results are preserved rather than marketed; model is not promoted.

## 2026-07-18 — Quality Gates

- `uv run ruff check .`: PASS.
- `uv run mypy src`: PASS, strict mode, 61 source files.
- `uv run pytest -q`: PASS, 18 tests in 3.17 seconds. Windows pytest temp cleanup emitted an
  atexit permission warning after success; project-local `--basetemp` was added afterward.

## 2026-07-18 — CUDA Runtime Repair and Hardware Smoke

- Added official UV `pytorch-cu126` explicit sources for Torch and Torchvision.
- `uv sync --extra dev`: PASS after downloading Torch 2.8.0+cu126.
- `weavevision doctor --json`: PASS; CUDA available, CUDA runtime 12.6, active device CUDA,
  RTX 4070 Laptop GPU 8188 MiB.
- Direct CUDA matrix smoke: PASS, finite result.
- `scripts/gpu_patchcore_smoke.py`: PASS_WITH_RESTRICTIONS after in-process fixture training;
  5 warmup + 20 measured 256x256 PatchCore forwards, p50 9.63 ms, p95 10.51 ms,
  p99 10.72 ms, peak allocated VRAM 271.88 MiB. No pickle artifact was loaded.
- `weavevision train --config configs/experiments/smoke_efficientad.yaml`: BLOCKED after GPU
  model initialization. The required 1,557,161,267-byte Imagenette auxiliary download failed
  at 925,022,719 bytes with `ContentTooShortError`. No partial result or model comparison claim
  was produced.
- First OpenVINO robustness pass: FAIL contract review. Deployment `pred_score` values were all
  1.0 because Anomalib's embedded postprocessor normalized them, while the external threshold
  artifact was calibrated in the training score space. The run is superseded and cannot be
  promoted. Factory defaults were changed to disable framework postprocessing/evaluation/
  visualization so WeaveVision owns one raw-score threshold contract.
- Replacement run `20260718_002106_weavevision_fixture_patchcore_uncommitted`: PASS. Raw-score
  OpenVINO artifact hash `0101b972...8fe73a`; fixture single/batch smoke PASS, 24 completed and
  zero failures. OpenVINO CPU model-call p50 125.12 ms, p95 127.60 ms (five measurements).
- Replacement sealed fixture evaluation: PASS_WITH_RESTRICTIONS. Image AUROC/AP 1.0 reflects
  only the deliberately simple synthetic anomaly separation; locked provisional threshold still
  produced TP=4/TN=0/FP=4/FN=0. Pixel AP 0.9505, IoU 0.0292. Model remains ineligible.
- Fixture robustness: PASS_WITH_RESTRICTIONS for brightness, contrast, noise, blur, JPEG,
  rotation, downsample, and occlusion score deltas; tile/coreset ablation NOT_RUN pending a
  real benchmark protocol.

## 2026-07-18 — M6/M9 Product and Release Checks

- Streamlit degraded-ready startup: PASS, `/_stcore/health` returned HTTP 200 `ok` with no
  active model.
- `uv build`: PASS; source distribution and wheel generated.
- `uv run pre-commit install`: PASS.
- Production TODO/FIXME/HACK/empty-pass scan: PASS.
- Expanded quality suite: `ruff` PASS, strict `mypy` PASS (62 source files), pytest PASS
  (47 tests), coverage 85.15%.
- `docker build -f Dockerfile.cpu ...`: BLOCKED; Docker Desktop Linux engine pipe is absent.
