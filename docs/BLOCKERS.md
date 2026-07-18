# Blockers

| ID | Status | Scope | Reason | Resolution |
|---|---|---|---|---|
| BLK-001 | BLOCKED | MVTec AD carpet benchmark | Dataset is not present and requires manual official license acceptance/download. | User provides the officially downloaded archive or extracted dataset. |
| BLK-002 | BLOCKED | Company pilot | No authorized company dataset or expert labels are present. | Import an authorized single-pattern pilot package. |
| BLK-003 | PASS_WITH_RESTRICTIONS | GPU hardware smoke | Torch 2.8.0+cu126 CUDA tensor and in-process fixture PatchCore forward passed on RTX 4070; p50 9.63 ms, p95 10.51 ms, peak allocated VRAM 271.88 MiB. This excludes preprocessing/reporting and is not a production benchmark. | Repeat full end-to-end latency on verified MVTec and company pilot images. |
| BLK-004 | BLOCKED | EfficientAD comparison | GPU training started, but the required 1.56 GB Imagenette auxiliary download failed at 925 MB with `ContentTooShortError`; no real MVTec carpet dataset exists for a meaningful same-split comparison. | Complete the official auxiliary download, provide MVTec carpet, and rerun both models on one immutable protocol. |
| BLK-005 | PASS_WITH_RESTRICTIONS | Fixture PatchCore | Real model pipeline passed, but normal-only provisional threshold classified all four normal fixture test images as anomaly. | Do not promote; use anomaly validation and real benchmark evidence. |
| BLK-006 | BLOCKED | CPU/CUDA Docker execution | Docker CLI 29.2.1 is installed, but Docker Desktop Linux daemon is not running (`dockerDesktopLinuxEngine` pipe missing). Dockerfiles and Compose profiles are present but image build cannot execute. | Start Docker Desktop, then run the documented CPU and GPU build/doctor commands. |
