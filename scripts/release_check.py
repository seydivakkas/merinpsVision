"""Fail-closed local release package audit."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

REQUIRED = [
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "pyproject.toml",
    "uv.lock",
    "docs/ARCHITECTURE.md",
    "docs/CLAIM_CONTRACT.md",
    "docs/DATASET_AND_LICENSE_REGISTER.md",
    "docs/COMPANY_PILOT_RUNBOOK.md",
    "docs/BLOCKERS.md",
    "docs/EXECUTION_LOG.md",
    "docs/FINAL_VERDICT.md",
]


def run(root: Path) -> dict[str, Any]:
    """Audit repository state, tracked files, docs, secrets, and production placeholders."""
    required_missing = [name for name in REQUIRED if not (root / name).is_file()]
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    forbidden_suffixes = {".ckpt", ".pt", ".pth", ".onnx", ".xml", ".bin", ".engine"}
    tracked_models = [
        name for name in tracked if Path(name).suffix.casefold() in forbidden_suffixes
    ]
    large_files = [
        name
        for name in tracked
        if (root / name).is_file() and (root / name).stat().st_size > 10 * 1024 * 1024
    ]
    production_markers = []
    marker = re.compile(r"\b(TODO|FIXME|HACK)\b")
    for path in (root / "src" / "weavevision").rglob("*.py"):
        if marker.search(path.read_text(encoding="utf-8")):
            production_markers.append(str(path.relative_to(root)))
    secret_pattern = re.compile(r"(?i)(api[_-]?key|secret|token)\s*=\s*['\"][^'\"]{8,}")
    suspected_secrets = []
    for name in tracked:
        path = root / name
        if path.is_file() and path.stat().st_size < 1_000_000:
            try:
                if secret_pattern.search(path.read_text(encoding="utf-8")):
                    suspected_secrets.append(name)
            except UnicodeDecodeError:
                continue
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    checks = {
        "required_docs": not required_missing,
        "no_tracked_model_weights": not tracked_models,
        "no_large_tracked_files": not large_files,
        "no_production_markers": not production_markers,
        "no_suspected_secrets": not suspected_secrets,
        "clean_git_state": not git_status,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "details": {
            "required_missing": required_missing,
            "tracked_models": tracked_models,
            "large_files": large_files,
            "production_markers": production_markers,
            "suspected_secrets": suspected_secrets,
            "git_status": git_status.splitlines(),
        },
    }


def main() -> None:
    """Run the release audit at repository root and write its evidence artifact."""
    root = Path(__file__).resolve().parents[1]
    result = run(root)
    destination = root / "artifacts" / "benchmarks" / "release_check.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
