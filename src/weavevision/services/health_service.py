"""Environment and dependency health diagnostics."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from weavevision.persistence.database import Database
from weavevision.settings import Settings


class HealthService:
    """Collect environment evidence without requiring a model or dataset."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def collect(self) -> dict[str, Any]:
        """Collect and persist a machine-readable system doctor report.

        Returns:
            Doctor payload with individual check outcomes and overall status.

        Side Effects:
            Creates configured writable directories, initializes SQLite, and writes JSON.
        """
        artifact_root = self.settings.resolved_artifacts_root()
        data_root = self.settings.resolved_data_root()
        artifact_root.mkdir(parents=True, exist_ok=True)
        data_root.mkdir(parents=True, exist_ok=True)
        db = Database(self.settings.resolved_database())
        device = self._torch_info()
        gpu = self._nvidia_info()
        checks = {
            "python_311": sys.version_info[:2] == (3, 11),
            "artifacts_writable": self._is_writable(artifact_root),
            "data_writable": self._is_writable(data_root),
            "database": db.health(),
        }
        payload: dict[str, Any] = {
            "schema_version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "python": {"version": platform.python_version(), "executable": sys.executable},
            "platform": {"system": platform.system(), "release": platform.release()},
            "hardware": {
                "cpu": platform.processor(),
                "ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
                "disk_free_gb": round(shutil.disk_usage(artifact_root).free / 1024**3, 2),
                "nvidia": gpu,
            },
            "runtime": device,
            "packages": {
                name: self._package_version(name)
                for name in ("weavevision", "torch", "anomalib", "streamlit", "pydantic")
            },
            "model_registry": {"status": "NOT_READY", "active_model": None},
            "fixture_smoke_inference": "NOT_RUN",
        }
        destination = artifact_root / "benchmarks" / "system_doctor.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        payload["artifact"] = str(destination)
        return payload

    @staticmethod
    def _package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    @staticmethod
    def _is_writable(path: Path) -> bool:
        probe = path / ".write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return True
        except OSError:
            return False

    @staticmethod
    def _torch_info() -> dict[str, Any]:
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
            return {
                "torch_available": True,
                "torch_version": torch.__version__,
                "cuda_available": cuda,
                "torch_cuda_version": torch.version.cuda,
                "active_device": "cuda" if cuda else "cpu",
            }
        except ImportError:
            return {
                "torch_available": False,
                "torch_version": None,
                "cuda_available": False,
                "torch_cuda_version": None,
                "active_device": "cpu",
            }

    @staticmethod
    def _nvidia_info() -> dict[str, Any] | None:
        command = shutil.which("nvidia-smi")
        if command is None:
            return None
        try:
            result = subprocess.run(
                [
                    command,
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            name, driver, memory = [part.strip() for part in result.stdout.strip().split(",", 2)]
            return {"name": name, "driver": driver, "memory_mb": int(memory)}
        except (OSError, subprocess.SubprocessError, ValueError):
            return {"status": "DETECTED_BUT_QUERY_FAILED"}
