"""Hash-verifying model registry with explicit promotion gates."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from weavevision.domain.enums import ModelStatus
from weavevision.domain.errors import ModelHashMismatchError, ModelNotReadyError
from weavevision.domain.schemas import ModelManifest
from weavevision.models.export import sha256_artifact


class ModelRegistry:
    """Manage immutable model manifests while keeping weights outside Git."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifests = root / "manifests"

    def register(self, manifest: ModelManifest) -> Path:
        """Integrity-check and atomically persist a model manifest."""
        artifact = self._resolve_artifact(manifest.artifact_path)
        if not artifact.is_file() or sha256_artifact(artifact) != manifest.artifact_sha256:
            raise ModelHashMismatchError(f"model artifact hash mismatch: {artifact}")
        self.manifests.mkdir(parents=True, exist_ok=True)
        destination = self.manifests / f"{manifest.model_id}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, destination)
        return destination

    def list(self) -> list[ModelManifest]:
        """Return all valid model manifests sorted newest first."""
        if not self.manifests.is_dir():
            return []
        values = [
            ModelManifest.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self.manifests.glob("*.json")
        ]
        return sorted(values, key=lambda item: item.created_at, reverse=True)

    def get(self, model_id: str, verify_hash: bool = True) -> ModelManifest:
        """Load a manifest and optionally verify its artifact hash."""
        path = self.manifests / f"{model_id}.json"
        if not path.is_file():
            raise ModelNotReadyError(f"model manifest not found: {model_id}")
        manifest = ModelManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if verify_hash:
            artifact = self._resolve_artifact(manifest.artifact_path)
            if not artifact.is_file() or sha256_artifact(artifact) != manifest.artifact_sha256:
                raise ModelHashMismatchError(f"model artifact hash mismatch: {artifact}")
        return manifest

    def active(self, company: bool = False) -> ModelManifest | None:
        """Return the unique active benchmark or company-pilot model."""
        target = ModelStatus.ACTIVE_COMPANY_PILOT if company else ModelStatus.ACTIVE_BENCHMARK
        active = [item for item in self.list() if item.status is target]
        if len(active) > 1:
            raise ModelNotReadyError(f"registry has multiple active models for {target.value}")
        return self.get(active[0].model_id) if active else None

    def promote(self, model_id: str, reason: str, *, company: bool = False) -> ModelManifest:
        """Promote an eligible validated model and retire the previous active model."""
        candidate = self.get(model_id)
        if candidate.status not in {ModelStatus.CANDIDATE, ModelStatus.VALIDATED}:
            raise ModelNotReadyError(f"model status is not promotable: {candidate.status.value}")
        if (
            not candidate.threshold_id
            or not candidate.metrics_path
            or not candidate.metrics_path.is_file()
        ):
            raise ModelNotReadyError("promotion requires locked threshold and metrics evidence")
        target = ModelStatus.ACTIVE_COMPANY_PILOT if company else ModelStatus.ACTIVE_BENCHMARK
        for current in self.list():
            if current.status is target:
                self.register(current.model_copy(update={"status": ModelStatus.RETIRED}))
        promoted = candidate.model_copy(
            update={
                "status": target,
                "promoted_at": datetime.now(UTC),
                "promotion_reason": reason,
            }
        )
        self.register(promoted)
        return promoted

    def health(self) -> dict[str, object]:
        """Report registry manifest count and active identities."""
        values = self.list()
        return {
            "manifests": len(values),
            "active_benchmark": next(
                (item.model_id for item in values if item.status is ModelStatus.ACTIVE_BENCHMARK),
                None,
            ),
            "active_company_pilot": next(
                (
                    item.model_id
                    for item in values
                    if item.status is ModelStatus.ACTIVE_COMPANY_PILOT
                ),
                None,
            ),
        }

    def _resolve_artifact(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.root.parent.parent / path).resolve()
