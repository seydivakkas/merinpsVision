"""Model registry, artifact integrity, health, and logging tests."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from weavevision.domain.enums import ModelStatus
from weavevision.domain.errors import ModelHashMismatchError, ModelNotReadyError
from weavevision.domain.schemas import ModelManifest
from weavevision.logging_config import configure_logging
from weavevision.models.export import sha256_artifact
from weavevision.models.registry import ModelRegistry
from weavevision.services.factory import load_active_analysis_service
from weavevision.services.health_service import HealthService
from weavevision.settings import load_settings


def isolated_settings(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Return settings whose mutable roots live in a test directory."""
    settings = load_settings()
    return settings.model_copy(
        update={
            "paths": settings.paths.model_copy(
                update={
                    "data_root": tmp_path / "data",
                    "artifacts_root": tmp_path / "artifacts",
                    "database": tmp_path / "artifacts" / "db.sqlite3",
                }
            )
        }
    )


def test_openvino_bundle_hash_changes_with_bin(tmp_path: Path) -> None:
    xml = tmp_path / "model.xml"
    binary = tmp_path / "model.bin"
    xml.write_text("<xml />", encoding="utf-8")
    binary.write_bytes(b"weights-a")
    first = sha256_artifact(xml)
    binary.write_bytes(b"weights-b")
    assert sha256_artifact(xml) != first


def test_registry_registers_and_detects_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"safe test bytes")
    manifest = ModelManifest(
        model_id="model_test",
        status=ModelStatus.CANDIDATE,
        algorithm="patchcore",
        dataset_manifest_sha256="a" * 64,
        training_run_id="run_test",
        config_sha256="b" * 64,
        artifact_path=artifact,
        artifact_sha256=sha256_artifact(artifact),
        created_at=datetime.now(UTC),
    )
    registry = ModelRegistry(tmp_path / "registry")
    registry.register(manifest)
    assert registry.get("model_test").model_id == "model_test"
    assert registry.health()["manifests"] == 1
    artifact.write_bytes(b"tampered")
    with pytest.raises(ModelHashMismatchError):
        registry.get("model_test")


def test_registry_rejects_ineligible_promotion(tmp_path: Path) -> None:
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"model")
    registry = ModelRegistry(tmp_path / "registry")
    registry.register(
        ModelManifest(
            model_id="candidate",
            status=ModelStatus.CANDIDATE,
            algorithm="patchcore",
            dataset_manifest_sha256="a" * 64,
            training_run_id="run",
            config_sha256="b" * 64,
            artifact_path=artifact,
            artifact_sha256=sha256_artifact(artifact),
            created_at=datetime.now(UTC),
        )
    )
    with pytest.raises(ModelNotReadyError, match="threshold"):
        registry.promote("candidate", "test")


def test_health_and_structured_logging(tmp_path: Path) -> None:
    settings = isolated_settings(tmp_path)
    result = HealthService(settings).collect()
    assert result["status"] == "PASS"
    logger = configure_logging(tmp_path / "logs" / "events.jsonl")
    logger.info("health complete", extra={"event": "doctor"})
    assert '"event":"doctor"' in (tmp_path / "logs" / "events.jsonl").read_text()


def test_service_factory_requires_active_model(tmp_path: Path) -> None:
    with pytest.raises(ModelNotReadyError):
        load_active_analysis_service(isolated_settings(tmp_path))
