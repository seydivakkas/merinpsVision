"""Dataset adapter and feedback service integration tests."""

from pathlib import Path

from weavevision.data.adapters.factory import adapter_from_config
from weavevision.domain.enums import FeedbackVerdict
from weavevision.services.feedback_service import FeedbackService
from weavevision.settings import load_settings

from .test_analysis_batch_reporting import service


def test_fixture_adapter_writes_verified_manifest() -> None:
    root = load_settings().project_root
    manifest = adapter_from_config(root / "configs" / "datasets" / "fixture.yaml").verify()
    assert manifest.dataset_id == "weavevision_fixture"
    assert manifest.manifest_sha256
    assert manifest.counts.images_total == 20


def test_feedback_persists_after_analysis(tmp_path: Path, textile_image: Path) -> None:
    analysis = service(tmp_path).analyze(textile_image)
    database = tmp_path / "artifacts" / "audit.sqlite3"
    feedback_id = FeedbackService(database).submit(
        analysis.analysis_id,
        "reviewer",
        FeedbackVerdict.CONFIRMED_NORMAL,
        "fixture review",
    )
    assert feedback_id.startswith("fb_")
