"""No-model degraded state smoke tests."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from weavevision.domain.enums import Decision
from weavevision.services.analysis_service import AnalysisService
from weavevision.settings import load_settings

pytestmark = pytest.mark.smoke


def test_tiny_invalid_quality_input_abstains_without_model(tmp_path: Path) -> None:
    path = tmp_path / "tiny.png"
    Image.fromarray(np.ones((16, 16, 3), dtype=np.uint8) * 100).save(path)
    settings = load_settings().model_copy(
        update={
            "paths": load_settings().paths.model_copy(
                update={
                    "artifacts_root": tmp_path / "artifacts",
                    "database": tmp_path / "artifacts" / "audit.sqlite3",
                }
            )
        }
    )
    result = AnalysisService(settings).analyze(path)
    assert result.prediction.decision is Decision.ABSTAIN
