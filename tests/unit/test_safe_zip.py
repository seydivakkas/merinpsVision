"""ZIP path traversal tests."""

from pathlib import Path
from zipfile import ZipFile

import pytest

from weavevision.services.batch_service import safe_extract_zip


def test_zip_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "blocked")
    with pytest.raises(ValueError, match="traversal"):
        safe_extract_zip(archive, tmp_path / "extract")
