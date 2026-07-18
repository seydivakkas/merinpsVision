"""Validate an authorized AITEX source path before normalized import."""

from __future__ import annotations

import argparse
from pathlib import Path

from weavevision.domain.errors import DatasetNotFoundError


def inspect_source(source: Path) -> dict[str, object]:
    """Inspect source images without copying, deleting, or splitting them."""
    if not source.is_dir():
        raise DatasetNotFoundError(f"AITEX source directory not found: {source}")
    images = sorted(
        path
        for path in source.rglob("*")
        if path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    return {
        "status": "READY_FOR_AUTHORIZED_IMPORT",
        "source": str(source.resolve()),
        "images": len(images),
    }


def main() -> None:
    """Print source inventory; transformation requires an explicit normalized-layout step."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(inspect_source(args.source))


if __name__ == "__main__":
    main()
