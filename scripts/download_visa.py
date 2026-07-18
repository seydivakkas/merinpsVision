"""Download VisA only from its official AWS Open Data registry location."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


def download(url: str, destination: Path) -> Path:
    """Download an explicitly supplied official URL without extracting it."""
    if not url.startswith("https://registry.opendata.aws/") and "amazonaws.com" not in url:
        raise ValueError("VisA downloads must use the official AWS Open Data source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    return destination


def main() -> None:
    """Parse official URL and archive destination."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(download(args.url, args.output))


if __name__ == "__main__":
    main()
