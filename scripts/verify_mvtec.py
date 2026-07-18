"""Verify a manually downloaded MVTec AD category."""

from __future__ import annotations

import argparse
from pathlib import Path

from weavevision.data.adapters.mvtec_ad import MVTecADAdapter


def main() -> None:
    """Run governed MVTec verification and print the manifest identity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--category", default="carpet")
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/mvtec_ad_carpet.json")
    )
    args = parser.parse_args()
    manifest = MVTecADAdapter(args.root, args.manifest, args.category).verify()
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
