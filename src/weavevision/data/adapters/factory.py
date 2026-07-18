"""Safe dataset adapter construction from validated YAML mappings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from weavevision.data.adapters.aitex import AITEXAdapter
from weavevision.data.adapters.base import DatasetAdapter
from weavevision.data.adapters.company import CompanyDatasetAdapter
from weavevision.data.adapters.fixture import FixtureAdapter
from weavevision.data.adapters.mvtec_ad import MVTecADAdapter
from weavevision.data.adapters.visa import VisAAdapter
from weavevision.domain.errors import ConfigError

ADAPTERS: dict[str, type[MVTecADAdapter]] = {
    "mvtec_ad": MVTecADAdapter,
    "fixture": FixtureAdapter,
    "visa": VisAAdapter,
    "aitex": AITEXAdapter,
    "company": CompanyDatasetAdapter,
}


def adapter_from_config(config_path: Path) -> DatasetAdapter:
    """Build an allow-listed adapter from a dataset YAML file."""
    try:
        payload: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        dataset = payload["dataset"]
        adapter_name = str(dataset["adapter"])
        adapter_type = ADAPTERS[adapter_name]
        return adapter_type(
            root=(config_path.parent.parent.parent / str(dataset["root"])).resolve(),
            manifest_path=(
                config_path.parent.parent.parent / str(dataset["manifest_path"])
            ).resolve(),
            category=str(dataset.get("category", "carpet")),
        )
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ConfigError(f"invalid dataset configuration {config_path}: {exc}") from exc
