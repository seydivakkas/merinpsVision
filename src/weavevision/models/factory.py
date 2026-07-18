"""Allow-listed Anomalib model construction without arbitrary imports."""

from __future__ import annotations

from typing import Any

from weavevision.domain.errors import ConfigError


def create_anomalib_model(name: str, config: dict[str, Any]) -> Any:
    """Create a supported Anomalib model from validated keyword arguments.

    Raises:
        ConfigError: If the algorithm or a supplied argument is unsupported.
    """
    if name == "patchcore":
        from anomalib.models import Patchcore

        allowed = {
            "backbone",
            "layers",
            "pre_trained",
            "coreset_sampling_ratio",
            "num_neighbors",
            "precision",
        }
        return Patchcore(
            **_allowed_kwargs(config, allowed, name),
            post_processor=False,
            evaluator=False,
            visualizer=False,
        )
    if name == "efficient_ad":
        from anomalib.models import EfficientAd

        allowed = {"imagenet_dir", "teacher_out_channels", "model_size", "padding", "pad_maps"}
        return EfficientAd(
            **_allowed_kwargs(config, allowed, name),
            post_processor=False,
            evaluator=False,
            visualizer=False,
        )
    if name == "padim":
        from anomalib.models import Padim

        allowed = {"backbone", "layers", "pre_trained", "n_features"}
        return Padim(
            **_allowed_kwargs(config, allowed, name),
            post_processor=False,
            evaluator=False,
            visualizer=False,
        )
    raise ConfigError(f"unsupported model algorithm: {name}")


def _allowed_kwargs(config: dict[str, Any], allowed: set[str], name: str) -> dict[str, Any]:
    payload = {key: value for key, value in config.items() if key != "name"}
    unknown = set(payload) - allowed
    if unknown:
        raise ConfigError(f"unsupported {name} arguments: {sorted(unknown)}")
    return payload
