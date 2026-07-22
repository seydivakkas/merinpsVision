"""Anomalib 2.5 adapter hiding framework objects from application services."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np

from weavevision.domain.errors import ConfigError, InferenceError, ModelNotReadyError
from weavevision.models.factory import create_anomalib_model


class AnomalibAdapter:
    """Train and infer supported Anomalib models through a stable local contract."""

    SUPPORTED_VERSION = "2.5.0"

    def __init__(
        self,
        model_id: str,
        algorithm: str,
        model_config: dict[str, Any],
        artifact_path: Path | None = None,
        device: str = "auto",
    ) -> None:
        self._model_id = model_id
        self.algorithm = algorithm
        self.model_config = model_config
        self.artifact_path = artifact_path
        self.device = device
        self._inferencer: Any | None = None
        self._check_version()

    @property
    def model_id(self) -> str:
        """Return immutable model identifier."""
        return self._model_id

    def fit_mvtec(
        self,
        dataset_root: Path,
        category: str,
        destination: Path,
        *,
        seed: int = 42,
        max_epochs: int | None = None,
        num_workers: int = 0,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        precision: str | None = None,
    ) -> tuple[Path, list[tuple[float, np.ndarray]]]:
        """Fit on normal MVTec-style training data and return validation predictions.

        Validation is derived only from the training directory via Anomalib's
        ``from_train`` mode. The sealed test directory is not used here.
        """
        from anomalib.data import MVTecAD
        from anomalib.engine import Engine

        model = create_anomalib_model(self.algorithm, self.model_config)
        datamodule = MVTecAD(
            root=dataset_root,
            category=category,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            val_split_mode="from_train",
            val_split_ratio=0.2,
            seed=seed,
        )
        engine_kwargs: dict[str, Any] = {
            "default_root_dir": destination.parent,
            "logger": False,
            "enable_progress_bar": False,
            "deterministic": True,
        }
        if max_epochs is not None:
            engine_kwargs["max_epochs"] = max_epochs
        if precision is not None:
            engine_kwargs["precision"] = precision
        if self.device == "cpu":
            engine_kwargs["accelerator"] = "cpu"
        engine = Engine(**engine_kwargs)
        engine.fit(model=model, datamodule=datamodule)
        destination.parent.mkdir(parents=True, exist_ok=True)
        engine.trainer.save_checkpoint(destination)
        exported = engine.export(
            model=model,
            export_type="openvino",
            export_root=destination.parent,
            model_file_name="model",
        )
        if exported is None or not exported.is_file():
            raise RuntimeError("Anomalib did not produce a Torch deployment artifact")
        self.artifact_path = exported
        predictions = engine.predict(
            model=model,
            dataloaders=datamodule.val_dataloader(),
            return_predictions=True,
        )
        return exported, _flatten_predictions(predictions)

    def predict_array(self, image_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        """Predict image score and pixel anomaly map from uint8 RGB data.

        Raises:
            ModelNotReadyError: If the adapter has no checkpoint.
            InferenceError: If Anomalib does not return required fields.
        """
        if self.artifact_path is None or not self.artifact_path.is_file():
            raise ModelNotReadyError("Anomalib checkpoint is not available")
        if self._inferencer is None:
            from anomalib.deploy import OpenVINOInferencer

            device = "CPU" if self.device == "cpu" else "AUTO"
            self._inferencer = OpenVINOInferencer(path=self.artifact_path, device=device)
        try:
            prediction = self._inferencer.predict(image_rgb)
            return _prediction_values(prediction)
        except (AttributeError, RuntimeError, ValueError) as exc:
            raise InferenceError(f"Anomalib inference failed: {exc}") from exc

    def export(self, destination: Path) -> Path:
        """Copy the immutable checkpoint to a requested registry location."""
        if self.artifact_path is None or not self.artifact_path.is_file():
            raise ModelNotReadyError("model artifact is not available for export")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.artifact_path.read_bytes())
        return destination

    @classmethod
    def _check_version(cls) -> None:
        installed = importlib.metadata.version("anomalib")
        if installed != cls.SUPPORTED_VERSION:
            raise ConfigError(
                f"Anomalib {cls.SUPPORTED_VERSION} required, found {installed}; adapter not loaded"
            )


def _flatten_predictions(predictions: Any) -> list[tuple[float, np.ndarray]]:
    if predictions is None:
        return []
    result: list[tuple[float, np.ndarray]] = []
    for batch in predictions:
        scores = _to_numpy(batch.pred_score)
        maps = _to_numpy(batch.anomaly_map)
        for index in range(scores.reshape(-1).size):
            result.append((float(scores.reshape(-1)[index]), np.squeeze(maps[index])))
    return result


def _prediction_values(prediction: Any) -> tuple[float, np.ndarray]:
    score = _to_numpy(prediction.pred_score).reshape(-1)
    anomaly_map = _to_numpy(prediction.anomaly_map)
    if score.size == 0 or anomaly_map.size == 0:
        raise ValueError("prediction contains empty score or anomaly map")
    return float(score[0]), np.squeeze(anomaly_map).astype(np.float32)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
