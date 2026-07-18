"""Structured local JSON logging without user image content."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as one privacy-conscious JSON object per line."""

    _context_fields = (
        "event",
        "run_id",
        "analysis_id",
        "batch_id",
        "model_id",
        "threshold_id",
        "dataset_id",
        "duration_ms",
        "device",
        "error_code",
    )

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record using the stable observability contract."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for field in self._context_fields:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(log_path: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure the package logger with stderr and JSONL file handlers.

    Args:
        log_path: Destination JSONL path.
        level: Minimum emitted logging level.

    Returns:
        Configured ``weavevision`` logger.

    Side Effects:
        Creates the log parent directory and replaces package handlers.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("weavevision")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger
