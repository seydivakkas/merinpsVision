"""Latency and memory benchmark helpers."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import psutil


def benchmark_callable(
    operation: Callable[[], object], warmup_runs: int = 10, measured_runs: int = 50
) -> dict[str, Any]:
    """Measure synchronized wall latency percentiles and process RAM."""
    if warmup_runs < 0 or measured_runs < 1:
        raise ValueError("invalid benchmark run counts")
    for _ in range(warmup_runs):
        operation()
    samples = []
    process = psutil.Process()
    peak_ram = process.memory_info().rss
    for _ in range(measured_runs):
        started = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - started) * 1000)
        peak_ram = max(peak_ram, process.memory_info().rss)
    values = np.asarray(samples)
    return {
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "latency_ms": {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "mean": float(values.mean()),
        },
        "peak_ram_mb": float(peak_ram / 1024**2),
        "throughput_images_per_second": float(1000 / values.mean()),
    }
