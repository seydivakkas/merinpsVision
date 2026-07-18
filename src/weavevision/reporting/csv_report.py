"""Flat CSV report output for single and batch results."""

from __future__ import annotations

import csv
from pathlib import Path

from weavevision.domain.schemas import AnalysisResult, BatchResult


def write_csv_report(result: AnalysisResult | BatchResult, destination: Path) -> Path:
    """Write core audit fields as a spreadsheet-safe CSV file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = result.results if isinstance(result, BatchResult) else [result]
    fields = [
        "analysis_id",
        "created_at",
        "filename",
        "sha256",
        "decision",
        "review_priority",
        "raw_anomaly_score",
        "anomaly_area_ratio",
        "region_count",
        "model_id",
        "threshold_id",
        "quality_status",
        "total_latency_ms",
    ]
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in values:
            writer.writerow(
                {
                    "analysis_id": item.analysis_id,
                    "created_at": item.created_at.isoformat(),
                    "filename": _safe_csv(item.source.filename),
                    "sha256": item.source.sha256,
                    "decision": item.prediction.decision.value,
                    "review_priority": item.prediction.review_priority.value,
                    "raw_anomaly_score": item.prediction.raw_anomaly_score,
                    "anomaly_area_ratio": item.prediction.anomaly_area_ratio,
                    "region_count": item.prediction.region_count,
                    "model_id": item.model.model_id if item.model else "",
                    "threshold_id": item.threshold.threshold_id if item.threshold else "",
                    "quality_status": item.quality_gate.status.value,
                    "total_latency_ms": item.timing_ms.total,
                }
            )
    return destination


def _safe_csv(value: str) -> str:
    return f"'{value}" if value.startswith(("=", "+", "-", "@")) else value
