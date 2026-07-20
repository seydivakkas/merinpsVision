"""Drift monitoring window orchestration service.

Orchestrates the statistical pipeline:
    raw metric values -> CUSUM/EWMA trend monitor -> DriftWindow record -> DB

No GPU, no I/O beyond the database write.  All statistical computation
is delegated to ``evaluation.trend_monitor`` and ``evaluation.psi``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import numpy as np

from weavevision.domain.enums import DriftPattern, TrendStatus
from weavevision.domain.schemas import DriftWindow
from weavevision.evaluation.alert_policy import DriftAlertPolicy
from weavevision.evaluation.trend_monitor import monitor_downward_drift
from weavevision.persistence.database import Database
from weavevision.persistence.repositories import DriftWindowRepository
from weavevision.settings import Settings


class DriftMonitorService:
    """Compute and persist one drift monitoring window.

    The service owns the boundary between raw metric observations and the
    audit trail.  It does not decide whether to open an incident -- that
    responsibility belongs to ``IncidentService``.

    Args:
        settings: Loaded application settings (drift thresholds come from here).
        database: Initialised ``Database`` instance (migrate() already called).
    """

    def __init__(self, settings: Settings, database: Database) -> None:
        self._settings = settings
        self._repo = DriftWindowRepository(database)
        self._policy = DriftAlertPolicy(
            minimum_signals=settings.drift.retraining_min_confirming_signals,
            ewma_lambda=settings.drift.ewma_lambda,
            ewma_limit_sigma=settings.drift.ewma_limit_sigma,
            cusum_k_sigma=settings.drift.cusum_k_sigma,
            cusum_h_sigma=settings.drift.cusum_h_sigma,
            psi_medium_threshold=settings.drift.psi_medium_threshold,
            psi_high_threshold=settings.drift.psi_high_threshold,
            sudden_drop_review_pp=settings.drift.sudden_drop_review_pp,
            sudden_drop_incident_pp=settings.drift.sudden_drop_incident_pp,
            sudden_drop_block_pp=settings.drift.sudden_drop_block_pp,
        )

    def run_window(
        self,
        metric_name: str,
        values: np.ndarray,
        baseline_mean: float,
        baseline_std: float,
        model_id: str,
        *,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        threshold_id: str | None = None,
        source_manifest_sha256: str | None = None,
    ) -> DriftWindow:
        """Run statistical monitors on *values* and persist the window record.

        Computes CUSUM + EWMA via the configured thresholds from
        ``configs/app.yaml -> drift``.  The worst-case alert state across
        the entire window is used as the ``trend_status``.

        Args:
            metric_name: Human-readable metric identifier (e.g. ``'image_ap50'``).
            values: Time-ordered metric observations (1-D NumPy, non-empty).
            baseline_mean: Operational-validation baseline mean.
            baseline_std: Operational-validation baseline standard deviation.
            model_id: Registry identifier of the model being monitored.
            window_start: Window opening timestamp (defaults to UTC now).
            window_end: Window closing timestamp (defaults to UTC now).
            threshold_id: Optional threshold registry identifier.
            source_manifest_sha256: Optional SHA-256 of the source manifest.

        Returns:
            The persisted ``DriftWindow`` schema instance.

        Raises:
            ValueError: Forwarded from ``monitor_downward_drift`` when
                ``values`` is empty or ``baseline_std <= 0``.
        """
        now = datetime.now(UTC)
        window_start = window_start or now
        window_end = window_end or now

        points = monitor_downward_drift(
            values,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            ewma_lambda=self._policy.ewma_lambda,
            ewma_limit_sigma=self._policy.ewma_limit_sigma,
            cusum_k_sigma=self._policy.cusum_k_sigma,
            cusum_h_sigma=self._policy.cusum_h_sigma,
        )

        # Aggregate: pick the worst-case trend_status across all points
        status_priority = {
            "STABLE": 0,
            "EWMA_ALERT": 1,
            "CUSUM_ALERT": 2,
            "BOTH_ALERT": 3,
        }
        worst = max(points, key=lambda p: status_priority.get(p.status, 0))

        trend_status = TrendStatus(worst.status)
        drift_pattern = _infer_drift_pattern(trend_status, len(points))

        window = DriftWindow(
            window_id=f"win_{uuid4().hex[:12]}",
            model_id=model_id,
            threshold_id=threshold_id,
            metric_name=metric_name,
            window_start=window_start,
            window_end=window_end,
            metric_value=float(values[-1]),
            ewma_value=worst.ewma,
            cusum_value=worst.cusum_down,
            trend_status=trend_status,
            drift_pattern=drift_pattern,
            source_manifest_sha256=source_manifest_sha256,
            created_at=now,
        )

        self._repo.save(window)
        return window

    @property
    def policy(self) -> DriftAlertPolicy:
        """Expose the resolved alert policy for downstream consumers."""
        return self._policy


def _infer_drift_pattern(status: TrendStatus, window_length: int) -> DriftPattern:
    """Heuristic: map a trend status + window length to a drift pattern.

    This is a lightweight heuristic for tagging -- not a rigorous classifier.
    Short windows with BOTH_ALERT are labelled SUDDEN; longer ones GRADUAL.

    Args:
        status: Worst-case trend status of the window.
        window_length: Number of observations in the window.

    Returns:
        A ``DriftPattern`` enum value.
    """
    if status == TrendStatus.STABLE:
        return DriftPattern.STABLE
    if status == TrendStatus.BOTH_ALERT and window_length <= 3:
        return DriftPattern.SUDDEN
    if status in (TrendStatus.CUSUM_ALERT, TrendStatus.BOTH_ALERT):
        return DriftPattern.GRADUAL
    # EWMA_ALERT alone -- likely technical noise
    return DriftPattern.TECHNICAL
