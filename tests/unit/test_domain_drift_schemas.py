"""Unit tests for drift lifecycle domain schemas (M1).

Each schema gets one valid construction test and one extra-field rejection test,
following the ConfigDict(extra="forbid") project convention.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from weavevision.domain.enums import (
    CanaryStatus,
    DriftPattern,
    ExperimentStatus,
    FeedbackVerdict,
    IncidentPriority,
    RetrainingStrategy,
    RollbackReason,
    TrendStatus,
)
from weavevision.domain.schemas import (
    CanaryEvaluation,
    DriftWindow,
    IncidentRecord,
    LabelingQueueItem,
    RetrainingRequest,
    RollbackEvent,
    TrendPoint,
    TriageDecision,
)

_NOW = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# DriftWindow
# ---------------------------------------------------------------------------


class TestDriftWindow:
    def _valid(self) -> dict:
        return {
            "window_id": "win_001",
            "model_id": "model_abc",
            "metric_name": "image_ap50",
            "window_start": _NOW,
            "window_end": _NOW,
            "trend_status": TrendStatus.STABLE,
            "drift_pattern": DriftPattern.STABLE,
            "created_at": _NOW,
        }

    def test_valid_construction(self) -> None:
        w = DriftWindow(**self._valid())
        assert w.window_id == "win_001"
        assert w.trend_status is TrendStatus.STABLE

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DriftWindow(**self._valid(), unknown_field="bad")

    def test_optional_metrics_default_to_none(self) -> None:
        w = DriftWindow(**self._valid())
        assert w.psi_value is None
        assert w.ewma_value is None


# ---------------------------------------------------------------------------
# TrendPoint
# ---------------------------------------------------------------------------


class TestTrendPoint:
    def _valid(self) -> dict:
        return {
            "index": 1,
            "value": 0.82,
            "ewma": 0.81,
            "ewma_lower_limit": 0.75,
            "cusum_down": 0.0,
            "ewma_alert": False,
            "cusum_alert": False,
            "status": TrendStatus.STABLE,
        }

    def test_valid_construction(self) -> None:
        tp = TrendPoint(**self._valid())
        assert tp.index == 1
        assert tp.cusum_down == 0.0

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrendPoint(**self._valid(), extra="nope")

    def test_index_must_be_ge_1(self) -> None:
        data = self._valid()
        data["index"] = 0
        with pytest.raises(ValidationError):
            TrendPoint(**data)

    def test_cusum_down_must_be_ge_0(self) -> None:
        data = self._valid()
        data["cusum_down"] = -0.1
        with pytest.raises(ValidationError):
            TrendPoint(**data)


# ---------------------------------------------------------------------------
# IncidentRecord
# ---------------------------------------------------------------------------


class TestIncidentRecord:
    def _valid(self) -> dict:
        return {
            "incident_id": "inc_001",
            "priority": IncidentPriority.P1_INCIDENT,
            "drift_pattern": DriftPattern.SUDDEN,
            "affected_window_id": "win_001",
            "model_id": "model_abc",
            "created_at": _NOW,
        }

    def test_valid_construction(self) -> None:
        ir = IncidentRecord(**self._valid())
        assert ir.priority is IncidentPriority.P1_INCIDENT
        assert ir.resolved_at is None

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IncidentRecord(**self._valid(), injected="evil")


# ---------------------------------------------------------------------------
# TriageDecision
# ---------------------------------------------------------------------------


class TestTriageDecision:
    def _valid(self) -> dict:
        return {
            "decision_id": "dec_001",
            "incident_id": "inc_001",
            "confirming_signal_count": 2,
            "created_at": _NOW,
        }

    def test_valid_construction(self) -> None:
        td = TriageDecision(**self._valid())
        assert td.confirming_signal_count == 2

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TriageDecision(**self._valid(), extra="x")

    def test_signal_count_le_6(self) -> None:
        data = self._valid()
        data["confirming_signal_count"] = 7
        with pytest.raises(ValidationError):
            TriageDecision(**data)

    def test_signal_count_ge_0(self) -> None:
        data = self._valid()
        data["confirming_signal_count"] = -1
        with pytest.raises(ValidationError):
            TriageDecision(**data)


# ---------------------------------------------------------------------------
# RetrainingRequest
# ---------------------------------------------------------------------------


class TestRetrainingRequest:
    def _valid(self) -> dict:
        return {
            "request_id": "req_001",
            "trigger_id": "inc_001",
            "strategy": RetrainingStrategy.FULL_RETRAIN,
            "target_model_family": "patchcore",
            "min_target_images": 200,
            "min_labeled_validation_images": 100,
            "status": ExperimentStatus.NOT_RUN,
            "created_at": _NOW,
        }

    def test_valid_construction(self) -> None:
        rr = RetrainingRequest(**self._valid())
        assert rr.strategy is RetrainingStrategy.FULL_RETRAIN

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrainingRequest(**self._valid(), bad="field")

    def test_min_target_images_gt_0(self) -> None:
        data = self._valid()
        data["min_target_images"] = 0
        with pytest.raises(ValidationError):
            RetrainingRequest(**data)


# ---------------------------------------------------------------------------
# LabelingQueueItem
# ---------------------------------------------------------------------------


class TestLabelingQueueItem:
    def _valid(self) -> dict:
        return {
            "item_id": "lq_001",
            "image_sha256": "abc123",
            "source_path": "/data/fixtures/normal/img_001.png",
            "priority_bucket": "P1",
            "selection_reason": "uncertainty",
            "created_at": _NOW,
        }

    def test_valid_construction(self) -> None:
        lq = LabelingQueueItem(**self._valid())
        assert lq.verdict is None

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LabelingQueueItem(**self._valid(), extra="x")

    def test_invalid_priority_bucket(self) -> None:
        data = self._valid()
        data["priority_bucket"] = "P5"
        with pytest.raises(ValidationError):
            LabelingQueueItem(**data)

    def test_verdict_can_be_feedback_verdict(self) -> None:
        data = self._valid()
        data["verdict"] = FeedbackVerdict.CONFIRMED_ANOMALY
        lq = LabelingQueueItem(**data)
        assert lq.verdict is FeedbackVerdict.CONFIRMED_ANOMALY


# ---------------------------------------------------------------------------
# CanaryEvaluation
# ---------------------------------------------------------------------------


class TestCanaryEvaluation:
    def _valid(self) -> dict:
        return {
            "canary_id": "can_001",
            "champion_model_id": "model_a",
            "challenger_model_id": "model_b",
            "sample_count": 100,
            "disagreement_rate": 0.05,
            "critical_recall_delta": -0.02,
            "latency_p95_ms": 42.0,
            "status": CanaryStatus.PASSED,
            "created_at": _NOW,
        }

    def test_valid_construction(self) -> None:
        ce = CanaryEvaluation(**self._valid())
        assert ce.status is CanaryStatus.PASSED

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CanaryEvaluation(**self._valid(), extra="x")

    def test_disagreement_rate_bounds(self) -> None:
        data = self._valid()
        data["disagreement_rate"] = 1.1
        with pytest.raises(ValidationError):
            CanaryEvaluation(**data)

    def test_latency_must_be_ge_0(self) -> None:
        data = self._valid()
        data["latency_p95_ms"] = -1.0
        with pytest.raises(ValidationError):
            CanaryEvaluation(**data)


# ---------------------------------------------------------------------------
# RollbackEvent
# ---------------------------------------------------------------------------


class TestRollbackEvent:
    def _valid(self) -> dict:
        return {
            "rollback_id": "rb_001",
            "from_model_id": "model_b",
            "to_model_id": "model_a",
            "reason": RollbackReason.RECALL_DROP,
            "triggered_by": "incident_service",
            "created_at": _NOW,
        }

    def test_valid_construction(self) -> None:
        rb = RollbackEvent(**self._valid())
        assert rb.reason is RollbackReason.RECALL_DROP
        assert rb.incident_id is None

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RollbackEvent(**self._valid(), extra="x")

    def test_incident_id_is_optional(self) -> None:
        data = self._valid()
        data["incident_id"] = "inc_001"
        rb = RollbackEvent(**data)
        assert rb.incident_id == "inc_001"
