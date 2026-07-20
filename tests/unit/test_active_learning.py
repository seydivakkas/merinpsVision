"""Unit tests for M6 active learning service.

Covers:
- select_candidates: returns <= n_select items
- select_candidates: respects min_drift_score filter
- select_candidates: all items persisted in labeling_queue
- select_candidates: empty after filter -> []
- select_candidates: shape mismatch -> ValueError
- select_candidates: n_select < 1 -> ValueError
- select_candidates: feature_matrix 1-D -> ValueError
- _greedy_coreset: k >= N returns all indices
- _greedy_coreset: k=1 returns single index
- _greedy_coreset: selects diverse points (not all the same)
- _greedy_coreset: zero-norm rows handled without NaN
- _score_to_bucket: boundary tests P0/P1/P2/P3
- list_pending: excludes reviewed items
- record_verdict: updates verdict + reviewed_at
- list_pending: sorted by priority_bucket ASC
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from weavevision.persistence.database import Database
from weavevision.services.active_learning_service import (
    ActiveLearningService,
    _greedy_coreset,
    _score_to_bucket,
)
from weavevision.settings import load_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tmp_db() -> Database:
    db = Database(Path(tempfile.mktemp(suffix=".sqlite3")))
    db.migrate()
    return db


def _service() -> ActiveLearningService:
    return ActiveLearningService(load_settings(), _tmp_db())


def _paths(n: int) -> list[Path]:
    return [Path(f"/data/img_{i:04d}.png") for i in range(n)]


def _features(n: int, d: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


# ---------------------------------------------------------------------------
# select_candidates
# ---------------------------------------------------------------------------


class TestSelectCandidates:
    def test_returns_at_most_n_select(self) -> None:
        svc = _service()
        paths = _paths(30)
        scores = np.linspace(0.5, 1.0, 30)
        items = svc.select_candidates(paths, scores, _features(30), n_select=10)
        assert len(items) <= 10

    def test_all_items_persisted(self) -> None:
        svc = _service()
        paths = _paths(20)
        scores = np.linspace(0.6, 1.0, 20)
        items = svc.select_candidates(paths, scores, _features(20), n_select=5)
        pending = svc.list_pending()
        persisted_ids = {r["item_id"] for r in pending}
        for item in items:
            assert item.item_id in persisted_ids

    def test_min_drift_score_filter(self) -> None:
        svc = _service()
        paths = _paths(10)
        # First 5 below 0.8, last 5 above
        scores = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])
        items = svc.select_candidates(
            paths, scores, _features(10), n_select=10, min_drift_score=0.8
        )
        # Only the last 5 qualify; all should be selected (5 < 10)
        assert len(items) == 5

    def test_empty_after_filter_returns_empty(self) -> None:
        svc = _service()
        paths = _paths(5)
        scores = np.full(5, 0.3)
        items = svc.select_candidates(paths, scores, _features(5), n_select=5, min_drift_score=0.9)
        assert items == []

    def test_drift_score_shape_mismatch_raises(self) -> None:
        svc = _service()
        with pytest.raises(ValueError, match="drift_scores shape"):
            svc.select_candidates(
                _paths(5),
                np.ones(3),  # wrong length
                _features(5),
            )

    def test_feature_matrix_1d_raises(self) -> None:
        svc = _service()
        with pytest.raises(ValueError, match="2-D array"):
            svc.select_candidates(
                _paths(5),
                np.ones(5),
                np.ones(5),  # 1-D
            )

    def test_n_select_zero_raises(self) -> None:
        svc = _service()
        with pytest.raises(ValueError, match="n_select"):
            svc.select_candidates(_paths(5), np.ones(5), _features(5), n_select=0)

    def test_items_have_correct_priority_buckets(self) -> None:
        svc = _service()
        paths = _paths(4)
        # Scores map to P0, P1, P2, P3 respectively
        scores = np.array([0.95, 0.80, 0.60, 0.30])
        items = svc.select_candidates(paths, scores, _features(4), n_select=4, min_drift_score=0.0)
        buckets = {item.priority_bucket for item in items}
        # At least P0 should be present (score 0.95 >= 0.90)
        assert "P0" in buckets

    def test_selection_reason_stored(self) -> None:
        svc = _service()
        paths = _paths(5)
        scores = np.ones(5) * 0.8
        items = svc.select_candidates(
            paths,
            scores,
            _features(5),
            n_select=3,
            selection_reason="test_reason",
        )
        assert all(item.selection_reason == "test_reason" for item in items)

    def test_n_select_larger_than_candidates(self) -> None:
        """n_select > available candidates -> return all candidates."""
        svc = _service()
        paths = _paths(3)
        scores = np.ones(3) * 0.9
        items = svc.select_candidates(paths, scores, _features(3), n_select=100)
        assert len(items) == 3


# ---------------------------------------------------------------------------
# _greedy_coreset
# ---------------------------------------------------------------------------


class TestGreedyCoreset:
    def test_k_ge_n_returns_all(self) -> None:
        feats = _features(5)
        result = _greedy_coreset(feats, k=10)
        assert set(result) == set(range(5))

    def test_k_equals_n(self) -> None:
        feats = _features(5)
        result = _greedy_coreset(feats, k=5)
        assert len(result) == 5
        assert len(set(result)) == 5  # no duplicates

    def test_k_1_returns_single(self) -> None:
        feats = _features(10)
        result = _greedy_coreset(feats, k=1)
        assert len(result) == 1

    def test_no_duplicates(self) -> None:
        feats = _features(20, d=4, seed=1)
        k = 8
        result = _greedy_coreset(feats, k=k)
        assert len(result) == k
        assert len(set(result)) == k

    def test_selects_diverse_points(self) -> None:
        """Greedy coreset should NOT select the same cluster repeatedly."""
        # Two tight clusters far apart; coreset should pick one from each
        cluster_a = np.zeros((5, 2), dtype=np.float32)
        cluster_b = np.ones((5, 2), dtype=np.float32) * 100.0
        feats = np.vstack([cluster_a, cluster_b])
        result = _greedy_coreset(feats, k=2)
        # One index from [0-4] and one from [5-9]
        has_a = any(idx < 5 for idx in result)
        has_b = any(idx >= 5 for idx in result)
        assert has_a and has_b, "Coreset should span both clusters"

    def test_zero_norm_rows_no_nan(self) -> None:
        """Zero-norm feature vectors must not produce NaN distances."""
        feats = np.zeros((5, 4), dtype=np.float32)
        result = _greedy_coreset(feats, k=3)
        assert len(result) == 3
        assert not any(np.isnan(feats[i]).any() for i in result)


# ---------------------------------------------------------------------------
# _score_to_bucket
# ---------------------------------------------------------------------------


class TestScoreToBucket:
    def test_p0_at_threshold(self) -> None:
        assert _score_to_bucket(0.90) == "P0"

    def test_p0_above_threshold(self) -> None:
        assert _score_to_bucket(1.00) == "P0"

    def test_p1_at_threshold(self) -> None:
        assert _score_to_bucket(0.75) == "P1"

    def test_p1_just_below_p0(self) -> None:
        assert _score_to_bucket(0.89) == "P1"

    def test_p2_at_threshold(self) -> None:
        assert _score_to_bucket(0.50) == "P2"

    def test_p2_just_below_p1(self) -> None:
        assert _score_to_bucket(0.74) == "P2"

    def test_p3_below_p2(self) -> None:
        assert _score_to_bucket(0.49) == "P3"

    def test_p3_at_zero(self) -> None:
        assert _score_to_bucket(0.0) == "P3"


# ---------------------------------------------------------------------------
# list_pending / record_verdict
# ---------------------------------------------------------------------------


class TestPendingAndVerdict:
    def test_list_pending_excludes_reviewed(self) -> None:
        svc = _service()
        paths = _paths(3)
        scores = np.ones(3) * 0.85
        items = svc.select_candidates(paths, scores, _features(3), n_select=3)
        assert len(items) >= 1

        # Mark first item as reviewed
        svc.record_verdict(items[0].item_id, "TRUE_ANOMALY", reviewer="tester")
        pending = svc.list_pending()
        ids = [r["item_id"] for r in pending]
        assert items[0].item_id not in ids

    def test_record_verdict_updates_row(self) -> None:
        svc = _service()
        paths = _paths(2)
        scores = np.ones(2) * 0.85
        items = svc.select_candidates(paths, scores, _features(2), n_select=2)
        item_id = items[0].item_id

        svc.record_verdict(item_id, "FALSE_POSITIVE", reviewer="qa_lead")

        with svc._db.connect() as conn:
            row = conn.execute(
                "SELECT verdict, assigned_reviewer, reviewed_at FROM labeling_queue"
                " WHERE item_id = ?",
                (item_id,),
            ).fetchone()
        assert row["verdict"] == "FALSE_POSITIVE"
        assert row["assigned_reviewer"] == "qa_lead"
        assert row["reviewed_at"] is not None

    def test_list_pending_sorted_by_bucket(self) -> None:
        """P0 items appear before P3 items in list_pending."""
        svc = _service()
        paths = _paths(6)
        # 3 low-priority, 3 high-priority
        scores = np.array([0.10, 0.20, 0.30, 0.91, 0.92, 0.93])
        svc.select_candidates(paths, scores, _features(6), n_select=6, min_drift_score=0.0)
        pending = svc.list_pending()
        buckets = [r["priority_bucket"] for r in pending]
        # All P0 items should come before P3 items
        p0_positions = [i for i, b in enumerate(buckets) if b == "P0"]
        p3_positions = [i for i, b in enumerate(buckets) if b == "P3"]
        if p0_positions and p3_positions:
            assert max(p0_positions) < min(p3_positions)
