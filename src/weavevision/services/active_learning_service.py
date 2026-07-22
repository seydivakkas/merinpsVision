"""Active learning labeling queue service.

Implements coreset-based candidate selection for expert labeling.
All computation is CPU-bound (NumPy/greedy coreset).  The GPU is never
touched here -- feature vectors come pre-computed from inference.

RTX 4070 Laptop constraint: coreset_ratio from configs/app.yaml
(TASK-042 fixed it to 0.02 to stay within 8 GB VRAM).  The
labeling queue itself is post-inference, so it has no VRAM impact.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from weavevision.domain.schemas import LabelingQueueItem
from weavevision.persistence.database import Database
from weavevision.settings import Settings

# Priority bucket thresholds (drift_score-based, initial policy).
# These are NOT read from app.yaml yet -- they are local constants that
# will be promoted to DriftPolicyConfig in a follow-up milestone.
_BUCKET_P0_THRESHOLD = 0.90
_BUCKET_P1_THRESHOLD = 0.75
_BUCKET_P2_THRESHOLD = 0.50


class ActiveLearningService:
    """Select and enqueue active-learning candidates for expert labeling.

    Selection strategy:
        1. Filter to ``drift_score >= min_drift_score`` candidates.
        2. Run greedy coreset on the feature matrix to maximise diversity.
        3. Assign priority bucket based on drift_score.
        4. Persist selected items to the ``labeling_queue`` table.

    Args:
        settings: Loaded application settings.
        database: Initialised ``Database`` instance (migrate() called).
    """

    def __init__(self, settings: Settings, database: Database) -> None:
        self._settings = settings
        self._db = database

    def select_candidates(
        self,
        image_paths: Sequence[Path],
        drift_scores: np.ndarray,
        feature_matrix: np.ndarray,
        *,
        n_select: int = 20,
        min_drift_score: float = 0.0,
        selection_reason: str = "drift_coreset",
    ) -> list[LabelingQueueItem]:
        """Select diverse high-drift candidates and enqueue them.

        Args:
            image_paths: Ordered sequence of source image paths.
            drift_scores: 1-D array of per-image drift scores, same order as
                *image_paths* (higher = more anomalous/drifted).
            feature_matrix: 2-D float array ``(N, D)`` of pre-computed
                feature embeddings, same order as *image_paths*.
            n_select: Maximum number of candidates to select.
            min_drift_score: Minimum drift_score to be considered as a
                candidate.  Images below this are ignored.
            selection_reason: Free-text label stored in the audit trail.

        Returns:
            List of persisted ``LabelingQueueItem`` instances.

        Raises:
            ValueError: Shapes are inconsistent or ``n_select < 1``.
        """
        n = len(image_paths)
        if drift_scores.shape != (n,):
            raise ValueError(f"drift_scores shape {drift_scores.shape} != ({n},)")
        if feature_matrix.ndim != 2 or feature_matrix.shape[0] != n:
            raise ValueError(
                f"feature_matrix must be (N={n}, D) 2-D array, got {feature_matrix.shape}"
            )
        if n_select < 1:
            raise ValueError("n_select must be >= 1")

        # Step 1: filter by drift threshold
        mask = drift_scores >= min_drift_score
        candidate_indices = np.where(mask)[0]

        if len(candidate_indices) == 0:
            return []

        candidate_features = feature_matrix[candidate_indices]
        candidate_scores = drift_scores[candidate_indices]

        # Step 2: greedy coreset (CPU)
        k = min(n_select, len(candidate_indices))
        selected_local = _greedy_coreset(candidate_features, k)
        selected_global = candidate_indices[selected_local]

        # Step 3: build and persist queue items
        items: list[LabelingQueueItem] = []
        now = datetime.now(UTC)

        for local_idx, global_idx in zip(selected_local, selected_global, strict=True):
            score = float(candidate_scores[local_idx])
            bucket = _score_to_bucket(score)
            sha256 = _path_to_pseudo_sha256(image_paths[global_idx])

            item = LabelingQueueItem(
                item_id=f"lq_{uuid4().hex[:12]}",
                image_sha256=sha256,
                source_path=image_paths[global_idx],
                priority_bucket=bucket,
                selection_reason=selection_reason,
                drift_score=score,
                created_at=now,
            )
            self._persist_item(item)
            items.append(item)

        return items

    def list_pending(self) -> list[dict[str, object]]:
        """Return all unreviewed queue items (verdict IS NULL).

        Returns:
            List of raw row dictionaries ordered by priority_bucket ASC,
            created_at ASC.
        """
        with self._db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM labeling_queue WHERE verdict IS NULL"
                " ORDER BY priority_bucket ASC, created_at ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def record_verdict(
        self,
        item_id: str,
        verdict: str,
        *,
        reviewer: str | None = None,
    ) -> None:
        """Update a queue item's verdict after expert review.

        Args:
            item_id: Identifier of the item being reviewed.
            verdict: Verdict string (``FeedbackVerdict.value``).
            reviewer: Optional reviewer identifier.
        """
        with self._db.connect() as conn:
            conn.execute(
                """UPDATE labeling_queue
                   SET verdict = ?, assigned_reviewer = ?, reviewed_at = ?
                   WHERE item_id = ?""",
                (verdict, reviewer, datetime.now(UTC).isoformat(), item_id),
            )

    def _persist_item(self, item: LabelingQueueItem) -> None:
        with self._db.connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO labeling_queue (
                    item_id, image_sha256, source_path, priority_bucket,
                    selection_reason, drift_score, uncertainty_score,
                    assigned_reviewer, verdict, created_at, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item.item_id,
                    item.image_sha256,
                    str(item.source_path),
                    item.priority_bucket,
                    item.selection_reason,
                    item.drift_score,
                    item.uncertainty_score,
                    item.assigned_reviewer,
                    item.verdict.value if item.verdict else None,
                    item.created_at.isoformat(),
                    item.reviewed_at.isoformat() if item.reviewed_at else None,
                ),
            )


# ---------------------------------------------------------------------------
# CPU-bound helpers (no GPU)
# ---------------------------------------------------------------------------


def _greedy_coreset(features: np.ndarray, k: int) -> np.ndarray:
    """Greedy farthest-point sampling for diverse candidate selection.

    Complexity: O(k * N) -- acceptable for labeling batches (<10 k images).
    All arithmetic is float32 on CPU; never touches GPU memory.

    Args:
        features: ``(N, D)`` float array of embeddings.
        k: Number of points to select (1 <= k <= N).

    Returns:
        ``(k,)`` integer index array into *features*.
    """
    n = len(features)
    if k >= n:
        return np.arange(n, dtype=np.intp)

    # L2-normalise to make cosine distance equivalent to Euclidean
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = features.astype(np.float32) / norms.astype(np.float32)

    selected = [int(np.argmax(normed[:, 0]))]  # deterministic seed
    min_dist: np.ndarray = np.full(n, np.inf, dtype=np.float32)

    for _ in range(k - 1):
        last = normed[selected[-1]]
        # Euclidean distance to the last selected point
        diffs = normed - last
        dist = np.einsum("ij,ij->i", diffs, diffs)
        np.minimum(min_dist, dist, out=min_dist)
        selected.append(int(np.argmax(min_dist)))

    return np.array(selected, dtype=np.intp)


def _score_to_bucket(drift_score: float) -> str:
    """Map a drift score to a priority bucket string.

    Args:
        drift_score: Value in [0.0, 1.0] (higher = more urgent).

    Returns:
        ``'P0'``, ``'P1'``, ``'P2'``, or ``'P3'``.
    """
    if drift_score >= _BUCKET_P0_THRESHOLD:
        return "P0"
    if drift_score >= _BUCKET_P1_THRESHOLD:
        return "P1"
    if drift_score >= _BUCKET_P2_THRESHOLD:
        return "P2"
    return "P3"


def _path_to_pseudo_sha256(path: Path) -> str:
    """Generate a deterministic pseudo-SHA256 from a path string.

    This is NOT a real content hash -- it is a placeholder for when
    pre-computed hashes are not available at queue time.

    Args:
        path: Source image path.

    Returns:
        64-character hex string based on the path string.
    """
    import hashlib

    return hashlib.sha256(str(path).encode()).hexdigest()
