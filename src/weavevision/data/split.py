"""Deterministic group-aware splitting before image tiling."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def deterministic_group_split(
    items: Sequence[T],
    groups: Sequence[str],
    validation_fraction: float,
    seed: int,
) -> tuple[list[T], list[T]]:
    """Split items by immutable parent group with deterministic assignment.

    Args:
        items: Items to split.
        groups: Parent identity for each item.
        validation_fraction: Fraction of unique groups assigned to validation.
        seed: Split seed included in the stable hash.

    Returns:
        Training and validation item lists.

    Raises:
        ValueError: If lengths or validation fraction are invalid.
    """
    if len(items) != len(groups):
        raise ValueError("items and groups must have identical lengths")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    grouped: dict[str, list[T]] = defaultdict(list)
    for item, group in zip(items, groups, strict=True):
        grouped[group].append(item)
    ranked = sorted(
        grouped,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
    )
    validation_count = max(1, round(len(ranked) * validation_fraction))
    if len(ranked) > 1:
        validation_count = min(validation_count, len(ranked) - 1)
    validation_groups = set(ranked[:validation_count])
    train = [
        item
        for group, members in grouped.items()
        if group not in validation_groups
        for item in members
    ]
    validation = [
        item for group, members in grouped.items() if group in validation_groups for item in members
    ]
    return train, validation
