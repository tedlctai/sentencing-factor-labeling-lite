"""Judgment-level splits.

Splits are always made over judgments, never over derived training entries, so
the same judgment cannot be on both sides.
"""
from __future__ import annotations

import random
from typing import List, Sequence, Tuple


def kfold(
    ids: Sequence[str],
    k: int = 2,
    seed: int = 0,
) -> List[Tuple[List[str], List[str]]]:
    """Return ``k`` (train_ids, test_ids) pairs; every id is in exactly one test fold."""
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    folds: List[List[str]] = [[] for _ in range(k)]
    for n, i in enumerate(order):
        folds[n % k].append(i)
    result = []
    for f in range(k):
        test = sorted(folds[f])
        train = sorted(i for g in range(k) if g != f for i in folds[g])
        result.append((train, test))
    return result


def assert_disjoint(train_ids: Sequence[str], test_ids: Sequence[str]) -> None:
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise AssertionError(f"{len(overlap)} judgments appear on both sides, e.g. {sorted(overlap)[:3]}")
