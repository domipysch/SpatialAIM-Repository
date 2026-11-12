"""
Choosing K from the score table: the best K, and the Pareto-optimal Ks.
"""

from __future__ import annotations
from collections.abc import Iterable
from itertools import combinations
import numpy as np
import pandas as pd
from ..registry import SELECTION_CRITERIA
from .scoring import column_values, combined_score


def best_ks(
    table: pd.DataFrame, keys: Iterable[str] | None = None
) -> dict[str, tuple[int, float] | None]:
    """The best ``(k, score)`` on all criteria ``keys`` (default
    ``SELECTION_CRITERIA``) and per criterion, as ``{"all": ..., "<key>": ...}``;
    ``None`` where no K has a finite score."""
    keys = tuple(keys or SELECTION_CRITERIA)
    out = {"all": _best_k(table, combined_score(table, keys))}
    for key in keys:
        out[key] = _best_k(table, column_values(table, key, len(table)))
    return out


def best_combination_ks(
    table: pd.DataFrame, keys: Iterable[str] | None = None
) -> dict[str, tuple[int, float] | None]:
    """The best ``(k, score)`` for every combination of 2 to n - 1 of the n
    criteria ``keys`` (default ``SELECTION_CRITERIA``), keyed
    ``"<key a>+<key b>+..."``; a combination's score is the harmonic mean of its
    criterion scores."""
    keys = tuple(keys or SELECTION_CRITERIA)
    return {
        "+".join(combo): _best_k(table, combined_score(table, combo))
        for size in range(2, len(keys))
        for combo in combinations(keys, size)
    }


def pareto_mask(table: pd.DataFrame, keys: Iterable[str] | None = None) -> np.ndarray:
    """Which Ks are Pareto-optimal on the criterion scores ``keys`` (default:
    ``SELECTION_CRITERIA``): no other K is at least as good on all and better on
    one. A NaN score counts as the worst value."""
    columns = tuple(keys or SELECTION_CRITERIA)
    n = len(table)
    if n == 0:
        return np.zeros(0, dtype=bool)
    points = np.column_stack([column_values(table, column, n) for column in columns])
    points = np.where(np.isfinite(points), points, -np.inf)

    keep = np.ones(n, dtype=bool)
    for i in range(n):
        others = np.delete(points, i, axis=0)
        if others.size == 0:
            continue
        dominated = np.all(others >= points[i], axis=1) & np.any(
            others > points[i], axis=1
        )
        keep[i] = not dominated.any()
    return keep


def _best_k(table: pd.DataFrame, values: np.ndarray) -> tuple[int, float] | None:
    """``(k, value)`` of the highest finite value, or ``None``. Ties go to the
    smallest K (the table is sorted by K and argmax takes the first)."""
    ks = table["k"].to_numpy(dtype=int)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    index = int(np.nanargmax(np.where(finite, values, -np.inf)))
    return int(ks[index]), float(values[index])
