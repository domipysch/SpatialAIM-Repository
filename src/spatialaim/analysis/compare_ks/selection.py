"""
Choosing K from the score table: the best K, and the Pareto-optimal Ks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .criteria import CRITERIA
from .scoring import column_values, overall


def best_ks(table: pd.DataFrame) -> dict[str, tuple[int, float] | None]:
    """The best ``(k, score)`` overall and per criterion, as
    ``{"overall": ..., "<criterion key>": ...}``; ``None`` for a criterion
    without any finite score."""
    out: dict[str, tuple[int, float] | None] = {
        "overall": _best_k(table, overall(table))
    }
    for criterion in CRITERIA:
        values = pd.to_numeric(table[criterion.key], errors="coerce").to_numpy(
            dtype=float
        )
        out[criterion.key] = _best_k(table, values)
    return out


def pareto_mask(table: pd.DataFrame, keys: tuple[str, ...] | None = None) -> np.ndarray:
    """Which Ks are Pareto-optimal on the criterion scores ``keys`` (default:
    all): no other K is at least as good on all and better on one. A NaN score
    counts as the worst value."""
    columns = keys or tuple(c.key for c in CRITERIA)
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
