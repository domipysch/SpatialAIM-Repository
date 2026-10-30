"""
Scores per K: each criterion's two curves combined by their harmonic mean, and
the overall score as the harmonic mean of the criterion scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from .criteria import CRITERIA, NULL_SUFFIX, Criterion


def score_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per K of ``df`` (``k_comparison.csv``): ``k``, and per criterion
    its combined score ``<key>`` and its shuffle null ``<key>_null``."""
    out = pd.DataFrame({"k": df["k"].astype(int).to_numpy()})
    for criterion in CRITERIA:
        out[criterion.key] = _combined(df, criterion)
        out[f"{criterion.key}{NULL_SUFFIX}"] = _combined_null(df, criterion)
    return out


def overall(table: pd.DataFrame, keys: tuple[str, ...] | None = None) -> np.ndarray:
    """Overall score per K: the harmonic mean of the criterion scores ``keys``
    (default: all). NaN where any of them is missing or not positive."""
    columns = keys or tuple(c.key for c in CRITERIA)
    n = len(table)
    points = np.column_stack([column_values(table, column, n) for column in columns])
    good = np.all(np.isfinite(points) & (points > 0), axis=1)
    safe = np.where(good[:, None], points, 1.0)
    return np.where(good, points.shape[1] / np.sum(1.0 / safe, axis=1), np.nan)


def curves(df: pd.DataFrame, criterion: Criterion) -> tuple[np.ndarray, np.ndarray]:
    """``(a, b)``: the criterion's two curves over K, as measured."""
    n = len(df)
    a, b = (column_values(df, column, n) for column in criterion.columns)
    return a, b


def harmonic_mean(a, b) -> np.ndarray:
    """Element-wise ``2ab / (a + b)``; 0 if either is 0, NaN if either is
    negative or NaN."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    total = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(total > 0, 2.0 * a * b / np.where(total > 0, total, 1.0), 0.0)
    bad = ~np.isfinite(a) | ~np.isfinite(b) | (a < 0) | (b < 0)
    return np.where(bad, np.nan, out)


def column_values(df: pd.DataFrame, column: str, n: int) -> np.ndarray:
    """``df[column]`` as floats, or all NaN if the column is missing."""
    if column not in df.columns:
        return np.full(n, np.nan)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def _combined(df: pd.DataFrame, criterion: Criterion) -> np.ndarray:
    """The criterion's score per K: harmonic mean of its two curves."""
    return harmonic_mean(*curves(df, criterion))


def _combined_null(df: pd.DataFrame, criterion: Criterion) -> np.ndarray:
    """The same for the curves' ``<column>_null`` columns. Only reconstruction
    has them; the others are all NaN."""
    n = len(df)
    a, b = (
        column_values(df, f"{column}{NULL_SUFFIX}", n) for column in criterion.columns
    )
    return harmonic_mean(a, b)
