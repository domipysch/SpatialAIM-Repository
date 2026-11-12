"""
Scores per K: each criterion's curves combined by their harmonic mean, and the
score on all criteria as the harmonic mean of the selection criteria's scores.
"""

from __future__ import annotations
from collections.abc import Iterable
import numpy as np
import pandas as pd
from ..criterion import Criterion
from ..registry import REGISTRY, SELECTION_CRITERIA
from .criteria import NULL_SUFFIX


def score_table(
    df: pd.DataFrame, criteria: Iterable[Criterion] | None = None
) -> pd.DataFrame:
    """One row per K of ``df`` (``k_comparison.csv``): ``k``, and per criterion
    (default: all registered) its combined score ``<key>`` and its shuffle null
    ``<key>_null``."""
    out = pd.DataFrame({"k": df["k"].astype(int).to_numpy()})
    n = len(df)
    for criterion in criteria or REGISTRY.values():
        out[criterion.key] = harmonic_mean(*curves(df, criterion))
        out[f"{criterion.key}{NULL_SUFFIX}"] = harmonic_mean(
            *(column_values(df, f"{c}{NULL_SUFFIX}", n) for c in criterion.columns)
        )
    return out


def combined_score(
    table: pd.DataFrame, keys: Iterable[str] | None = None
) -> np.ndarray:
    """Score per K on several criteria: the harmonic mean of the criterion
    scores ``keys`` (default: ``SELECTION_CRITERIA``). NaN where any of them is
    missing or not positive."""
    columns = tuple(keys or SELECTION_CRITERIA)
    n = len(table)
    points = np.column_stack([column_values(table, column, n) for column in columns])
    good = np.all(np.isfinite(points) & (points > 0), axis=1)
    safe = np.where(good[:, None], points, 1.0)
    return np.where(good, points.shape[1] / np.sum(1.0 / safe, axis=1), np.nan)


def curves(df: pd.DataFrame, criterion: Criterion) -> tuple[np.ndarray, ...]:
    """The criterion's curves over K, one per column, as measured."""
    n = len(df)
    return tuple(column_values(df, column, n) for column in criterion.columns)


def harmonic_mean(*values) -> np.ndarray:
    """Element-wise harmonic mean of the arrays ``values``; 0 if any is 0, NaN
    if any is negative or NaN."""
    points = np.stack([np.asarray(v, dtype=float) for v in values])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = len(values) / np.sum(1.0 / points, axis=0)
    bad = np.any(~np.isfinite(points) | (points < 0), axis=0)
    return np.where(bad, np.nan, out)


def column_values(df: pd.DataFrame, column: str, n: int) -> np.ndarray:
    """``df[column]`` as floats, or all NaN if the column is missing."""
    if column not in df.columns:
        return np.full(n, np.nan)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
