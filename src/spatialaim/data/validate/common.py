"""
Checks shared by the validators. Each returns its findings as lists of
messages: errors (SpatialAIM will fail) and warnings.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse

__all__ = [
    "colored",
    "status_of",
    "is_intable",
    "to_int_safe",
    "load_X",
    "raw_count_sanity",
    "check_gene_names_uppercase",
    "check_gene_names_unique",
    "check_count_matrix",
    "check_index_counts",
]


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_gene_names_uppercase(adata: ad.AnnData, label: str) -> list[str]:
    """Error if any gene name is not uppercase."""
    non_upper = [g for g in adata.var_names if g != g.upper()]
    if non_upper:
        return [
            f"{label}: {len(non_upper)} gene names are not uppercase "
            f"(e.g. {non_upper[:3]}) - uppercase them, e.g. "
            f"`adata.var_names = adata.var_names.str.upper()`"
        ]
    return []


def check_gene_names_unique(adata: ad.AnnData, label: str) -> list[str]:
    """Error if any gene name occurs twice."""
    counts = pd.Series(adata.var_names.tolist()).value_counts()
    dups = counts[counts > 1]
    if not dups.empty:
        examples = dups.index.tolist()[:3]
        return [
            f"{label}: {len(dups)} duplicate gene name(s) "
            f"(e.g. {examples}) - rename or remove duplicates"
        ]
    return []


def check_count_matrix(X: np.ndarray, label: str) -> tuple[list[str], list[str]]:
    """``(errors, warns)``: NaN or negative values are errors; signs that ``X``
    is not raw counts are warnings (see :func:`raw_count_sanity`)."""
    errors: list[str] = []
    if np.isnan(X).any():
        errors.append(f"{label}: NaN values in X")
    elif (X < 0).any():
        errors.append(f"{label}: negative values in X")
    return errors, raw_count_sanity(X, label)


def raw_count_sanity(X: np.ndarray, label: str) -> list[str]:
    """Warnings if ``X`` looks normalised, log-transformed, non-integer, or has
    all-zero rows."""
    warns: list[str] = []
    row_sums = X.sum(axis=1)
    if row_sums.mean() > 0:
        cv = row_sums.std() / row_sums.mean()
        if cv < 0.01:
            warns.append(
                f"{label}: X appears library-size normalized (row-sum CV={cv:.4f})"
            )
    if X.max() < 20:
        warns.append(f"{label}: X max={X.max():.3f} - may be log1p-transformed")
    sample = X.flatten()[:1000]
    if (np.abs(sample - np.round(sample)) > 1e-4).any():
        warns.append(f"{label}: X contains non-integer values - expected raw counts")
    n_zero = (row_sums == 0).sum()
    if n_zero > 0:
        warns.append(f"{label}: {n_zero} observation(s) with all-zero counts")
    return warns


def check_index_counts(
    index_row: pd.Series | None,
    label: str,
    expectations: list[tuple[str, str, int]],
) -> list[str]:
    """Warnings where an ``index.csv`` count differs from the data.

    ``expectations``: ``(index.csv column, what, actual value)``. Empty or
    non-integer index cells are skipped.
    """
    if index_row is None:
        return []
    warns: list[str] = []
    for idx_col, what, actual in expectations:
        val = index_row.get(idx_col, "")
        if is_intable(val) and to_int_safe(val) != actual:
            warns.append(
                f"{label}: index.csv {idx_col}={val} != actual {what}={actual}"
            )
    return warns


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_X(adata: ad.AnnData) -> np.ndarray:
    """``adata.X`` as a dense array."""
    X = adata.X
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


def is_intable(x) -> bool:
    """Whether ``x`` is an integer or a string holding one."""
    try:
        if isinstance(x, (int, np.integer)):
            return True
        if isinstance(x, str):
            int(x)
            return True
    except Exception:
        return False
    return False


def to_int_safe(x) -> int:
    """``int(x)``; raises ``ValueError`` unless :func:`is_intable`."""
    if is_intable(x):
        return int(x)
    raise ValueError("not intable")


def status_of(errors: list[str], warns: list[str]) -> str:
    """``"ERROR"``, ``"WARN"`` or ``"OK"``."""
    if errors:
        return "ERROR"
    return "WARN" if warns else "OK"


def colored(text: str, status: str) -> str:
    """``text`` in the terminal colour of ``status``."""
    codes = {"OK": "\033[92m", "WARN": "\033[93m", "ERROR": "\033[91m"}
    reset = "\033[0m"
    code = codes.get(status, "")
    return f"{code}{text}{reset}"
