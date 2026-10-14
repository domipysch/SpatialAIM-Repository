"""
Cosine similarity.
"""

from __future__ import annotations
import numpy as np


def cosine_similarity(A: np.ndarray, B: np.ndarray, axis: int) -> np.ndarray:
    """Cosine similarity of ``A`` and ``B`` per column (``axis=0``) or per row
    (``axis=1``); 0 where a vector is all zero."""
    dot = np.sum(A * B, axis=axis)
    denom = np.linalg.norm(A, axis=axis) * np.linalg.norm(B, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        cs = np.where(denom == 0.0, 0.0, dot / denom)
    return np.clip(cs, -1.0, 1.0)
