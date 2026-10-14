"""
Small array helpers shared across the package.
"""

from __future__ import annotations

import numpy as np

__all__ = ["to_dense"]


def to_dense(x, dtype=np.float32) -> np.ndarray:
    """``x`` (an AnnData, a sparse or a dense matrix) as a dense array."""
    if hasattr(x, "X"):
        x = x.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=dtype)
