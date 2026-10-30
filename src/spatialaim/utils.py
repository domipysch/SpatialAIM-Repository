"""
Small array helpers shared across the package.
"""

from __future__ import annotations

import numpy as np

__all__ = ["box_stats", "to_dense"]


def to_dense(x, dtype=np.float32) -> np.ndarray:
    """``x`` (an AnnData, a sparse or a dense matrix) as a dense array."""
    if hasattr(x, "X"):
        x = x.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=dtype)


def box_stats(values) -> dict[str, float | int | None]:
    """The numbers a box plot of ``values`` is drawn from, computed exactly as
    Plotly does for raw data (``quartilemethod="linear"``): quartiles by Hazen
    interpolation, and whiskers (``lowerfence`` / ``upperfence``) at the most
    extreme values within 1.5 IQR of the box. Non-finite values are ignored;
    without any finite value, every statistic is ``None``."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    v = v[np.isfinite(v)]
    keys = ("min", "lowerfence", "q1", "median", "q3", "upperfence", "max", "mean")
    if v.size == 0:
        return {"n": 0, **{key: None for key in keys}}
    q1, median, q3 = np.quantile(v, [0.25, 0.5, 0.75], method="hazen")
    iqr = q3 - q1
    lowerfence = min(q1, v[v >= q1 - 1.5 * iqr][0])
    upperfence = max(q3, v[v <= q3 + 1.5 * iqr][-1])
    stats = (v[0], lowerfence, q1, median, q3, upperfence, v[-1], v.mean())
    return {"n": int(v.size), **{key: float(x) for key, x in zip(keys, stats)}}
