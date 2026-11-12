"""
Collect the metrics every K's analysis wrote into one table, one row per K.
Nothing is recomputed; a metric missing for a K is NaN.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..registry import REGISTRY


def collect_k_metrics(output_folder: Path, ks: list[int]) -> pd.DataFrame:
    """The metrics of the swept ``ks`` under the run root ``output_folder``,
    one row per K, sorted by K."""
    rows = [_k_metrics(output_folder / f"k_{k:03d}" / "analysis", k) for k in ks]
    return pd.DataFrame(rows).sort_values("k", ignore_index=True)


def _k_metrics(data_dir: Path, k: int) -> dict[str, float]:
    """One K's row: ``k`` and the columns of every registered criterion."""
    row: dict[str, float] = {"k": k}
    for criterion in REGISTRY.values():
        row.update(criterion.collect(data_dir))
    return row
