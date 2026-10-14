"""
Collect the metrics every K's analysis wrote into one table, one row per K.
Nothing is recomputed; a metric missing for a K is NaN.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


def collect_k_metrics(output_folder: Path, ks: list[int]) -> pd.DataFrame:
    """The metrics of the swept ``ks`` under the run root ``output_folder``,
    one row per K, sorted by K."""
    rows = [
        _k_metrics(output_folder / f"k_{k:03d}" / "analysis" / "data", k) for k in ks
    ]
    return pd.DataFrame(rows).sort_values("k", ignore_index=True)


def _k_metrics(data_dir: Path, k: int) -> dict[str, float]:
    """One K's row, read from its ``analysis/data`` folder.

    ``shape`` is the shape-consistency score: NaN without a segmentation mask,
    and not part of the K selection. Only the cosine similarities carry a
    ``*_null`` column: the two z-scores are already measured against the label
    shuffle (null 0 by construction).
    """
    cossim = _read_cossim(data_dir / "cossim_summary.csv")
    topology = _load_json(data_dir / "topology_metrics.json")
    modularity = _load_json(data_dir / "modularity_metrics.json")
    shape = _load_json(data_dir / "shape_consistency_metrics.json")
    # Stored as null when a K has too few states for the test (e.g. K = 1).
    nhood = topology.get("nhood_enrichment") or {}
    purity = topology.get("local_purity") or {}
    return {
        "k": k,
        "cossim_hard_raw_spot": cossim("median_spot"),
        "cossim_hard_raw_gene": cossim("median_gene"),
        "cossim_hard_raw_spot_null": cossim("median_spot_null"),
        "cossim_hard_raw_gene_null": cossim("median_gene_null"),
        "nhood_mean_self_zscore": _finite(nhood.get("mean_self_zscore")),
        "local_purity_zscore": _finite(purity.get("z_score")),
        "modularity_shared": _finite(modularity.get("modularity_shared")),
        "modularity_st_expression": _finite(modularity.get("modularity_st_expression")),
        "shape": _finite(shape.get("shape_consistency")),
    }


def _read_cossim(path: Path):
    """A lookup ``column -> float`` into the ``hard-raw`` row of
    ``cossim_summary.csv``; NaN for a missing file, row or column."""
    table = pd.read_csv(path, index_col=0) if path.exists() else pd.DataFrame()

    def cell(column: str) -> float:
        if "hard-raw" not in table.index or column not in table.columns:
            return float("nan")
        return _finite(table.loc["hard-raw", column])

    return cell


def _load_json(path: Path) -> dict:
    """A JSON dict, or ``{}`` when the file is absent or unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _finite(x) -> float:
    """``x`` as a float, with inf / NaN / None / non-numbers as NaN.

    z-scores can be inf when a permutation null has zero variance (a tiny
    state); as NaN they leave a gap in the GUI's plots instead of blowing up an
    autoscaled axis.
    """
    try:
        f = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")
