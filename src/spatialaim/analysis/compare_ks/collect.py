"""
Collect the metrics every K's analysis wrote into one table, one row per K.
Nothing is recomputed; a metric missing for a K is NaN.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..reconstruction import COSSIM_METRICS_JSON, COSSIM_SUMMARY_CSV_LEGACY


def collect_k_metrics(output_folder: Path, ks: list[int]) -> pd.DataFrame:
    """The metrics of the swept ``ks`` under the run root ``output_folder``,
    one row per K, sorted by K."""
    rows = [_k_metrics(output_folder / f"k_{k:03d}" / "analysis", k) for k in ks]
    return pd.DataFrame(rows).sort_values("k", ignore_index=True)


def _k_metrics(data_dir: Path, k: int) -> dict[str, float]:
    """One K's row, read from its ``analysis`` folder.

    ``shape`` is the shape-consistency score: NaN without a segmentation mask,
    and not part of the K selection. Only the cosine similarities carry a
    ``*_null`` column: the two assortativities are chance-corrected already
    (0 = randomly placed labels by construction). Outputs written before them
    hold only the old z-scores, so their assortativity columns are NaN; those
    written before the flat layout nest them (``{"value"}`` / ``{"mean"}``).
    """
    cossim = _read_cossim(data_dir)
    topology = _load_json(data_dir / "topology_metrics.json")
    modularity = _load_json(data_dir / "modularity_metrics.json")
    shape = _load_json(data_dir / "shape_consistency_metrics.json")
    overall = topology.get("spatial_assort_global")
    if isinstance(overall, dict):
        overall = overall.get("value")
    per_state = topology.get(
        "spatial_assort_celltype_mean",
        (topology.get("spatial_assort_celltype") or {}).get("mean"),
    )
    return {
        "k": k,
        "cossim_hard_raw_spot": cossim("median_spot"),
        "cossim_hard_raw_gene": cossim("median_gene"),
        "cossim_hard_raw_spot_null": cossim("median_spot_null"),
        "cossim_hard_raw_gene_null": cossim("median_gene_null"),
        "spatial_assort_celltype": _finite(per_state),
        "spatial_assort_global": _finite(overall),
        "modularity_shared": _finite(modularity.get("modularity_shared")),
        "modularity_st_expression": _finite(modularity.get("modularity_st_expression")),
        "shape": _finite(shape.get("shape_consistency")),
    }


def _read_cossim(data_dir: Path):
    """A lookup ``median_{spot,gene}[_null] -> float`` into the ``hard-raw``
    entry of ``cossim_metrics.json`` (or of the ``cossim_summary.csv`` older
    outputs wrote); NaN for a missing file or entry."""
    hard_raw = _load_json(data_dir / COSSIM_METRICS_JSON).get("hard-raw")
    if hard_raw is None:
        legacy = data_dir / COSSIM_SUMMARY_CSV_LEGACY
        table = pd.read_csv(legacy, index_col=0) if legacy.exists() else None
        row = (
            {}
            if table is None or "hard-raw" not in table.index
            else table.loc["hard-raw"]
        )
        return lambda column: _finite(row.get(column))

    def cell(column: str) -> float:
        side = column.removeprefix("median_").removesuffix("_null")
        if column.endswith("_null"):
            return _finite((hard_raw.get("null") or {}).get(f"median_{side}"))
        return _finite((hard_raw.get(f"per_{side}") or {}).get("median"))

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

    As NaN, an undefined metric leaves a gap in the GUI's plots instead of
    blowing up an autoscaled axis.
    """
    try:
        f = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")
