"""
``compare_k_runs``: collect, score and select, then write ``k_comparison.csv``
and ``k_selection.json`` at the run root.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import pandas as pd
from ..registry import REGISTRY, SELECTION_CRITERIA
from .collect import collect_k_metrics
from .scoring import combined_score, score_table
from .selection import best_combination_ks, best_ks

logger = logging.getLogger(__name__)


K_COMPARISON_CSV = "k_comparison.csv"
K_SELECTION_JSON = "k_selection.json"


def compare_k_runs(output_folder: Path, ks: list[int]) -> None:
    """Write ``k_comparison.csv`` and ``k_selection.json`` for the swept ``ks``
    at the run root ``output_folder``."""
    output_folder = Path(output_folder)
    keys = SELECTION_CRITERIA
    metrics = collect_k_metrics(output_folder, ks)
    scores = score_table(metrics, [REGISTRY[k] for k in keys])

    csv_path = output_folder / K_COMPARISON_CSV
    _with_scores(metrics, scores, keys).to_csv(csv_path, index=False)
    logger.info("K-sweep comparison written to %s", csv_path)

    _write_selection(output_folder / K_SELECTION_JSON, scores, keys)


def _with_scores(
    metrics: pd.DataFrame, scores: pd.DataFrame, keys: tuple[str, ...]
) -> pd.DataFrame:
    """``metrics`` plus the score column of each selection criterion and
    their combined score ``all``.

    The combined scores' own shuffle nulls are left out: only reconstruction has
    one, and the GUI recomputes it from the measured null columns.
    """
    out = metrics.copy()
    for key in keys:
        out[key] = scores[key].to_numpy()
    out["all"] = combined_score(scores, keys)
    return out


def _write_selection(
    json_path: Path, scores: pd.DataFrame, keys: tuple[str, ...]
) -> None:
    """Write the best K with its score: per selection criterion (``single``),
    on all of them (``all``) and per combination of 2 to n - 1 of them
    (``combinations``); ``None`` where nothing has a finite score."""
    best = best_ks(scores, keys)
    proposal = {
        "single": {key: _as_json(best[key]) for key in keys},
        "all": _as_json(best["all"]),
        "combinations": {
            name: _as_json(hit)
            for name, hit in best_combination_ks(scores, keys).items()
        },
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(proposal, fh, indent=2)
    logger.info("K selection written to %s (all: %s)", json_path, proposal["all"])


def _as_json(hit: tuple[int, float] | None) -> dict | None:
    """``{"k", "score"}`` of a hit, or ``None`` without one."""
    return None if hit is None else {"k": hit[0], "score": hit[1]}
