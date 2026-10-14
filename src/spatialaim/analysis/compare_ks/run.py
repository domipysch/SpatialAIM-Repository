"""
``compare_k_runs``: collect, score and select, then write ``k_comparison.csv``
and ``k_selection.json`` at the run root.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import pandas as pd
from .collect import collect_k_metrics
from .criteria import CRITERIA
from .scoring import overall, score_table
from .selection import best_ks

logger = logging.getLogger(__name__)


K_COMPARISON_CSV = "k_comparison.csv"
K_SELECTION_JSON = "k_selection.json"


def compare_k_runs(output_folder: Path, ks: list[int]) -> None:
    """Write ``k_comparison.csv`` and ``k_selection.json`` for the swept ``ks``
    at the run root ``output_folder``."""
    output_folder = Path(output_folder)
    metrics = collect_k_metrics(output_folder, ks)
    scores = score_table(metrics)

    csv_path = output_folder / K_COMPARISON_CSV
    _with_scores(metrics, scores).to_csv(csv_path, index=False)
    logger.info("K-sweep comparison written to %s", csv_path)

    _write_selection(output_folder / K_SELECTION_JSON, scores)


def _with_scores(metrics: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """``metrics`` plus one combined-score column per criterion and ``overall``.

    The combined scores' own shuffle nulls are left out: only reconstruction has
    one, and the GUI recomputes it from the measured null columns.
    """
    out = metrics.copy()
    for criterion in CRITERIA:
        out[criterion.key] = scores[criterion.key].to_numpy()
    out["overall"] = overall(scores)
    return out


def _write_selection(json_path: Path, scores: pd.DataFrame) -> None:
    """Write the best K and its score per criterion and overall (``None`` when
    a criterion has no finite score) - the same proposal the GUI computes."""
    best = best_ks(scores)
    proposal = {
        "best": {
            name: None if hit is None else {"k": hit[0], "score": hit[1]}
            for name, hit in best.items()
        }
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(proposal, fh, indent=2)
    chosen = ", ".join(
        f"{name}={hit[0]}" for name, hit in best.items() if hit is not None
    )
    logger.info("K selection written to %s (%s)", json_path, chosen)
