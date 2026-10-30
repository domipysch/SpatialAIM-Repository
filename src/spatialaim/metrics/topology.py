"""
Spatial-organisation metrics over state labels on the spatial graph: overall and per state.

Both are chance-corrected like Cohen's kappa: (observed − chance) / (1 − chance),
1 = no edge leaves its state, 0 = as mixed as randomly placed labels.

Global assortativity:
Newman’s assortativity coefficient for discrete labels (Newman 2003, Phys. Rev. E 67, 026126, eq. 2).

Per-state assortativity: Adapted per state.

"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Assortativity:
    """``value``: the global assortativity r; ``per_state``: κ_s, indexed by
    state (NaN for a state without spots)."""

    value: float
    per_state: np.ndarray

    @property
    def mean_state(self) -> float:
        """Mean κ_s over the states it is defined for (NaN if none)."""
        finite = self.per_state[np.isfinite(self.per_state)]
        return float(finite.mean()) if finite.size else float("nan")


def assortativity(labels: np.ndarray, connectivities) -> Assortativity | None:
    """Assortativity of the integer state ``labels`` on the graph's edges
    (unweighted; each stored edge counts once, so a directed kNN graph works).

    With e_st the share of edges from state s to state t, a_s = Σ_t e_st and
    b_s = Σ_t e_ts:

    - global: r = (Σ e_ss − Σ a_s b_s) / (1 − Σ a_s b_s)
    - per state: κ_s = (p_s − b_s) / (1 − b_s), p_s = e_ss / a_s

    r is the mean of the κ_s weighted by a_s (1 − b_s). Returns ``None`` for a
    graph without edges; values are NaN where undefined (a single state).
    """
    A = connectivities.tocoo()
    if A.nnz == 0:
        return None
    n_states = int(labels.max()) + 1
    src, dst = labels[A.row], labels[A.col]
    same = src == dst
    n_edges = float(A.nnz)
    a = np.bincount(src, minlength=n_states) / n_edges
    b = np.bincount(dst, minlength=n_states) / n_edges
    e_same = np.bincount(src[same], minlength=n_states) / n_edges

    observed = float(e_same.sum())
    chance = float(np.dot(a, b))
    with np.errstate(invalid="ignore", divide="ignore"):
        value = (observed - chance) / (1.0 - chance) if chance < 1.0 else np.nan
        purity = np.where(a > 0, e_same / a, np.nan)
        kappa = np.where((a > 0) & (b < 1.0), (purity - b) / (1.0 - b), np.nan)

    return Assortativity(value=float(value), per_state=kappa)
