"""
Spatial-organisation metrics over state labels on the spatial graph.
"""

from __future__ import annotations
import numpy as np


def local_spatial_purity(labels: np.ndarray, connectivities) -> float:
    """Share of the graph's edges that join two spots of the same state."""
    A = connectivities.tocoo()
    if A.nnz == 0:
        return float("nan")
    return float((labels[A.row] == labels[A.col]).mean())


def self_nhood_enrichment(
    labels: np.ndarray, connectivities, n_perm: int, rng: np.random.Generator
) -> np.ndarray | None:
    """Per state, how much more often its spots border each other than under
    ``n_perm`` label shuffles, as a z-score (the diagonal of squidpy's
    neighbourhood enrichment).

    Returns an array indexed by state (NaN for a state without spots or with a
    constant null), or ``None`` with fewer than 2 states.
    """
    if len(np.unique(labels)) < 2:
        return None

    A = connectivities.tocoo()
    row, col = A.row, A.col
    n_states = int(labels.max()) + 1

    def within_state_edges(states: np.ndarray) -> np.ndarray:
        """Per state, the number of edges with both ends in that state."""
        same = states[row] == states[col]
        return np.bincount(states[row][same], minlength=n_states).astype(np.float64)

    observed = within_state_edges(labels)
    null = np.stack(
        [within_state_edges(rng.permutation(labels)) for _ in range(n_perm)]
    )
    mean = null.mean(axis=0)
    std = null.std(axis=0)

    z = np.full(n_states, np.nan)
    ok = std > 0
    z[ok] = (observed[ok] - mean[ok]) / std[ok]
    return z
