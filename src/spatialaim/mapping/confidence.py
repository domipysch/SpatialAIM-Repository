"""
Per-spot mapping confidence in [0, 1]: how decisively a spot was assigned.
"""

import math

import numpy as np

# Number of best states the margin confidence compares.
N_TOP_STATES = 4


def top_margin_confidence(score: np.ndarray, n: int = N_TOP_STATES) -> np.ndarray:
    """``1 - d1 / mean(d2..dn)`` per spot, from the ``n`` lowest scores
    ``d1 <= d2 <= ... <= dn`` of ``score`` (spots x states, lower = better).

    ~1 when the best state is far ahead, ~0 when the top states are tied;
    1 with a single state.
    """
    eps = 1e-8
    n_spots, k = score.shape
    if k == 1:
        return np.ones(n_spots, dtype=np.float32)
    n = min(n, k)
    top = np.sort(score, axis=1)[:, :n]
    best = top[:, 0]
    rest = top[:, 1:].mean(axis=1)
    conf = 1.0 - best / (rest + eps)
    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def entropy_confidence(p: np.ndarray) -> np.ndarray:
    """``1 - entropy / log(K)`` per spot of the soft mapping ``p`` (rows sum
    to 1): 1 when all weight is on one state, 0 when it is spread evenly;
    1 with a single state."""
    eps = 1e-12
    n_spots, k = p.shape
    if k == 1:
        return np.ones(n_spots, dtype=np.float32)
    entropy = -(p * np.log(p + eps)).sum(axis=1)
    conf = 1.0 - entropy / math.log(k)
    return np.clip(conf, 0.0, 1.0).astype(np.float32)
