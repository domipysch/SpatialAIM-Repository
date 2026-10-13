"""Spatial-organisation metrics over spot state labels: a label-shuffling
permutation test, local spatial purity, and the per-node same-label neighbour
count (used on the ST expression graph for transcriptional coherence)."""

from __future__ import annotations

import logging
from typing import Callable
import numpy as np

logger = logging.getLogger(__name__)


def permutation_test(
    observed: float,
    labels: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_perm: int,
    rng: np.random.Generator,
) -> dict:
    """Shuffle ``labels`` ``n_perm`` times, recomputing ``metric_fn`` each time.

    Returns a dict with ``observed``, ``p_value`` (fraction of null >= observed),
    ``z_score``, ``null_mean`` and ``null_std``; non-finite cases yield NaNs.
    """
    if not np.isfinite(observed):
        return {
            "observed": float(observed),
            "p_value": float("nan"),
            "z_score": float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
        }
    perm = np.array([metric_fn(rng.permutation(labels)) for _ in range(n_perm)])
    valid = perm[np.isfinite(perm)]
    if len(valid) == 0:
        return {
            "observed": float(observed),
            "p_value": float("nan"),
            "z_score": float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
        }
    return {
        "observed": float(observed),
        "p_value": float((valid >= observed).mean()),
        "z_score": float((observed - valid.mean()) / (valid.std() + 1e-12)),
        "null_mean": float(valid.mean()),
        "null_std": float(valid.std()),
    }


def local_spatial_purity(labels: np.ndarray, connectivities) -> float:
    """Fraction of neighbour pairs in the spatial graph that share a state label.

    ``connectivities`` is a (n_spots x n_spots) sparse adjacency (e.g. squidpy's
    ``spatial_connectivities``); each stored edge is counted once, ignoring its
    weight. Returns NaN for an empty graph.
    """
    A = connectivities.tocoo()
    if A.nnz == 0:
        return float("nan")
    return float((labels[A.row] == labels[A.col]).mean())


def same_label_neighbours(
    labels: np.ndarray, connectivities
) -> tuple[np.ndarray, np.ndarray]:
    """Per node: how many of its graph neighbours share its label, and how many
    neighbours it has.

    A node's neighbours are the stored entries of its row in ``connectivities``
    (any graph: the directed spatial KNN, the symmetric expression KNN), each
    counted once whatever its weight, the node itself excluded. Returns
    ``(same, degree)``, two int arrays of length n_nodes.
    """
    A = connectivities.tocoo()
    off = A.row != A.col
    row, col = A.row[off], A.col[off]
    n = A.shape[0]
    same = np.bincount(
        row, weights=(labels[row] == labels[col]).astype(np.float64), minlength=n
    )
    degree = np.bincount(row, minlength=n)
    return same.astype(np.int64), degree.astype(np.int64)
