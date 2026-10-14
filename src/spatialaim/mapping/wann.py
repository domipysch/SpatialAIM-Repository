"""
WANN mapper: reliability-weighted adaptive kNN (Di Salvo et al., TMLR 2025).

Reference cells are the labelled training set (label = state at this K) and
spots the test set. Per K:

1. Each reference cell gets a reliability eta = 1/k', with k' the smallest
   neighbourhood size whose kNN vote among the other cells recovers its own
   state (k' = 11, 13, ..., 51; 1/51 if none does).
2. Each spot takes k_T = 1/eta of its nearest reference cell and votes over its
   k_T nearest cells, each weighted by its reliability.

Neighbours are ranked by cosine similarity on the shared genes. The rankings do
not depend on K, so they are computed once in ``prepare``.
"""

import logging

import numpy as np

from spatialaim.adata_schema import (
    OBS_START_CLUSTER,
    UNS_SHARED_GENES,
)
from spatialaim.utils import to_dense

from .base import SpotStateMapper
from .confidence import entropy_confidence

logger = logging.getLogger(__name__)

# Neighbourhood sizes probed for the reliability (paper defaults; odd sizes).
K_MIN = 11
K_MAX = 51
K_STEP = 2

# Tie-break in the reliability vote: the neighbour at rank j counts
# 1 + RANK_EPS * (nn_max - j), so a tie goes to the state with nearer members.
# Small enough never to overturn a real majority.
RANK_EPS = 1e-4


class WANNMapper(SpotStateMapper):
    """Reliability-weighted adaptive-kNN label transfer (soft P)."""

    eps: float = 1e-8
    name = "wann"

    def prepare(self, adata_sc, adata_st, labels_by_k) -> None:
        """Find, once for the sweep, each reference cell's and each spot's
        nearest reference cells (cosine on the shared genes)."""
        super().prepare(adata_sc, adata_st, labels_by_k)

        # The probed sizes, limited by the number of other reference cells.
        n_cells = adata_sc.n_obs
        nn_max = min(K_MAX, max(1, n_cells - 1))
        k_lo = min(K_MIN, nn_max)
        grid = list(range(k_lo, nn_max + 1, K_STEP)) or [nn_max]
        nn_max = grid[-1]
        if grid[-1] != K_MAX:
            logger.info(
                "%s: reliability grid clamped to k'=%d..%d (n_cells=%d)",
                type(self).__name__,
                grid[0],
                grid[-1],
                n_cells,
            )
        self._grid = grid
        self._nn_max = nn_max

        shared = self.adata_sc.uns[UNS_SHARED_GENES]
        Zs = self._spatial_data_matrix()  # spots x genes
        Zc = np.asarray(
            to_dense(self.adata_sc[:, shared]), dtype=np.float32
        )  # cells x genes
        Zs = Zs / (np.linalg.norm(Zs, axis=1, keepdims=True) + self.eps)
        Zc = Zc / (np.linalg.norm(Zc, axis=1, keepdims=True) + self.eps)

        # Nearest reference cells, nearest first: per cell (itself excluded)
        # and per spot.
        self._ref_nbr_idx = self._cosine_topk(Zc, Zc, nn_max, exclude_self=True)
        self._spot_nbr_idx = self._cosine_topk(Zs, Zc, nn_max, exclude_self=False)
        self._spot_nn = self._spot_nbr_idx[:, 0]
        self._start_cluster = (
            self.adata_sc.obs[OBS_START_CLUSTER].astype(int).to_numpy()
        )

    def map(self, start_cluster_to_state, k) -> tuple[np.ndarray, np.ndarray]:
        """Soft P (spots x states, rows sum to 1) from the reliability-weighted
        vote, and its :func:`entropy_confidence`."""
        labels = np.asarray(start_cluster_to_state, dtype=np.int64)
        y = labels[self._start_cluster]  # state of each reference cell
        eta = self._reliability(y, k)

        # Each spot's neighbourhood size: 1/eta of its nearest cell.
        k_t = np.clip(
            np.round(1.0 / eta[self._spot_nn]).astype(np.int64), 1, self._nn_max
        )

        # Vote over the first k_t neighbours, weighted by their reliability.
        n_spots = self._spot_nbr_idx.shape[0]
        nbr_states = y[self._spot_nbr_idx]
        nbr_eta = eta[self._spot_nbr_idx]
        keep = np.arange(self._nn_max)[None, :] < k_t[:, None]
        weights = (nbr_eta * keep).astype(np.float32)

        P = np.zeros((n_spots, k), dtype=np.float32)
        rows = np.broadcast_to(np.arange(n_spots)[:, None], nbr_states.shape)
        np.add.at(P, (rows, nbr_states), weights)
        P = P / np.clip(P.sum(axis=1, keepdims=True), self.eps, None)
        return P, entropy_confidence(P)

    def _reliability(self, y: np.ndarray, k: int) -> np.ndarray:
        """Reliability eta = 1/k' of every reference cell, given the cells'
        states ``y``: the first k' in the grid at which the (rank-tie-broken)
        vote of its k' nearest cells gives its own state; 1/grid[-1] if none."""
        n_cells = y.shape[0]
        nbr_states = y[self._ref_nbr_idx]
        counts = np.zeros((n_cells, k), dtype=np.float32)
        eta = np.full(n_cells, 1.0 / self._grid[-1], dtype=np.float32)
        resolved = np.zeros(n_cells, dtype=bool)
        rows = np.arange(n_cells)

        prev = 0
        for kp in self._grid:
            # Add neighbours prev..kp-1 to the running vote.
            for j in range(prev, kp):
                w = 1.0 + RANK_EPS * (self._nn_max - j)
                np.add.at(counts, (rows, nbr_states[:, j]), np.float32(w))
            prev = kp
            pred = np.argmax(counts, axis=1)
            newly = (pred == y) & ~resolved
            eta[newly] = np.float32(1.0 / kp)
            resolved |= newly
            if resolved.all():
                break
        return eta

    @staticmethod
    def _cosine_topk(
        Q: np.ndarray,
        R: np.ndarray,
        k: int,
        exclude_self: bool,
        batch_rows: int = 4096,
    ) -> np.ndarray:
        """Indices of the ``k`` rows of ``R`` most similar to each row of ``Q``,
        nearest first. Both must be L2-normalised. Computed in batches of rows
        to limit memory; ``exclude_self`` (for ``Q is R``) skips each row itself.
        """
        out = []
        for start in range(0, Q.shape[0], batch_rows):
            q = Q[start : start + batch_rows]
            sim = q @ R.T
            if exclude_self:
                rows = np.arange(q.shape[0])
                cols = np.arange(start, start + q.shape[0])
                sim[rows, cols] = -np.inf
            out.append(_topk_indices(sim, k))
        return np.concatenate(out, axis=0)


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Column indices of the ``k`` largest values per row, largest first."""
    k = min(k, scores.shape[1])
    part = np.argpartition(scores, -k, axis=1)[:, -k:]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)
