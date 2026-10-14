"""
Reference aggregates per start cluster, and the per-state expression profiles
they pool into at a given K.
"""

import numpy as np
from anndata import AnnData

from spatialaim.adata_schema import (
    OBS_START_CLUSTER,
    OBSM_LOGNORM_SHARED_GENES,
    UNS_SHARED_GENES,
    UNS_START_CLUSTER_CENTROIDS_SHARED_GENES,
    UNS_START_CLUSTER_CENTROIDS_SHARED_GENES_NORM,
    UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES,
    UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES_NORM,
    UNS_START_CLUSTER_SIZES,
)


def compute_start_cluster_aggregates(adata_sc: AnnData) -> None:
    """Per start cluster, on the shared genes: cell count, expression sum and
    mean (centroid), for raw counts and for lognorm values. Stored in
    ``adata_sc.uns``, one row per start cluster.

    An all-zero raw centroid is set to a tiny constant, so the tree can still
    compute a distance to it.
    """
    raw = _dense_float32(adata_sc[:, adata_sc.uns[UNS_SHARED_GENES]].X)
    norm = _dense_float32(adata_sc.obsm[OBSM_LOGNORM_SHARED_GENES])
    labels = adata_sc.obs[OBS_START_CLUSTER].astype(int).to_numpy()
    n_clusters = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=n_clusters).astype(np.float32)

    expr_sums = _sum_by_cluster(raw, labels, n_clusters)
    centroids = expr_sums / (sizes[:, None] + 1e-8)
    zero_rows = np.where(~centroids.any(axis=1))[0]
    if zero_rows.size:
        centroids[zero_rows] = 1e-6

    expr_sums_norm = _sum_by_cluster(norm, labels, n_clusters)
    centroids_norm = expr_sums_norm / (sizes[:, None] + 1e-8)

    adata_sc.uns[UNS_START_CLUSTER_SIZES] = sizes
    adata_sc.uns[UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES] = expr_sums
    adata_sc.uns[UNS_START_CLUSTER_CENTROIDS_SHARED_GENES] = centroids
    adata_sc.uns[UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES_NORM] = expr_sums_norm
    adata_sc.uns[UNS_START_CLUSTER_CENTROIDS_SHARED_GENES_NORM] = centroids_norm


def assemble_state_profiles_shared_genes(
    labels_k: np.ndarray,
    k: int,
    adata_sc: AnnData,
    eps: float = 1e-8,
) -> np.ndarray:
    """Mean raw expression per state (k x shared genes), pooled from the start
    clusters: a state's summed counts divided by its number of cells.

    ``labels_k`` maps each start cluster to its state at this K. Needs
    :func:`compute_start_cluster_aggregates` to have run.
    """
    expr_sums = adata_sc.uns[UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES]
    sizes = adata_sc.uns[UNS_START_CLUSTER_SIZES]

    state_sums = np.zeros((k, expr_sums.shape[1]), dtype=np.float32)
    np.add.at(state_sums, labels_k, expr_sums)
    state_sizes = np.zeros(k, dtype=np.float32)
    np.add.at(state_sizes, labels_k, sizes)

    profiles = state_sums / (state_sizes[:, None] + eps)
    return profiles.astype(np.float32, copy=False)


def _dense_float32(x) -> np.ndarray:
    """``x`` (dense or sparse) as a dense float32 array."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _sum_by_cluster(
    expr: np.ndarray, labels: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Sum the rows of ``expr`` per cluster label."""
    sums = np.zeros((n_clusters, expr.shape[1]), dtype=np.float32)
    np.add.at(sums, labels, expr)
    return sums
