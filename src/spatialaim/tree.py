"""
The agglomeration tree (hierarchical clustering) over the start clusters, and its cut at a given K.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from spatialaim.config import LINKAGE_METHODS


def build_agglomeration_tree(
    centroids: np.ndarray,
    method: str = LINKAGE_METHODS[0],
    eps: float = 1e-8,
) -> np.ndarray:
    """Linkage matrix over the start-cluster centroids (one row per cluster),
    following the tree construction of Grabski et al. (2023, sc-SHC):

    1. scale each centroid to relative gene frequencies (row sums to 1), which
       removes sequencing depth - mean or summed counts give the same result;
    2. Euclidean distances between these profiles;
    3. agglomerate with ``method`` (one of ``LINKAGE_METHODS``):

       - ``"average"`` (UPGMA, default) on the distances as they are;
       - ``"ward"``, reproducing R's ``hclust(..., method="ward.D")`` as in
         sc-SHC. scipy's ward is R's ward.D2, so it gets ``sqrt(distances)``
         (``ward.D(d) == ward.D2(sqrt(d))``).

    Every start cluster counts equally, whatever its size.
    """
    if method not in LINKAGE_METHODS:
        raise ValueError(f"method must be one of {LINKAGE_METHODS}, got {method!r}")
    profiles = np.asarray(centroids, dtype=np.float64)
    if profiles.shape[0] < 2:
        raise ValueError(
            f"need at least 2 start clusters to build a tree, got {profiles.shape[0]}"
        )
    if not np.isfinite(profiles).all():
        raise ValueError("centroids contain non-finite values")

    profiles = profiles / (profiles.sum(axis=1, keepdims=True) + eps)
    distances = pdist(profiles, metric="euclidean")
    if method == "ward":
        distances = np.sqrt(distances)
    return linkage(distances, method=method)


def labels_at_k(linkage_z: np.ndarray, k: int, n_start_clusters: int) -> np.ndarray:
    """Cut the tree into ``k`` states: the state (0..k-1) of every start cluster.
    For ``k >= n_start_clusters`` every start cluster is its own state."""
    if k >= n_start_clusters:
        return np.arange(n_start_clusters, dtype=int)
    raw = fcluster(linkage_z, t=k, criterion="maxclust")
    _, remapped = np.unique(raw, return_inverse=True)
    return remapped.astype(int)
