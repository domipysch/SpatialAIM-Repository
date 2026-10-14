"""
Start clusters of the reference (Leiden over-clustering or an annotation),
the reference's neighbour graphs and shared-gene Leiden clustering.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from spatialaim.adata_schema import (
    LAYER_LOGNORM,
    OBS_LEIDEN_SHARED_GENES,
    OBS_START_CLUSTER,
    OBSM_LOGNORM_SHARED_GENES,
    OBSM_PCA,
    OBSM_PCA_SHARED_GENES,
    OBSP_CONNECTIVITIES_SHARED_GENES,
    OBSP_DISTANCES_SHARED_GENES,
    UNS_LEIDEN_NUMBER_STATES_SHARED_GENES,
    UNS_LEIDEN_RESOLUTION_ALL_GENES,
    UNS_LEIDEN_RESOLUTION_SHARED_GENES,
    UNS_N_START_CLUSTERS,
    UNS_NEIGHBORS_SHARED_GENES,
    UNS_SHARED_GENES,
    UNS_START_CLUSTER_NAMES,
)

logger = logging.getLogger(__name__)


def build_neighbor_graph_all_genes(
    adata_sc: AnnData,
    n_pca_comps: int = 30,
) -> None:
    """PCA and kNN graph of the reference on all genes (lognorm layer), stored
    in ``adata_sc``. Built in both start-cluster modes: the all-gene UMAP and the
    modularity metric use it too."""
    n = min(n_pca_comps, adata_sc.n_obs - 1, adata_sc.n_vars - 1)
    sc.pp.pca(adata_sc, n_comps=n, layer=LAYER_LOGNORM, key_added=OBSM_PCA)
    sc.pp.neighbors(adata_sc, use_rep=OBSM_PCA)


def run_leiden_clustering_all_genes(
    adata_sc: AnnData,
    resolution: float,
    n_pca_comps: int = 30,
):
    """Over-cluster the reference with Leiden on all genes; the clusters become
    the start clusters (named ``cluster_<i>``)."""
    build_neighbor_graph_all_genes(adata_sc, n_pca_comps=n_pca_comps)
    sc.tl.leiden(
        adata_sc,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        key_added=OBS_START_CLUSTER,
    )
    number_of_clusters = len(
        np.unique(adata_sc.obs[OBS_START_CLUSTER].astype(int).values)
    )
    logger.info(
        "Leiden clustering (resolution=%.2f): %d start clusters",
        resolution,
        number_of_clusters,
    )
    adata_sc.uns[UNS_LEIDEN_RESOLUTION_ALL_GENES] = resolution
    adata_sc.uns[UNS_N_START_CLUSTERS] = number_of_clusters
    adata_sc.uns[UNS_START_CLUSTER_NAMES] = [
        f"cluster_{i}" for i in range(number_of_clusters)
    ]


def drop_unannotated(adata_sc: AnnData, cell_type_key: str) -> AnnData:
    """``adata_sc`` without the cells that have no label in
    ``obs[cell_type_key]``. Returns the same object if all cells are labelled,
    else a copy. Raises ``KeyError`` if the column does not exist."""
    if cell_type_key not in adata_sc.obs:
        raise KeyError(
            f"cell_type_key {cell_type_key!r} not in obs; "
            f"available: {list(adata_sc.obs.columns)}"
        )
    labelled = adata_sc.obs[cell_type_key].notna().to_numpy()
    if labelled.all():
        return adata_sc
    logger.warning(
        "Dropping %d of %d cell(s) with no %r label",
        int((~labelled).sum()),
        adata_sc.n_obs,
        cell_type_key,
    )
    return adata_sc[labelled].copy()


def set_start_clusters_from_annotation(
    adata_sc: AnnData, cell_type_key: str
) -> list[str]:
    """Use the cell types in ``obs[cell_type_key]`` as the start clusters (in
    sorted category order) and return their names. Run :func:`drop_unannotated`
    first."""
    types = pd.Categorical(adata_sc.obs[cell_type_key].astype(str))
    names = [str(c) for c in types.categories]

    adata_sc.obs[OBS_START_CLUSTER] = np.asarray(types.codes, dtype=np.int64)
    adata_sc.uns[UNS_N_START_CLUSTERS] = len(names)
    adata_sc.uns[UNS_START_CLUSTER_NAMES] = names
    logger.info(
        "Start clusters from annotation %r: %d type(s)", cell_type_key, len(names)
    )
    return names


def run_leiden_clustering_shared_genes(
    adata_sc: AnnData,
    resolution: float,
    n_pca_comps: int = 30,
):
    """Leiden-cluster the reference on the shared genes only. Not a start
    partition: the modularity metric and the shared-gene UMAP use it."""
    _build_neighbor_graph_shared_genes(adata_sc, n_pca_comps=n_pca_comps)
    sc.tl.leiden(
        adata_sc,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        neighbors_key=UNS_NEIGHBORS_SHARED_GENES,
        key_added=OBS_LEIDEN_SHARED_GENES,
    )
    number_of_clusters = len(
        np.unique(adata_sc.obs[OBS_LEIDEN_SHARED_GENES].astype(int).values)
    )
    logger.info(
        "Leiden clustering on shared genes (resolution=%.2f): %d clusters",
        resolution,
        number_of_clusters,
    )
    adata_sc.uns[UNS_LEIDEN_RESOLUTION_SHARED_GENES] = resolution
    adata_sc.uns[UNS_LEIDEN_NUMBER_STATES_SHARED_GENES] = number_of_clusters


def _build_neighbor_graph_shared_genes(
    adata_sc: AnnData,
    n_pca_comps: int = 30,
) -> None:
    """PCA and kNN graph of the reference on the shared genes (lognorm), stored
    in ``adata_sc`` under the shared-gene keys."""
    # Computed on a temporary shared-gene AnnData, then copied back.
    adata_shared = AnnData(
        X=adata_sc.obsm[OBSM_LOGNORM_SHARED_GENES],
        obs=adata_sc.obs[[]].copy(),
        var=pd.DataFrame(index=adata_sc.uns[UNS_SHARED_GENES]),
    )
    n = min(n_pca_comps, adata_shared.n_obs - 1, adata_shared.n_vars - 1)
    sc.pp.pca(adata_shared, n_comps=n, key_added=OBSM_PCA_SHARED_GENES)
    sc.pp.neighbors(
        adata_shared,
        use_rep=OBSM_PCA_SHARED_GENES,
        key_added=UNS_NEIGHBORS_SHARED_GENES,
    )

    # The uns entry names the obsp keys, so scanpy's leiden / umap can use this
    # graph on adata_sc via neighbors_key=UNS_NEIGHBORS_SHARED_GENES.
    adata_sc.obsm[OBSM_PCA_SHARED_GENES] = adata_shared.obsm[OBSM_PCA_SHARED_GENES]
    adata_sc.uns[UNS_NEIGHBORS_SHARED_GENES] = adata_shared.uns[
        UNS_NEIGHBORS_SHARED_GENES
    ]
    for key in (OBSP_DISTANCES_SHARED_GENES, OBSP_CONNECTIVITIES_SHARED_GENES):
        adata_sc.obsp[key] = adata_shared.obsp[key]
