"""
Convention for the sc/st AnnData objects.
``L`` = number of start clusters
``G_shared`` = number of shared genes,
``K`` = number of states at the current cut.
"""

# -------------------- adata_sc --------------------


# (n_cells,) int — start-cluster label (0..L-1): the all-gene Leiden over-cluster, or the annotated type in start-from-annotation mode
OBS_START_CLUSTER = "start_cluster"

# (n_cells,) int — Leiden label from shared genes only
OBS_LEIDEN_SHARED_GENES = "leiden_shared_genes"

# (n_cells,) str categorical — cell's state at the current K
OBS_COMPUTED_STATE = "computed_state"

# int — start-cluster count L
UNS_N_START_CLUSTERS = "n_start_clusters"

# list[str] — display name per start cluster, index-aligned to OBS_START_CLUSTER: the annotated type names, or "cluster_<i>" in Leiden mode
UNS_START_CLUSTER_NAMES = "start_cluster_names"

# float — resolution of the all-gene Leiden over-clustering (absent in start-from-annotation mode)
UNS_LEIDEN_RESOLUTION_ALL_GENES = "leiden_resolution"

# float — resolution (shared genes)
UNS_LEIDEN_RESOLUTION_SHARED_GENES = "leiden_resolution_shared_genes"

# int — cluster count (shared genes)
UNS_LEIDEN_NUMBER_STATES_SHARED_GENES = "leiden_number_states_shared_genes"

# float — modularity of the shared-gene Leiden partition (K-independent; cached across the sweep)
UNS_MODULARITY_SHARED_LEIDEN = "modularity_shared_leiden"

# (n_cells x n_comps) — PCA of the shared-gene lognorm matrix
OBSM_PCA_SHARED_GENES = "X_pca_shared_genes"

# dict — scanpy neighbors params for the shared-gene graph
UNS_NEIGHBORS_SHARED_GENES = "neighbors_shared_genes"

# (n_cells x n_cells) sparse — neighbor distances
OBSP_DISTANCES_SHARED_GENES = "neighbors_shared_genes_distances"

# (n_cells x n_cells) sparse — neighbor connectivities
OBSP_CONNECTIVITIES_SHARED_GENES = "neighbors_shared_genes_connectivities"

# per-start-cluster aggregates, all column-aligned to UNS_SHARED_GENES

# list[str] — sc/st gene intersection; column order for every *_SHARED array
UNS_SHARED_GENES = "shared_genes"

# (L,) — cells per start cluster
UNS_START_CLUSTER_SIZES = "start_cluster_sizes"

# (L x G_shared) — summed raw expression per start cluster
UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES = "start_cluster_expr_sums_shared"

# (L x G_shared) — mean raw expression per start cluster (all-zero rows set to 1e-6)
UNS_START_CLUSTER_CENTROIDS_SHARED_GENES = "start_cluster_centroids_shared"

# (L x G_shared) — summed lognorm expression per start cluster
UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES_NORM = "start_cluster_expr_sums_shared_norm"

# (L x G_shared) — mean lognorm expression per start cluster
UNS_START_CLUSTER_CENTROIDS_SHARED_GENES_NORM = "start_cluster_centroids_shared_norm"


# -------------------- adata_st --------------------


# obsm/obs — one K's spot->state mapping

# (n_spots x 2) — spatial coordinates
OBSM_SPATIAL = "spatial"

# (n_spots x K) — soft spot->state assignment P at the current K
OBSM_MAPPING_SOFT = "mapping_soft"

# (n_spots,) int — argmax state index (0..K-1) at the current K
OBS_MAPPING_HARD = "mapping_hard"

# (n_spots,) str categorical — hard state label as a category, for squidpy neighbourhood tools (nhood_enrichment)
OBS_MAPPING_STATE_CAT = "mapping_state"

# (n_spots,) float in [0,1] — per-spot assignment confidence at the current K; absent when the mapper defines no confidence (reference)
OBS_MAPPING_CONFIDENCE = "mapping_confidence"

# obsm/obsp/uns — K-independent spatial/expression graphs, built once and reused across the K-sweep

# (n_spots x n_spots) sparse — squidpy spatial KNN graph (local purity + neighbourhood enrichment)
OBSP_SPATIAL_CONNECTIVITIES = "spatial_connectivities"

# (n_spots x n_spots) sparse — ST all-gene expression KNN graph for the mapping modularity
OBSP_ST_EXPR_CONNECTIVITIES = "expr_connectivities"

# (n_spots,) int — the cell's label in the optional segmentation mask, 0 = none (input, set at curation)
OBS_SEGMENTATION_ID = "segmentation_id"

# float — µm per mask pixel, optional (input, set at curation)
UNS_SEGMENTATION_PIXEL_SIZE = "segmentation_pixel_size_um"

# (n_spots x 3) — z-scored log(area), eccentricity, solidity from the segmentation mask; NaN rows = no segmentation; absent without a mask
OBSM_SHAPE_FEATURES = "shape_features"

# dict {zscore, count} — squidpy neighbourhood-enrichment result over the mapped states
UNS_NHOOD_ENRICHMENT = f"{OBS_MAPPING_STATE_CAT}_nhood_enrichment"


# -------------------- shared across adata_sc and adata_st --------------------


# (n_obs x n_vars) — normalize_total(1e4) + log1p over all genes
LAYER_LOGNORM = "lognorm"

# (n_obs x G_shared) — normalize_total(1e4) + log1p over shared genes only, aligned to UNS_SHARED_GENES
OBSM_LOGNORM_SHARED_GENES = "lognorm_shared"


# -------------------- scanpy default keys, referenced by name --------------------


# (n_cells x n_comps) — PCA of the all-gene lognorm layer
OBSM_PCA = "X_pca"

# (n_cells x 2) — UMAP from the all-gene neighbor graph
OBSM_UMAP = "X_umap"

# (n_cells x 2) — UMAP from the shared-gene neighbor graph
OBSM_UMAP_SHARED_GENES = "X_umap_shared_genes"
