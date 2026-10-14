"""
Entrypoint for SpatialAIM for one sc/ST pair.
"""

import logging
from pathlib import Path
import anndata
import scanpy as sc
from spatialaim.adata_schema import (
    LAYER_LOGNORM,
    OBSM_LOGNORM_SHARED_GENES,
    OBSM_UMAP,
    OBSM_UMAP_SHARED_GENES,
    UNS_N_START_CLUSTERS,
    UNS_NEIGHBORS_SHARED_GENES,
    UNS_SHARED_GENES,
    UNS_START_CLUSTER_CENTROIDS_SHARED_GENES,
)
from spatialaim.analysis.run import run_analysis
from spatialaim.analysis.compare_ks import compare_k_runs
from spatialaim.analysis.shape_consistency import (
    prepare_shape_features,
    write_shape_features,
)
from spatialaim.config import LINKAGE_METHODS
from .aggregation import compute_start_cluster_aggregates
from .clustering import (
    build_neighbor_graph_all_genes,
    drop_unannotated,
    run_leiden_clustering_all_genes,
    run_leiden_clustering_shared_genes,
    set_start_clusters_from_annotation,
)
from .io import (
    read_reference_scaffold,
    reference_scaffold_key,
    write_reference_scaffold,
    write_run_outputs,
    write_start_clustering,
)
from .mapping import SpotStateMapper
from .tree import build_agglomeration_tree, labels_at_k

logger = logging.getLogger(__name__)


def run(
    sc_path: Path,
    st_path: Path,
    root_output_folder: Path,
    mapper: SpotStateMapper,
    linkage_method: str = LINKAGE_METHODS[0],
    leiden_resolution: float = 3.0,
    k_min: int | None = None,
    k_max: int | None = None,
    k_step: int = 1,
    start_from_annotation: str | None = None,
    segmentation_path: Path | None = None,
):
    """Run the full sweep for one sc/ST pair.

    1. Build the reference scaffold (start clusters, aggregates, UMAPs), or
       reuse the one cached in ``root_output_folder`` by an earlier mapper.
    2. Build the agglomeration tree over the start clusters.
    3. For every K from high to low: cut the tree, map the spots onto the
       states, write the results to ``<root>/<mapper>/k_<kkk>/`` and analyse
       them.
    4. Compare all Ks (``k_comparison.csv``, ``k_selection.json``).

    ``start_from_annotation`` names an ``obs`` column whose cell types replace
    the Leiden start clusters (unlabelled cells are dropped). Such runs need
    their own ``root_output_folder``, otherwise they overwrite the Leiden runs.
    ``segmentation_path`` is the ST slice's optional segmentation mask; with it
    the analysis also scores shape consistency.
    """
    root_output_folder = Path(root_output_folder)
    root_output_folder.mkdir(parents=True, exist_ok=True)
    mapping_output_folder = root_output_folder / mapper.name
    mapping_output_folder.mkdir(parents=True, exist_ok=True)

    adata_sc = anndata.read_h5ad(sc_path)
    adata_st = anndata.read_h5ad(st_path)
    if segmentation_path is not None:
        write_shape_features(
            prepare_shape_features(adata_st, segmentation_path), mapping_output_folder
        )
    if start_from_annotation is not None:
        # Before preprocessing, so every computed block covers the same cells.
        adata_sc = drop_unannotated(adata_sc, start_from_annotation)

    # 1. Reference scaffold
    shared_genes = list(adata_sc.var_names.intersection(adata_st.var_names))
    key = reference_scaffold_key(
        sc_path, st_path, shared_genes, leiden_resolution, start_from_annotation
    )
    cached = read_reference_scaffold(root_output_folder, key)
    if cached is not None:
        adata_sc = cached
        _preprocess_st(adata_st, adata_sc.uns[UNS_SHARED_GENES])
    else:
        pre_processing(adata_sc, adata_st)
        build_reference_scaffold(
            adata_sc, leiden_resolution, start_from_annotation=start_from_annotation
        )
        write_reference_scaffold(root_output_folder, adata_sc, key)
    write_start_clustering(mapping_output_folder, adata_sc)

    # 2. Tree and its cuts
    tree = build_agglomeration_tree(
        adata_sc.uns[UNS_START_CLUSTER_CENTROIDS_SHARED_GENES],
        method=linkage_method,
    )
    n_start_clusters = adata_sc.uns[UNS_N_START_CLUSTERS]
    ks = _ks_to_sweep(n_start_clusters, k_min, k_max, k_step)
    labels_by_k = {k: labels_at_k(tree, k, n_start_clusters) for k in ks}

    # 3. Map and analyse every K
    mapper.prepare(adata_sc, adata_st, labels_by_k)
    try:
        for k in ks:
            spot_to_state, confidence = mapper.map(labels_by_k[k], k)
            run_dir = mapping_output_folder / f"k_{k:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_run_outputs(
                run_dir=run_dir,
                spot_to_state=spot_to_state,
                confidence=confidence,
                labels_k=labels_by_k[k],
                n_start_clusters=n_start_clusters,
                k=k,
                adata_st=adata_st,
            )
            logger.info("K=%3d mapped -> %s", k, run_dir)
            run_analysis(adata_sc, adata_st, run_dir)
    finally:
        mapper.close()  # e.g. stops a reference aligner's worker process

    # 4. Compare the Ks
    logger.info("Comparing K-runs...")
    compare_k_runs(mapping_output_folder, ks)


def _ks_to_sweep(
    n_start_clusters: int, k_min: int | None, k_max: int | None, k_step: int
) -> list[int]:
    """The Ks to sweep, from high to low: from ``k_max`` (at most the number of
    start clusters) down to ``k_min`` (at least 1) in steps of ``k_step``."""
    k_hi = min(n_start_clusters, k_max) if k_max else n_start_clusters
    k_lo = max(1, k_min) if k_min else 1
    ks = list(range(k_hi, k_lo - 1, -k_step))
    logger.info(
        "Sweeping K from %d down to %d (step %d): %d levels",
        k_hi,
        k_lo,
        k_step,
        len(ks),
    )
    return ks


# --------------------------------------------------------------------------- #
# Reference scaffold (also used by the GUI)
# --------------------------------------------------------------------------- #
def pre_processing(
    adata_sc: anndata.AnnData,
    adata_st: anndata.AnnData,
):
    """Normalise and log-transform both datasets, leaving ``.X`` as raw counts.

    Adds to each a lognorm layer over all genes and a lognorm block over the
    shared genes (``obsm``), and records the shared genes in ``adata_sc.uns``.
    """
    shared_genes = list(adata_sc.var_names.intersection(adata_st.var_names))
    _preprocess_sc(adata_sc, shared_genes)
    _preprocess_st(adata_st, shared_genes)


def build_reference_scaffold(
    adata_sc: anndata.AnnData,
    leiden_resolution: float,
    start_from_annotation: str | None = None,
) -> None:
    """The mapper-independent part of a sweep, on a pre-processed ``adata_sc``:
    start clusters (Leiden, or the ``start_from_annotation`` column), the
    shared-gene Leiden clustering, the start-cluster aggregates and both UMAPs.
    """
    if start_from_annotation is None:
        logger.info("Computing Leiden over-clustering...")
        run_leiden_clustering_all_genes(adata_sc, resolution=leiden_resolution)
    else:
        logger.info("Start clusters from annotation %r...", start_from_annotation)
        # No Leiden here, but the graph is still needed by the metrics and UMAP.
        build_neighbor_graph_all_genes(adata_sc)
        set_start_clusters_from_annotation(adata_sc, start_from_annotation)

    run_leiden_clustering_shared_genes(adata_sc, resolution=leiden_resolution)
    compute_start_cluster_aggregates(adata_sc)

    # UMAPs are only for the analysis and the GUI.
    sc.tl.umap(adata_sc, key_added=OBSM_UMAP)
    sc.tl.umap(
        adata_sc,
        neighbors_key=UNS_NEIGHBORS_SHARED_GENES,
        key_added=OBSM_UMAP_SHARED_GENES,
    )


def _preprocess_sc(adata_sc: anndata.AnnData, shared_genes) -> None:
    """The reference half of :func:`pre_processing`."""
    adata_sc.uns[UNS_SHARED_GENES] = list(shared_genes)
    _lognorm_all(adata_sc)
    _lognorm_shared(adata_sc, shared_genes)


def _preprocess_st(adata_st: anndata.AnnData, shared_genes) -> None:
    """The spatial half of :func:`pre_processing`."""
    _lognorm_all(adata_st)
    _lognorm_shared(adata_st, shared_genes)


def _lognorm_all(adata: anndata.AnnData) -> None:
    """Add ``layers[LAYER_LOGNORM]``: all genes, normalised to 10,000 counts
    per cell and log1p-transformed."""
    adata.layers[LAYER_LOGNORM] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4, layer=LAYER_LOGNORM)
    sc.pp.log1p(adata, layer=LAYER_LOGNORM)


def _lognorm_shared(adata: anndata.AnnData, shared_genes) -> None:
    """Add ``obsm[OBSM_LOGNORM_SHARED_GENES]``: like the lognorm layer, but on
    the shared genes only (normalised by their counts alone, so it cannot be a
    layer)."""
    adata_shared = adata[:, list(shared_genes)].copy()
    adata_shared.uns.pop("log1p", None)  # avoids a scanpy warning
    sc.pp.normalize_total(adata_shared, target_sum=1e4)
    sc.pp.log1p(adata_shared)
    adata.obsm[OBSM_LOGNORM_SHARED_GENES] = adata_shared.X
