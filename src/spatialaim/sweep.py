"""
Entrypoint for SpatialAIM for one sc/ST pair.
"""

import logging
from dataclasses import asdict
from pathlib import Path
import anndata
import scanpy as sc
import yaml
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
from spatialaim.config import SpatialAIMConfig
from .aggregation import compute_start_cluster_aggregates
from .clustering import (
    check_annotation,
    run_leiden_clustering_all_genes,
    run_leiden_clustering_shared_genes,
    set_start_clusters_from_annotation,
    set_start_clusters_from_leiden,
)
from .io import (
    read_reference_scaffold,
    reference_scaffold_key,
    start_folder_name,
    write_reference_scaffold,
    write_run_outputs,
    write_start_clustering,
)
from .tree import build_agglomeration_tree, labels_at_k

logger = logging.getLogger(__name__)


def run(
    sc_path: Path,
    st_path: Path,
    root_output_folder: Path,
    config: SpatialAIMConfig,
    segmentation_path: Path | None = None,
):
    """Run the full sweep for one sc/ST pair, once per start: the Leiden
    over-clustering (if ``config.leiden_start``) and each annotation in
    ``config.annotations``.

    1. Check every annotation (it must exist and label every cell), before
       anything is computed.
    2. Build the reference scaffold (everything that does not depend on the
       start), or reuse the one cached in ``root_output_folder``.
    3. Per start, into ``<root>/<start>/<mapper>/`` (see
       :func:`spatialaim.io.start_folder_name`): set the start clusters (saved
       as ``<root>/<start>/start_clustering.h5ad``, shared by its mappers), build
       the agglomeration tree over them, and for every K from high to low cut
       the tree, map the spots onto the states, write and analyse the result
       (``k_<kkk>/``); then compare all Ks (``k_comparison.csv``,
       ``k_selection.json``).

    ``segmentation_path`` is the ST slice's optional segmentation mask; with it
    the analysis also scores shape consistency, and the per-cell shape features
    are written once to ``<root>/cell_shape_features.csv``.
    """
    annotations = list(config.annotations)
    starts = [None, *annotations] if config.leiden_start else annotations
    if not starts:
        raise ValueError("no start: leiden_start is off and no annotation given")
    folders = [start_folder_name(s) for s in starts]
    if len(set(folders)) != len(folders):
        raise ValueError(f"starts {starts} are not distinct (start folders {folders})")

    root_output_folder = Path(root_output_folder)
    root_output_folder.mkdir(parents=True, exist_ok=True)

    # Load data
    adata_sc = anndata.read_h5ad(sc_path)
    adata_st = anndata.read_h5ad(st_path)

    # 1. Fail before the expensive part if any annotation is invalid.
    for annotation in annotations:
        check_annotation(adata_sc, annotation)
    if segmentation_path is not None:
        write_shape_features(
            prepare_shape_features(adata_st, segmentation_path), root_output_folder
        )

    # 2. Reference scaffold
    shared_genes = list(adata_sc.var_names.intersection(adata_st.var_names))
    key = reference_scaffold_key(
        sc_path, st_path, shared_genes, config.leiden_resolution
    )
    cached = read_reference_scaffold(root_output_folder, key)
    if cached is not None:
        adata_sc = cached
        _preprocess_st(adata_st, adata_sc.uns[UNS_SHARED_GENES])
    else:
        pre_processing(adata_sc, adata_st)
        build_reference_scaffold(adata_sc, config.leiden_resolution)
        adata_sc = restrict_to_shared_genes(adata_sc)
        write_reference_scaffold(root_output_folder, adata_sc, key)

    # 3. One sweep per start
    for start_from_annotation, folder in zip(starts, folders):
        logger.info("Start %s...", folder)
        mapping_output_folder = root_output_folder / folder / config.mapping
        mapping_output_folder.mkdir(parents=True, exist_ok=True)
        _write_config(mapping_output_folder, config, start_from_annotation)
        set_start_clusters(adata_sc, start_from_annotation)
        write_start_clustering(root_output_folder / folder, adata_sc)
        _sweep_one_start(adata_sc, adata_st, mapping_output_folder, config)


def _write_config(
    mapping_output_folder: Path,
    config: SpatialAIMConfig,
    start_from_annotation: str | None,
) -> None:
    """Save the settings of one start's sweep as ``config.yaml``: the config,
    with ``leiden_start`` and ``annotations`` replaced by this start
    (``start_from_annotation``: ``None`` = Leiden, else the obs column)."""
    settings = asdict(config)
    settings.pop("leiden_start")
    settings.pop("annotations")
    settings["start_from_annotation"] = start_from_annotation
    with open(mapping_output_folder / "config.yaml", "w") as f:
        yaml.safe_dump(settings, f, sort_keys=False)


def _sweep_one_start(
    adata_sc: anndata.AnnData,
    adata_st: anndata.AnnData,
    mapping_output_folder: Path,
    config: SpatialAIMConfig,
) -> None:
    """The tree, the K-sweep and the K comparison for the start clusters set
    in ``adata_sc``, into ``mapping_output_folder``."""
    # Tree and its cuts
    tree = build_agglomeration_tree(
        adata_sc.uns[UNS_START_CLUSTER_CENTROIDS_SHARED_GENES],
        method=config.linkage_method,
    )
    n_start_clusters = adata_sc.uns[UNS_N_START_CLUSTERS]
    ks = _ks_to_sweep(n_start_clusters, config.k_min, config.k_max, config.k_step)
    labels_by_k = {k: labels_at_k(tree, k, n_start_clusters) for k in ks}

    # Map and analyse every K
    mapper = config.build_mapper()
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

    # Compare the Ks
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
) -> None:
    """The start- and mapper-independent part of a sweep, on a pre-processed
    ``adata_sc``: the all-gene Leiden over-clustering, the shared-gene Leiden
    clustering and both UMAPs. :func:`set_start_clusters` adds a start."""
    logger.info("Computing Leiden over-clustering...")
    run_leiden_clustering_all_genes(adata_sc, resolution=leiden_resolution)
    run_leiden_clustering_shared_genes(adata_sc, resolution=leiden_resolution)

    # UMAPs are only for the analysis and the GUI.
    sc.tl.umap(adata_sc, key_added=OBSM_UMAP)
    sc.tl.umap(
        adata_sc,
        neighbors_key=UNS_NEIGHBORS_SHARED_GENES,
        key_added=OBSM_UMAP_SHARED_GENES,
    )


def restrict_to_shared_genes(adata_sc: anndata.AnnData) -> anndata.AnnData:
    """The built scaffold on the shared genes only (``X``, the lognorm layer and
    the PCA loadings lose every other gene). Nothing reads another gene once the
    all-gene PCA and graph are built, and the lognorm values stay normalised over
    all genes."""
    return adata_sc[:, list(adata_sc.uns[UNS_SHARED_GENES])].copy()


def set_start_clusters(
    adata_sc: anndata.AnnData, start_from_annotation: str | None
) -> None:
    """Set the start clusters on a reference scaffold, in place: the Leiden
    over-clusters (``None``) or the types of the annotation
    ``start_from_annotation``, plus their aggregates. Overwrites the previous
    start."""
    if start_from_annotation is None:
        set_start_clusters_from_leiden(adata_sc)
    else:
        check_annotation(adata_sc, start_from_annotation)
        set_start_clusters_from_annotation(adata_sc, start_from_annotation)
    compute_start_cluster_aggregates(adata_sc)


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
