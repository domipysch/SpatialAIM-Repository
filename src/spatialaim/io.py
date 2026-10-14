"""
Start clustering, each K's mapping,
cached reference scaffold shared by all mappers of a pair.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
import anndata
import numpy as np
import pandas as pd
from spatialaim.adata_schema import (
    OBS_MAPPING_CONFIDENCE,
    OBS_START_CLUSTER,
    UNS_N_START_CLUSTERS,
    UNS_START_CLUSTER_NAMES,
)

logger = logging.getLogger(__name__)

START_CLUSTERING_FILENAME = "start_clustering.h5ad"
START_CLUSTER_TO_STATE_FILENAME = "start_cluster_to_state.csv"

# Bump on any change to what the scaffold contains, so old caches are rebuilt.
REFERENCE_SCAFFOLD_FORMAT_VERSION = 2
_SCAFFOLD_H5AD = "reference_scaffold.h5ad"
_SCAFFOLD_META = "reference_scaffold.meta.json"


def write_start_clustering(output_folder: Path, adata_sc: anndata.AnnData) -> None:
    """Write ``start_clustering.h5ad``: each reference cell's start cluster
    (``obs["start_cluster"]``, an integer) and its name
    (``obs["start_cluster_name"]``: the annotated type, or ``cluster_<i>``)."""
    labels = adata_sc.obs[OBS_START_CLUSTER].astype(int).to_numpy()
    names = [str(n) for n in adata_sc.uns[UNS_START_CLUSTER_NAMES]]
    if len(names) != int(adata_sc.uns[UNS_N_START_CLUSTERS]):
        raise ValueError(
            f"{UNS_START_CLUSTER_NAMES} has {len(names)} entries but "
            f"{UNS_N_START_CLUSTERS} is {adata_sc.uns[UNS_N_START_CLUSTERS]}"
        )
    obs = pd.DataFrame(
        {
            "start_cluster": labels,
            # Categorical already, so anndata does not log a conversion on write.
            "start_cluster_name": pd.Categorical([names[i] for i in labels]),
        },
        index=adata_sc.obs_names,
    )
    anndata.AnnData(X=np.zeros((len(labels), 0), dtype=np.float32), obs=obs).write_h5ad(
        output_folder / START_CLUSTERING_FILENAME
    )


def write_run_outputs(
    run_dir: Path,
    spot_to_state: np.ndarray,
    labels_k: np.ndarray,
    n_start_clusters: int,
    k: int,
    adata_st: anndata.AnnData,
    confidence: np.ndarray | None = None,
) -> None:
    """Write one K's results to ``run_dir``:

    - ``start_cluster_to_state.csv``: the state of every start cluster;
    - ``spot_to_state_mapping_soft.h5ad``: the soft mapping (spots x states),
      plus the per-spot ``confidence`` in ``obs`` if the mapper gives one.
    """
    pd.DataFrame(
        {"start_cluster": np.arange(n_start_clusters), "state": labels_k}
    ).to_csv(run_dir / START_CLUSTER_TO_STATE_FILENAME, index=False)

    obs = pd.DataFrame(index=adata_st.obs_names)
    if confidence is not None:
        obs[OBS_MAPPING_CONFIDENCE] = np.asarray(confidence)
    anndata.AnnData(
        X=np.asarray(spot_to_state, dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"state_{i}" for i in range(k)]),
    ).write_h5ad(run_dir / "spot_to_state_mapping_soft.h5ad")


def reference_scaffold_key(
    sc_path: Path,
    st_path: Path,
    shared_genes,
    leiden_resolution: float,
    start_from_annotation: str | None = None,
) -> dict:
    """The key a cached scaffold must match to be reused: both input files
    (path, size, modification time), a hash of the shared genes, the Leiden
    resolution and the start-cluster source."""
    sc_stat = Path(sc_path).stat()
    st_stat = Path(st_path).stat()
    genes = sorted(str(g) for g in shared_genes)
    gene_hash = hashlib.sha256("\x00".join(genes).encode("utf-8")).hexdigest()
    return {
        "format_version": REFERENCE_SCAFFOLD_FORMAT_VERSION,
        "sc_path": str(Path(sc_path).resolve()),
        "sc_size": sc_stat.st_size,
        "sc_mtime_ns": sc_stat.st_mtime_ns,
        "st_path": str(Path(st_path).resolve()),
        "st_size": st_stat.st_size,
        "st_mtime_ns": st_stat.st_mtime_ns,
        "shared_genes_sha256": gene_hash,
        "leiden_resolution": float(leiden_resolution),
        "start_from_annotation": start_from_annotation,
    }


def read_reference_scaffold(cache_dir: Path, key: dict) -> anndata.AnnData | None:
    """The cached scaffold in ``cache_dir`` if it was stored under ``key``,
    else ``None`` (also when the cache is missing or unreadable)."""
    cache_dir = Path(cache_dir)
    h5ad_path = cache_dir / _SCAFFOLD_H5AD
    meta_path = cache_dir / _SCAFFOLD_META
    if not (h5ad_path.exists() and meta_path.exists()):
        return None
    try:
        with open(meta_path, encoding="utf-8") as fh:
            stored = json.load(fh)
        if stored != key:
            return None
        adata_sc = anndata.read_h5ad(h5ad_path)
    except Exception:  # noqa: BLE001 - a corrupt cache is just a miss
        logger.warning("Ignoring unreadable reference scaffold cache in %s", cache_dir)
        return None
    logger.info("Reusing cached reference scaffold from %s", h5ad_path)
    return adata_sc


def write_reference_scaffold(
    cache_dir: Path, adata_sc: anndata.AnnData, key: dict
) -> None:
    """Cache the scaffold and its ``key`` in ``cache_dir``.

    Each file is written to a temporary name and then moved into place, h5ad
    first, so an existing meta file always means a complete h5ad. Failures are
    only logged: the sweep goes on without a cache.
    """
    cache_dir = Path(cache_dir)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = f".{os.getpid()}.tmp"
        h5ad_path = cache_dir / _SCAFFOLD_H5AD
        meta_path = cache_dir / _SCAFFOLD_META
        h5ad_tmp = h5ad_path.with_suffix(h5ad_path.suffix + suffix)
        meta_tmp = meta_path.with_suffix(meta_path.suffix + suffix)

        adata_sc.write_h5ad(h5ad_tmp)
        os.replace(h5ad_tmp, h5ad_path)
        with open(meta_tmp, "w", encoding="utf-8") as fh:
            json.dump(key, fh)
        os.replace(meta_tmp, meta_path)
        logger.info("Wrote reference scaffold cache to %s", h5ad_path)
    except Exception:  # noqa: BLE001 - caching is best-effort
        logger.warning(
            "Could not write reference scaffold cache to %s; continuing without cache.",
            cache_dir,
            exc_info=True,
        )
