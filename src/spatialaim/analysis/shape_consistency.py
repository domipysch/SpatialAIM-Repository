"""Shape consistency of the mapping: how uniform the cell shapes within each
mapped state are, from the ST slice's optional cell segmentation mask.

:func:`prepare_shape_features` runs once per sweep, right after the ST data is
read: it measures the cells in the mask and caches their z-scored shape features
in ``adata_st.obsm``. :func:`analyse_shape_consistency` runs per K and scores the
hard spot->state labels on them (see :mod:`spatialaim.metrics.shape_consistency`).
Without a mask the per-K file is still written, with NaN scores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from anndata import AnnData
from scipy import sparse

from spatialaim.adata_schema import (
    OBS_MAPPING_HARD,
    OBS_SEGMENTATION_ID,
    OBSM_SHAPE_FEATURES,
    OBSP_SHAPE_CONNECTIVITIES,
)
from spatialaim.metrics.shape_consistency import (
    shape_features,
    shape_knn_graph,
    shape_modularity,
    state_variance_score,
    valid_cells,
    zscore,
)

logger = logging.getLogger(__name__)

SHAPE_CONSISTENCY_JSON = "shape_consistency_metrics.json"


def prepare_shape_features(adata_st: AnnData, segmentation_path: Path) -> None:
    """Measure every segmented cell in the mask and cache the z-scored features.

    Requires: adata_st.obs["segmentation_id"] (the cell's mask label, 0 = none).
    Adds: adata_st.obsm[OBSM_SHAPE_FEATURES] (n_spots x 3, NaN rows for cells
    without segmentation).
    """
    import tifffile

    if OBS_SEGMENTATION_ID not in adata_st.obs:
        raise KeyError(
            f"a segmentation mask is given but adata_st.obs[{OBS_SEGMENTATION_ID!r}] "
            "is missing"
        )
    logger.info("Measuring cell shapes in %s...", Path(segmentation_path).name)
    mask = tifffile.imread(segmentation_path)
    raw = shape_features(mask, adata_st.obs[OBS_SEGMENTATION_ID].to_numpy())
    del mask
    adata_st.obsm[OBSM_SHAPE_FEATURES] = zscore(raw)
    logger.info(
        "  shape features for %d of %d cells", int(valid_cells(raw).sum()), len(raw)
    )


def _shape_connectivities(adata_st: AnnData, valid: np.ndarray):
    """The shape kNN graph over the valid cells, built once and cached in obsp
    (full n_spots x n_spots, no edges at cells without features)."""
    if OBSP_SHAPE_CONNECTIVITIES not in adata_st.obsp:
        sub = shape_knn_graph(adata_st.obsm[OBSM_SHAPE_FEATURES][valid])
        index = np.flatnonzero(valid)
        coo = sparse.coo_matrix(sub)
        adata_st.obsp[OBSP_SHAPE_CONNECTIVITIES] = sparse.csr_matrix(
            (coo.data, (index[coo.row], index[coo.col])),
            shape=(adata_st.n_obs, adata_st.n_obs),
        )
    return adata_st.obsp[OBSP_SHAPE_CONNECTIVITIES][valid][:, valid]


def analyse_shape_consistency(adata_st: AnnData, output_data_dir: Path) -> None:
    """Both shape-consistency variants for this K's hard mapping.

    Requires: adata_st.obs[OBS_MAPPING_HARD]; adata_st.obsm[OBSM_SHAPE_FEATURES]
        if the slice has a segmentation mask.
    Writes shape_consistency_metrics.json under output_data_dir.
    """
    result = {
        "shape_consistency_variance": float("nan"),
        "shape_consistency_modularity": float("nan"),
        "n_cells_used": 0,
        "n_cells_total": int(adata_st.n_obs),
    }
    features = np.asarray(
        adata_st.obsm.get(OBSM_SHAPE_FEATURES, np.empty((adata_st.n_obs, 0)))
    )
    valid = valid_cells(features) & (features.shape[1] > 0)
    if valid.sum() > 2:  # a kNN graph needs a few cells
        labels = adata_st.obs[OBS_MAPPING_HARD].astype(int).to_numpy()[valid]
        result["n_cells_used"] = int(valid.sum())
        result["shape_consistency_variance"] = state_variance_score(
            features[valid], labels
        )
        result["shape_consistency_modularity"] = shape_modularity(
            _shape_connectivities(adata_st, valid), labels
        )

    with open(output_data_dir / SHAPE_CONSISTENCY_JSON, "w") as f:
        json.dump(result, f, indent=4)
