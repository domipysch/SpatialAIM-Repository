"""
Shape consistency of one K's mapping, from the ST slice's optional segmentation
mask. The cells are measured once per sweep (:func:`prepare_shape_features`);
each K is then scored on them (:func:`analyse_shape_consistency`).
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from anndata import AnnData
from spatialaim.adata_schema import (
    OBS_MAPPING_HARD,
    OBS_SEGMENTATION_ID,
    OBSM_SHAPE_FEATURES,
    UNS_SEGMENTATION_PIXEL_SIZE,
)
from spatialaim.metrics.shape_consistency import (
    shape_features,
    state_variance_score,
    valid_cells,
    zscore,
)

logger = logging.getLogger(__name__)

SHAPE_CONSISTENCY_JSON = "shape_consistency_metrics.json"
# Raw per-cell features at the run root, plotted by the GUI.
SHAPE_FEATURES_CSV = "cell_shape_features.csv"


def prepare_shape_features(adata_st: AnnData, segmentation_path: Path) -> pd.DataFrame:
    """Measure every cell in the mask (linked via ``obs["segmentation_id"]``)
    and store the z-scored features in ``adata_st.obsm[OBSM_SHAPE_FEATURES]``.

    Returns the raw features per cell: ``area_px``, ``area_um2`` (NaN without a
    pixel size), ``eccentricity``, ``solidity``; NaN for unsegmented cells.
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
    area_px = np.exp(raw[:, 0])
    pixel_um = adata_st.uns.get(UNS_SEGMENTATION_PIXEL_SIZE)
    return pd.DataFrame(
        {
            "area_px": area_px,
            "area_um2": area_px * float(pixel_um) ** 2 if pixel_um else np.nan,
            "eccentricity": raw[:, 1],
            "solidity": raw[:, 2],
        },
        index=pd.Index(adata_st.obs_names, name="cell"),
    )


def write_shape_features(table: pd.DataFrame, run_root: Path) -> None:
    """Write the table of :func:`prepare_shape_features` to the run root."""
    table.to_csv(Path(run_root) / SHAPE_FEATURES_CSV)


def analyse_shape_consistency(adata_st: AnnData, output_data_dir: Path) -> None:
    """Write ``shape_consistency_metrics.json``: the shape consistency of the
    spots' states and how many cells it used. NaN without a mask."""
    result = {
        "shape_consistency": float("nan"),
        "n_cells_used": 0,
        "n_cells_total": int(adata_st.n_obs),
    }
    features = np.asarray(
        adata_st.obsm.get(OBSM_SHAPE_FEATURES, np.empty((adata_st.n_obs, 0)))
    )
    valid = valid_cells(features) & (features.shape[1] > 0)
    if valid.sum() >= 2:
        labels = adata_st.obs[OBS_MAPPING_HARD].astype(int).to_numpy()[valid]
        result["n_cells_used"] = int(valid.sum())
        result["shape_consistency"] = state_variance_score(features[valid], labels)

    with open(output_data_dir / SHAPE_CONSISTENCY_JSON, "w") as f:
        json.dump(result, f, indent=4)
