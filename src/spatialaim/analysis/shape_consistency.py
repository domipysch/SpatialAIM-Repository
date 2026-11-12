"""
Shape consistency of one K's mapping: how uniform the cell shapes within each
mapped state are, from the ST slice's optional segmentation mask.

Each segmented cell is described by log(area), eccentricity and solidity
(CellProfiler's MeasureObjectSizeShape definitions, via scikit-image), z-scored
over the slice. The cells are measured once per sweep
(:func:`prepare_shape_features`); each K is then scored on them.
"""

from __future__ import annotations
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
from .criterion import Criterion, finite

logger = logging.getLogger(__name__)

SHAPE_CONSISTENCY_JSON = "shape_consistency_metrics.json"
# Raw per-cell features at the run root, plotted by the GUI.
SHAPE_FEATURES_CSV = "cell_shape_features.csv"
SHAPE_FEATURES = ("log_area", "eccentricity", "solidity")


class ShapeConsistency(Criterion):
    """How uniform the cell shapes within each mapped state are."""

    key = "shape_consistency"
    label = "Shape consistency"
    short = "Shape"
    unit = "1 − state / total shape variance"
    filename = SHAPE_CONSISTENCY_JSON
    columns = ("shape",)
    curve_labels = ("shape consistency",)

    def analyse(self, adata_sc, adata_st, start_cluster_to_state) -> dict:
        """The shape consistency of the spots' states and how many cells it
        used. NaN without a mask."""
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
        return result

    def collect(self, data_dir: Path) -> dict[str, float]:
        """The shape-consistency score (NaN without a segmentation mask)."""
        return {"shape": finite(self.load(data_dir).get("shape_consistency"))}


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


def write_shape_features(table: pd.DataFrame, pair_output: Path) -> None:
    """Write the table of :func:`prepare_shape_features` (6 significant digits)
    to the pair output, once for all starts and mappers."""
    table.to_csv(Path(pair_output) / SHAPE_FEATURES_CSV, float_format="%.6g")


def shape_features(mask: np.ndarray, segmentation_id: np.ndarray) -> np.ndarray:
    """Raw shape features (cells x ``SHAPE_FEATURES``), area in pixels.

    ``mask`` is the 2D label image (0 = background) and ``segmentation_id[i]``
    the mask label of cell i; cells with id 0 get a NaN row.
    """
    from skimage.measure import regionprops_table

    props = pd.DataFrame(
        regionprops_table(
            mask, properties=("label", "area", "eccentricity", "solidity")
        )
    ).set_index("label")
    table = pd.DataFrame(
        {
            "log_area": np.log(props["area"].to_numpy(dtype=np.float64)),
            "eccentricity": props["eccentricity"].to_numpy(dtype=np.float64),
            "solidity": props["solidity"].to_numpy(dtype=np.float64),
        },
        index=props.index,
    )
    # Label 0 is not in the table, so unsegmented cells become NaN.
    return table.reindex(np.asarray(segmentation_id, dtype=np.int64)).to_numpy(
        dtype=np.float64
    )


def zscore(features: np.ndarray) -> np.ndarray:
    """Each column z-scored over its non-NaN rows; NaN rows stay NaN."""
    features = np.asarray(features, dtype=np.float64)
    mean = np.nanmean(features, axis=0)
    std = np.nanstd(features, axis=0)
    return (features - mean) / np.where(std > 0, std, 1.0)


def valid_cells(features: np.ndarray) -> np.ndarray:
    """Boolean mask of the cells whose features are all defined."""
    return np.isfinite(features).all(axis=1)


def state_variance_score(
    features: np.ndarray, labels: np.ndarray, min_cells: int = 2
) -> float:
    """``1 - mean state variance / total variance``.

    A state's variance is its cells' summed squared distance to the state mean
    divided by (cells - 1); all states count equally in the mean. 1 = all cells
    of a state share one shape, 0 = chance level, < 0 = more varied than chance.

    ``features`` (cells x d) and ``labels`` cover the same cells, NaN rows
    removed. States with fewer than ``min_cells`` cells (at least 2) are skipped;
    NaN if none is left or the total variance is 0.
    """
    frame = pd.DataFrame(np.asarray(features, dtype=np.float64))
    labels = np.asarray(labels)
    if len(frame) < 2:
        return float("nan")
    total = float(frame.var(ddof=1).sum())
    sizes = frame.groupby(labels).size()
    deviation = frame - frame.groupby(labels).transform("mean")
    squared = (deviation**2).sum(axis=1).groupby(labels).sum()
    keep = sizes >= max(min_cells, 2)
    per_state = squared[keep] / (sizes[keep] - 1)
    if not len(per_state) or total <= 0:
        return float("nan")
    return 1.0 - float(per_state.mean()) / total
