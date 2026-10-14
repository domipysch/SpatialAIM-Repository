"""
Shape consistency: how uniform the cell shapes within each mapped state are.

Each segmented cell is described by log(area), eccentricity and solidity
(CellProfiler's MeasureObjectSizeShape definitions, via scikit-image), z-scored
over the slice.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

SHAPE_FEATURES = ("log_area", "eccentricity", "solidity")


def shape_features(mask: np.ndarray, segmentation_id: np.ndarray) -> np.ndarray:
    """Raw shape features (cells x ``SHAPE_FEATURES``), area in pixels.

    ``mask`` is the 2D label image (0 = background) and ``segmentation_id[i]``
    the mask label of cell i; cells with id 0 get a NaN row.
    """
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
