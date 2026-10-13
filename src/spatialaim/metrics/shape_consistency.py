"""Shape consistency: how uniform the cell shapes within each mapped state are.

Each segmented ST cell is described by three shape features of its outline in the
segmentation mask -- log(area), eccentricity and solidity, as defined by
CellProfiler's MeasureObjectSizeShape and computed with scikit-image
``regionprops_table`` -- z-scored over all segmented cells of the slice, so the
three count equally (variance 1 each, 3 in total). A mapping groups the cells by
their hard state label; two variants score how consistent the shapes within
those groups are:

* ``state_variance_score`` -- 1 - (the shape variance of each state, averaged
  over the states so each state counts equally) / total variance: 1 = all cells
  of every state share one shape, 0 = each state as varied as the whole slice
  (chance level),
* ``shape_modularity`` -- modularity of the state labels on a kNN graph of the
  cells in shape-feature space.

Pure computation: the mask is read and the features are cached by
:mod:`spatialaim.analysis.shape_consistency`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from skimage.measure import regionprops_table

from .biology import compute_modularity

SHAPE_FEATURES = ("log_area", "eccentricity", "solidity")
N_NEIGHBORS = 15


def shape_features(mask: np.ndarray, segmentation_id: np.ndarray) -> np.ndarray:
    """(n_cells x 3) raw shape features, one row per entry of ``segmentation_id``.

    ``mask`` is the 2D label image (0 = background); ``segmentation_id[i]`` is the
    mask label of cell i, 0 for a cell without segmentation, which gets a NaN row.
    Columns as in ``SHAPE_FEATURES``: log(area in pixels), eccentricity, solidity.
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
    ids = np.asarray(segmentation_id, dtype=np.int64)
    # label 0 is never a row of ``table``, so cells without segmentation become NaN
    return table.reindex(ids).to_numpy(dtype=np.float64)


def zscore(features: np.ndarray) -> np.ndarray:
    """Each column z-scored over its finite rows; NaN rows stay NaN."""
    features = np.asarray(features, dtype=np.float64)
    mean = np.nanmean(features, axis=0)
    std = np.nanstd(features, axis=0)
    return (features - mean) / np.where(std > 0, std, 1.0)


def valid_cells(features: np.ndarray) -> np.ndarray:
    """Boolean mask of the cells with all shape features defined."""
    return np.isfinite(features).all(axis=1)


def state_variance_score(
    features: np.ndarray, labels: np.ndarray, min_cells: int = 2
) -> float:
    """1 - mean state variance / total variance; higher = more consistent.

    ``features`` (n x d) and ``labels`` (n,) over the same cells, NaN rows already
    removed. A state's variance is the summed squared distance of its cells to the
    state's mean vector divided by (cells - 1) -- the unbiased estimate, so a small
    state is not made to look more consistent than a large one by its size alone.
    The states' variances are averaged with equal weight, whatever their size, and
    divided by the total variance of all cells (the same estimate with one state;
    d = 3 for z-scored features). 1 = every state's cells share one shape, 0 =
    states as varied as the whole slice (chance level); below 0 = more varied.
    States with fewer than ``min_cells`` cells (at least 2: a single cell has no
    variance) are left out; NaN if none is left or the total variance is 0.
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


def shape_knn_graph(features: np.ndarray, n_neighbors: int = N_NEIGHBORS):
    """Connectivities (n x n, sparse) of a kNN graph in shape-feature space.

    ``features`` without NaN rows. scanpy ``neighbors`` directly on the features
    (no PCA), the same graph construction as the ST expression graph.
    """
    tmp = AnnData(X=np.zeros((len(features), 1), dtype=np.float32))
    tmp.obsm["X_shape"] = np.asarray(features, dtype=np.float32)
    sc.pp.neighbors(
        tmp, n_neighbors=min(n_neighbors, len(features) - 1), use_rep="X_shape"
    )
    return tmp.obsp["connectivities"]


def shape_modularity(connectivities, labels: np.ndarray) -> float:
    """Modularity of ``labels`` on the shape kNN graph ``connectivities``."""
    graph = AnnData(
        X=np.zeros((connectivities.shape[0], 1), dtype=np.float32),
        obsp={"connectivities": sparse.csr_matrix(connectivities)},
    )
    return compute_modularity(graph, np.asarray(labels))
