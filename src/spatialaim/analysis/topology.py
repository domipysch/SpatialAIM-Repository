"""
Spatial organisation of one K's mapping: the assortativity of the spots' states
on the spatial kNN graph, overall and per state (chance-corrected, no shuffles).
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import anndata as ad
import numpy as np
import squidpy as sq
from spatialaim.adata_schema import (
    OBS_MAPPING_HARD,
    OBSM_SPATIAL,
    OBSP_SPATIAL_CONNECTIVITIES,
)
from spatialaim.metrics.topology import assortativity

logger = logging.getLogger(__name__)


K_SPATIAL = 6  # neighbours in the spatial kNN graph


def analyse_spatial_organization(
    adata_st: ad.AnnData,
    output_data_dir: Path,
    k: int = K_SPATIAL,
):
    """Write ``topology_metrics.json``: the global assortativity, the mean
    per-state assortativity, and per mapped state its assortativity (for the
    GUI's cell-type modal). Undefined values are NaN."""
    spot_states = np.asarray(adata_st.obs[OBS_MAPPING_HARD].to_numpy(), dtype=int)
    coords = np.asarray(adata_st.obsm[OBSM_SPATIAL])
    assert len(coords) == len(
        spot_states
    ), "Spatial organisation skipped: coordinates length mismatch "
    assert len(spot_states) > k, "too few spots"

    a = assortativity(spot_states, _spatial_connectivities(adata_st, k))
    if a is None:
        result = {
            "spatial_assort_global": float("nan"),
            "spatial_assort_celltype_mean": float("nan"),
            "per_state": {},
        }
    else:
        result = {
            "spatial_assort_global": a.value,
            "spatial_assort_celltype_mean": a.mean_state,
            "per_state": {
                str(int(s)): float(a.per_state[s]) for s in np.unique(spot_states)
            },
        }
    with open(output_data_dir / "topology_metrics.json", "w") as f:
        json.dump(result, f, indent=4)


def _spatial_connectivities(adata_st: ad.AnnData, k: int):
    """The spatial kNN graph of the spots (squidpy). Built on the first K and
    kept in ``adata_st.obsp``, since the coordinates do not change."""
    if OBSP_SPATIAL_CONNECTIVITIES not in adata_st.obsp:
        sq.gr.spatial_neighbors_knn(adata_st, n_neighs=k)
    return adata_st.obsp[OBSP_SPATIAL_CONNECTIVITIES]
