"""
Spatial organisation of one K's mapping: local purity and neighbourhood
enrichment of the spots' states, each against label shuffles.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Callable
import anndata as ad
import numpy as np
import squidpy as sq
from spatialaim.adata_schema import (
    OBS_MAPPING_HARD,
    OBSM_SPATIAL,
    OBSP_SPATIAL_CONNECTIVITIES,
)
from spatialaim.metrics.topology import local_spatial_purity, self_nhood_enrichment

logger = logging.getLogger(__name__)


K_SPATIAL = 6  # neighbours in the spatial kNN graph
N_PERM_SPATIAL = 100  # label shuffles for the null


def analyse_spatial_organization(
    adata_st: ad.AnnData,
    output_data_dir: Path,
    k: int = K_SPATIAL,
    n_perm: int = N_PERM_SPATIAL,
    seed: int = 0,
):
    """Write ``topology_metrics.json``: local purity with its shuffle test, the
    mean self neighbourhood-enrichment z-score, and per state its own self
    z-score (for the GUI). Undefined values are NaN / null."""
    spot_states = adata_st.obs[OBS_MAPPING_HARD].to_numpy()
    coords = np.asarray(adata_st.obsm[OBSM_SPATIAL])
    assert len(coords) == len(
        spot_states
    ), "Spatial organisation skipped: coordinates length mismatch "
    assert len(spot_states) > k, "too few spots"

    conn = _spatial_connectivities(adata_st, k)

    rng = np.random.default_rng(seed)
    obs_lsp = local_spatial_purity(spot_states, conn)
    local_purity = _permutation_test(
        obs_lsp, spot_states, lambda l: local_spatial_purity(l, conn), n_perm, rng
    )

    self_z = self_nhood_enrichment(
        spot_states, conn, n_perm, np.random.default_rng(seed)
    )
    nhood = (
        None
        if self_z is None
        else {"mean_self_zscore": _mean_self_z(self_z, spot_states)}
    )

    result = {
        "local_purity": local_purity,
        "nhood_enrichment": nhood,
        "per_state": {
            str(int(s)): {
                "self_zscore": (
                    float(self_z[s]) if self_z is not None else float("nan")
                )
            }
            for s in np.unique(spot_states)
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


def _permutation_test(
    observed: float,
    labels: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_perm: int,
    rng: np.random.Generator,
) -> dict:
    """Compare ``observed`` with ``metric_fn`` on ``n_perm`` shuffles of
    ``labels``: ``observed``, ``p_value`` (share of shuffles >= observed),
    ``z_score``, ``null_mean``, ``null_std``. NaN if ``observed`` or every
    shuffle is not finite."""
    _nan_result = {
        "observed": float(observed),
        "p_value": float("nan"),
        "z_score": float("nan"),
        "null_mean": float("nan"),
        "null_std": float("nan"),
    }
    if not np.isfinite(observed):
        return _nan_result
    perm = np.array([metric_fn(rng.permutation(labels)) for _ in range(n_perm)])
    valid = perm[np.isfinite(perm)]
    if len(valid) == 0:
        return _nan_result
    return {
        "observed": float(observed),
        "p_value": float((valid >= observed).mean()),
        "z_score": float((observed - valid.mean()) / (valid.std() + 1e-12)),
        "null_mean": float(valid.mean()),
        "null_std": float(valid.std()),
    }


def _mean_self_z(self_z: np.ndarray, spot_states: np.ndarray) -> float:
    """Mean self z-score over the mapped states (NaN if none is finite)."""
    present = self_z[np.unique(spot_states)]
    return float(np.nanmean(present)) if np.isfinite(present).any() else float("nan")
