"""
Spatial organisation of one K's mapping: the assortativity of the spots' states
on the spatial kNN graph, overall and per state (chance-corrected, no shuffles).

Both are chance-corrected like Cohen's kappa: (observed − chance) / (1 − chance),
1 = no edge leaves its state, 0 = as mixed as randomly placed labels.

Global assortativity: Newman's assortativity coefficient for discrete labels
(Newman 2003, Phys. Rev. E 67, 026126, eq. 2). Per-state assortativity: the
same, adapted per state.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
import anndata as ad
import numpy as np
from spatialaim.adata_schema import (
    OBS_MAPPING_HARD,
    OBSM_SPATIAL,
    OBSP_SPATIAL_CONNECTIVITIES,
)
from .criterion import Criterion, finite

logger = logging.getLogger(__name__)


K_SPATIAL = 6  # neighbours in the spatial kNN graph


class SpatialOrganisation(Criterion):
    """Assortativity of the spots' states on the spatial kNN graph."""

    # Both curves are chance-corrected (0 = random labels, 1 = no edge leaves
    # its state), so they share one scale across K and starts without rescaling.
    key = "spatial"
    label = "Spatial organisation"
    short = "Spatial org."
    unit = "assortativity"
    filename = "topology_metrics.json"
    columns = ("spatial_assort_celltype", "spatial_assort_global")
    curve_labels = ("per cell type (mean)", "global")

    k: int = K_SPATIAL

    def analyse(self, adata_sc, adata_st, start_cluster_to_state) -> dict:
        """The global assortativity, the mean per-state assortativity, and per
        mapped state its assortativity (for the GUI's cell-type modal).
        Undefined values are NaN."""
        spot_states = np.asarray(adata_st.obs[OBS_MAPPING_HARD].to_numpy(), dtype=int)
        coords = np.asarray(adata_st.obsm[OBSM_SPATIAL])
        assert len(coords) == len(
            spot_states
        ), "Spatial organisation skipped: coordinates length mismatch "
        assert len(spot_states) > self.k, "too few spots"

        a = assortativity(spot_states, _spatial_connectivities(adata_st, self.k))
        if a is None:
            return {
                "spatial_assort_global": float("nan"),
                "spatial_assort_celltype_mean": float("nan"),
                "per_state": {},
            }
        return {
            "spatial_assort_global": a.value,
            "spatial_assort_celltype_mean": a.mean_state,
            "per_state": {
                str(int(s)): float(a.per_state[s]) for s in np.unique(spot_states)
            },
        }

    def collect(self, data_dir: Path) -> dict[str, float]:
        """The mean per-state and the global assortativity. Outputs written
        before the flat layout nest them (``{"value"}`` / ``{"mean"}``); those
        holding only the old z-scores give NaN."""
        topology = self.load(data_dir)
        overall = topology.get("spatial_assort_global")
        if isinstance(overall, dict):
            overall = overall.get("value")
        per_state = topology.get(
            "spatial_assort_celltype_mean",
            (topology.get("spatial_assort_celltype") or {}).get("mean"),
        )
        return {
            "spatial_assort_celltype": finite(per_state),
            "spatial_assort_global": finite(overall),
        }


@dataclass(frozen=True)
class Assortativity:
    """``value``: the global assortativity r; ``per_state``: κ_s, indexed by
    state (NaN for a state without spots)."""

    value: float
    per_state: np.ndarray

    @property
    def mean_state(self) -> float:
        """Mean κ_s over the states it is defined for (NaN if none)."""
        finite_states = self.per_state[np.isfinite(self.per_state)]
        return float(finite_states.mean()) if finite_states.size else float("nan")


def assortativity(labels: np.ndarray, connectivities) -> Assortativity | None:
    """Assortativity of the integer state ``labels`` on the graph's edges
    (unweighted; each stored edge counts once, so a directed kNN graph works).

    With e_st the share of edges from state s to state t, a_s = Σ_t e_st and
    b_s = Σ_t e_ts:

    - global: r = (Σ e_ss − Σ a_s b_s) / (1 − Σ a_s b_s)
    - per state: κ_s = (p_s − b_s) / (1 − b_s), p_s = e_ss / a_s

    r is the mean of the κ_s weighted by a_s (1 − b_s). Returns ``None`` for a
    graph without edges; values are NaN where undefined (a single state).
    """
    A = connectivities.tocoo()
    if A.nnz == 0:
        return None
    n_states = int(labels.max()) + 1
    src, dst = labels[A.row], labels[A.col]
    same = src == dst
    n_edges = float(A.nnz)
    a = np.bincount(src, minlength=n_states) / n_edges
    b = np.bincount(dst, minlength=n_states) / n_edges
    e_same = np.bincount(src[same], minlength=n_states) / n_edges

    observed = float(e_same.sum())
    chance = float(np.dot(a, b))
    with np.errstate(invalid="ignore", divide="ignore"):
        value = (observed - chance) / (1.0 - chance) if chance < 1.0 else np.nan
        purity = np.where(a > 0, e_same / a, np.nan)
        kappa = np.where((a > 0) & (b < 1.0), (purity - b) / (1.0 - b), np.nan)

    return Assortativity(value=float(value), per_state=kappa)


def _spatial_connectivities(adata_st: ad.AnnData, k: int):
    """The spatial kNN graph of the spots (squidpy). Built on the first K and
    kept in ``adata_st.obsp``, since the coordinates do not change."""
    if OBSP_SPATIAL_CONNECTIVITIES not in adata_st.obsp:
        import squidpy as sq

        sq.gr.spatial_neighbors_knn(adata_st, n_neighs=k)
    return adata_st.obsp[OBSP_SPATIAL_CONNECTIVITIES]
