"""
Transcriptional coherence of one K's mapping: modularities of the states on the
reference and ST expression graphs, and the per-state coherence for the GUI.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from spatialaim.adata_schema import (
    LAYER_LOGNORM,
    OBS_COMPUTED_STATE,
    OBS_MAPPING_HARD,
    OBSP_CONNECTIVITIES_SHARED_GENES,
    OBSP_ST_EXPR_CONNECTIVITIES,
)
from spatialaim.metrics.modularity import compute_modularity

logger = logging.getLogger(__name__)


def analyse_modularities(
    adata_sc: AnnData,
    adata_st: AnnData,
    output_data_dir: Path,
):
    """Write ``modularity_metrics.json`` and ``coherence_per_spot.csv``.

    The modularity of the states on the reference's all-gene and shared-gene
    kNN graphs and on the ST expression kNN graph, plus per state the chance
    level of its coherence (see :func:`_per_state_coherence`).
    """
    cell_states = adata_sc.obs[OBS_COMPUTED_STATE].astype(int).to_numpy()

    modularity_all = compute_modularity(adata_sc, cell_states)
    modularity_shared = compute_modularity(
        adata_sc, cell_states, obsp_key=OBSP_CONNECTIVITIES_SHARED_GENES
    )
    modularity_st_expression = _spatial_expression_modularity(adata_st)
    per_state = _per_state_coherence(adata_st, cell_states, output_data_dir)

    with open(output_data_dir / "modularity_metrics.json", "w") as f:
        json.dump(
            {
                "modularity_all": float(modularity_all),
                "modularity_shared": float(modularity_shared),
                "modularity_st_expression": float(modularity_st_expression),
                "per_state": per_state,
            },
            f,
            indent=4,
        )


def _spatial_expression_modularity(adata_st: AnnData) -> float:
    """Modularity of the spots' states on a kNN graph of the ST expression
    (all genes, lognorm, PCA). The graph is built on the first K and kept in
    ``adata_st.obsp``."""
    if OBSP_ST_EXPR_CONNECTIVITIES not in adata_st.obsp:
        # Built on a copy so the PCA and neighbour keys stay off adata_st.
        st_expr = AnnData(
            X=adata_st.layers[LAYER_LOGNORM].copy(),
            obs=adata_st.obs[[]].copy(),
            var=adata_st.var[[]].copy(),
        )
        n = min(30, st_expr.n_obs - 1, st_expr.n_vars - 1)
        sc.pp.pca(st_expr, n_comps=n)
        sc.pp.neighbors(st_expr)
        adata_st.obsp[OBSP_ST_EXPR_CONNECTIVITIES] = st_expr.obsp["connectivities"]

    graph = AnnData(
        X=np.zeros((adata_st.n_obs, 1), dtype=np.float32),
        obsp={"connectivities": adata_st.obsp[OBSP_ST_EXPR_CONNECTIVITIES]},
    )
    hard_labels = adata_st.obs[OBS_MAPPING_HARD].astype(int).to_numpy()
    return compute_modularity(graph, hard_labels)


def _per_state_coherence(
    adata_st: AnnData, cell_states: np.ndarray, output_data_dir: Path
) -> dict:
    """Per spot, the share of its ST-expression neighbours mapped to its own
    state (written to ``coherence_per_spot.csv``). Returns per state its chance
    level, the state's share of spots; every reference state gets an entry,
    also one without spots."""
    hard = adata_st.obs[OBS_MAPPING_HARD].astype(int).to_numpy()
    share = _same_label_share(hard, adata_st.obsp[OBSP_ST_EXPR_CONNECTIVITIES])
    pd.DataFrame({"id": adata_st.obs_names, "same_label_share": share}).to_csv(
        output_data_dir / "coherence_per_spot.csv", index=False
    )
    return {
        str(int(s)): {"chance": float(np.mean(hard == s))}
        for s in np.union1d(np.unique(hard), np.unique(cell_states))
    }


def _same_label_share(labels: np.ndarray, connectivities) -> np.ndarray:
    """Per node, the share of its graph neighbours with its own label (NaN
    without neighbours). Neighbours are the stored entries of its row, weights
    ignored, itself excluded."""
    A = connectivities.tocoo()
    off = A.row != A.col
    row, col = A.row[off], A.col[off]
    n = A.shape[0]
    same = np.bincount(
        row, weights=(labels[row] == labels[col]).astype(np.float64), minlength=n
    ).astype(np.int64)
    degree = np.bincount(row, minlength=n).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(degree > 0, same / degree, np.nan)
