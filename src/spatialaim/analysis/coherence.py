"""
Transcriptional coherence of one K's mapping: modularities of the states on the
reference and ST expression graphs.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
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
    """Write ``modularity_metrics.json``: the modularity of the states on the
    reference's all-gene and shared-gene kNN graphs and on the ST expression
    kNN graph."""
    cell_states = adata_sc.obs[OBS_COMPUTED_STATE].astype(int).to_numpy()

    modularity_all = compute_modularity(adata_sc, cell_states)
    modularity_shared = compute_modularity(
        adata_sc, cell_states, obsp_key=OBSP_CONNECTIVITIES_SHARED_GENES
    )
    modularity_st_expression = _spatial_expression_modularity(adata_st)

    with open(output_data_dir / "modularity_metrics.json", "w") as f:
        json.dump(
            {
                "modularity_all": float(modularity_all),
                "modularity_shared": float(modularity_shared),
                "modularity_st_expression": float(modularity_st_expression),
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
