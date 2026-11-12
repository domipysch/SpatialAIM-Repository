"""
Transcriptional coherence of one K's mapping: modularities of the states on the
reference and ST expression graphs.
"""

from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from anndata import AnnData
from spatialaim.adata_schema import (
    LAYER_LOGNORM,
    OBS_COMPUTED_STATE,
    OBS_MAPPING_HARD,
    OBSP_CONNECTIVITIES_SHARED_GENES,
    OBSP_ST_EXPR_CONNECTIVITIES,
)
from .criterion import Criterion, finite

logger = logging.getLogger(__name__)


class Coherence(Criterion):
    """Modularity of the states on the ST and reference expression graphs."""

    key = "coherence"
    label = "Transcriptional coherence"
    short = "Coherence"
    unit = "modularity"
    filename = "modularity_metrics.json"
    columns = ("modularity_st_expression", "modularity_shared")
    curve_labels = ("ST expression (mapping)", "SC shared-gene (reference)")
    axis_from_zero = True

    def analyse(self, adata_sc, adata_st, start_cluster_to_state) -> dict:
        """The modularity of the states on the reference's all-gene and
        shared-gene kNN graphs and on the ST expression kNN graph."""
        cell_states = adata_sc.obs[OBS_COMPUTED_STATE].astype(int).to_numpy()
        return {
            "modularity_all": compute_modularity(adata_sc, cell_states),
            "modularity_shared": compute_modularity(
                adata_sc, cell_states, obsp_key=OBSP_CONNECTIVITIES_SHARED_GENES
            ),
            "modularity_st_expression": _spatial_expression_modularity(adata_st),
        }

    def collect(self, data_dir: Path) -> dict[str, float]:
        """The shared-gene reference and the ST expression modularity."""
        modularity = self.load(data_dir)
        return {
            "modularity_shared": finite(modularity.get("modularity_shared")),
            "modularity_st_expression": finite(
                modularity.get("modularity_st_expression")
            ),
        }


def compute_modularity(
    adata: AnnData, labels: np.ndarray, obsp_key: str = "connectivities"
) -> float:
    """Modularity of ``labels`` on the kNN graph ``adata.obsp[obsp_key]``.
    NaN if igraph is not installed or the graph does not exist."""
    g = _igraph_from_obsp(adata, obsp_key)
    if g is None:
        return float("nan")
    return float(g.modularity(np.asarray(labels).tolist()))


def _igraph_from_obsp(adata: AnnData, obsp_key: str):
    """``adata.obsp[obsp_key]`` as an undirected igraph without duplicate
    edges or self-loops; ``None`` (with a warning) if igraph or the graph is
    missing."""
    try:
        import igraph as ig
    except ImportError:
        logger.warning("igraph not available — skipping modularity")
        return None

    if obsp_key not in adata.obsp:
        logger.warning("No precomputed neighbors graph — skipping modularity")
        return None

    A = adata.obsp[obsp_key].tocoo()
    edges = list(zip(A.row.tolist(), A.col.tolist()))
    g = ig.Graph(n=adata.n_obs, edges=edges, directed=False)
    g.simplify()
    return g


def _spatial_expression_modularity(adata_st: AnnData) -> float:
    """Modularity of the spots' states on a kNN graph of the ST expression
    (all genes, lognorm, PCA). The graph is built on the first K and kept in
    ``adata_st.obsp``."""
    if OBSP_ST_EXPR_CONNECTIVITIES not in adata_st.obsp:
        import scanpy as sc

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
