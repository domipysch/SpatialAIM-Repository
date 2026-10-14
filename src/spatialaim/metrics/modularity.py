"""
Modularity of a partition on a kNN graph.
"""

from __future__ import annotations
import logging
import numpy as np
from anndata import AnnData

logger = logging.getLogger(__name__)


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
