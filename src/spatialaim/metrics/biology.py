"""Graph modularity of a partition of the reference / ST KNN graphs."""

from __future__ import annotations

import logging

import numpy as np
from anndata import AnnData

logger = logging.getLogger(__name__)


def _igraph_from_obsp(adata: AnnData, obsp_key: str):
    """The undirected simple igraph of ``adata.obsp[obsp_key]``, or ``None``.

    ``None`` (with a warning) when igraph is not installed or the graph has not
    been computed, so every caller can degrade to NaN.
    """
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


def compute_modularity(
    adata: AnnData, labels: np.ndarray, obsp_key: str = "connectivities"
) -> float:
    """Modularity of ``labels`` on the precomputed KNN graph in
    ``adata.obsp[obsp_key]``. Requires igraph; returns NaN if igraph or the graph
    is unavailable."""
    g = _igraph_from_obsp(adata, obsp_key)
    if g is None:
        return float("nan")
    return float(g.modularity(np.asarray(labels).tolist()))
