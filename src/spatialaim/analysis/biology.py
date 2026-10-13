"""Computed-state modularity and per-state transcriptional coherence for the
post-mapping analysis."""

from __future__ import annotations

import logging
import json
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
from spatialaim.metrics.biology import compute_modularity
from spatialaim.metrics.topology import same_label_neighbours

logger = logging.getLogger(__name__)


def _spatial_expression_modularity(adata_st: AnnData) -> float:
    """Modularity of the hard spot->state labels on the ST expression KNN graph.

    Measures transcriptional coherence of the mapping within the spatial data:
    builds an expression KNN graph over all genes of ``adata_st`` (PCA of the
    lognorm layer + scanpy neighbors, matching the reference all-genes graph)
    and computes the modularity of ``adata_st.obs[OBS_MAPPING_HARD]`` on it.

    Requires: adata_st.layers[LAYER_LOGNORM], adata_st.obs[OBS_MAPPING_HARD].
    """
    # The expression KNN graph is K-independent (only the hard labels change per
    # K), so build it once and cache the connectivities on adata_st.
    if OBSP_ST_EXPR_CONNECTIVITIES not in adata_st.obsp:
        # Throwaway AnnData so the PCA / neighbor graph never touches adata_st.
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


def analyse_modularities(
    adata_sc: AnnData,
    adata_st: AnnData,
    output_data_dir: Path,
):
    """Modularity of the computed-state partition on the reference KNN graphs.

    Also measures transcriptional coherence of the mapping within the spatial
    data via ``_spatial_expression_modularity`` (hard spot->state labels on the
    ST expression KNN graph over all genes), and per state via
    ``_per_state_coherence`` (also writes coherence_per_spot.csv).

    Requires: adata_sc.obs[OBS_COMPUTED_STATE],
        adata_sc.obsp[OBSP_CONNECTIVITIES_SHARED_GENES],
        adata_st.layers[LAYER_LOGNORM], adata_st.obs[OBS_MAPPING_HARD].
    Writes modularity_metrics.json under output_data_dir.
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


def _same_label_share(labels: np.ndarray, connectivities) -> np.ndarray:
    """Per node, the share of its KNN-graph neighbours that carry its label
    (NaN for a node without neighbours)."""
    same, degree = same_label_neighbours(labels, connectivities)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(degree > 0, same / degree, np.nan)


def _per_state_coherence(
    adata_st: AnnData, cell_states: np.ndarray, output_data_dir: Path
) -> dict:
    """Transcriptional coherence per state: for every spot, the share of its
    neighbours on the ST expression KNN graph (the graph of
    ``modularity_st_expression``) that the mapping gave the same state, written
    per spot to ``coherence_per_spot.csv``, plus per state the chance level of
    that share (the state's share of spots). Each link counts once, as in
    ``compute_modularity``. Every state of the reference (``cell_states``) gets
    an entry, also one the mapping assigned no spots.

    Requires: adata_st.obsp[OBSP_ST_EXPR_CONNECTIVITIES] (built by
    ``_spatial_expression_modularity``), adata_st.obs[OBS_MAPPING_HARD].
    """
    hard = adata_st.obs[OBS_MAPPING_HARD].astype(int).to_numpy()
    share = _same_label_share(hard, adata_st.obsp[OBSP_ST_EXPR_CONNECTIVITIES])
    pd.DataFrame({"id": adata_st.obs_names, "same_label_share": share}).to_csv(
        output_data_dir / "coherence_per_spot.csv", index=False
    )

    return {
        str(int(s)): {"chance": float(np.mean(hard == s))}
        for s in np.union1d(np.unique(hard), np.unique(cell_states))
    }
