"""
Load one K's saved mapping and attach it to the sc / ST AnnData.
"""

from __future__ import annotations

import logging
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from spatialaim.adata_schema import (
    OBS_COMPUTED_STATE,
    OBS_MAPPING_HARD,
    OBS_START_CLUSTER,
    OBSM_MAPPING_SOFT,
)
from spatialaim.io import START_CLUSTER_TO_STATE_FILENAME

logger = logging.getLogger(__name__)


def load_start_cluster_to_state(run_dir: Path) -> np.ndarray:
    """The state of every start cluster, from ``start_cluster_to_state.csv``."""
    path = Path(run_dir) / START_CLUSTER_TO_STATE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Required mapping output missing: {path}")
    return pd.read_csv(path)["state"].to_numpy()


def infer_cell_to_state_cluster(adata_sc: AnnData, start_cluster_to_state: np.ndarray):
    """Set ``adata_sc.obs[OBS_COMPUTED_STATE]``: each reference cell's state,
    via its start cluster."""
    start_cluster_idx = adata_sc.obs[OBS_START_CLUSTER].astype(int).to_numpy()
    cell_states = start_cluster_to_state[start_cluster_idx]
    adata_sc.obs[OBS_COMPUTED_STATE] = pd.Categorical(cell_states.astype(str))


def load_spot_to_state_mapping_soft_and_hard(run_dir: Path, adata_st: AnnData) -> None:
    """Set ``adata_st.obsm[OBSM_MAPPING_SOFT]`` (the soft mapping, spots x
    states) and ``adata_st.obs[OBS_MAPPING_HARD]`` (each spot's most likely
    state) from ``spot_to_state_mapping_soft.h5ad``. Raises if its spots are
    not those of ``adata_st`` in the same order."""
    mapping_path = Path(run_dir) / "spot_to_state_mapping_soft.h5ad"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Required mapping output missing: {mapping_path}")

    mapping_ad = ad.read_h5ad(mapping_path)
    if not mapping_ad.obs_names.equals(adata_st.obs_names):
        raise ValueError(
            f"Spot order in {mapping_path} does not match adata_st.obs_names."
        )
    adata_st.obsm[OBSM_MAPPING_SOFT] = np.asarray(mapping_ad.X, dtype=np.float64)
    adata_st.obs[OBS_MAPPING_HARD] = adata_st.obsm[OBSM_MAPPING_SOFT].argmax(axis=1)
