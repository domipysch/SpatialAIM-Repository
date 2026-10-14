"""
The post-mapping analysis of a mapping for one specific K.
"""

from __future__ import annotations
import logging
from pathlib import Path
from anndata import AnnData
from .coherence import analyse_modularities
from .loading import (
    infer_cell_to_state_cluster,
    load_spot_to_state_mapping_soft_and_hard,
    load_start_cluster_to_state,
)
from .reconstruction import analyse_reconstruction
from .shape_consistency import analyse_shape_consistency
from .spatial_organization import analyse_spatial_organization

logger = logging.getLogger(__name__)


def run_analysis(adata_sc: AnnData, adata_st: AnnData, run_dir: Path) -> None:
    """Analyse the mapping saved in ``run_dir`` (one K) and write the metrics
    to ``run_dir/analysis/data/``: spatial organisation, reconstruction,
    coherence and shape consistency."""
    run_dir = Path(run_dir)
    data_dir = run_dir / "analysis" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    start_cluster_to_state = load_start_cluster_to_state(run_dir)
    infer_cell_to_state_cluster(adata_sc, start_cluster_to_state)
    load_spot_to_state_mapping_soft_and_hard(run_dir, adata_st)

    analyse_spatial_organization(adata_st, data_dir)
    analyse_reconstruction(adata_sc, adata_st, start_cluster_to_state, data_dir)
    analyse_modularities(adata_sc, adata_st, data_dir)
    analyse_shape_consistency(adata_st, data_dir)
