"""
The post-mapping analysis of a mapping for one specific K.
"""

from __future__ import annotations
import logging
from pathlib import Path
from anndata import AnnData
from .loading import (
    infer_cell_to_state_cluster,
    load_spot_to_state_mapping_soft_and_hard,
    load_start_cluster_to_state,
)
from .registry import REGISTRY

logger = logging.getLogger(__name__)


def run_analysis(adata_sc: AnnData, adata_st: AnnData, run_dir: Path) -> None:
    """Analyse the mapping saved in ``run_dir`` (one K) with every registered
    criterion and write the metrics to ``run_dir/analysis/``."""
    run_dir = Path(run_dir)
    data_dir = run_dir / "analysis"
    data_dir.mkdir(parents=True, exist_ok=True)

    start_cluster_to_state = load_start_cluster_to_state(run_dir)
    infer_cell_to_state_cluster(adata_sc, start_cluster_to_state)
    load_spot_to_state_mapping_soft_and_hard(run_dir, adata_st)

    for criterion in REGISTRY.values():
        criterion.run(adata_sc, adata_st, start_cluster_to_state, data_dir)
