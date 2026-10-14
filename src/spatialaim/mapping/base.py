"""
The ``SpotStateMapper`` interface every spot->state mapper implements.
"""

from abc import ABC, abstractmethod

import anndata as ad
import numpy as np

from spatialaim.adata_schema import UNS_SHARED_GENES
from spatialaim.utils import to_dense

from ..aggregation import assemble_state_profiles_shared_genes


class SpotStateMapper(ABC):
    """Maps ST spots onto the states of each K. Subclasses set ``name`` and
    implement ``map``.

    The sweep calls ``prepare`` once, then ``map`` for every K, then ``close``.
    """

    name: str
    adata_sc: ad.AnnData
    adata_st: ad.AnnData

    def prepare(self, adata_sc, adata_st, labels_by_k) -> None:
        """Store the pre-processed sc/ST data for ``map``. ``labels_by_k`` maps
        each K to its start-cluster -> state cut, for mappers that need all Ks
        up front. Overrides must call ``super().prepare(...)``."""
        self.adata_sc = adata_sc
        self.adata_st = adata_st

    @abstractmethod
    def map(self, start_cluster_to_state, k) -> tuple[np.ndarray, np.ndarray | None]:
        """Map the spots onto the ``k`` states of this cut.

        Returns ``(P, confidence)``: ``P`` is spots x states, ``confidence`` the
        per-spot confidence in [0, 1] (higher = more decisive), or ``None`` if
        the mapper has none.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release resources held for the sweep (e.g. a worker process). Called
        once after the K-loop; does nothing by default."""

    # Helpers for subclasses
    def _spatial_data_matrix(self) -> np.ndarray:
        """The ST counts on the shared genes, dense float32 (spots x genes)."""
        z_shared = to_dense(self.adata_st[:, self.adata_sc.uns[UNS_SHARED_GENES]])
        return np.asarray(z_shared, dtype=np.float32)

    def _state_profiles(self, start_cluster_to_state, k) -> np.ndarray:
        """Mean raw expression per state (k x shared genes)."""
        return assemble_state_profiles_shared_genes(
            start_cluster_to_state, k, self.adata_sc
        )
