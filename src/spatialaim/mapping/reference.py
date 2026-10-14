"""
Reference mapper: runs an external aligner (Tangram / TACCO / DOT) in its own
conda env for every K.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from spatialaim.adata_schema import OBS_START_CLUSTER, OBSM_SPATIAL, UNS_SHARED_GENES
from spatialaim.reference_aligners.registry import REFERENCE_ALIGNERS, run_aligner

from .base import SpotStateMapper

logger = logging.getLogger(__name__)


class ReferenceMapper(SpotStateMapper):
    """Maps the spots with Tangram, TACCO or DOT.

    The aligners map spots onto cell types, so at each K the reference cells
    are labelled with their state at that K and the aligner's output columns
    are the states. Each K is one ``conda run`` of the aligner on a
    shared-gene sc/ST pair written once by ``prepare``.
    """

    def __init__(self, reference_method: str = "tangram") -> None:
        if reference_method not in REFERENCE_ALIGNERS:
            raise ValueError(
                f"reference_method must be one of {tuple(REFERENCE_ALIGNERS)}, "
                f"got {reference_method!r}"
            )
        self.reference_method = reference_method
        # Named after the aligner, so its results and config.yaml share one
        # output folder and the three aligners do not overwrite each other.
        self.name = reference_method
        self._prepared = False

    def prepare(self, adata_sc, adata_st, labels_by_k) -> None:
        """Write the shared-gene sc and ST files the aligner reads, to a
        temporary folder that lives as long as this mapper.

        The sc file has one categorical obs column per K (``state_k<kkk>``)
        with each cell's state, so no file is rewritten during the sweep.
        """
        super().prepare(adata_sc, adata_st, labels_by_k)
        shared = list(adata_sc.uns[UNS_SHARED_GENES])
        self._tmpdir = tempfile.TemporaryDirectory(prefix="aim_reference_")
        self._workdir = Path(self._tmpdir.name)
        self._sc_path = self._workdir / "sc_ref.h5ad"
        self._st_path = self._workdir / "st_ref.h5ad"

        # sc: each cell's state at every K
        start_cluster = adata_sc.obs[OBS_START_CLUSTER].astype(int).to_numpy()
        sc_obs = pd.DataFrame(index=adata_sc.obs_names)
        for k, labels_k in labels_by_k.items():
            cell_states = np.asarray(labels_k)[start_cluster]
            sc_obs[self._state_key(k)] = pd.Categorical(cell_states.astype(str))
        ad.AnnData(
            X=adata_sc[:, shared].X.copy(),
            obs=sc_obs,
            var=pd.DataFrame(index=shared),
        ).write_h5ad(self._sc_path)

        # ST: with the spatial coordinates, which DOT uses
        st_obsm = {}
        if OBSM_SPATIAL in adata_st.obsm:
            st_obsm[OBSM_SPATIAL] = np.asarray(adata_st.obsm[OBSM_SPATIAL])
        ad.AnnData(
            X=adata_st[:, shared].X.copy(),
            obs=pd.DataFrame(index=adata_st.obs_names),
            var=pd.DataFrame(index=shared),
            obsm=st_obsm or None,
        ).write_h5ad(self._st_path)

        self._st_obs_names = [str(s) for s in adata_st.obs_names]
        self._prepared = True
        logger.info(
            "ReferenceMapper[%s] prepared shared-gene inputs (%d genes, %d K-levels) at %s",
            self.reference_method,
            len(shared),
            len(labels_by_k),
            self._workdir,
        )

    def map(self, start_cluster_to_state, k) -> tuple[np.ndarray, None]:
        """Run the aligner for this K. Returns ``(P, None)``: the aligners give
        no confidence. For K = 1 every spot simply gets the one state."""
        if not self._prepared:
            raise RuntimeError(
                "ReferenceMapper.prepare(...) must run before map(); the sweep "
                "calls it once before the K-loop."
            )
        k = int(k)
        n_spots = int(self.adata_st.n_obs)
        if k < 2:
            return np.ones((n_spots, 1), dtype=np.float32), None

        out_dir = self._workdir / self._state_key(k)
        logger.info("ReferenceMapper[%s] K=%d", self.reference_method, k)
        out_path = run_aligner(
            self.reference_method,
            self._sc_path,
            self._st_path,
            out_dir,
            self._state_key(k),
        )
        return self._read_mapping(out_path, k), None

    @staticmethod
    def _state_key(k: int) -> str:
        """The sc obs column with the cells' states at ``k``."""
        return f"state_k{k:03d}"

    def _read_mapping(self, path: Path, k: int) -> np.ndarray:
        """The aligner's mapping as spots x K, in ST spot order; states the
        aligner did not output get zero columns. Values are kept as the aligner
        wrote them (not re-normalised)."""
        if not path.exists():
            raise RuntimeError(f"{self.reference_method} produced no mapping at {path}")
        mp = ad.read_h5ad(path)
        X = mp.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        frame = pd.DataFrame(
            X, index=mp.obs_names.astype(str), columns=mp.var_names.astype(str)
        )
        frame = frame.reindex(
            index=self._st_obs_names,
            columns=[str(i) for i in range(k)],
            fill_value=0.0,
        )
        return frame.to_numpy(dtype=np.float32)
