"""
Nearest-centroid mapper: each spot goes to its most cosine-similar state.
"""

import numpy as np

from .base import SpotStateMapper
from .confidence import top_margin_confidence


class NearestCentroidMapper(SpotStateMapper):
    """Parameter-free nearest-centroid mapping (one-hot P)."""

    eps: float = 1e-8
    name = "nearest_centroid"

    def map(self, start_cluster_to_state, k) -> tuple[np.ndarray, np.ndarray]:
        """Assign each spot to the state whose mean profile is most
        cosine-similar. Returns a one-hot P and the margin-based confidence
        (see :func:`top_margin_confidence`)."""
        Z_shared = self._spatial_data_matrix()
        M_shared = self._state_profiles(start_cluster_to_state, k)

        Zn = Z_shared / (np.linalg.norm(Z_shared, axis=1, keepdims=True) + self.eps)
        Mn = M_shared / (np.linalg.norm(M_shared, axis=1, keepdims=True) + self.eps)
        sim = Zn @ Mn.T  # spots x states
        spot_state = np.argmax(sim, axis=1)

        P = np.zeros((Z_shared.shape[0], M_shared.shape[0]), dtype=np.float32)
        P[np.arange(P.shape[0]), spot_state] = 1.0
        confidence = top_margin_confidence(1.0 - sim)  # cosine distance
        return P, confidence
