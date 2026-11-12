"""
The index of all criteria. Every registered criterion is analysed for every K;
those in ``SELECTION_CRITERIA`` also select K.
"""

from __future__ import annotations
from .coherence import Coherence
from .criterion import Criterion
from .reconstruction import Reconstruction
from .shape_consistency import ShapeConsistency
from .topology import SpatialOrganisation

REGISTRY: dict[str, Criterion] = {
    c.key: c
    for c in (Reconstruction(), SpatialOrganisation(), Coherence(), ShapeConsistency())
}

# The keys of the criteria whose scores, and every combination of them, select K.
SELECTION_CRITERIA: tuple[str, ...] = ("recon", "spatial", "coherence")
