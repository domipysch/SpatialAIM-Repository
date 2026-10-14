"""
Run configuration for SpatialAIM
"""

from __future__ import annotations
from dataclasses import dataclass
from spatialaim.reference_aligners.registry import REFERENCE_ALIGNERS
from .mapping import (
    SpotStateMapper,
    NearestCentroidMapper,
    ReferenceMapper,
    WANNMapper,
)

_INPROCESS_METHODS = ("nearest_centroid", "wann")
_REFERENCE_METHODS = tuple(REFERENCE_ALIGNERS)
MAPPING_CHOICES = _INPROCESS_METHODS + _REFERENCE_METHODS

LINKAGE_METHODS = ("average", "ward")


@dataclass
class SpatialAIMConfig:

    mapping: str = "nearest_centroid"
    leiden_resolution: float = 3.0
    # Agglomeration-tree linkage: "average" (UPGMA) or "ward" (balanced states).
    linkage_method: str = LINKAGE_METHODS[0]
    # Start clusters (what the agglomeration tree is built over): None = Leiden
    # over-clustering; an obs column name = use that annotation instead, in which
    # case no all-gene Leiden is computed and K sweeps from the number of types.
    start_from_annotation: str | None = None
    # K sweep range
    k_min: int | None = None
    k_max: int | None = None
    k_step: int = 1

    def build_mapper(self) -> SpotStateMapper:
        """Build the ``SpotStateMapper`` named by ``self.mapping``.

        The mapper modules are imported here, lazily, so merely reading
        ``SpatialAIMConfig``/``MAPPING_CHOICES`` does not import them.
        """

        inprocess: dict[str, type[SpotStateMapper]] = {
            NearestCentroidMapper.name: NearestCentroidMapper,
            WANNMapper.name: WANNMapper,
        }
        assert tuple(inprocess) == _INPROCESS_METHODS, (
            "MAPPING_CHOICES out of sync with mapper class names: "
            f"{tuple(inprocess)} != {_INPROCESS_METHODS}"
        )

        if self.mapping in _REFERENCE_METHODS:
            return ReferenceMapper(reference_method=self.mapping)
        if self.mapping not in inprocess:
            raise ValueError(
                f"mapping must be one of {MAPPING_CHOICES}, got {self.mapping!r}"
            )
        return inprocess[self.mapping]()
