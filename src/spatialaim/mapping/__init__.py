"""
Spot->state mappers: the ``SpotStateMapper`` interface and its implementations.
"""

from .base import SpotStateMapper
from .nearest_centroid import NearestCentroidMapper
from .wann import WANNMapper
from .reference import ReferenceMapper

__all__ = [
    "SpotStateMapper",
    "NearestCentroidMapper",
    "WANNMapper",
    "ReferenceMapper",
]
