"""SpatialAIM (Annotation Independent Mapping): over-cluster an scRNA reference, agglomerate
the subclusters into K states, and map ST spots onto those states across a K sweep.
Public surface: ``SpatialAIMConfig``, ``MAPPING_CHOICES``, ``run``, ``SpotStateMapper``.

These names are exposed lazily (PEP 562) so that importing ``spatialaim``, or a
submodule such as ``spatialaim.reference_aligners.run_tangram`` from inside a reference
aligner's own conda env, does not pull in the heavy sweep/analysis stack
(scanpy, squidpy, ...), which is only present in ``spatialaim_env``. The heavy modules
load on first attribute access instead.
"""

from importlib import import_module
from typing import TYPE_CHECKING

_LAZY = {
    "SpatialAIMConfig": ("spatialaim.config", "SpatialAIMConfig"),
    "LINKAGE_METHODS": ("spatialaim.config", "LINKAGE_METHODS"),
    "MAPPING_CHOICES": ("spatialaim.config", "MAPPING_CHOICES"),
    "SpotStateMapper": ("spatialaim.mapping", "SpotStateMapper"),
    "run": ("spatialaim.sweep", "run"),
}

__all__ = [
    "SpatialAIMConfig",
    "LINKAGE_METHODS",
    "MAPPING_CHOICES",
    "run",
    "SpotStateMapper",
]


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(import_module(module_name), attr)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # let type checkers and IDEs see the real symbols
    from spatialaim.config import LINKAGE_METHODS, MAPPING_CHOICES, SpatialAIMConfig
    from spatialaim.mapping import SpotStateMapper
    from spatialaim.sweep import run
