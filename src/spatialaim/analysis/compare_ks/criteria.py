"""
The criteria that select K, in the shape the GUI reads them.
The index of all criteria is :mod:`spatialaim.analysis.registry`.
"""

from __future__ import annotations
from itertools import combinations
from ..criterion import Criterion
from ..registry import REGISTRY, SELECTION_CRITERIA

# Suffix of a column's label-shuffle null, e.g. ``cossim_hard_raw_spot_null``.
NULL_SUFFIX = "_null"

CRITERIA: tuple[Criterion, ...] = tuple(REGISTRY[k] for k in SELECTION_CRITERIA)

CRITERION: dict[str, Criterion] = REGISTRY

# The criterion-vs-criterion scatter panels (x, y).
SCATTER_PAIRS: tuple[tuple[str, str], ...] = tuple(combinations(SELECTION_CRITERIA, 2))
