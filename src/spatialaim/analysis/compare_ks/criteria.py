"""
The K-selection criteria: each is measured by two curves over K.
"""

from __future__ import annotations

from dataclasses import dataclass

# Suffix of a column's label-shuffle null, e.g. ``cossim_hard_raw_spot_null``.
NULL_SUFFIX = "_null"


@dataclass(frozen=True)
class Criterion:
    """One K-selection criterion: two measured curves combined into one score."""

    key: str  # column of the combined score in the score table
    label: str  # axis / card title
    short: str  # short label for the scatter-panel titles
    columns: tuple[str, str]  # the two ``k_comparison.csv`` columns
    curve_labels: tuple[str, str]  # legend labels for those columns
    unit: str  # y-axis title of the line card
    axis_from_zero: bool = False  # start this criterion's scatter axis at 0


CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        key="recon",
        label="Reconstruction cossim",
        short="Reconstruction",
        columns=("cossim_hard_raw_spot", "cossim_hard_raw_gene"),
        curve_labels=("raw · spot", "raw · gene"),
        unit="cosine similarity",
    ),
    # Both are chance-corrected (0 = random labels, 1 = no edge leaves its
    # state), so they share one scale across K and starts without rescaling.
    Criterion(
        key="spatial",
        label="Spatial organisation",
        short="Spatial org.",
        columns=("spatial_assort_celltype", "spatial_assort_global"),
        curve_labels=("per cell type (mean)", "global"),
        unit="assortativity",
    ),
    Criterion(
        key="coherence",
        label="Transcriptional coherence",
        short="Coherence",
        columns=("modularity_st_expression", "modularity_shared"),
        curve_labels=("ST expression (mapping)", "SC shared-gene (reference)"),
        unit="modularity",
        axis_from_zero=True,
    ),
)

CRITERION = {c.key: c for c in CRITERIA}

# The criterion-vs-criterion scatter panels (x, y).
SCATTER_PAIRS: tuple[tuple[str, str], ...] = (
    ("recon", "spatial"),
    ("recon", "coherence"),
    ("spatial", "coherence"),
)
