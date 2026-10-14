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
    scale_to_max: bool = False  # divide each curve by its max over K first
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
    # The two z-scores differ in scale (tens vs. hundreds), hence scale_to_max.
    Criterion(
        key="spatial",
        label="Spatial organisation",
        short="Spatial org.",
        columns=("nhood_mean_self_zscore", "local_purity_zscore"),
        curve_labels=("nhood enrichment z", "local purity z"),
        unit="z-score normalized",
        scale_to_max=True,
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
