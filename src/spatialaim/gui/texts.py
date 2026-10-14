"""
User-facing texts used in more than one place: the mapper labels and the help
behind each report card's "?".
"""

from __future__ import annotations

from spatialaim.mapping.confidence import N_TOP_STATES

# Display-only labels for the mapper checkboxes; the keys stay the raw names.
MAPPER_LABELS = {
    "nearest_centroid": "nearest_centroid (default)",
    "wann": "wann (preserve cell type fractions)",
}

# Card help texts, opened by each report card's "?". The Mapping-confidence one is
# per method and built by confidence_caption instead.
HELP_FRACTIONS = """
For each computed cell type you see two bars:

- **Reference (solid)**: the fraction of reference cells belonging to that cell type.
  (It follows from the tree cut alone, so it is the same for every mapping
  method.)
- **Spatial (hatched)**: the fraction of spatial cells this method assigned to that
  cell type.

The two need not match: the reference and the tissue section need not contain the
same mix of cells.
"""

HELP_RECONSTRUCTION = """
Each spatial cell's gene expression is predicted from the cell type profiles it
was mapped onto, and compared against the measured ST expression of that same
spatial cell over the shared genes, by cosine similarity (0–1, higher is better).

There are two panels: gene-wise and cell-wise cosine similarity. In each panel
there are four boxes:

- **soft** predicts gene expression from the original spatial cell × cell-types
  probability matrix, so a spatial cell is a weighted blend of profiles; **hard**
  predicts from the single winning cell type's profile alone.
- **raw** compares raw counts; **norm** compares log-normalised expression
  (total-count normalised to 1e4, then log1p, over the shared genes).
"""

HELP_SPATIAL_ORG = """
Quantify whether the mapped cell types form coherent tissue structure.

Start with a spatial 6nn-graph: connect each spatial cell to its 6 nearest
neighbours. Then compute:

- **Local purity**: the fraction of links where two neighbours carry the same
  cell type (z-scored against random null permutations).
- **Self neighbourhood enrichment**: mean over all cell types of the number of
  edges connecting two spatial cells of that cell type (z-scored against null
  permutations).

For both z-scores, 0 means the mapping is spatially no better arranged than
random relabelling; clearly positive means it forms spatial domains.
"""

HELP_SHAPE = """
For every segmented spatial cell in the mask, we compute:

- **Area**: the cell's size (µm² when the mask's pixel size is known, else pixels;
  log scale).
- **Eccentricity**: how elongated it is — 0 for a circle, towards 1 for a line.
- **Solidity**: area / area of its convex hull — 1 for a convex cell, lower for a
  dented or lobed outline, eg. a star-shape.

**Shape consistency** sums this up for the mapping: the three features are
z-scored over all segmented cells, each cell type's shape variance is
computed and averaged over the cell types (each type with at least two cells counts
equally), and the score is 1 − that mean variance / the variance of all cells.
1 = all cells of each type share one shape, 0 = the types are as varied as the
whole tissue (chance level).
"""

_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def confidence_caption(mapper: str) -> str:
    """A short, per-method explanation of how each spot's confidence is defined,
    shown in the "Mapping confidence per spot" card's help modal.

    The reference aligners write no confidence of their own, so their card falls
    back to the soft mapping they return; the caption says so rather than letting
    the two quantities look interchangeable.
    """
    if mapper == "nearest_centroid":
        dn = f"d{str(N_TOP_STATES).translate(_SUBSCRIPT)}"  # e.g. "d₄"
        return (
            "Each spatial cell is assigned to its most cosine-similar cell-type centroid; "
            "its "
            f"confidence is the winner's *relative margin* over the top-"
            f"{N_TOP_STATES} rivals in cosine-distance space — for the "
            f"{N_TOP_STATES} smallest distances d₁ ≤ … ≤ {dn}, "
            "1 − d₁ ⁄ mean(d₂…)."
        )
    if mapper == "wann":
        return (
            "Each spatial cell gets a reliability-weighted soft vote over the cell types; its "
            "confidence is how one-hot that vote is — 1 (normalised Shannon entropy)."
        )
    return (
        f"**{MAPPER_LABELS.get(mapper, mapper)}** returns a soft cells × cell-types "
        "mapping and no confidence of its own, so what is shown is the **maximum "
        "value per cell in that returned soft mapping**: how much of a cell's weight "
        "the winning cell type holds."
    )
