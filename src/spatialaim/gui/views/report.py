"""
A mapper tab's report: a grid of cards (cell-type fractions, reconstruction,
mapping confidence, spatial organisation and, with a segmentation, shape
features), each with its "?" help.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import streamlit as st

from spatialaim.gui import data_access
from spatialaim.gui.loaders import load_scaffold, load_soft, shape_features
from spatialaim.gui.render.report import (
    render_confidence_figure,
    render_fractions_figure,
    render_onehot_figure,
    render_reconstruction_figure,
)
from spatialaim.gui.render.shape import render_shape_features_figure
from spatialaim.gui.texts import (
    HELP_FRACTIONS,
    HELP_RECONSTRUCTION,
    HELP_SHAPE,
    HELP_SPATIAL_ORG,
    confidence_caption,
)
from spatialaim.gui.views.cards import Card, plot_card, render_card_grid, render_metrics

# Which fields the Spatial-organisation card shows per metric group, and in what
# order — the z-score first, since it is the number that is comparable across K;
# the raw fraction and the null it was scored against come after. Fields left out
# here (nhood_enrichment's n_states) are not displayed; the analysis still writes
# them. A group missing from this map is shown whole, in its stored order.
_TOPOLOGY_FIELDS: dict[str, tuple[str, ...]] = {
    "local_purity": ("z_score", "p_value", "observed", "null_mean", "null_std"),
    "nhood_enrichment": ("mean_self_zscore",),
}


def topology_groups(metrics: dict) -> dict[str, dict]:
    """The nested metric groups of ``topology_metrics.json``, shaped for display.

    Drops any top-level scalar and, for a known group,
    keeps only the fields in ``_TOPOLOGY_FIELDS`` in that order (skipping
    absent ones). ``nhood_enrichment`` is null when there are too few mapped
    states and then drops out. ``per_state`` feeds the cell-type modal, not
    this card.
    """
    out: dict[str, dict] = {}
    for group, vals in metrics.items():
        if not isinstance(vals, dict) or group == "per_state":
            continue
        order = _TOPOLOGY_FIELDS.get(group)
        out[group] = vals if order is None else {f: vals[f] for f in order if f in vals}
    return out


def render_report(mapper: str, root: Path, k: int, args: argparse.Namespace) -> None:
    """The per-mapper report as a grid of compact cards, each an interactive
    Plotly figure or a small metrics table (no matplotlib)."""
    P, hard, confidence = load_soft(str(root), k)

    # The fractions card is simply skipped if the scaffold can't be built.
    adata_sc = load_scaffold(args)

    def _confidence() -> None:
        # nearest_centroid and wann each define their own per-spot confidence and
        # write it out; the reference aligners do not, so their card shows the max
        # per spot of the soft mapping they return instead (see the caption).
        fig = (
            render_confidence_figure(confidence)
            if confidence is not None
            else render_onehot_figure(P.max(axis=1))
        )
        # No caption: the per-method explanation is what the card's ``?`` opens.
        plot_card(
            fig,
            key=f"card_conf_{mapper}",
            stem=f"{mapper}_k{k:03d}_confidence",
        )

    def _spatial_org() -> None:
        metrics = data_access.load_data_json(root, k, "topology_metrics.json")
        if not metrics:
            st.info("topology_metrics.json not found for this K.")
            return
        # Drop the top-level scalar table; keep the metric groups (local purity,
        # neighbourhood enrichment), each shaped by _TOPOLOGY_FIELDS.
        nested = topology_groups(metrics)
        if nested:
            render_metrics(st, nested)
        else:
            st.info("No spatial-organisation sub-metrics for this K.")

    # Assemble the cards. Only include those that have data; each carries the text
    # its ``?`` opens.
    # No Modularity card: the one mapping-dependent value it held now rides in the
    # spatial panel's subplot title (see ``render.headline._mod_suffix``).
    cards: list[Card] = []
    if adata_sc is not None:
        cards.append(
            Card(
                "Cell type fractions",
                lambda: plot_card(
                    render_fractions_figure(adata_sc, root, k, hard),
                    key=f"card_frac_{mapper}",
                    stem=f"{mapper}_k{k:03d}_fractions",
                ),
                help=HELP_FRACTIONS,
                key=f"frac_{mapper}",
            )
        )
    cossim = data_access.load_cossim_distributions(root, k)
    if cossim:
        cards.append(
            Card(
                "Reconstruction Cosine Similarity",
                lambda: plot_card(
                    render_reconstruction_figure(cossim),
                    key=f"card_recon_{mapper}",
                    stem=f"{mapper}_k{k:03d}_reconstruction",
                ),
                help=HELP_RECONSTRUCTION,
                key=f"recon_{mapper}",
            )
        )
    cards.append(
        Card(
            "Mapping confidence per cell",
            _confidence,
            help=confidence_caption(mapper),
            key=f"conf_{mapper}",
        )
    )
    cards.append(
        Card(
            "Spatial organisation",
            _spatial_org,
            help=HELP_SPATIAL_ORG,
            key=f"topo_{mapper}",
        )
    )

    # Only for a sweep run with a segmentation mask (it wrote cell_shape_features.csv).
    features = shape_features(root)
    if features is not None:

        def _shape() -> None:
            metrics = (
                data_access.load_data_json(root, k, "shape_consistency_metrics.json")
                or {}
            )
            score = metrics.get("shape_consistency", float("nan"))
            st.markdown(
                f"Shape consistency: **{score:.3f}** "
                f"({metrics.get('n_cells_used', 0):,} segmented cells)"
                if score == score
                else "Shape consistency: not available for this K"
            )
            plot_card(
                render_shape_features_figure(root, k, hard, features),
                key=f"card_shape_{mapper}",
                stem=f"{mapper}_k{k:03d}_shape_features",
            )

        cards.append(
            Card(
                "Shape features",
                _shape,
                help=HELP_SHAPE,
                key=f"shape_{mapper}",
                wide=True,
            )
        )

    render_card_grid(cards)
