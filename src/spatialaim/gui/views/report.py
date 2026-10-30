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
    confidence_caption,
)
from spatialaim.gui.views.cards import Card, plot_card, render_card_grid


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
    cossim = data_access.load_cossim_boxplots(root, k)
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
