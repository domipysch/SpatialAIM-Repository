"""
The Compare tab: one reference UMAP over the selected mappers' spatial maps,
then report cards that put every selected mapper in one plot or table.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import streamlit as st

from spatialaim.gui import data_access, state_names
from spatialaim.gui.loaders import load_scaffold, load_soft, spatial_coords
from spatialaim.gui.render.headline import render_compare_figure
from spatialaim.gui.render.report import (
    render_compare_fractions_figure,
    render_compare_reconstruction_figure,
)
from spatialaim.gui.views.cards import (
    Card,
    export_popover,
    plot_card,
    render_card_grid,
)
from spatialaim.gui.views.controls import Controls
from spatialaim.gui.views.report import topology_groups
from spatialaim.gui.widgets.headline_plot import headline_plot


@st.fragment
def compare_tab(mappers: list[str], args: argparse.Namespace, ctrl: Controls) -> None:
    """Compare several methods at once: one shared reference UMAP followed by each
    selected method's spatial map side by side.

    The shared-gene UMAP toggle and confidence options do not apply here (this tab
    always shows the single all-gene reference UMAP with no confidence colouring).
    """
    st.subheader("Side-by-side comparison")

    selected = st.pills(
        "Methods to compare",
        mappers,
        selection_mode="multi",
        default=list(mappers),
        key="cmp_methods",
        help="Pick two or more computed methods; their spatial maps are shown "
        "side by side under one reference UMAP.",
    )
    if len(selected) < 2:
        st.info("Select at least two methods to compare.")
        return

    # Keep the stable MAPPING_CHOICES order that `mappers` already carries.
    selected = [m for m in mappers if m in selected]
    k = ctrl.k

    coords = spatial_coords(str(args.stdata))
    if coords is None:
        st.info("ST data has no obsm['spatial'] — nothing to compare spatially.")
        return

    # Fall back to spatial-only if the reference UMAP can't be built.
    adata_sc = load_scaffold(
        args, warn="Scaffold build failed — reference UMAP unavailable"
    )

    # Load each selected method's hard assignment for this K (skip any missing K).
    hards: list[np.ndarray] = []
    valid: list[str] = []
    for m in selected:
        root = data_access.run_root(args.output_dir, m)
        if k not in data_access.list_ks(root):
            continue
        _P, hard, _conf = load_soft(str(root), k)
        hards.append(hard)
        valid.append(m)
    if len(valid) < 2:
        st.info(f"Fewer than two selected methods have K={k} available.")
        return

    # One figure: reference UMAP centred on top, spatial maps in a 2-wide grid
    # below, all sharing a single interactive state legend.
    ref_root = data_access.run_root(args.output_dir, valid[0]) if adata_sc else None
    fig = render_compare_figure(
        coords, hards, valid, k, adata_sc=adata_sc, root=ref_root
    )
    headline_plot(fig, key="cmp_headline")
    export_popover(fig, key="cmp_headline_exp", stem=f"compare_spatial_k{k:03d}")

    # Report sections below the plots — the same UI cards as a single method tab,
    # but with every selected method combined into one plot/table per section.
    st.divider()
    st.subheader("Report")
    _compare_sections(valid, hards, args, k)


def _compare_sections(
    valid: list[str], hards: list[np.ndarray], args: argparse.Namespace, k: int
) -> None:
    """Combined report cards for the Compare tab: each section overlays all
    selected methods in a single Plotly figure or a table with one column per
    method."""
    roots = {m: data_access.run_root(args.output_dir, m) for m in valid}

    # Gather each method's per-K data once.
    cossims: dict[str, dict] = {}
    spot_fracs: dict[str, list[float]] = {}
    for m, hard in zip(valid, hards):
        cossims[m] = data_access.load_cossim_distributions(roots[m], k)
        spot_fracs[m] = [float(np.mean(hard == s)) for s in range(k)]

    def _reconstruction() -> None:
        fig = render_compare_reconstruction_figure(cossims)
        if fig is None:
            st.info("No reconstruction cosine-similarity data for these methods.")
        else:
            plot_card(fig, key="cmp_sec_recon", stem=f"compare_reconstruction_k{k:03d}")

    def _fractions() -> None:
        plot_card(
            render_compare_fractions_figure(
                spot_fracs, k, names=state_names.load(args.output_dir, k)
            ),
            key="cmp_sec_frac",
            stem=f"compare_fractions_k{k:03d}",
            caption="Cell fractions are identical across methods (see a method tab).",
        )

    def _spatial_org() -> None:
        # The same groups and field order as a method tab's card (_TOPOLOGY_FIELDS),
        # flattened to one row per field with methods as columns.
        cols: dict[str, dict] = {}
        for m in valid:
            topo = (
                data_access.load_data_json(roots[m], k, "topology_metrics.json") or {}
            )
            flat: dict[str, float] = {}
            for grp, sub in topology_groups(topo).items():
                for name, val in sub.items():
                    flat[f"{grp}.{name}"] = val
            cols[m] = flat
        df = pd.DataFrame(cols)
        if df.empty:
            st.info("No spatial-organisation metrics for these methods.")
        else:
            st.dataframe(df.round(4), width="stretch")

    def _modularity() -> None:
        # Only the mapping-dependent modularity, methods as columns.
        row: dict[str, float] = {}
        for m in valid:
            mod = (
                data_access.load_data_json(roots[m], k, "modularity_metrics.json") or {}
            )
            row[m] = mod.get("modularity_st_expression", float("nan"))
        df = pd.DataFrame({"modularity_st_expression": row}).T
        st.dataframe(df.round(4), width="stretch")

    cards: list[Card] = [
        Card("Reconstruction Cosine Similarity", _reconstruction, key="cmp_recon"),
        Card("Cell type fractions", _fractions, key="cmp_frac"),
    ]
    cards.append(Card("Spatial organisation", _spatial_org, key="cmp_topo"))
    cards.append(Card("Modularity", _modularity, key="cmp_mod"))

    render_card_grid(cards)
