"""
The "Single-cell reference" tab: the mapper-independent clustering side -- the
start clustering and computed cell types on the reference UMAPs, the marker
genes (with the cell-type naming dialog), and the start clusters merged per type.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import streamlit as st
from anndata import AnnData

from spatialaim.gui import data_access, state_names
from spatialaim.gui.loaders import load_scaffold
from spatialaim.gui.render.common import state_palette
from spatialaim.gui.render.headline import (
    render_headline_figure,
    render_reference_umaps_figure,
)
from spatialaim.gui.render.markers import render_state_markers_figure
from spatialaim.gui.render.report import render_start_cluster_merge_figure
from spatialaim.gui.views.cards import export_popover, plot_card
from spatialaim.gui.views.cell_type_modal import open_cell_type_dialog, reopened_state
from spatialaim.gui.views.controls import Controls
from spatialaim.gui.widgets.headline_plot import headline_plot


@st.dialog("Set cell type names", width="large")
def _state_names_dialog(
    adata_sc: AnnData, ref_root: Path, output_dir: Path, k: int
) -> None:
    """Name this K's computed cell types, next to the UMAP that shows them.

    The names belong to ``(output_dir, K)`` and are shared by every mapper (they
    all cut the same tree), so one edit here relabels every tab at this K. A field
    left empty keeps that cell type's default ``Cell type <n>``.
    """
    left, right = st.columns([3, 2], gap="medium")

    with left:
        try:
            # coords=None -> the reference UMAP alone, which is all this modal
            # needs; hard/confidence are only read by the spatial panel.
            fig = render_headline_figure(
                None, np.array([]), None, 0.0, k, adata_sc=adata_sc, root=ref_root
            )
            fig.update_layout(height=430, margin=dict(l=0, r=0, t=60, b=0))
            st.plotly_chart(fig, width="stretch", key="namesdlg_umap")
        except Exception as exc:  # noqa: BLE001 - the table stays usable without it
            st.info(f"UMAP unavailable: {exc}")

    palette = state_palette(k)
    current = state_names.load(output_dir, k)
    with right:
        st.caption("Leave a field empty to keep the default label.")
        values: dict[int, str] = {}
        for s in range(k):
            # A flex row rather than nested columns, which Streamlit only allows
            # one level deep and this is already inside a column.
            row = st.container(horizontal=True, vertical_alignment="center")
            row.markdown(
                f'<span style="color:{palette.get(s, "#b8b8b8")};font-size:20px">'
                f"●</span>&nbsp;Cell type {s}",
                unsafe_allow_html=True,
            )
            values[s] = row.text_input(
                f"Name for cell type {s}",
                value=current.get(s, ""),
                key=f"namesdlg_{k}_{s}",
                label_visibility="collapsed",
                placeholder=f"Cell type {s}",
            )

    st.divider()
    save, cancel = st.columns(2)
    if save.button("Save", type="primary", width="stretch", key="namesdlg_save"):
        state_names.save(output_dir, k, values)
        st.rerun()
    if cancel.button("Cancel", width="stretch", key="namesdlg_cancel"):
        st.rerun()


@st.fragment
def reference_tab(mappers: list[str], args: argparse.Namespace, ctrl: Controls) -> None:
    """The single-cell reference view: clustering-side plots that don't depend on
    the spot-mapping method.

    The tree cut (``start_cluster_to_state``) and the reference expression are identical
    across mappers, so this reads them from any finished mapper that has the
    currently selected K. Shows three UMAPs on top (start clustering, and the
    computed states on the all-gene and shared-gene embeddings), then the
    mapper-independent sections rendered as interactive Plotly.
    """
    st.subheader("Single-cell reference")

    # Any finished mapper carrying this K works — pick the first.
    ref_mapper = next(
        (
            m
            for m in mappers
            if ctrl.k in data_access.list_ks(data_access.run_root(args.output_dir, m))
        ),
        None,
    )
    if ref_mapper is None:
        st.info("No finished method has this K available yet.")
        return
    ref_root = data_access.run_root(args.output_dir, ref_mapper)

    adata_sc = load_scaffold(
        args, warn="Scaffold build failed — reference view unavailable"
    )
    if adata_sc is None:
        return

    fig_umaps = render_reference_umaps_figure(adata_sc, ref_root, ctrl.k)
    menu_state = headline_plot(fig_umaps, key="ref_umaps", context_menu=True)
    if menu_state is None:
        menu_state = reopened_state("ref_umaps", ctrl.k)
    if menu_state is not None:
        open_cell_type_dialog(
            adata_sc, ref_root, ctrl.k, menu_state, args.output_dir, origin="ref_umaps"
        )
    export_popover(
        fig_umaps, key="ref_umaps_exp", stem=f"reference_umaps_k{ctrl.k:03d}"
    )
    st.caption(
        f"K = {ctrl.k}. The start clustering and its tree cut are shared across "
        f"methods (shown from **{ref_mapper}**)."
    )

    st.divider()
    with st.expander("Cell type marker genes", expanded=False):
        n_genes = st.number_input(
            "Genes per cell type",
            min_value=1,
            max_value=25,
            value=5,
            step=1,
            key="ref_markers_n",
            help="Top-ranked shared genes shown per computed cell type "
            "(scanpy rank_genes_groups, Wilcoxon, each type vs. the rest, on "
            "the lognorm layer).",
        )
        fig_mk = render_state_markers_figure(
            adata_sc, ref_root, ctrl.k, n_genes=int(n_genes)
        )
        if fig_mk is None:
            st.info(
                "Marker genes need at least two shared genes and two cell types "
                "with two or more cells each."
            )
        else:
            plot_card(
                fig_mk,
                key="ref_markers",
                stem=f"reference_marker_genes_k{ctrl.k:03d}_n{int(n_genes)}",
                caption="Dot size: fraction of the cell type's cells expressing "
                "the gene. Colour: mean lognorm expression, scaled 0–1 per gene "
                "across cell types. Hover for the Wilcoxon score and adjusted "
                "p-value.",
            )
        if st.button(
            "Set cell type names",
            icon=":material/edit:",
            key="ref_name_states",
            help="Give the computed cell types names, used everywhere in the UI.",
        ):
            _state_names_dialog(adata_sc, ref_root, args.output_dir, ctrl.k)
    with st.expander("Start clusters merged per cell type", expanded=False):
        plot_card(
            render_start_cluster_merge_figure(adata_sc, ref_root, ctrl.k),
            key="ref_start_cluster_merge",
            stem=f"reference_start_cluster_merge_k{ctrl.k:03d}",
        )
