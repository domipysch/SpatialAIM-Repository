"""
The cell-type modal, opened by a right click on a cell type in a headline or
reference plot: the type's name, its maps, and the "General", "Gene expression"
and (with a segmentation) "Shape features" views.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st
from anndata import AnnData

from spatialaim.gui import state_names
from spatialaim.gui.loaders import shape_features
from spatialaim.gui.render.cell_type import (
    render_cell_type_maps_figure,
    render_gene_distribution_figure,
    render_gene_maps_figure,
    render_self_enrichment_figure,
    render_state_coherence_figure,
    render_state_confidence_figure,
)
from spatialaim.gui.render.common import SpatialContext
from spatialaim.gui.render.markers import state_marker_table
from spatialaim.gui.render.report import render_fractions_figure
from spatialaim.gui.render.shape import render_state_shape_figure

# One decorated dialog per title, kept across runs: an open dialog is re-run as
# a stored fragment (see ``cards._card_help_dialog``), so its function object
# must stay the same from run to run; decorating anew on each call would rebuild
# it.
_CELL_TYPE_DIALOGS: dict[str, Callable] = {}

# Set when the modal saves a new name: the app reruns so every label (and the
# modal's own title) picks the name up, and the plot that opened the modal
# reopens it from this ``(origin, k, state)``.
_CTDLG_REOPEN = "_ctdlg_reopen"


def open_cell_type_dialog(
    adata_sc: AnnData,
    root: Path,
    k: int,
    state: int,
    output_dir: Path,
    spatial: SpatialContext | None = None,
    *,
    origin: str,
) -> None:
    """Open the cell-type modal for ``state``, titled with the type's name.
    ``spatial`` (mapper tabs) adds the spatial rankings, maps and violins;
    ``origin`` is the key of the plot that opened it (see ``reopened_state``)."""
    title = state_names.label(state, state_names.load(output_dir, k))
    if title not in _CELL_TYPE_DIALOGS:
        _CELL_TYPE_DIALOGS[title] = st.dialog(title, width="large")(
            _cell_type_dialog_body
        )
    _CELL_TYPE_DIALOGS[title](adata_sc, root, k, state, output_dir, spatial, origin)


def reopened_state(origin: str, k: int) -> int | None:
    """The state whose modal ``origin`` should reopen after a rename, if any."""
    pending = st.session_state.get(_CTDLG_REOPEN)
    if pending and pending[0] == origin and pending[1] == k:
        del st.session_state[_CTDLG_REOPEN]
        return int(pending[2])
    return None


# The modal's gene lists: (key, label, from the spatial ranking?, direction).
_MARKER_LISTS = {
    "up": ("↑ Upregulated", False, "up"),
    "up_st": ("↑ Upregulated · ST", True, "up"),
    "down": ("↓ Downregulated", False, "down"),
    "down_st": ("↓ Downregulated · ST", True, "down"),
}

# The modal's views, picked in its left bar: key -> (label, icon). "genes" lists
# the marker genes beneath it.
_CTDLG_VIEWS = {
    "general": ("General", ":material/insights:"),
    "genes": ("Gene expression", ":material/genetics:"),
}


def _cell_type_dialog_body(
    adata_sc: AnnData,
    root: Path,
    k: int,
    state: int,
    output_dir: Path,
    spatial: SpatialContext | None,
    origin: str,
) -> None:
    """The cell-type modal (right click on a cell type in a headline / reference
    plot). A header row of two cards: a name field for the type (the same names
    the "Set cell type names" dialog edits) beside the type highlighted on the
    all-gene reference UMAP (and, in a mapper tab, the spatial map). Below, a
    switch of views -- "General" (selected on opening,
    ``_cell_type_general_view``), "Gene expression", whose marker-gene list
    (``_cell_type_gene_list``) sits in a card left of its plots, and, with a
    segmentation, "Shape features" -- each drawn as cards over the full width."""
    # "large" is too narrow for two UMAPs beside the gene list; widen only this
    # dialog (matched through the keyed container inside it).
    st.html(
        "<style>div[role='dialog']:has(.st-key-ctdlg_root)"
        "{width:min(1400px,95vw);max-width:95vw}"
        # A tight gene list, with horizontal room around each GeneCards link.
        ".st-key-ctdlg_genes{gap:0.1rem}"
        ".st-key-ctdlg_genes button,.st-key-ctdlg_genes a"
        "{min-height:1.9rem;padding-top:0.1rem;padding-bottom:0.1rem}"
        "div[class*='st-key-ctdlg_gc_']{padding:0 0.4rem}"
        # Breathing room above and below the view switch.
        ".st-key-ctdlg_switch{margin:0.5rem 0 0.25rem}</style>"
    )
    # View and gene are per mapper too: the spatial lists depend on the mapping.
    view_key = f"ctdlg_view_{root.name}_{k}_{state}"
    gene_key = f"ctdlg_gene_{root.name}_{k}_{state}"
    # The shape view only exists for a mapping whose sweep had a segmentation.
    shape = shape_features(root) if spatial is not None else None
    views = dict(_CTDLG_VIEWS)
    if shape is not None:
        views["shape"] = ("Shape features", ":material/pentagon:")
    if st.session_state.get(view_key) not in views:
        st.session_state[view_key] = "general"
    view = st.session_state[view_key]

    def _show(v: str, g: str | None = None) -> None:
        st.session_state[view_key] = v
        if g is not None:
            st.session_state[gene_key] = g

    with st.container(key="ctdlg_root"):
        head_left, head_right = st.columns(
            [1, 4], gap="medium", vertical_alignment="center"
        )
        with head_left, st.container(border=True):
            # A form, so Enter and the button both save in one step (a plain
            # input only commits its value on Enter, before any save).
            current = state_names.load(output_dir, k).get(int(state), "")
            with st.form(f"ctdlg_name_form_{k}_{state}", border=False):
                new_name = st.text_input(
                    "Set cell type name",
                    value=current,
                    placeholder=f"Cell type {state}",
                    key=f"ctdlg_name_{k}_{state}",
                )
                saved = st.form_submit_button(
                    "Save name",
                    icon=":material/edit:",
                    width="stretch",
                )
            if saved and new_name.strip() != current:
                state_names.set_name(output_dir, k, state, new_name)
                st.session_state[_CTDLG_REOPEN] = (origin, k, int(state))
                st.rerun()
        with head_right, st.container(border=True):
            st.plotly_chart(
                render_cell_type_maps_figure(adata_sc, root, k, state, spatial=spatial),
                width="stretch",
                key="ctdlg_types",
            )

        # The view switch is keyed on ``view_key`` itself; deselecting the
        # current view leaves None, which the check above turns into "general".
        with st.container(key="ctdlg_switch"):
            st.segmented_control(
                "View",
                list(views),
                format_func=lambda v: f"{views[v][1]} {views[v][0]}",
                key=view_key,
                label_visibility="collapsed",
                width="stretch",
            )
        if view == "general":
            _cell_type_general_view(adata_sc, root, k, state, spatial)
        elif view == "shape" and shape is not None and spatial is not None:
            _cell_type_shape_view(root, k, state, spatial, shape)
        else:
            nav, body = st.columns([1, 4], gap="medium")
            with nav, st.container(border=True):
                gene = _cell_type_gene_list(
                    adata_sc, root, k, state, spatial, gene_key, _show
                )
            with body:
                if gene:
                    with st.container(border=True):
                        st.plotly_chart(
                            render_gene_maps_figure(
                                adata_sc, root, k, gene, spatial=spatial
                            ),
                            width="stretch",
                            key="ctdlg_expr",
                        )
                    with st.container(border=True):
                        st.plotly_chart(
                            render_gene_distribution_figure(
                                adata_sc, root, k, state, gene, spatial=spatial
                            ),
                            width="stretch",
                            key="ctdlg_violin",
                        )


def _modal_card(title: str):
    """A bordered card of the cell-type modal with a bold title: the modal's
    counterpart of the report's ``render_card_grid`` cards."""
    card = st.container(border=True)
    card.markdown(f"**{title}**")
    return card


def _cell_type_gene_list(
    adata_sc: AnnData,
    root: Path,
    k: int,
    state: int,
    spatial: SpatialContext | None,
    gene_key: str,
    show: Callable[..., None],
) -> str | None:
    """The modal's marker-gene list (under "Gene expression"): a dropdown of the
    type's lists (10 most up- or downregulated in the reference and, in a mapper
    tab, among the spots assigned to it), each gene a button selecting it plus a
    GeneCards link. Returns the selected gene -- the first listed one until a
    gene is clicked -- or ``None`` when there is none."""
    tables = {False: state_marker_table(adata_sc, root, k, state)}
    if spatial is not None:
        tables[True] = state_marker_table(adata_sc, root, k, state, spatial=spatial)
    options = ["up", "up_st", "down", "down_st"] if spatial else ["up", "down"]
    # Name the source on every entry once there are two of them.
    labels = {
        key: (lab if spatial is None or from_st else lab + " · scRNA")
        for key, (lab, from_st, _d) in _MARKER_LISTS.items()
    }
    st.markdown("**Marker genes**")
    choice = st.selectbox(
        "Marker genes",
        options,
        format_func=labels.get,
        key=f"ctdlg_dir_{k}_{state}_{bool(spatial)}",
        label_visibility="collapsed",
    )
    _lab, from_st, direction = _MARKER_LISTS[choice]
    table = tables.get(from_st)
    if table is None:
        st.info(
            "No marker ranking for this cell type here (too few "
            + ("assigned cells)." if from_st else "cells).")
        )
        return st.session_state.get(gene_key)
    rows = table[direction]
    if not rows:
        st.caption("none")
    gene = st.session_state.get(gene_key) or (rows[0]["gene"] if rows else None)
    where = "assigned cells" if from_st else "cells"
    genes_box = st.container(key="ctdlg_genes")
    for r in rows:
        link_col, gene_col = genes_box.columns(
            [1, 6], gap="small", vertical_alignment="center"
        )
        gene_col.button(
            f"{r['gene']}  ({r['score']:+.1f})",
            key=f"ctdlg_{k}_{state}_{choice}_{r['gene']}",
            type="primary" if r["gene"] == gene else "tertiary",
            width="stretch",
            on_click=show,
            args=("genes", r["gene"]),
            help=f"Wilcoxon score {r['score']:+.1f} · adjusted p "
            f"{r['padj']:.1e} · "
            f"expressed in {r['frac']:.0%} of this type's {where}",
        )
        link_col.link_button(
            "",
            f"https://www.genecards.org/card/{r['gene']}",
            key=f"ctdlg_gc_{k}_{state}_{choice}_{r['gene']}",
            icon=":material/open_in_new:",
            type="tertiary",
            help=f"Open {r['gene']} on GeneCards",
        )
    st.caption(f"Shared genes, Wilcoxon vs. all other {where}; score in brackets.")
    return gene


def _cell_type_general_view(
    adata_sc: AnnData,
    root: Path,
    k: int,
    state: int,
    spatial: SpatialContext | None,
) -> None:
    """The cell-type modal's "General" view. A per-type bar plot beside a
    this-type-vs-all-spots violin in each row, every other type greyed out in the
    bars: the cell-type fractions beside the mapping confidence, and the self
    neighbourhood-enrichment z-score (spatial organisation) beside the
    transcriptional coherence. The reference tab has no mapping, so it shows the
    fractions alone."""
    fractions = render_fractions_figure(
        adata_sc,
        root,
        k,
        spatial.hard if spatial is not None else None,
        highlight=state,
    )
    if spatial is None:
        with _modal_card("Cell type fractions"):
            st.plotly_chart(fractions, width="stretch", key="ctdlg_frac")
        return

    frac_col, conf_col = st.columns([3, 2], gap="medium")
    with frac_col, _modal_card("Cell type fractions"):
        st.plotly_chart(fractions, width="stretch", key="ctdlg_frac")
    with conf_col, _modal_card(f"Mapping {spatial.confidence_title.lower()}"):
        st.plotly_chart(
            render_state_confidence_figure(root, k, state, spatial),
            width="stretch",
            key="ctdlg_conf",
        )
    z_col, coh_col = st.columns([3, 2], gap="medium")
    with z_col, _modal_card("Spatial organization"):
        st.plotly_chart(
            render_self_enrichment_figure(root, k, highlight=state),
            width="stretch",
            key="ctdlg_selfz",
        )
    with coh_col, _modal_card("Transcriptional coherence"):
        st.plotly_chart(
            render_state_coherence_figure(root, k, state, spatial),
            width="stretch",
            key="ctdlg_coherence",
        )


def _cell_type_shape_view(
    root: Path,
    k: int,
    state: int,
    spatial: SpatialContext,
    features: pd.DataFrame,
) -> None:
    """The cell-type modal's "Shape features" view: area, eccentricity and
    solidity of the type's cells vs. all cells."""
    with _modal_card("Shape features"):
        st.plotly_chart(
            render_state_shape_figure(root, k, state, spatial.hard, features),
            width="stretch",
            key="ctdlg_shape",
        )
