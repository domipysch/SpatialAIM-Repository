"""
The "Inspect genes" modal, opened from the button under a mapper tab's headline
plot: every shared gene in an alphabetical list, the selected gene's expression
on the reference UMAP and the spatial map, and its violins per cell type at K.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from anndata import AnnData

from spatialaim.gui.render.cell_type import (
    render_gene_by_type_figure,
    render_gene_maps_figure,
)
from spatialaim.gui.render.common import SpatialContext
from spatialaim.gui.render.markers import shared_genes

_GENE_KEY = "genedlg_gene"


@st.dialog("Inspect genes", width="large")
def open_gene_dialog(
    adata_sc: AnnData,
    root: Path,
    k: int,
    spatial: SpatialContext | None,
) -> None:
    """The modal: a filterable gene list in a card left of the gene's maps and
    its per-cell-type violins (reference cells and assigned spots)."""
    # Widened like the cell-type modal (matched through the keyed container).
    st.html(
        "<style>div[role='dialog']:has(.st-key-genedlg_root)"
        "{width:min(1400px,95vw);max-width:95vw}"
        ".st-key-genedlg_list{gap:0.1rem}"
        ".st-key-genedlg_list button"
        "{min-height:1.9rem;padding-top:0.1rem;padding-bottom:0.1rem}</style>"
    )
    genes = sorted(shared_genes(adata_sc))
    if not genes:
        st.info("No shared genes in the reference scaffold.")
        return
    if st.session_state.get(_GENE_KEY) not in genes:
        st.session_state[_GENE_KEY] = genes[0]
    gene = st.session_state[_GENE_KEY]

    def _select(g: str) -> None:
        st.session_state[_GENE_KEY] = g

    with st.container(key="genedlg_root"):
        nav, body = st.columns([1, 4], gap="medium")
        with nav, st.container(border=True):
            st.markdown(f"**Shared genes** ({len(genes)})")
            query = st.text_input(
                "Filter genes",
                key="genedlg_filter",
                placeholder="Filter…",
                label_visibility="collapsed",
            )
            shown = [g for g in genes if query.strip().upper() in g]
            with st.container(height=700, border=False, key="genedlg_list"):
                for g in shown:
                    st.button(
                        g,
                        key=f"genedlg_{g}",
                        type="primary" if g == gene else "tertiary",
                        width="stretch",
                        on_click=_select,
                        args=(g,),
                    )
                if not shown:
                    st.caption("No match.")
        with body:
            with st.container(border=True):
                st.plotly_chart(
                    render_gene_maps_figure(adata_sc, root, k, gene, spatial=spatial),
                    width="stretch",
                    key="genedlg_expr",
                )
            with st.container(border=True):
                st.plotly_chart(
                    render_gene_by_type_figure(
                        adata_sc, root, k, gene, spatial=spatial
                    ),
                    width="stretch",
                    key="genedlg_violin",
                )
