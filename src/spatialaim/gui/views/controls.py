"""
The shared display controls above the tab bar (K slider, shared-gene UMAP
toggle), which drive every result tab at once.
"""

from __future__ import annotations

from typing import NamedTuple

import streamlit as st

from spatialaim.gui.widgets.slider import live_select_slider


class Controls(NamedTuple):
    """Shared display controls that drive every result tab at once.

    Only K and the shared-gene-UMAP toggle are global; the confidence threshold
    and "Plot confidence" checkbox are per-tab (rendered inside each mapper tab,
    where they can be hidden when that mapper has no confidence).
    """

    k: int
    show_shared_umap: bool


def shared_controls(ks: list[int]) -> Controls:
    """Render the K slider and the shared-gene-UMAP toggle once, above the tabs."""
    c1, c2 = st.columns([3, 2], vertical_alignment="center")
    with c1:
        k = live_select_slider("K (number of cell types)", ks, ks[0], key="ctrl_k")
    with c2:
        show_shared_umap = st.checkbox(
            "Show shared-gene-only UMAP",
            value=False,
            key="ctrl_shared_umap",
            help="Add the shared-gene (ST-overlap) UMAP as a third panel, left of "
            "the all-gene UMAP.",
        )
    return Controls(int(k), bool(show_shared_umap))


def set_ctrl_k(idx: int) -> None:
    """on_click: move the shared K slider to option index ``idx`` (runs before the
    slider re-instantiates, so writing its session-state key is allowed)."""
    st.session_state["ctrl_k"] = {"value": int(idx)}
