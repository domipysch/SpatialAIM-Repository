"""
The display controls of the result pages (K slider, shared-gene UMAP toggle).

Each page renders its own controls, but their values live in plain session-state
keys (``_view_*``) as well: Streamlit drops a widget's state on a page that does
not render it, and the chosen K and toggle should survive a trip to another page.
"""

from __future__ import annotations

from typing import NamedTuple

import streamlit as st

from spatialaim.gui.widgets.slider import live_select_slider

_K_KEY = "ctrl_k"  # the K slider's widget state ({"value": option index})
_UMAP_KEY = "ctrl_shared_umap"
_VIEW_K = "_view_k"  # the K on show
_VIEW_KS = "_view_ks"  # the K options it was last shown among
_VIEW_UMAP = "_view_shared_umap"


class Controls(NamedTuple):
    """Display controls that drive a result page.

    Only K and the shared-gene-UMAP toggle are page-wide; the confidence
    threshold and "Plot confidence" checkbox are per mapper view (rendered inside
    it, where they can be hidden when that mapper has no confidence).
    """

    k: int
    show_shared_umap: bool


def k_slider(ks: list[int], default_k: int | None = None) -> int:
    """The K slider over ``ks``. It keeps the K on show while the options stay the
    same; when they change (another start or mapper), it moves to the same K if
    offered, else to ``default_k`` (else the first option)."""
    if st.session_state.get(_VIEW_KS) != ks:
        st.session_state[_VIEW_KS] = list(ks)
        k = st.session_state.get(_VIEW_K)
        if k not in ks:
            k = default_k if default_k in ks else ks[0]
        st.session_state[_K_KEY] = {"value": ks.index(k)}
    remembered = st.session_state.get(_VIEW_K)
    value = remembered if remembered in ks else default_k
    k = int(live_select_slider("K (number of cell types)", ks, value, key=_K_KEY))
    st.session_state[_VIEW_K] = k
    return k


def shared_umap_toggle() -> bool:
    """The shared-gene-UMAP checkbox, remembered across pages."""
    if _UMAP_KEY not in st.session_state:
        st.session_state[_UMAP_KEY] = bool(st.session_state.get(_VIEW_UMAP, False))
    show = st.checkbox(
        "Show shared-gene-only UMAP",
        key=_UMAP_KEY,
        help="Add the shared-gene (ST-overlap) UMAP as a third panel, left of "
        "the all-gene UMAP.",
    )
    st.session_state[_VIEW_UMAP] = bool(show)
    return bool(show)


def select_k(ks: list[int], k: int | None) -> None:
    """Show ``k`` among the options ``ks`` (a no-op for ``None`` or a K not
    offered). Safe in a widget callback: it runs before the slider re-renders."""
    if k is None or k not in ks:
        return
    st.session_state[_VIEW_KS] = list(ks)
    st.session_state[_VIEW_K] = int(k)
    st.session_state[_K_KEY] = {"value": ks.index(k)}


def set_ctrl_k(idx: int) -> None:
    """on_click: move the K slider to option index ``idx`` of its current options
    (runs before the slider re-instantiates, so writing its state is allowed)."""
    st.session_state[_K_KEY] = {"value": int(idx)}
