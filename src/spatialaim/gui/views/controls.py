"""
The display controls of the result pages (K slider, K navigation buttons,
shared-gene UMAP toggle).

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

    Only K is page-wide; the shared-gene-UMAP toggle, the confidence threshold and
    the "Plot confidence" checkbox are per mapper view (rendered under its plots).
    """

    k: int


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


def k_nav_buttons(ks: list[int], k: int, best: int | None, pareto: list[int]) -> None:
    """Three buttons beside the K slider: the next lower Pareto-optimal K, back to
    the best K on all criteria, and the next higher Pareto-optimal K. Each is disabled
    when it has nowhere to go."""
    lower = max((p for p in pareto if p < k and p in ks), default=None)
    higher = min((p for p in pareto if p > k and p in ks), default=None)
    row = st.container(horizontal=True, gap="small")
    for label, target, key, help_ in (
        ("← Pareto K", lower, "k_nav_lower", "Next lower Pareto-optimal K"),
        (
            "↺ Best K",
            best if best != k else None,
            "k_nav_best",
            "Best K on all criteria",
        ),
        ("Pareto K →", higher, "k_nav_higher", "Next higher Pareto-optimal K"),
    ):
        row.button(
            label,
            key=key,
            disabled=target is None or target not in ks,
            on_click=select_k,
            args=(ks, target),
            help=help_ + (f" ({target})" if target is not None else ""),
            width="stretch",
        )


def toggle_button(
    label: str, key: str, *, default: bool = False, help: str = ""
) -> bool:
    """An on/off switch drawn as a plain button whose box icon shows the state
    (ticked in the primary colour while on, an empty grey box while off). The value lives in plain session
    state under ``key``."""
    if key not in st.session_state:
        st.session_state[key] = default

    def _flip() -> None:
        st.session_state[key] = not st.session_state[key]

    on = bool(st.session_state[key])
    # A plain button either way; only the box icon carries the state: ticked in
    # the theme's primary colour while on, an empty grey box while off.
    box = (
        ":primary[:material/check_box:]"
        if on
        else ":gray[:material/check_box_outline_blank:]"
    )
    st.button(
        f"{box} {label}",
        key=f"{key}_btn",
        on_click=_flip,
        help=help or None,
    )
    return on


def shared_umap_state() -> bool:
    """The shared-gene-UMAP toggle's current value, readable before its button
    renders (so the plot can be laid out first)."""
    if _UMAP_KEY not in st.session_state:
        st.session_state[_UMAP_KEY] = bool(st.session_state.get(_VIEW_UMAP, False))
    return bool(st.session_state[_UMAP_KEY])


def shared_umap_toggle() -> bool:
    """The shared-gene-UMAP toggle button, remembered across pages."""
    show = toggle_button(
        "Shared-gene UMAP",
        _UMAP_KEY,
        default=shared_umap_state(),
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
