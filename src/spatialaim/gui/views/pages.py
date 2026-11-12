"""
The result pages behind the top navigation: Explore (one starting point × method
at one K), Compare (several methods of one starting point), and the
single-cell Reference. (The Overview lives in ``views.overview``.)

Each page opens with a control bar in the order the results nest -- starting
clusters, then method, then K. The chosen start and method are remembered in
plain session-state keys (``_view_start``, ``_view_mapper``), so they carry over
between pages; Streamlit drops the state of a widget its page does not render.
"""

from __future__ import annotations

import argparse
from typing import Callable, NamedTuple

import streamlit as st

from spatialaim import MAPPING_CHOICES
from spatialaim.gui import compute, data_access, demo
from spatialaim.gui.render.markers import marker_poller
from spatialaim.gui.views.choose_k import best_k_all, pareto_ks
from spatialaim.gui.views.compare_tab import compare_tab
from spatialaim.gui.views.controls import (
    Controls,
    k_nav_buttons,
    k_slider,
    select_k,
)
from spatialaim.gui.views.mapper_tab import mapper_tab
from spatialaim.gui.views.reference_tab import reference_tab

_VIEW_START = "_view_start"  # start folder name on show
_VIEW_MAPPER = "_view_mapper"  # mapper on show (Explore)
_START_WIDGET = "view_start_select"
_MAPPER_WIDGET = "view_mapper_select"


class ViewContext(NamedTuple):
    """What every page needs: the run settings, the run queue, and the starts."""

    settings: argparse.Namespace
    runs: dict[str, compute.MapperRun]
    queue: list[str]
    # The starts on disk plus those a running or queued sweep is about to write.
    starts: list[data_access.Start]


def focus(start: data_access.Start, mapper: str) -> None:
    """Show ``mapper`` in ``start`` at its best K on all criteria (on the next render of
    the Explore page). Safe in a callback and before ``st.switch_page``."""
    st.session_state[_VIEW_START] = start.folder.name
    st.session_state[_VIEW_MAPPER] = mapper
    st.session_state.pop(_START_WIDGET, None)
    st.session_state.pop(_MAPPER_WIDGET, None)
    root = data_access.run_root(start.folder, mapper)
    select_k(data_access.list_ks(root), best_k_all(root))


def _start_args(ctx: ViewContext, start: data_access.Start) -> argparse.Namespace:
    """The run settings pointed at one start, as the result views expect them."""
    return argparse.Namespace(
        **{
            **vars(ctx.settings),
            "output_dir": start.folder,
            "start_from_annotation": start.annotation,
        }
    )


def _finished_mappers(start: data_access.Start, ctx: ViewContext) -> list[str]:
    """Mappers whose sweep of ``start`` is complete (MAPPING_CHOICES order)."""
    return [
        m
        for m in data_access.list_mappers(start.folder)
        if compute.start_complete(ctx.settings.pair_dir, m, start.annotation)
        and (m not in ctx.runs or ctx.runs[m].finished(start.annotation))
    ]


def _explore_mappers(start: data_access.Start, ctx: ViewContext) -> list[str]:
    """Mappers worth a view in ``start``: those with results there, the one
    sweeping it, and queued ones that still lack it (MAPPING_CHOICES order)."""
    present = set(data_access.list_mappers(start.folder))
    for m, run in ctx.runs.items():
        if run.is_running() and start.annotation in run.starts:
            present.add(m)
    for m in ctx.queue:
        if not compute.start_complete(ctx.settings.pair_dir, m, start.annotation):
            present.add(m)
    return [m for m in MAPPING_CHOICES if m in present]


def _remembered(widget: str, view: str, options: list[str]) -> None:
    """Seed ``widget``'s state from the remembered ``view`` value (or the first
    option) before it renders."""
    if st.session_state.get(widget) not in options:
        value = st.session_state.get(view)
        st.session_state[widget] = value if value in options else options[0]


def _start_select(
    starts: list[data_access.Start], on_change: Callable[[], None] | None = None
) -> data_access.Start:
    """The starting-clusters dropdown (Leiden first)."""
    names = [s.folder.name for s in starts]
    _remembered(_START_WIDGET, _VIEW_START, names)

    def _changed() -> None:
        st.session_state[_VIEW_START] = st.session_state[_START_WIDGET]
        if on_change is not None:
            on_change()

    st.selectbox(
        "Starting clusters",
        names,
        format_func=lambda n: starts[names.index(n)].label,
        key=_START_WIDGET,
        on_change=_changed,
        help="The clusters the agglomeration tree starts from: the Leiden "
        "over-clustering, or an annotation of the reference. Each start has its "
        "own tree, so its cell types, K range and names are its own.",
    )
    name = st.session_state[_START_WIDGET]
    st.session_state[_VIEW_START] = name
    return starts[names.index(name)]


def _no_results(ctx: ViewContext) -> None:
    """The notice of a page with nothing to show yet."""
    busy = bool(ctx.queue) or any(r.is_running() for r in ctx.runs.values())
    st.info(
        "Computing… results appear here as soon as a method has finished."
        if busy
        else (
            "No finished results yet."
            if demo.ACTIVE
            else "No finished results yet. Select methods in the sidebar and click **Run**."
        )
    )


# -- Explore -----------------------------------------------------------------


def explore_page(ctx: ViewContext) -> None:
    """One starting point × method at one K: the control bar, a status line, and
    the method's results (or its progress while it computes)."""

    def _best_k_of_selection() -> None:
        # On a start or method switch, land on that combination's best K.
        start = next(
            s for s in ctx.starts if s.folder.name == st.session_state[_VIEW_START]
        )
        mapper = st.session_state.get(_MAPPER_WIDGET) or st.session_state.get(
            _VIEW_MAPPER
        )
        if mapper is None:
            return
        root = data_access.run_root(start.folder, mapper)
        select_k(data_access.list_ks(root), best_k_all(root))

    col_start, col_mapper, col_k, col_nav = st.columns(
        [2, 2, 9, 3], vertical_alignment="bottom"
    )
    with col_start:
        start = _start_select(ctx.starts, on_change=_best_k_of_selection)
    mappers = _explore_mappers(start, ctx)
    if not mappers:
        st.info(f"No method has been run from {start.label} yet.")
        return
    with col_mapper:
        _remembered(_MAPPER_WIDGET, _VIEW_MAPPER, mappers)

        def _mapper_changed() -> None:
            st.session_state[_VIEW_MAPPER] = st.session_state[_MAPPER_WIDGET]
            _best_k_of_selection()

        mapper = st.selectbox(
            "Method", mappers, key=_MAPPER_WIDGET, on_change=_mapper_changed
        )
        st.session_state[_VIEW_MAPPER] = mapper

    args = _start_args(ctx, start)
    root = data_access.run_root(start.folder, mapper)
    run = ctx.runs.get(mapper)
    done = compute.start_complete(ctx.settings.pair_dir, mapper, start.annotation) and (
        run is None or run.finished(start.annotation)
    )
    ks = data_access.list_ks(root) if done else []
    best = best_k_all(root) if done else None
    ctrl = None
    if ks:
        with col_k:
            k = k_slider(ks, best)
        with col_nav:
            k_nav_buttons(ks, k, best, pareto_ks(root))
        ctrl = Controls(k)
        note = (
            " (best on all criteria)"
            if k == best
            else (f", best K on all criteria = {best}" if best else "")
        )
        st.caption(f"**{start.label}** · **{mapper}** · K = {k}{note}")
    mapper_tab(mapper, args, ctx.runs, ctx.queue, ctrl, ks)
    marker_poller()


# -- Compare -----------------------------------------------------------------


def compare_page(ctx: ViewContext) -> None:
    """Several methods of one starting point side by side at one K."""
    starts = [s for s in ctx.starts if len(_finished_mappers(s, ctx)) >= 2]
    if not starts:
        st.info("Compare needs at least two finished methods from one starting point.")
        return
    col_start, col_k = st.columns([2, 6], vertical_alignment="bottom")
    with col_start:
        start = _start_select(starts)
    mappers = _finished_mappers(start, ctx)
    ks = sorted(
        {
            k
            for m in mappers
            for k in data_access.list_ks(data_access.run_root(start.folder, m))
        }
    )
    with col_k:
        k = k_slider(ks, best_k_all(data_access.run_root(start.folder, mappers[0])))
    compare_tab(mappers, _start_args(ctx, start), Controls(k))
    marker_poller()


# -- Reference ---------------------------------------------------------------


def reference_page(ctx: ViewContext) -> None:
    """The single-cell reference of one starting point at one K."""
    starts = [s for s in ctx.starts if _finished_mappers(s, ctx)]
    if not starts:
        _no_results(ctx)
        return
    col_start, col_k = st.columns([2, 6], vertical_alignment="bottom")
    with col_start:
        start = _start_select(starts)
    mappers = _finished_mappers(start, ctx)
    ks = sorted(
        {
            k
            for m in mappers
            for k in data_access.list_ks(data_access.run_root(start.folder, m))
        }
    )
    with col_k:
        k = k_slider(ks, best_k_all(data_access.run_root(start.folder, mappers[0])))
    reference_tab(mappers, _start_args(ctx, start), Controls(k))
    marker_poller()
