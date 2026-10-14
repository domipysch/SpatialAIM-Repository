"""
Streamlit entry point of the SpatialAIM GUI, run by ``spatialaim gui`` (see
``__main__.py``) and re-executed top to bottom on every rerun.

Renders the sidebar, starts the queued mapper sweeps one after another, and lays
out the result tabs: the single-cell reference, one tab per mapper, and Compare.
While a sweep runs it polls, so progress and new tabs appear on their own.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import streamlit as st

from spatialaim import MAPPING_CHOICES
from spatialaim.gui import compute, data_access
from spatialaim.gui.sidebar import render_sidebar, render_sidebar_footer
from spatialaim.gui.views.choose_k import best_overall_k
from spatialaim.gui.views.compare_tab import compare_tab
from spatialaim.gui.views.controls import set_ctrl_k, shared_controls
from spatialaim.gui.views.mapper_tab import mapper_tab
from spatialaim.gui.views.reference_tab import reference_tab

# Session-state key of the tab bar; holds the active tab's label.
_TABS_KEY = "result_tabs"


def _tab_change_callback(
    output_dir: Path, tab_mappers: list[str], ks_all: list[int]
) -> Callable[[], None]:
    """``on_change`` for the tab bar: opening a mapper's tab moves the shared K
    slider onto that mapper's best overall K, so each tab lands on the K its own
    sweep proposes instead of on whatever the previous tab was showing.

    Runs as a widget callback, i.e. before the script body re-instantiates the
    slider, so writing its session-state key is allowed. Nothing happens on the
    reference and Compare tabs, or while a mapper has no scored K yet — the K on
    show then simply carries over.
    """

    def _on_change() -> None:
        mapper = st.session_state.get(_TABS_KEY)
        if mapper not in tab_mappers:
            return
        k = best_overall_k(data_access.run_root(output_dir, str(mapper)))
        if k is not None and k in ks_all:
            set_ctrl_k(ks_all.index(k))

    return _on_change


def _render_body(
    settings: argparse.Namespace,
    runs: dict,
    queue: list,
    computed: list[str],
    running_mapper: str | None,
    active: bool,
    tab_mappers: list[str],
    comparable: list[str],
) -> None:
    """Render the main content area (idle prompt, computing notice, or the tab
    set). Called inside a single ``st.empty()`` placeholder so it is replaced
    atomically each run."""
    if not tab_mappers:
        # Render EXACTLY ONE element here. The polling loop reruns via st.rerun(),
        # which does not clear stale elements, so if the idle branch emitted more
        # elements than the tabs branch, the extra ones (e.g. this prompt) would
        # linger below the tabs for the whole computation. A single st.info is a
        # different element type than st.tabs, so it is fully replaced when the
        # first tab appears.
        if active or st.session_state.get("_run_requested"):
            st.info(
                "### SpatialAIM results explorer\nComputing… tabs will appear as methods start."
            )
        else:
            st.info(
                "### SpatialAIM results explorer\n"
                "Select one or more methods in the sidebar and click **Run** to begin."
            )
        return

    # Tabs exist now — the pending-run flag has done its job.
    st.session_state.pop("_run_requested", None)
    # Shared display controls (K + shared-gene-UMAP toggle) live above the tab
    # bar and drive every tab at once. K options are the finished methods' common
    # set (all sweeps share the reference, so it matches).
    finished = [m for m in tab_mappers if m in computed and m != running_mapper]
    ks_all = sorted(
        {
            kk
            for m in finished
            for kk in data_access.list_ks(data_access.run_root(settings.output_dir, m))
        }
    )
    ctrl = shared_controls(ks_all) if ks_all else None

    # Tab order: the "Single-cell reference" tab (clustering-side,
    # mapper-independent) first, then a divider, then the per-method tabs, then
    # the Compare tab (>=2 finished methods). The reference tab appears once any
    # method has finished.
    ref_available = ctrl is not None and len(finished) >= 1
    compare_available = len(comparable) >= 2 and ctrl is not None

    if ref_available:
        # st.tabs has no divider, so draw a vertical rule on the right edge of the
        # first tab (the reference tab) to set it apart from the methods.
        st.markdown(
            "<style>"
            'div[data-baseweb="tab-list"] button[role="tab"]:first-of-type{'
            "border-right:2px solid rgba(140,140,140,0.35);"
            "margin-right:0.75rem;padding-right:1rem;}"
            "</style>",
            unsafe_allow_html=True,
        )

    labels: list[str] = []
    if ref_available:
        labels.append("Single-cell reference")
    labels += list(tab_mappers)
    if compare_available:
        labels.append("⇄ Compare")

    tabs = st.tabs(
        labels,
        key=_TABS_KEY,
        on_change=_tab_change_callback(settings.output_dir, tab_mappers, ks_all),
    )
    idx = 0
    if ref_available:
        with tabs[idx]:
            reference_tab(finished, settings, ctrl)
        idx += 1
    for mapper in tab_mappers:
        with tabs[idx]:
            mapper_tab(mapper, settings, runs, queue, ctrl, ks_all)
        idx += 1
    if compare_available:
        with tabs[idx]:
            compare_tab(comparable, settings, ctrl)


def _start_next_queued(
    settings: argparse.Namespace, runs: dict[str, compute.MapperRun], queue: list[str]
) -> bool:
    """Sequential execution: start the next queued mapper when none is running.
    Returns whether one was started."""
    if any(r.is_running() for r in runs.values()) or not queue:
        return False
    nxt = queue.pop(0)
    runs[nxt] = compute.MapperRun(
        nxt,
        settings.scdata,
        settings.stdata,
        settings.output_dir,
        settings.k_min,
        settings.k_max,
        settings.k_step,
        linkage_method=settings.linkage_method,
        start_from_annotation=settings.start_from_annotation,
        segmentation_path=settings.segmentation,
    ).start()
    return True


def main() -> None:
    """One run of the app: sidebar, queue, result tabs, and the polling rerun."""
    st.set_page_config(page_title="SpatialAIM GUI", layout="wide")
    settings = render_sidebar()
    render_sidebar_footer()

    if settings is None:
        st.title("SpatialAIM results explorer")
        st.info(
            "Set the scRNA path (.h5ad format), ST path (.h5ad format), and "
            "output directory in the sidebar to begin."
        )
        return

    runs: dict[str, compute.MapperRun] = st.session_state.setdefault("runs", {})
    queue: list[str] = st.session_state.setdefault("queue", [])

    just_started = _start_next_queued(settings, runs, queue)

    computed = data_access.list_mappers(settings.output_dir)
    running_mapper = next((m for m, r in runs.items() if r.is_running()), None)
    active = running_mapper is not None or bool(queue)
    # A tab per method that is computed, running, or queued (MAPPING_CHOICES order).
    tab_set = set(computed) | set(runs.keys()) | set(queue)
    tab_mappers = [m for m in MAPPING_CHOICES if m in tab_set]
    # Comparable = finished on disk (exclude the one still running).
    comparable = [m for m in tab_mappers if m in computed and m != running_mapper]

    # Render the whole body into one st.empty() placeholder. Because st.rerun()
    # (used by the polling loop below) does NOT clear stale elements, an st.info
    # rendered on the idle run would otherwise linger on screen for the entire
    # computation. A single placeholder is re-filled atomically each run, so the
    # idle prompt is replaced the instant the first tab renders.
    with st.empty().container():
        _render_body(
            settings,
            runs,
            queue,
            computed,
            running_mapper,
            active,
            tab_mappers,
            comparable,
        )

    # While work is pending/running, refresh so progress advances and new tabs
    # flip from "waiting" to results.
    if just_started:
        # Finish this run immediately (no sleep) so Streamlit clears the now-stale
        # idle "select methods" prompt right away, instead of leaving it on screen
        # for the polling interval below. The next run does the actual polling.
        st.rerun()
    elif queue or any(r.is_running() for r in runs.values()):
        time.sleep(1.5)
        st.rerun()


# Streamlit executes this module top-to-bottom on every rerun.
main()
