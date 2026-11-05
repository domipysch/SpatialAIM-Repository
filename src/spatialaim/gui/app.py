"""
Streamlit entry point of the SpatialAIM GUI, run by ``spatialaim gui`` (see
``__main__.py``) and re-executed top to bottom on every rerun.

Renders the sidebar, starts the queued mapper sweeps one after another, and lays
out the results as pages behind a top navigation: the Overview of every
starting point × method, Explore (one starting point × method at one K),
Compare (several methods of one starting point) and the single-cell Reference.
While a sweep runs it polls, so progress and new results appear on their own.
The online demo (``demo.ACTIVE``) shows the same pages over precomputed results,
without the run queue, plus its imprint and privacy policy.
"""

from __future__ import annotations

import time

import streamlit as st

from spatialaim.gui import compute, data_access, demo
from spatialaim.gui.sidebar import render_sidebar, render_sidebar_footer
from spatialaim.gui.views.overview import overview_view
from spatialaim.gui.views.pages import (
    ViewContext,
    compare_page,
    explore_page,
    focus,
    reference_page,
)


def _start_next_queued(
    settings, runs: dict[str, compute.MapperRun], queue: list[str]
) -> bool:
    """Sequential execution: start the next queued mapper when none is running,
    on the starts it still lacks (decided now, so annotations added while it
    waited are included). A queued mapper with nothing left is dropped.
    Returns whether one was started."""
    if any(r.is_running() for r in runs.values()):
        return False
    while queue:
        nxt = queue.pop(0)
        starts = compute.missing_starts(settings.pair_dir, nxt, settings.annotations)
        if not starts:
            continue
        runs[nxt] = compute.MapperRun(
            nxt,
            settings.scdata,
            settings.stdata,
            settings.pair_dir,
            settings.k_min,
            settings.k_max,
            settings.k_step,
            linkage_method=settings.linkage_method,
            starts=starts,
            segmentation_path=settings.segmentation,
        ).start()
        return True
    return False


def _n_finished(start: data_access.Start) -> int:
    """How many mappers have a complete sweep of ``start`` on disk."""
    return sum(
        (data_access.run_root(start.folder, m) / "k_comparison.csv").exists()
        for m in data_access.list_mappers(start.folder)
    )


def _pages(ctx: ViewContext, disk_starts: list[data_access.Start]) -> list:
    """The navigation's pages, landing on Explore (which also shows the progress
    of a running sweep). The Overview and Reference join once a sweep has
    finished, Compare once one start has two finished methods."""
    per_start = [_n_finished(s) for s in disk_starts]
    n_finished = sum(per_start)
    explore = st.Page(
        lambda: explore_page(ctx),
        title="Explore",
        icon=":material/travel_explore:",
        url_path="explore",
        default=True,
    )

    def _open(start: data_access.Start, mapper: str) -> None:
        focus(start, mapper)
        st.switch_page(explore)

    pages = []
    if n_finished:
        pages.append(
            st.Page(
                lambda: overview_view(disk_starts, _open),
                title="Overview",
                icon=":material/leaderboard:",
                url_path="overview",
            )
        )
    pages.append(explore)
    if max(per_start, default=0) >= 2:
        pages.append(
            st.Page(
                lambda: compare_page(ctx),
                title="Compare",
                icon=":material/compare_arrows:",
                url_path="compare",
            )
        )
    if n_finished:
        pages.append(
            st.Page(
                lambda: reference_page(ctx),
                title="Single-cell reference",
                icon=":material/scatter_plot:",
                url_path="reference",
            )
        )
    return pages


def demo_main() -> None:
    """One run of the online demo: the pair picker and the result pages of the
    chosen pair, landing on Explore. Nothing runs, so there is no queue or
    polling."""
    st.set_page_config(page_title="SpatialAIM demo", layout="wide")
    settings = render_sidebar()
    legal = demo.legal_pages()
    if settings is None:
        pages = [st.Page(demo.no_data_view, title="SpatialAIM", default=True)]
    else:
        starts = data_access.list_starts(settings.pair_dir)
        pages = _pages(ViewContext(settings, {}, [], starts), starts)
    nav = st.navigation([*pages, *legal], position="top")
    render_sidebar_footer(legal)
    nav.run()


def main() -> None:
    """One run of the app: sidebar, queue, the result pages, and the polling
    rerun."""
    if demo.ACTIVE:
        demo_main()
        return
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

    # The starts on disk, plus those a running or queued sweep is about to write,
    # so Explore can show its progress before their folders exist.
    planned: list[str | None] = []
    running = next((r for r in runs.values() if r.is_running()), None)
    if running is not None:
        planned += running.starts
    if queue:
        planned += [None, *settings.annotations]
    starts = data_access.list_starts(settings.pair_dir, planned)

    # Render the whole body into one st.empty() placeholder. Because st.rerun()
    # (used by the polling loop below) does NOT clear stale elements, an st.info
    # rendered on the idle run would otherwise linger on screen for the entire
    # computation. A single placeholder is re-filled atomically each run.
    with st.empty().container():
        if not starts:
            busy = bool(queue) or running is not None
            if busy or st.session_state.get("_run_requested"):
                st.info(
                    "### SpatialAIM results explorer\n"
                    "Computing… results will appear as methods start."
                )
            else:
                st.info(
                    "### SpatialAIM results explorer\n"
                    "Select one or more methods in the sidebar and click **Run** "
                    "to begin."
                )
        else:
            st.session_state.pop("_run_requested", None)
            ctx = ViewContext(settings, runs, queue, starts)
            disk_starts = data_access.list_starts(settings.pair_dir)
            st.navigation(_pages(ctx, disk_starts), position="top").run()
    _poll(just_started, runs, queue)


def _poll(just_started: bool, runs: dict, queue: list) -> None:
    """While work is pending/running, refresh so progress advances and new
    results flip from "waiting" to shown."""
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
