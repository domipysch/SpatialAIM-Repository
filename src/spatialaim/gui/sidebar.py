"""
The sidebar: data paths (step 1), the annotations to start from besides the
Leiden over-clustering, linkage and mapping methods to run (step 2), and the
footer with the version and the feedback link. In the online demo
(``demo.ACTIVE``) step 1 is a pick of a precomputed pair and step 2 only shows
what was computed for it.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import streamlit as st
from packaging.version import InvalidVersion, Version

from spatialaim import LINKAGE_METHODS, MAPPING_CHOICES
from spatialaim.config import _INPROCESS_METHODS
from spatialaim.gui import compute, data_access, demo
from spatialaim.gui.loaders import obs_columns
from spatialaim.gui.texts import MAPPER_LABELS
from spatialaim.gui.validation import (
    data_requirements_dialog,
    request_data_help,
    validation_row,
)
from spatialaim.reference_aligners.registry import available_reference_aligners

# Everything is configured in the sidebar — the launcher only forwards the server
# port. The K range is fixed to the full sweep (k_min = 1, k_max = L, where L is
# the start-cluster count discovered at runtime) in steps of 1. The agglomeration
# linkage is chosen in the sidebar, from the same LINKAGE_METHODS the
# `--linkage_method` CLI flag offers.
_DEFAULT_K_MIN: int = 1
_DEFAULT_K_MAX: int | None = None
_DEFAULT_K_STEP = 1

_ANNOTATIONS_LABEL = "Also start from annotations"
_LEIDEN_ONLY = "Start from Leiden over-clustering only"
_ANNOTATIONS_HELP = (
    "SpatialAIM always starts from a Leiden over-clustering of the reference. Each "
    "annotation picked here adds a sweep that starts from its cell types instead; "
    "the results view switches between these starting points. Offered are the "
    "scRNA-seq obs columns that label every cell. Annotations can be added after a "
    "run: the next run computes every selected method on the starting points it "
    "is missing. Starting points with results stay selected."
)

_LEGACY_NOTICE = (
    "This output folder holds results from an earlier SpatialAIM version. They can "
    "be viewed, but new runs need a new output folder."
)

# Display-only labels; the raw method names go into the run config.
_LINKAGE_LABELS = {
    "average": "average - allow for outlier cell types",
    "ward": "ward - more balanced cell types",
}
_LINKAGE_HELP = (
    "Linkage for the agglomeration tree over the start clusters. "
    "'average' (UPGMA) peels small tight groups off a dominant cell type; "
    "'ward' carries a size term and tends to produce balanced cell types."
)

# The sidebar's keyed inputs keep their values for the whole session.
# st.switch_page (the Overview's Open button) reruns without the frontend's
# widget states, and Streamlit then drops every widget's value that is not
# persisted -- the sidebar would come back empty.
_PERSIST = "session"

_LOCKED_HELP = (
    "Fixed by the runs already in this output folder: every persisted tree cut is "
    "tied to it. Point the output directory somewhere new to choose again."
)


@st.cache_data(show_spinner=False)
def _selectable_methods() -> list[str]:
    """MAPPING_CHOICES restricted to methods that can actually run here: the
    in-process mappers are always available; a reference aligner is offered only
    if its conda env is currently installed. Cached so conda is queried once per
    session, not on every rerun."""
    available = set(_INPROCESS_METHODS) | set(available_reference_aligners())
    return [m for m in MAPPING_CHOICES if m in available]


def _browse(kind: str, key: str) -> None:
    """on_click callback: open a native file/folder dialog and store the chosen
    path under session-state ``key``.

    Works because the Streamlit server runs on the user's own machine, so the
    tkinter dialog appears on their screen. Any failure (e.g. no display on a
    remote host) is surfaced as a sidebar warning; typing the path still works.
    """
    st.session_state.pop("_browse_error", None)
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            if kind == "dir":
                chosen = filedialog.askdirectory(title="Select output directory")
            elif kind == "tif":
                chosen = filedialog.askopenfilename(
                    title="Select segmentation mask",
                    filetypes=[("TIFF", "*.tif *.tiff"), ("All files", "*.*")],
                )
            else:
                chosen = filedialog.askopenfilename(
                    title="Select .h5ad file",
                    filetypes=[("AnnData h5ad", "*.h5ad"), ("All files", "*.*")],
                )
        finally:
            root.update()
            root.destroy()
    except Exception as exc:  # noqa: BLE001
        st.session_state["_browse_error"] = f"File dialog unavailable: {exc}"
        return
    if chosen:
        st.session_state[key] = chosen


def _clear_session() -> None:
    """Reset paths, scaffold settings, method selection and the run queue for a
    clean restart.

    Runs as a button ``on_click`` callback (before widgets re-instantiate), which
    is the only point Streamlit allows resetting widget-backed keys. On-disk
    results are left untouched; any background sweep keeps running but is no
    longer tracked.
    """
    st.session_state["cfg_sc"] = ""
    st.session_state["cfg_st"] = ""
    st.session_state["cfg_seg"] = ""
    st.session_state["cfg_out"] = ""
    st.session_state["cfg_linkage"] = LINKAGE_METHODS[0]
    st.session_state["cfg_annotations"] = []
    st.session_state["runs"] = {}
    st.session_state["queue"] = []
    st.session_state.pop("_run_requested", None)
    # Forget which pair was validated, so the next selection reports again.
    st.session_state.pop("_validation_seen", None)
    st.session_state.pop("_validation_reopen", None)
    st.session_state.pop("_show_data_help", None)
    _drop_view_state()


def _drop_view_state() -> None:
    """Drop transient widget state: the method checkboxes, the result pages'
    controls and remembered view (start, method, K), and every per-view /
    best-K control."""
    for k in list(st.session_state.keys()):
        if k.startswith(
            (
                "add_",
                "done_",
                "tab_",
                "cmp",
                "ctrl_",
                "bestk_",
                "ks_",
                "view_",
                "_view_",
                "ov_",
            )
        ):
            del st.session_state[k]


def _run_lock(out_path: Path) -> dict | None:
    """The settings an output folder's existing runs pin down, or ``None`` when it
    holds no runs yet: its starts' annotations, the linkage, and whether it is a
    folder from an earlier version (``legacy``, which takes no new runs).

    Read back from the mappers' ``config.yaml``; a linkage no config records
    comes back as ``None`` and stays editable.
    """
    starts = data_access.list_starts(out_path)
    if not starts:
        return None
    linkages = (data_access.linkage_method_from_config(s.folder) for s in starts)
    return {
        "legacy": data_access.is_legacy_layout(out_path),
        "annotations": [s.annotation for s in starts if s.annotation is not None],
        "linkage_method": next((m for m in linkages if m), None),
    }


def _annotation_row(sc_str: str, lock: dict | None) -> list[str]:
    """Sidebar picker for the annotations to start from besides the Leiden
    over-clustering; returns the chosen obs columns. With nothing picked it reads
    "Start from Leiden over-clustering only".

    Annotations can be added at any time, like mappers: the next run sweeps every
    mapper of the folder on the starts it lacks. The annotations the folder
    already holds runs for stay selected (removing one puts it straight back),
    since their results stay on disk either way.
    """
    recorded = list(lock["annotations"]) if lock is not None else []
    sc_path = Path(sc_str.strip()) if sc_str.strip() else None
    available = (
        obs_columns(str(sc_path)) if sc_path is not None and sc_path.is_file() else []
    )
    annotations = [*recorded, *(a for a in available if a not in recorded)]
    if not annotations:
        st.sidebar.multiselect(
            _ANNOTATIONS_LABEL,
            [],
            disabled=True,
            placeholder=f"{_LEIDEN_ONLY} (no complete annotation in the scRNA-seq data)",
            help=_ANNOTATIONS_HELP,
        )
        return []
    # A choice made for another scRNA-seq file may not exist in this one; the
    # recorded starts are always part of the choice.
    chosen_before = st.session_state.get("cfg_annotations", [])
    st.session_state["cfg_annotations"] = [
        *recorded,
        *(a for a in chosen_before if a in annotations and a not in recorded),
    ]
    chosen = st.sidebar.multiselect(
        _ANNOTATIONS_LABEL,
        annotations,
        key="cfg_annotations",
        persist_state=_PERSIST,
        placeholder=_LEIDEN_ONLY,
        help=_ANNOTATIONS_HELP,
    )
    return [str(a) for a in chosen]


def _linkage_method_row(lock: dict | None) -> str:
    """Sidebar picker for the agglomeration linkage, disabled and pinned to the
    recorded value once the output folder holds runs."""

    def _fmt(method: str) -> str:
        return _LINKAGE_LABELS.get(str(method), str(method))

    if lock is not None and lock["linkage_method"]:
        return str(
            st.sidebar.selectbox(
                "Linkage method",
                [lock["linkage_method"]],
                format_func=_fmt,
                disabled=True,
                help=_LOCKED_HELP,
            )
        )
    return str(
        st.sidebar.selectbox(
            "Linkage method",
            LINKAGE_METHODS,
            format_func=_fmt,
            key="cfg_linkage",
            persist_state=_PERSIST,
            help=_LINKAGE_HELP,
        )
    )


# Centre the icon inside the (stretched) Browse buttons; Streamlit left-aligns
# button labels by default. The Clear button sits on the step-1 header row and
# is shrunk to a compact secondary control.
_SIDEBAR_CSS = (
    "<style>"
    "div[class*='st-key-browse_'] button{"
    "display:flex;justify-content:center;align-items:center;padding:0;}"
    "div[class*='st-key-browse_'] button>div{"
    "display:flex;justify-content:center;align-items:center;width:100%;}"
    "div[class*='st-key-browse_'] button p{margin:0;text-align:center;}"
    # Info + Clear share one container so nothing (no column gap, no leftover
    # column width) can wedge itself between them: a right-aligned flex row
    # whose items keep their natural width.
    "div[class*='st-key-hdr_btns']{display:flex;flex-direction:row;"
    "justify-content:flex-end;align-items:center;gap:0.25rem;width:100%;}"
    "div[class*='st-key-hdr_btns']>div{width:auto;flex:0 0 auto;}"
    "div[class*='st-key-hdr_btns'] button{"
    "padding:0.1rem 0.4rem;min-height:0;width:auto;white-space:nowrap;}"
    "div[class*='st-key-hdr_btns'] button p{font-size:0.75rem;margin:0;}"
    # Group label for the method list: matches a Streamlit widget label
    # (0.875rem) so it lines up with 'Linkage method' above it.
    ".spatialaim-group-label{font-size:0.875rem;line-height:1.6;margin-bottom:0.25rem;}"
    # Theme background like the inputs, not the grey sidebar surface. There
    # is no Streamlit CSS variable for it (--st-* only reaches custom-component
    # shadow roots), but Streamlit sets `color-scheme` on the app root from the
    # active theme, so light-dark() follows a theme switch without a rerun.
    # The two values are Streamlit's default backgroundColor per mode.
    "div[class*='st-key-methods_box']{padding:0.5rem 0.75rem;"
    "background-color:light-dark(#ffffff,#0e1117);}"
    # Selected multiselect values (sidebar annotations, Overview filters): a
    # light tint of the primary blue (gui.__main__.PRIMARY_COLOR) with blue text
    # instead of Streamlit's solid primary-colour tag.
    "[data-testid='stMultiSelectTagsContainer'] [data-tag]{"
    "background-color:light-dark(rgba(28,131,225,0.12),rgba(28,131,225,0.25)) !important;"
    "color:light-dark(#1659a8,#8cc4ff) !important;}"
    "[data-testid='stMultiSelectTagsContainer'] [data-tag] button{color:inherit !important;}"
    # Streamlit's multiselect offers "Select all" (and "Select N matches" while
    # filtering) with no option to turn it off; the annotation picker should
    # not have it. Its dropdown renders outside the sidebar, so this applies to
    # every multiselect of the app.
    # The list is virtualized: each row sits in its own absolutely positioned
    # 40px wrapper, so the bulk row's wrapper is hidden and the rest of the list
    # is pulled up by one row (40px = Streamlit's dropdownItemHeight).
    "div[data-testid='stMultiSelectDropdown'] div:has(> [data-key='__select_all__']),"
    "div[data-testid='stMultiSelectDropdown'] div:has(> [data-key='__select_matches__'])"
    "{display:none !important;}"
    "div[data-testid='stMultiSelectDropdown'] [role='listbox']:has("
    "[data-key='__select_all__'],[data-key='__select_matches__'])>div"
    "{margin-top:-40px;}"
    "div[class*='st-key-methods_box'] label{font-size:0.875rem;}"
    # Even rhythm inside the box: the container gap is 0, so every row's
    # spacing comes from these margins alone.
    "div[class*='st-key-methods_box'] label{padding:0.15rem 0;}"
    ".spatialaim-mapper-group{font-size:0.7rem;font-weight:600;letter-spacing:0.04em;"
    "text-transform:uppercase;opacity:0.55;margin:0.5rem 0 1rem;}"
    ".spatialaim-mapper-group.spatialaim-first{margin-top:0;}"
    "</style>"
)


def _data_header() -> bool:
    """Step 1's header row with the Info and Clear buttons. Opens the input
    requirements dialog when Info was clicked and returns whether it did."""
    col_hdr, col_btns = st.sidebar.columns([3, 2], vertical_alignment="center")
    col_hdr.subheader("1. Select data")
    with col_btns:
        btns = st.container(key="hdr_btns")
    btns.button(
        "Info",
        key="info_btn",
        on_click=request_data_help,
        help="Expected input data",
    )
    btns.button(
        "Clear",
        key="clear_btn",
        on_click=_clear_session,
        help="Reset all inputs, the method selection and the run queue. "
        "Computed results on disk are kept.",
    )
    # Opening the requirements card takes precedence over the validation modal:
    # Streamlit allows one dialog per rerun.
    show_data_help = st.session_state.pop("_show_data_help", False)
    if show_data_help:
        data_requirements_dialog()
    return show_data_help


def _path_row(label: str, key: str, kind: str, browse_key: str) -> str:
    """A path text field with an icon-only Browse button to its right
    (bottom-aligned so it lines up with the input, not the label)."""
    col_in, col_btn = st.sidebar.columns([5, 1], vertical_alignment="bottom")
    val = col_in.text_input(label, key=key, persist_state=_PERSIST)
    col_btn.button(
        "📁",
        key=browse_key,
        on_click=_browse,
        args=(kind, key),
        help="Browse…",
        width="stretch",
    )
    return val


def _optional_path(text: str) -> Path | None:
    """The path typed into a field, or ``None`` when it is empty."""
    return Path(text.strip()) if text.strip() else None


def _method_rows(
    runs: dict[str, compute.MapperRun],
    queue: list[str],
    computed: list[str],
    running_mapper: str | None,
    missing: dict[str, int],
) -> list[str]:
    """The mapping-method checkboxes, grouped into baseline and reference
    mappers; returns the methods ticked to run.

    Every choice stays listed; its state decides whether the row is live. A method
    that is computed, running or queued is locked on: its row shows ticked and
    disabled instead of leaving the list. A computed method that lacks some of
    the selected starting points (``missing``: their number) says so; the next
    run fills them in.
    """
    taken = set(computed) | set(queue)
    if running_mapper:
        taken.add(running_mapper)
    selectable = set(_selectable_methods())

    st.sidebar.markdown(
        "<div class='spatialaim-group-label'>Select mapping methods to run "
        "(1 or more)</div>",
        unsafe_allow_html=True,
    )
    # nearest_centroid is the default mapper, so it starts ticked; a manual
    # untick sticks (setdefault only fills the key in on a fresh session).
    st.session_state.setdefault("add_nearest_centroid", True)
    box = st.sidebar.container(border=True, gap=0, key="methods_box")
    selected = []
    heading_shown: set[str] = set()
    for m in MAPPING_CHOICES:
        heading = "Baseline mappers" if m in _INPROCESS_METHODS else "Reference mappers"
        if heading not in heading_shown:
            first = "" if heading_shown else " spatialaim-first"
            heading_shown.add(heading)
            box.markdown(
                f"<div class='spatialaim-mapper-group{first}'>{heading}</div>",
                unsafe_allow_html=True,
            )
        label = MAPPER_LABELS.get(m, m)
        if m in taken:
            # A running mapper already has config.yaml on disk (so it is in
            # `computed`); don't call it done until its thread finishes.
            if m == running_mapper:
                state = "⏳ running"
            elif m in queue:
                state = "🕒 queued"
            elif missing.get(m):
                n = missing[m]
                state = f"{n} starting point{'s' if n > 1 else ''} to run"
            else:
                state = "computed"
            # Own key namespace: `add_<m>` may already hold the user's tick.
            box.checkbox(
                f"{label} — {state}", value=True, disabled=True, key=f"done_{m}"
            )
        elif m in selectable:
            if box.checkbox(label, key=f"add_{m}", persist_state=_PERSIST):
                selected.append(m)
        else:
            box.checkbox(
                f"{label} — conda env not found", disabled=True, key=f"add_{m}"
            )
    return selected


def render_sidebar() -> argparse.Namespace | None:
    """Render the sidebar (data paths, annotation starts, linkage, method
    selection + run queue).

    Returns the run settings once the paths are valid, else ``None``. Their
    ``output_dir`` and ``pair_dir`` are both the output folder; the app points
    ``output_dir`` at the start folder on show.
    """
    if demo.ACTIVE:
        return _render_demo_sidebar()
    st.sidebar.title("SpatialAIM GUI")

    # -- step 1: data paths ----------------------------------------------
    st.sidebar.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)
    show_data_help = _data_header()
    st.session_state.setdefault("cfg_sc", "")
    st.session_state.setdefault("cfg_st", "")
    st.session_state.setdefault("cfg_seg", "")
    st.session_state.setdefault("cfg_out", "")
    st.session_state.setdefault("cfg_linkage", LINKAGE_METHODS[0])

    sc_str = _path_row("scRNA-seq .h5ad path", "cfg_sc", "file", "browse_sc")
    st_str = _path_row("ST .h5ad path", "cfg_st", "file", "browse_st")
    seg_str = _path_row("Segmentation path (optional)", "cfg_seg", "tif", "browse_seg")
    out_str = _path_row("Output directory", "cfg_out", "dir", "browse_out")
    if st.session_state.get("_browse_error"):
        st.sidebar.warning(st.session_state.pop("_browse_error"))

    sc_path = _optional_path(sc_str)
    st_path = _optional_path(st_str)
    out_path = _optional_path(out_str)
    # Optional: an empty field means no mask. A set path is validated with the
    # pair (a missing file shows up there as an error) and only passed on to the
    # runs when it points at a file.
    seg_path = _optional_path(seg_str)

    incomplete = (
        sc_path is None
        or not sc_path.is_file()
        or st_path is None
        or not st_path.is_file()
        or out_path is None
    )

    # Both paths point at real files: validate the pair before anything is
    # configured, so contract violations surface here and not deep in a sweep.
    if not incomplete:
        validation_row(sc_path, st_path, seg_path, allow_dialog=not show_data_help)

    # -- step 2: scaffold knobs + methods ---------------------------------
    # The header stays visible while step 1 is incomplete; its controls need the
    # paths (the annotations read the scRNA-seq obs, the rest lock against the output
    # folder), so they are replaced by a hint until then.
    st.sidebar.divider()
    st.sidebar.subheader("2. Configure method")
    if incomplete:
        st.sidebar.caption("Select data first.")
        return None
    out_path.mkdir(parents=True, exist_ok=True)

    # Settings the runs are built from lock once this folder holds runs.
    lock = _run_lock(out_path)
    legacy = lock is not None and lock["legacy"]
    if legacy:
        st.sidebar.info(_LEGACY_NOTICE)
    linkage_method = _linkage_method_row(lock)
    annotations = _annotation_row(sc_str, lock)

    settings = argparse.Namespace(
        scdata=sc_path,
        stdata=st_path,
        segmentation=seg_path if seg_path is not None and seg_path.is_file() else None,
        output_dir=out_path,
        pair_dir=out_path,
        k_min=_DEFAULT_K_MIN,
        k_max=_DEFAULT_K_MAX,
        k_step=_DEFAULT_K_STEP,
        linkage_method=str(linkage_method),
        annotations=annotations,
        legacy=legacy,
    )

    runs: dict[str, compute.MapperRun] = st.session_state.setdefault("runs", {})
    queue: list[str] = st.session_state.setdefault("queue", [])
    computed = data_access.list_pair_mappers(out_path)
    running = next((r for r in runs.values() if r.is_running()), None)
    running_mapper = running.mapper if running is not None else None
    # Per computed method, the selected starting points it has not finished and
    # that no running or queued sweep of it will cover.
    missing: dict[str, int] = {}
    if not legacy:
        for m in computed:
            if m in queue:
                continue
            todo = compute.missing_starts(out_path, m, annotations)
            if m == running_mapper:
                todo = [s for s in todo if s not in running.starts]
            if todo:
                missing[m] = len(todo)
    selected = _method_rows(runs, queue, computed, running_mapper, missing)

    st.sidebar.divider()
    if st.sidebar.button(
        "Run",
        disabled=not (selected or missing) or legacy,
        width="stretch",
        help="The selected methods compute one after another, each from the "
        "Leiden over-clustering and from every chosen annotation; methods with "
        "results only compute the starting points they are missing.",
    ):
        for m in [*selected, *missing]:
            if m not in queue:
                queue.append(m)
        # Flag a pending run so the empty-state prompt never flashes before the
        # first tab appears; cleared once any tab renders. No st.rerun() here:
        # letting main() continue in this same run starts the method and renders
        # its progress tab immediately, so the idle "select methods" prompt is
        # replaced in place instead of lingering until the next run.
        st.session_state["_run_requested"] = True

    # Surface failures.
    # A run that failed in a later start has already written its earlier ones,
    # so a failure is shown whether or not the mapper has results on disk.
    for m, r in list(runs.items()):
        if r.error and not r.is_running():
            st.sidebar.error(f"'{m}' failed — see terminal.")

    return settings


def _render_demo_sidebar() -> argparse.Namespace | None:
    """The demo's sidebar: pick a precomputed pair, then see how it was computed.

    Returns the same settings as ``render_sidebar`` (``None`` without any pair),
    pinned to what the pair output holds, so the result pages read it like a
    finished local run.
    """
    demo.banner()
    st.sidebar.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)
    st.sidebar.title("SpatialAIM")
    st.sidebar.subheader("1. Select data")
    pairs = demo.list_pairs()
    if not pairs:
        st.sidebar.caption("No precomputed results available.")
        return None
    by_id = {p.pair_id: p for p in pairs}
    pair_id = st.sidebar.selectbox(
        "Dataset pair (scRNA-seq × ST)",
        list(by_id),
        format_func=lambda i: by_id[i].label,
        key="demo_pair",
        persist_state=_PERSIST,
        on_change=_drop_view_state,
        help="Public scRNA-seq references and spatial slices, each mapped in "
        "advance. Choosing another pair resets the result pages.",
    )
    pair = by_id[pair_id]
    # scRNA-seq × ST authors; once when both sides come from the same authors.
    authors = list(dict.fromkeys(a for a in (pair.author, pair.st_author) if a))
    details = " · ".join(x for x in (pair.tissue, " × ".join(authors)) if x)
    if details:
        st.sidebar.caption(details)
    links = [
        f"{side} " + " · ".join(f"[{text}]({url})" for text, url in parts if url)
        for side, parts in (
            ("scRNA-seq:", (("paper", pair.sc_paper), ("data", pair.sc_data))),
            ("ST:", (("paper", pair.st_paper), ("data", pair.st_data))),
        )
        if any(url for _, url in parts)
    ]
    if links:
        st.sidebar.caption(" &nbsp;|&nbsp; ".join(links))

    st.sidebar.divider()
    st.sidebar.subheader("2. Method")
    linkage_method = str(
        st.sidebar.selectbox(
            "Linkage method",
            list(pair.outputs),
            format_func=lambda m: _LINKAGE_LABELS.get(m, m),
            key=f"demo_linkage_{pair.pair_id}",
            persist_state=_PERSIST,
            on_change=_drop_view_state,
            disabled=len(pair.outputs) == 1,
            help=f"{_LINKAGE_HELP} Changing it resets the result pages.",
        )
    )
    pair_dir = pair.outputs[linkage_method]
    view_key = f"{pair.pair_id}_{linkage_method}"
    lock = _run_lock(pair_dir)
    annotations = lock["annotations"] if lock is not None else []
    # The full sidebar's controls, disabled and showing what was computed.
    st.sidebar.multiselect(
        _ANNOTATIONS_LABEL,
        annotations,
        default=annotations,
        disabled=True,
        placeholder=_LEIDEN_ONLY,
        key=f"demo_annotations_{view_key}",
        help="SpatialAIM always starts from a Leiden over-clustering of the "
        "reference; these annotations were additionally used as starting points.",
    )
    _demo_method_rows(data_access.list_pair_mappers(pair_dir), view_key)
    return argparse.Namespace(
        scdata=pair.sc_path,
        stdata=pair.st_path,
        segmentation=pair.segmentation,
        output_dir=pair_dir,
        pair_dir=pair_dir,
        k_min=_DEFAULT_K_MIN,
        k_max=_DEFAULT_K_MAX,
        k_step=_DEFAULT_K_STEP,
        linkage_method=str(linkage_method),
        annotations=annotations,
        legacy=lock is not None and lock["legacy"],
    )


def _demo_method_rows(computed: list[str], view_key: str) -> None:
    """The demo's read-only method list: ``_method_rows``' box with every method
    disabled, ticked where it was computed for the pair output."""
    st.sidebar.markdown(
        "<div class='spatialaim-group-label'>Computed mapping methods</div>",
        unsafe_allow_html=True,
    )
    box = st.sidebar.container(border=True, gap=0, key="methods_box")
    heading_shown: set[str] = set()
    for m in MAPPING_CHOICES:
        heading = "Baseline mappers" if m in _INPROCESS_METHODS else "Reference mappers"
        if heading not in heading_shown:
            first = "" if heading_shown else " spatialaim-first"
            heading_shown.add(heading)
            box.markdown(
                f"<div class='spatialaim-mapper-group{first}'>{heading}</div>",
                unsafe_allow_html=True,
            )
        box.checkbox(
            MAPPER_LABELS.get(m, m),
            value=m in computed,
            disabled=True,
            key=f"demo_method_{m}_{view_key}",
        )


@st.dialog("Feedback")
def _feedback_dialog() -> None:
    """Where to send ideas and bug reports — the tool has no in-app channel."""
    st.markdown(
        "We are happy about feedback to make this tool more useful to you! "
        "If you have any ideas, wishes or you find a bug simply write a mail to "
        "[dominik.pysch@fau.de](mailto:dominik.pysch@fau.de). Thank you!"
    )
    if st.button("Close", key="feedback_close"):
        st.rerun()


_FOOTER_CSS = (
    "<style>"
    "div[data-testid='stSidebarContent']{display:flex;flex-direction:column;}"
    "div[data-testid='stSidebarUserContent']{flex:1 0 auto;display:flex;"
    "flex-direction:column;padding-bottom:1rem;}"
    "div[data-testid='stSidebarUserContent']>div{display:flex;"
    "flex-direction:column;flex:1 0 auto;}"
    # Direct child only: the selector would otherwise stretch every nested
    # vertical block (the methods card, the header button row) as well.
    "div[data-testid='stSidebarUserContent']>div>div[data-testid='stVerticalBlock']"
    "{flex:1 0 auto;}"
    # Streamlit may wrap a keyed container in a layout div; push whichever
    # one is the sidebar column's child.
    "div[data-testid='stLayoutWrapper']:has(> div[class*='st-key-sidebar_footer']),"
    "div[class*='st-key-sidebar_footer']{margin-top:auto;}"
    "div[class*='st-key-feedback_btn'] button{opacity:0.5;}"
    "div[class*='st-key-feedback_btn'] button:hover{opacity:1;}"
    "div[class*='st-key-feedback_btn'] button p{font-size:0.8rem;}"
    # The demo's legal links: plain dimmed text like the Feedback link, without
    # the page-link pill (also not on the page currently shown).
    "div[class*='st-key-sidebar_footer'] a{opacity:0.5;padding:0;"
    "background:none !important;min-height:0;}"
    "div[class*='st-key-sidebar_footer'] a:hover{opacity:1;}"
    "div[class*='st-key-sidebar_footer'] a p,"
    "div[class*='st-key-sidebar_footer'] a span{font-size:0.8rem;font-weight:400;}"
    # Every footer item keeps its natural width, so each row reads left to right.
    "div[class*='st-key-sidebar_footer'] div[class*='stElementContainer']"
    "{flex:0 0 auto !important;width:auto !important;}"
    # A hairline above the demo's footer.
    "div[class*='st-key-sidebar_footer']:has(a){border-top:1px solid "
    "light-dark(rgba(49,51,63,0.15),rgba(250,250,250,0.15));padding-top:0.5rem;}"
    "</style>"
)


def render_sidebar_footer(legal_pages: list | None = None) -> None:
    """Feedback link at the very bottom of the sidebar, and in the demo the
    links to the imprint and privacy policy (``legal_pages``).

    Called from ``main`` rather than from ``render_sidebar`` so it sits below both of
    that function's exits — including the early one taken while the data paths
    are still incomplete.

    A ``tertiary`` button is borderless plain text, so this reads as a link, not
    a control; the CSS only shrinks and dims it further, since it is a footnote
    rather than part of the workflow.

    Pinning it to the bottom edge takes a chain of stretches, because Streamlit
    lays the sidebar out as static blocks sized to their content: the content
    column has to fill the height left over by the sidebar header, its wrapper
    and the root vertical block have to pass that height on, and only then does
    ``margin-top:auto`` on this one element push it down. Streamlit's own
    ``padding-bottom`` on the content column reserves ~96px that would otherwise
    leave the link floating well above the edge, so it is cut to 1rem.
    """
    st.sidebar.markdown(_FOOTER_CSS, unsafe_allow_html=True)
    footer = st.sidebar.container(gap="xsmall", key="sidebar_footer")
    row = footer.container(horizontal=True, vertical_alignment="center")
    full_version = _app_version()
    row.caption(
        f"v{_release_version(full_version)}", help=f"spatial-aim {full_version}"
    )
    if row.button(
        "Feedback",
        icon=":material/chat_bubble:",
        type="tertiary",
        key="feedback_btn",
    ):
        _feedback_dialog()
    if legal_pages:
        legal = footer.container(
            horizontal=True, vertical_alignment="center", gap="medium"
        )
        for page in legal_pages:
            legal.page_link(page, label=page.title, width="content")


def _app_version() -> str:
    """The installed ``spatial-aim`` version (from the git tag)."""
    try:
        return version("spatial-aim")
    except PackageNotFoundError:  # source tree without install
        return "0.0.0"


def _release_version(full: str) -> str:
    """``X.Y.Z`` of a version, without dev / local parts
    (``1.0.1.dev2+g05a8291`` -> ``1.0.1``)."""
    try:
        return Version(full).base_version
    except InvalidVersion:
        return full
