"""
The sidebar: data paths (step 1), the start clusters, linkage and mapping methods
to run (step 2), and the footer with the version and the feedback link.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import streamlit as st
from packaging.version import InvalidVersion, Version

from spatialaim import LINKAGE_METHODS, MAPPING_CHOICES
from spatialaim.config import _INPROCESS_METHODS
from spatialaim.gui import compute, data_access
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

# Sidebar label for the default start clusters (no annotation column chosen).
_LEIDEN_START_LABEL = "No (default, use Leiden over-clustering)"
_NO_ANNOTATION_LABEL = "No annotations found in scRNA data"

# Display-only labels; the raw method names go into the run config.
_LINKAGE_LABELS = {
    "average": "average - allow for outlier cell types",
    "ward": "ward - more balanced cell types",
}

_LOCKED_HELP = (
    "Fixed by the runs already in this output folder: the cached reference scaffold "
    "and every persisted tree cut are tied to it. Point the output directory "
    "somewhere new to choose again."
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
    st.session_state["cfg_start"] = _LEIDEN_START_LABEL
    st.session_state["runs"] = {}
    st.session_state["queue"] = []
    st.session_state.pop("_run_requested", None)
    # Forget which pair was validated, so the next selection reports again.
    st.session_state.pop("_validation_seen", None)
    st.session_state.pop("_validation_reopen", None)
    st.session_state.pop("_show_data_help", None)
    # Drop transient widget state: the method checkboxes, shared controls, and
    # every per-tab / best-K control.
    for k in list(st.session_state.keys()):
        if k.startswith(("add_", "done_", "tab_", "cmp", "ctrl_", "bestk_", "ks_")):
            del st.session_state[k]


def _run_lock(out_str: str) -> dict | None:
    """The scaffold-defining settings an output folder's existing runs pin down, or
    ``None`` when it holds no runs yet.

    Read back from any mapper's ``config.yaml``; a value it does not record
    comes back as ``None`` and stays editable.
    """
    out_path = Path(out_str.strip()) if out_str.strip() else None
    if out_path is None or not data_access.list_mappers(out_path):
        return None
    return {
        "start_from_annotation": data_access.start_from_annotation_from_config(
            out_path
        ),
        "linkage_method": data_access.linkage_method_from_config(out_path),
    }


def _start_cluster_row(sc_str: str, lock: dict | None) -> str | None:
    """Sidebar picker for the start clusters; returns the annotation column or
    ``None`` for the Leiden over-clustering.

    Once an output dir holds a run, its recorded choice is shown as a disabled
    dropdown rather than letting the two modes mix in one folder.
    """
    help_text = (
        "The agglomeration tree is built over Leiden over-clusters by default; an "
        "scRNA obs column uses that annotation's cell types instead."
    )

    def _fmt(option: str) -> str:
        return option if option == _LEIDEN_START_LABEL else f".obs key {option}"

    if lock is not None:
        recorded = lock["start_from_annotation"]
        st.sidebar.selectbox(
            "Override starting clusters",
            [recorded or _LEIDEN_START_LABEL],
            format_func=_fmt,
            disabled=True,
            help=_LOCKED_HELP,
        )
        return recorded

    sc_path = Path(sc_str.strip()) if sc_str.strip() else None
    annotations = (
        obs_columns(str(sc_path)) if sc_path is not None and sc_path.is_file() else []
    )
    if not annotations:
        # Nothing to override with: Streamlit can only grey out a whole selectbox,
        # not a single option, so the box itself carries the message.
        st.sidebar.selectbox(
            "Override starting clusters",
            [_NO_ANNOTATION_LABEL],
            disabled=True,
            help=help_text,
        )
        return None
    choice = st.sidebar.selectbox(
        "Override starting clusters",
        [_LEIDEN_START_LABEL, *annotations],
        format_func=_fmt,
        key="cfg_start",
        help=help_text,
    )
    return None if choice == _LEIDEN_START_LABEL else str(choice)


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
            help="Linkage for the agglomeration tree over the start clusters. "
            "'average' (UPGMA) peels small tight groups off a dominant cell type; "
            "'ward' carries a size term and tends to produce balanced cell types.",
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
    val = col_in.text_input(label, key=key)
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
) -> list[str]:
    """The mapping-method checkboxes, grouped into baseline and reference
    mappers; returns the methods ticked to run.

    Every choice stays listed; its state decides whether the row is live. A method
    that is computed, running or queued is locked on: its row shows ticked and
    disabled instead of leaving the list.
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
            elif m in computed:
                state = "computed"
            else:
                state = "🕒 queued"
            # Own key namespace: `add_<m>` may already hold the user's tick.
            box.checkbox(
                f"{label} — {state}", value=True, disabled=True, key=f"done_{m}"
            )
        elif m in selectable:
            if box.checkbox(label, key=f"add_{m}"):
                selected.append(m)
        else:
            box.checkbox(
                f"{label} — conda env not found", disabled=True, key=f"add_{m}"
            )
    return selected


def render_sidebar() -> argparse.Namespace | None:
    """Render the sidebar (data paths, start clusters, linkage, method selection +
    run queue).

    Returns the run settings once the paths are valid, else ``None``.
    """
    st.sidebar.title("SpatialAIM GUI")

    # -- step 1: data paths ----------------------------------------------
    st.sidebar.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)
    show_data_help = _data_header()
    st.session_state.setdefault("cfg_sc", "")
    st.session_state.setdefault("cfg_st", "")
    st.session_state.setdefault("cfg_seg", "")
    st.session_state.setdefault("cfg_out", "")
    st.session_state.setdefault("cfg_linkage", LINKAGE_METHODS[0])

    sc_str = _path_row("scRNA .h5ad path", "cfg_sc", "file", "browse_sc")
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
    # paths (start clusters read the scRNA obs, the rest lock against the output
    # folder), so they are replaced by a hint until then.
    st.sidebar.divider()
    st.sidebar.subheader("2. Configure method")
    if incomplete:
        st.sidebar.caption("Select data first.")
        return None
    out_path.mkdir(parents=True, exist_ok=True)

    # Settings the scaffold is built from lock once this folder holds runs.
    lock = _run_lock(out_str)
    start_from_annotation = _start_cluster_row(sc_str, lock)
    linkage_method = _linkage_method_row(lock)

    settings = argparse.Namespace(
        scdata=sc_path,
        stdata=st_path,
        segmentation=seg_path if seg_path is not None and seg_path.is_file() else None,
        output_dir=out_path,
        k_min=_DEFAULT_K_MIN,
        k_max=_DEFAULT_K_MAX,
        k_step=_DEFAULT_K_STEP,
        linkage_method=str(linkage_method),
        start_from_annotation=start_from_annotation,
    )

    runs: dict[str, compute.MapperRun] = st.session_state.setdefault("runs", {})
    queue: list[str] = st.session_state.setdefault("queue", [])
    computed = data_access.list_mappers(out_path)
    running_mapper = next((m for m, r in runs.items() if r.is_running()), None)
    selected = _method_rows(runs, queue, computed, running_mapper)

    st.sidebar.divider()
    if st.sidebar.button(
        "Run",
        disabled=not selected,
        width="stretch",
        help="The selected methods compute one after another.",
    ):
        for m in selected:
            if m not in queue:
                queue.append(m)
        # Flag a pending run so the empty-state prompt never flashes before the
        # first tab appears; cleared once any tab renders. No st.rerun() here:
        # letting main() continue in this same run starts the method and renders
        # its progress tab immediately, so the idle "select methods" prompt is
        # replaced in place instead of lingering until the next run.
        st.session_state["_run_requested"] = True

    # Surface failures.
    for m, r in list(runs.items()):
        if r.error and not r.is_running() and m not in computed:
            st.sidebar.error(f"'{m}' failed — see terminal.")

    return settings


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
    "</style>"
)


def render_sidebar_footer() -> None:
    """Feedback link at the very bottom of the sidebar.

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
    footer = st.sidebar.container(
        horizontal=True, vertical_alignment="center", key="sidebar_footer"
    )
    full_version = _app_version()
    footer.caption(
        f"v{_release_version(full_version)}", help=f"spatial-aim {full_version}"
    )
    if footer.button(
        "Feedback",
        icon=":material/chat_bubble:",
        type="tertiary",
        key="feedback_btn",
    ):
        _feedback_dialog()


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
