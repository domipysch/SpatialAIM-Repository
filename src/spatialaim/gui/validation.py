"""
The sidebar's input checks: the selected pair validated with the checks behind
``spatialaim validate`` (reused as-is), its status line and findings dialog, and
the "Info" dialog stating what SpatialAIM expects as input.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from spatialaim.data.validate import PairFindings, check_pair


@st.cache_data(show_spinner="Validating dataset pair…")
def _validate_pair(
    sc_path_str: str, st_path_str: str, seg_path_str: str = ""
) -> PairFindings:
    """Findings for the selected sc/ST pair (and the ST slice's segmentation mask,
    if one is given), from ``spatialaim.data.validate``.

    Cached per path combination: the checks read both count matrices, so they run
    once per selection rather than on every rerun.
    """
    return check_pair(
        Path(sc_path_str),
        Path(st_path_str),
        segmentation_path=Path(seg_path_str) if seg_path_str else None,
    )


def _subject_heading(key: str) -> str:
    kind, _, name = key.partition(":")
    return {
        "sc": f"scRNA reference — {name}",
        "st": f"ST slice — {name}",
    }.get(kind, "Pair (shared genes)")


@st.dialog("Dataset validation", width="large")
def _validation_dialog(findings: PairFindings, with_segmentation: bool) -> None:
    """Modal listing what the validators found. Purely informative — closing it
    leaves the pair selected and the run configurable."""
    if findings.errors:
        st.markdown(
            "The selected pair violates the SpatialAIM input contract. You can continue, "
            "but the run is likely to fail or to produce meaningless results."
        )
    else:
        st.markdown(
            "The selected pair is usable, but the checks flagged the following."
        )

    for subject in findings.subjects:
        if not (subject.errors or subject.warns):
            continue
        kind = subject.key.partition(":")[0]
        st.markdown(f"**{_subject_heading(subject.key)}**")
        for msg in subject.errors:
            st.error(msg.removeprefix(f"{kind}: "), icon="❌")
        for msg in subject.warns:
            st.warning(msg.removeprefix(f"{kind}: "), icon="⚠️")

    seg_arg = " --segmentation <mask.tif>" if with_segmentation else ""
    st.caption(
        "Same checks as "
        f"`spatialaim validate --scdata <sc.h5ad> --stdata <st.h5ad>{seg_arg}`."
    )
    if st.button("Continue", type="primary", key="validation_close"):
        st.rerun()


def _reopen_validation() -> None:
    st.session_state["_validation_reopen"] = True


def request_data_help() -> None:
    st.session_state["_show_data_help"] = True


@st.dialog("What SpatialAIM expects as input", width="large")
def data_requirements_dialog() -> None:
    """Reference card for the two input files and the pair they form.

    States the contract the validators enforce - keep the two in step when a
    check in ``spatialaim.data.validate`` changes.
    """
    st.markdown(
        "SpatialAIM maps **scRNA-seq reference data** onto **imaging-based high-resolution spatial transcriptomics data** (e.g. MERFISH / Xenium / seqFISH / osmFISH). "
        "Both are expected as `.h5ad` (AnnData) files holding **raw counts**."
    )

    st.markdown("**Required input**")
    st.markdown("""
| | scRNA reference | ST slice |
|---|---|---|
| file type | `.h5ad` | `.h5ad` |
| **`X`** | raw counts, cells × genes | raw counts, "cells" × genes |
| **`var_names`** | gene symbols, **UPPERCASE** and unique | gene symbols, **UPPERCASE** and unique |
| **`obsm["spatial"]`** | - | `(n_cells, 2)` x/y coordinates |
""")

    st.markdown("**Annotations are optional**")
    st.markdown(
        "In default settings SpatialAIM does not need one — it over-clusters the reference itself. An scRNA `obs` "
        "column with cell types can still be chosen under *Override starting "
        "clusters* to build the tree over that annotation instead."
    )
    st.markdown("""
| | scRNA reference |
|---|---|
| **`obs["<YOUR-CELLTYPE-KEY>"]`** | cell type annotation columns |
""")

    st.markdown("**Segmentation mask is optional**")
    st.markdown(
        "A cell segmentation of the ST slice adds the **shape features** of the "
        "spatial cells (area, eccentricity, solidity) and the **shape "
        "consistency** of the mapping: how uniform the cell shapes within each "
        "mapped cell type are."
    )
    st.markdown(
        "Segmentation mask: 2D `.tif` / `.tiff`, dtype **`uint32`**. `0` = "
        "background, cells labelled `1 … max` **without gaps**; labels without a "
        "cell in the ST data are allowed."
    )
    st.markdown("""
| | ST slice |
|---|---|
| **`obs["segmentation_id"]`** | integer, each cell's label in the mask, `0` = cell not segmented; non-zero ids unique and present in the mask |
| **`uns["segmentation_pixel_size_um"]`** | optional: µm per mask pixel (cell areas are then shown in µm², else in pixels) |
""")


def validation_row(
    sc_path: Path,
    st_path: Path,
    seg_path: Path | None = None,
    *,
    allow_dialog: bool = True,
) -> None:
    """Validate the selected pair (with the optional segmentation mask) and report
    the verdict above the step-2 divider.

    Clean pairs get a one-line tick; anything else opens the modal once per pair
    (re-openable via *Details*). Findings never block: the user closes the modal
    and configures the run as before. ``allow_dialog=False`` keeps the status line
    but suppresses the modal for this rerun, when another dialog already owns it.
    """
    seg_str = str(seg_path) if seg_path is not None else ""
    key = f"{sc_path}||{st_path}||{seg_str}"
    try:
        findings = _validate_pair(str(sc_path), str(st_path), seg_str)
    except Exception as exc:  # noqa: BLE001 - a failed check must not kill the app
        st.sidebar.caption(f"⚠️ Could not validate the dataset pair: {exc}")
        return

    if findings.ok:
        what = (
            "Dataset pair and segmentation" if seg_path is not None else "Dataset pair"
        )
        st.sidebar.caption(f"✅ {what} match requirements")
        return

    n_err, n_warn = len(findings.errors), len(findings.warns)
    counts = [f"{n_err} error(s)"] if n_err else []
    counts += [f"{n_warn} warning(s)"] if n_warn else []
    col_txt, col_btn = st.sidebar.columns([3, 1], vertical_alignment="center")
    col_txt.caption(f"{'❌' if n_err else '⚠️'} Validation: {', '.join(counts)}")
    col_btn.button(
        "Details",
        key="validation_details",
        on_click=_reopen_validation,
        width="stretch",
    )

    # Auto-open once per pair; afterwards only on demand.
    seen = st.session_state.get("_validation_seen")
    wanted = st.session_state.pop("_validation_reopen", False) or seen != key
    if wanted and allow_dialog:
        st.session_state["_validation_seen"] = key
        _validation_dialog(findings, seg_path is not None)
