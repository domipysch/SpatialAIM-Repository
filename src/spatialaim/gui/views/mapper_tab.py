"""
One mapper's tab: its run progress while computing, then the "Choose K" card,
the headline UMAP(s) + spatial map with the confidence controls, and the report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import streamlit as st

from spatialaim.adata_schema import OBSM_UMAP_SHARED_GENES
from spatialaim.gui import compute, data_access
from spatialaim.gui.loaders import load_scaffold, load_soft, spatial_coords
from spatialaim.gui.render.common import SpatialContext
from spatialaim.gui.render.headline import render_headline_figure
from spatialaim.gui.views.cards import export_popover
from spatialaim.gui.views.cell_type_modal import open_cell_type_dialog, reopened_state
from spatialaim.gui.views.choose_k import render_choose_k
from spatialaim.gui.views.controls import Controls
from spatialaim.gui.views.report import render_report
from spatialaim.gui.widgets.headline_plot import headline_plot
from spatialaim.gui.widgets.slider import live_slider

# Shown under the mapper tabs' headline plot (not the reference tab's UMAPs).
_PLOT_CLICK_HINT = (
    "**Left click:** Show / hide cell type · **Right click:** Inspect cell type"
)


def _headline(
    mapper: str,
    root: Path,
    k: int,
    args: argparse.Namespace,
    *,
    key_prefix: str,
    show_shared_umap: bool = False,
    conf_controls: bool = False,
    threshold: float = 0.0,
    plot_confidence: bool = False,
) -> None:
    """UMAP(s) + live spatial plot sharing one state legend.

    When ``conf_controls`` is set and the mapper wrote per-spot confidence, the
    confidence-threshold slider and the "Plot confidence" checkbox are rendered
    directly below the plots, aligned under the (rightmost) spatial panel, and
    their values drive this tab's spatial colouring. Otherwise ``threshold`` /
    ``plot_confidence`` are taken from the arguments — the Compare tab passes
    neither and sets ``conf_controls=False``, so it never shows the controls.
    """
    _P, hard, confidence = load_soft(str(root), k)
    coords = spatial_coords(str(args.stdata))
    have_conf = confidence is not None
    have_spatial = coords is not None

    # Build the reference scaffold for the UMAP panel.
    adata_sc = load_scaffold(args, warn="Scaffold build failed — UMAP unavailable")

    have_umap = adata_sc is not None
    # Mirror render_headline_figure's panel logic so the confidence
    # controls line up under the correct (rightmost) column.
    have_shared = (
        have_umap and show_shared_umap and OBSM_UMAP_SHARED_GENES in adata_sc.obsm
    )
    n_panels = int(have_shared) + int(have_umap) + int(have_spatial)

    # Reserve the plot's slot; the confidence controls render *below* it so the
    # figure stays on top and the controls sit under the spatial panel.
    plot_slot = st.container()

    if conf_controls and have_conf and have_spatial and n_panels:
        control_cols = st.columns(n_panels)
        with control_cols[-1]:
            threshold = live_slider(
                "Confidence threshold",
                0.0,
                1.0,
                0.01,
                0.0,
                key=f"{key_prefix}_thr",
            )
            plot_confidence = st.checkbox(
                "Plot confidence",
                value=False,
                key=f"{key_prefix}_pconf",
                help="Colour cells by confidence intensity instead of assigned "
                "cell type.",
            )

    # One figure, all subplots (UMAP(s) + spatial) sharing a single state legend.
    # Rendered by a client-side component so clicking a legend entry OR any point
    # toggles that state (single click) / isolates it (double click); inactive
    # states are drawn in a light colour rather than hidden.
    if coords is None and adata_sc is None:
        with plot_slot:
            st.info("Nothing to plot: no spatial coords and no UMAP scaffold.")
        return

    try:
        fig = render_headline_figure(
            coords,
            hard,
            confidence,
            threshold,
            k,
            adata_sc=adata_sc,
            root=root,
            plot_confidence=plot_confidence,
            show_shared_umap=show_shared_umap,
            st_path=str(args.stdata),
        )
    except Exception as exc:  # noqa: BLE001
        with plot_slot:
            st.warning(f"UMAP rendering failed — showing spatial only: {exc}")
        fig = render_headline_figure(
            coords,
            hard,
            confidence,
            threshold,
            k,
            adata_sc=None,
            root=root,
            plot_confidence=plot_confidence,
        )
    with plot_slot:
        origin = f"{key_prefix}_headline"
        menu_state = headline_plot(fig, key=origin, context_menu=adata_sc is not None)
        if menu_state is None:
            menu_state = reopened_state(origin, k)
        if menu_state is not None and adata_sc is not None:
            # The reference aligners write no confidence; as in the report's
            # confidence card, their soft mapping's max per spot stands in.
            spatial = (
                SpatialContext(
                    str(args.stdata),
                    hard,
                    coords,
                    confidence if have_conf else _P.max(axis=1),
                    "Confidence" if have_conf else "Max probability",
                )
                if have_spatial and len(hard) == len(coords)
                else None
            )
            open_cell_type_dialog(
                adata_sc, root, k, menu_state, args.output_dir, spatial, origin=origin
            )
        if adata_sc is not None:
            st.caption(_PLOT_CLICK_HINT)
        # stem carries the figure identity (K + threshold + confidence toggle) so
        # the export cache invalidates when any of them changes (see export_popover).
        stem = f"{key_prefix}_k{k:03d}_thr{int(round(threshold * 100)):03d}"
        if plot_confidence:
            stem += "_conf"
        export_popover(fig, key=f"{key_prefix}_headline_exp", stem=stem)

    if coords is None:
        st.caption("ST data has no obsm['spatial'] — no spatial plot.")


def _render_progress(mapper: str, run: compute.MapperRun) -> None:
    """Progress bar for a mapper currently being computed."""
    done, expected = run.n_done(), run.n_expected()
    if expected:
        st.progress(
            min(done / expected, 1.0), text=f"Computing {mapper}: {done}/{expected} K"
        )
    else:
        # Before the first K folder the sweep is still building the reference
        # scaffold — which only over-clusters when the start clusters are Leiden's.
        stage = (
            "over-clustering…"
            if run.start_from_annotation is None
            else f"start clusters from {run.start_from_annotation}…"
        )
        st.progress(0.0, text=f"Computing {mapper}: {stage}")
    st.caption("This tab will show results when the run finishes.")


@st.fragment
def mapper_tab(
    mapper: str,
    args: argparse.Namespace,
    runs: dict,
    queue: list,
    ctrl: Controls | None,
    ks_all: list[int],
) -> None:
    # This tab is a fragment: its own widgets (confidence-threshold slider, export
    # popovers, pareto toggle) rerun only this tab, not every sibling tab. A
    # linked-plot K pick, though, must move the *global* K slider and refresh every
    # tab, so it sets this flag and we escalate to a full app rerun here.
    if st.session_state.pop("_ctrl_k_pick_pending", False):
        st.rerun(scope="app")

    root = data_access.run_root(args.output_dir, mapper)
    run = runs.get(mapper)

    # In-progress / queued / failed states take precedence over any partial
    # on-disk output.
    if run is not None and run.is_running():
        _render_progress(mapper, run)
        return
    if mapper in queue:
        st.info(
            f"🕒 Waiting for another method to finish before computing **{mapper}**…"
        )
        return
    if run is not None and run.error and not data_access.list_ks(root):
        st.error(f"'{mapper}' failed to compute — see the terminal for details.")
        return

    ks = data_access.list_ks(root)
    if not ks:
        st.info("No K folders found yet for this mapper.")
        return
    if ctrl is None or ctrl.k not in ks:
        st.info("K not available for this method at the current shared setting.")
        return

    # A little air between the tab bar and the card.
    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
    render_choose_k(mapper, root, ks_all, current_k=ctrl.k)

    _headline(
        mapper,
        root,
        ctrl.k,
        args,
        key_prefix=f"tab_{mapper}",
        show_shared_umap=ctrl.show_shared_umap,
        conf_controls=True,
    )

    st.divider()
    st.subheader("Report")
    render_report(mapper, root, ctrl.k, args)
