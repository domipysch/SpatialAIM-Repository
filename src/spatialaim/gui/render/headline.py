"""
The large multi-panel figures, each sharing one cell-type legend across its
panels: a mapper tab's UMAP(s) + spatial map, the Compare tab's UMAP over every
mapper's spatial map, and the reference tab's three UMAPs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from anndata import AnnData
from plotly.subplots import make_subplots

from spatialaim.adata_schema import OBSM_UMAP, OBSM_UMAP_SHARED_GENES
from spatialaim.analysis.loading import (
    infer_cell_to_state_cluster,
    load_start_cluster_to_state,
)
from spatialaim.gui import data_access
from spatialaim.gui.render.common import (
    STATE_LEGEND,
    TAB20,
    SpatialContext,
    hex_color,
    base_layout,
    names_for,
    state_palette,
    style_scatter_axes,
)
from spatialaim.gui.render.markers import top_genes_per_state
from spatialaim.gui.render.scatter import (
    add_spatial_traces,
    add_start_cluster_umap_traces,
    add_umap_traces,
)


def _mod_suffix(mod: dict | None, key: str) -> str:
    """`` (modularity = 0.123)`` for modularity ``key`` in ``mod`` (else ``''``),
    appended to a subplot title so each UMAP shows its own modularity."""
    v = (mod or {}).get(key)
    return f"  (modularity = {v:.3f})" if isinstance(v, (int, float)) else ""


def render_headline_figure(
    coords: np.ndarray | None,
    hard: np.ndarray,
    confidence: np.ndarray | None,
    threshold: float,
    k: int,
    *,
    adata_sc: AnnData | None = None,
    root: Path | None = None,
    plot_confidence: bool = False,
    show_shared_umap: bool = False,
    dot_size_umap: float = 3.0,
    dot_size_spatial: float = 3.0,
    st_path: str | None = None,
) -> go.Figure:
    """One interactive figure holding the UMAP(s) and the spatial map as
    side-by-side subplots that share a single state legend.

    The reference (all-gene) UMAP is included when ``adata_sc`` (and ``root``)
    are given; with ``show_shared_umap`` a second, shared-gene UMAP is added to
    its left. The spatial panel is included when ``coords`` is not None. Every
    panel colours states from one ``state_palette(k)`` keyed by state id, and a
    state's traces share a ``legendgroup`` so a single click toggles it
    everywhere.

    With ``plot_confidence`` the spatial panel is coloured by a continuous
    confidence scale (with a colourbar) instead of by state.

    ``dot_size_*`` are the *un-zoomed* marker sizes. ``widgets.headline_plot``
    grows them again as a panel is zoomed into, so these are the sizes that apply
    at full extent and in a static export.

    With ``st_path`` (the ST file) every hover box also names the state's top
    spatial marker genes, ranked over the spots ``hard`` assigns to it.
    """
    palette = state_palette(k)
    have_umap = adata_sc is not None and root is not None
    have_shared = (
        have_umap and show_shared_umap and OBSM_UMAP_SHARED_GENES in adata_sc.obsm
    )
    have_spatial = coords is not None
    conf_mode = plot_confidence and have_spatial and confidence is not None

    mod = (
        data_access.load_data_json(root, k, "modularity_metrics.json")
        if root is not None
        else None
    )
    titles: list[str] = []
    if have_shared:
        titles.append(
            "Shared-gene UMAP — computed cell types"
            + _mod_suffix(mod, "modularity_shared")
        )
    if have_umap:
        titles.append(
            "Reference UMAP — computed cell types" + _mod_suffix(mod, "modularity_all")
        )
    if have_spatial:
        if conf_mode:
            spatial_title = "Spatial confidence"
        else:
            spatial_title = "Mapped cell types"
            if confidence is not None and threshold > 0.0:
                spatial_title += f" (confidence ≥ {threshold:.2f})"
            # The mapping-dependent modularity belongs to this panel, the way each
            # UMAP title carries its own reference-graph modularity above. Left off
            # the confidence view, where it would read as a property of the
            # confidence rather than of the mapping.
            spatial_title += _mod_suffix(mod, "modularity_st_expression")
        titles.append(spatial_title)

    n_cols = max(1, len(titles))
    spacing = 0.06 if n_cols >= 3 else 0.08
    fig = make_subplots(
        rows=1, cols=n_cols, subplot_titles=titles, horizontal_spacing=spacing
    )

    # Equal aspect (scaleanchor + constrain="domain") gives every panel a true
    # 1:1 ratio by shrinking it within its own subplot cell, centred with
    # whitespace. Because it only shrinks the domain it never overlaps neighbours,
    # so it is applied uniformly for any column count (the client no longer
    # post-corrects the aspect ratio).
    equal_aspect = True

    legend_shown: set[int] = set()
    col = 1
    # Every panel's hover box names the state's most enriched genes; needs the
    # scaffold, so it is empty (and the hover unchanged) on a spatial-only figure.
    spatial = (
        SpatialContext(st_path, np.asarray(hard), coords)
        if st_path is not None and have_spatial and len(hard) == len(coords)
        else None
    )
    top_genes = top_genes_per_state(
        adata_sc if have_umap else None, root, k, spatial=spatial
    )
    names = names_for(root, k)
    # Each adder pins its own subplot axes.
    if have_shared:
        add_umap_traces(
            fig,
            adata_sc,
            root,
            k,
            col=col,
            palette=palette,
            legend_shown=legend_shown,
            dot_size=dot_size_umap,
            umap_key=OBSM_UMAP_SHARED_GENES,
            equal_aspect=equal_aspect,
            top_genes=top_genes,
            names=names,
        )
        col += 1
    if have_umap:
        add_umap_traces(
            fig,
            adata_sc,
            root,
            k,
            col=col,
            palette=palette,
            legend_shown=legend_shown,
            dot_size=dot_size_umap,
            umap_key=OBSM_UMAP,
            equal_aspect=equal_aspect,
            top_genes=top_genes,
            names=names,
        )
        col += 1
    if have_spatial:
        add_spatial_traces(
            fig,
            coords,
            hard,
            confidence,
            threshold,
            col=col,
            palette=palette,
            legend_shown=legend_shown,
            dot_size=dot_size_spatial,
            plot_confidence=plot_confidence,
            equal_aspect=equal_aspect,
            top_genes=top_genes,
            names=names,
        )

    # Keep the state legend as a fixed horizontal strip on top in every mode, so
    # it never jumps when the confidence colourbar appears on the right. The
    # ``y`` offset leaves a little vertical gap between the legend and the plots.
    return base_layout(
        fig,
        height=720,
        legend={**STATE_LEGEND, "y": 1.12},
        margin=dict(l=10, r=10, t=70, b=10),
    )


def render_annotation_figure(
    adata_sc: AnnData,
    labels: np.ndarray,
    annotation: str,
    *,
    coords: np.ndarray | None = None,
    st_labels: np.ndarray | None = None,
    show_shared_umap: bool = False,
    reference_layout: bool = False,
    colors: dict[str, str] | None = None,
    dot_size: float = 3.0,
) -> go.Figure:
    """The annotation ``labels`` (one per scaffold cell) on the reference UMAP(s)
    and, given ``coords`` and ``st_labels``, the annotation start's identity-cut
    mapping on the spatial map. Independent of K; one colour per annotated type
    across all panels (sorted type order, as the annotation start numbers them).

    With ``reference_layout`` the panels line up under
    ``render_reference_umaps_figure``: an empty first column (under the start
    clustering), then the all-gene and the shared-gene UMAP, at its height.

    ``colors`` (type -> hex) overrides the default tab20-by-type colours; a type
    it leaves out is drawn grey."""
    types = sorted(
        set(labels.tolist()) | set([] if st_labels is None else st_labels.tolist())
    )
    if colors is None:
        colors = {t: TAB20[i % 20] for i, t in enumerate(types)}
    else:
        colors = {t: hex_color(colors.get(t)) for t in types}
    have_shared = OBSM_UMAP_SHARED_GENES in adata_sc.obsm
    all_gene = (
        f"Reference UMAP — {annotation}",
        np.asarray(adata_sc.obsm[OBSM_UMAP]),
        labels,
        False,
    )
    # ``None`` = an empty column (reference layout only).
    panels: list[tuple[str, np.ndarray, np.ndarray, bool] | None] = []
    if reference_layout:
        panels = [None, all_gene]
        if have_shared:
            panels.append(
                (
                    f"Shared-gene UMAP — {annotation}",
                    np.asarray(adata_sc.obsm[OBSM_UMAP_SHARED_GENES]),
                    labels,
                    False,
                )
            )
    elif show_shared_umap and have_shared:
        panels.append(
            (
                f"Shared-gene UMAP — {annotation}",
                np.asarray(adata_sc.obsm[OBSM_UMAP_SHARED_GENES]),
                labels,
                False,
            )
        )
    if not reference_layout:
        panels.append(all_gene)
    if coords is not None and st_labels is not None:
        panels.append((f"Mapped {annotation}", coords, st_labels, True))

    n_cols = len(panels)
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=[p[0] if p is not None else "" for p in panels],
        horizontal_spacing=0.06 if n_cols >= 3 else 0.08,
    )
    shown: set[str] = set()
    for col, panel in enumerate(panels, start=1):
        if panel is None:
            fig.update_xaxes(visible=False, row=1, col=col)
            fig.update_yaxes(visible=False, row=1, col=col)
            continue
        _title, xy, lab, spatial = panel
        for t in types:
            mask = lab == t
            if not mask.any():
                continue
            fig.add_trace(
                go.Scattergl(
                    x=xy[mask, 0],
                    y=xy[mask, 1],
                    mode="markers",
                    marker=dict(size=dot_size, color=colors[t], opacity=1.0),
                    name=t,
                    legendgroup=t,
                    showlegend=t not in shown,
                    hovertemplate=f"{t}<extra></extra>",
                ),
                row=1,
                col=col,
            )
            shown.add(t)
        style_scatter_axes(
            fig,
            row=1,
            col=col,
            xref=fig.data[-1].xaxis or "x",
            coords=xy,
            xtitle="x" if spatial else "UMAP1",
            ytitle="y" if spatial else "UMAP2",
            equal_aspect=True,
            reversed_y=spatial,
            hide_ticks=not spatial,
        )
    return base_layout(
        fig,
        height=520 if reference_layout else 720,
        legend={**STATE_LEGEND, "title": annotation, "y": 1.12},
        margin=dict(l=10, r=10, t=70, b=10),
    )


def render_compare_figure(
    coords: np.ndarray,
    hards: list[np.ndarray],
    mapper_names: list[str],
    k: int,
    *,
    adata_sc: AnnData | None = None,
    root: Path | None = None,
    dot_size_umap: float = 3.0,
    dot_size_spatial: float = 3.0,
) -> go.Figure:
    """One reference UMAP centred on top, then the selected mappers' spatial maps
    in a grid below (two per row), all in one figure with a single shared state
    legend — so one legend click toggles a state across every panel at once.

    The spatial coordinates are shared across mappers; only each mapper's hard
    assignment (``hards[i]`` for ``mapper_names[i]``) differs. No shared-gene UMAP
    and no confidence colouring here.
    """
    palette = state_palette(k)
    have_umap = adata_sc is not None and root is not None
    n = len(mapper_names)
    n_spatial_rows = (n + 1) // 2

    # Grid: (optional) row 1 = UMAP spanning both columns so equal-aspect centres
    # it with whitespace; each later row holds up to two spatial panels.
    specs: list[list] = []
    if have_umap:
        specs.append([{"colspan": 2}, None])
    for r in range(n_spatial_rows):
        specs.append([{}, {}] if n - r * 2 >= 2 else [{}, None])
    n_rows = len(specs)

    titles: list[str] = []
    if have_umap:
        titles.append("Reference UMAP — computed cell types")
    titles += [f"Spatial — {m}" for m in mapper_names]

    # Per-row pixel budgets (row_heights normalises these into proportions): the
    # spatial rows are taller than the UMAP row so the maps get more room.
    umap_px = 420
    spatial_px = 640
    row_heights = ([umap_px] if have_umap else []) + [spatial_px] * n_spatial_rows
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        specs=specs,
        subplot_titles=titles,
        row_heights=row_heights,
        horizontal_spacing=0.06,
        vertical_spacing=(min(0.1, 0.8 / (n_rows - 1)) if n_rows > 1 else 0.0),
    )

    legend_shown: set[int] = set()
    umap_row = 1 if have_umap else 0
    # The tree cut is shared across mappers, so one set of top genes and one set
    # of names serve every panel here.
    top_genes = top_genes_per_state(adata_sc if have_umap else None, root, k)
    names = names_for(root, k)
    if have_umap:
        add_umap_traces(
            fig,
            adata_sc,
            root,
            k,
            row=1,
            col=1,
            palette=palette,
            legend_shown=legend_shown,
            dot_size=dot_size_umap,
            umap_key=OBSM_UMAP,
            top_genes=top_genes,
            names=names,
        )
    for j, hard in enumerate(hards):
        add_spatial_traces(
            fig,
            coords,
            hard,
            None,
            0.0,
            row=umap_row + 1 + j // 2,
            col=1 + j % 2,
            palette=palette,
            legend_shown=legend_shown,
            dot_size=dot_size_spatial,
            plot_confidence=False,
            top_genes=top_genes,
            names=names,
        )

    height = (umap_px if have_umap else 0) + spatial_px * n_spatial_rows + 60
    return base_layout(
        fig,
        height=height,
        legend={**STATE_LEGEND, "y": 1.02},
        margin=dict(l=10, r=10, t=60, b=10),
    )


def render_reference_umaps_figure(
    adata_sc: AnnData,
    root: Path,
    k: int,
    *,
    dot_size_umap: float = 3.0,
) -> go.Figure:
    """Three reference UMAPs sharing one state legend: the start clustering
    (left), the computed states on the all-gene UMAP (middle), and the computed
    states on the shared-gene UMAP (right, dropped if that embedding is absent).

    Rendered by the same client-side component as the headline plot, so one
    legend click toggles a state across all three panels (the start clusters
    grey out with the state they merged into).
    """
    palette = state_palette(k)
    start_cluster_to_state = load_start_cluster_to_state(data_access.k_dir(root, k))
    infer_cell_to_state_cluster(adata_sc, start_cluster_to_state)
    have_shared = OBSM_UMAP_SHARED_GENES in adata_sc.obsm

    mod = data_access.load_data_json(root, k, "modularity_metrics.json")
    titles = [
        "Start clustering",
        "Computed cell types — reference UMAP" + _mod_suffix(mod, "modularity_all"),
    ]
    if have_shared:
        titles.append(
            "Computed cell types — shared-gene UMAP"
            + _mod_suffix(mod, "modularity_shared")
        )
    n_cols = len(titles)
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.06 if n_cols >= 3 else 0.08,
    )

    legend_shown: set[int] = set()
    top_genes = top_genes_per_state(adata_sc, root, k)
    names = names_for(root, k)
    # Left panel shows start clusters, not cell types, so it gets no top genes —
    # only the names, for the "-> <cell type>" line in its hover.
    add_start_cluster_umap_traces(
        fig,
        adata_sc,
        start_cluster_to_state,
        col=1,
        dot_size=dot_size_umap,
        state_labels=names,
    )
    add_umap_traces(
        fig,
        adata_sc,
        root,
        k,
        col=2,
        palette=palette,
        legend_shown=legend_shown,
        dot_size=dot_size_umap,
        umap_key=OBSM_UMAP,
        top_genes=top_genes,
        names=names,
    )
    if have_shared:
        add_umap_traces(
            fig,
            adata_sc,
            root,
            k,
            col=3,
            palette=palette,
            legend_shown=legend_shown,
            dot_size=dot_size_umap,
            umap_key=OBSM_UMAP_SHARED_GENES,
            top_genes=top_genes,
            names=names,
        )

    return base_layout(
        fig,
        height=520,
        legend={**STATE_LEGEND, "y": 1.12},
        margin=dict(l=10, r=10, t=70, b=10),
    )
