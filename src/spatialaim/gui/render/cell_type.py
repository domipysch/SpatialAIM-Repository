"""
The cell-type modal's figures (right click on a cell type): the type highlighted
on the reference UMAP and the spatial map, a gene's expression on both, and the
type-vs-all violins and per-type bars of its "General" view.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from anndata import AnnData
from plotly.subplots import make_subplots

from spatialaim.adata_schema import OBSM_UMAP
from spatialaim.gui import data_access
from spatialaim.gui.render.common import (
    BACKDROP_GREY,
    SpatialContext,
    add_violin,
    base_layout,
    card_layout,
    hex_color,
    names_for,
    state_palette,
    style_scatter_axes,
)
from spatialaim.gui.render.markers import gene_expression, spatial_gene_expression
from spatialaim.gui.render.scatter import umap_panel_data
from spatialaim.gui.state_names import label as state_label

# Lognorm 0 is drawn light grey so expressing cells stand out, then reds.
_EXPR_SCALE = [[0.0, "#e3e3e3"], [0.02, "#fcbba1"], [0.5, "#ef3b2c"], [1.0, "#67000d"]]


class _MapPanel(NamedTuple):
    """One scatter panel of the cell-type modal: its points, which of them are
    highlighted (or their expression), and how its axes are styled."""

    title: str
    coords: np.ndarray
    values: np.ndarray  # bool "in this cell type" mask, or per-point expression
    axis_titles: tuple[str, str]
    reversed_y: bool


def _modal_coords(
    adata_sc: AnnData,
    root: Path,
    k: int,
    spatial: SpatialContext | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """The reference UMAP coordinates and per-cell states, plus the spot
    coordinates when ``spatial`` carries them (else ``None``)."""
    data = umap_panel_data(adata_sc, str(root), int(k), OBSM_UMAP, False)
    spot_coords = (
        np.asarray(spatial.coords)
        if spatial is not None and spatial.coords is not None
        else None
    )
    return data, spot_coords


def _map_row_figure(
    panels: list[_MapPanel],
    *,
    color: str | None,
    dot_size: float,
    height: int = 430,
) -> go.Figure:
    """One row of equal-aspect scatter panels. With ``color`` each panel's
    ``values`` is a mask drawn in that colour over every other point in grey;
    without, ``values`` is an expression colouring every point (highest drawn on
    top so expressing points are never buried), each panel with its own
    colourbar beside it, since the two normalisations are not on one scale."""
    fig = make_subplots(
        rows=1,
        cols=len(panels),
        subplot_titles=[p.title for p in panels],
        horizontal_spacing=0.12,
    )
    for col, p in enumerate(panels, start=1):
        if color is not None:
            for mask, c in ((~p.values, BACKDROP_GREY), (p.values, color)):
                fig.add_trace(
                    go.Scattergl(
                        x=p.coords[mask, 0],
                        y=p.coords[mask, 1],
                        mode="markers",
                        marker=dict(size=dot_size, color=c),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=1,
                    col=col,
                )
        else:
            order = np.argsort(p.values, kind="stable")
            domain = fig.layout["xaxis" if col == 1 else f"xaxis{col}"].domain
            fig.add_trace(
                go.Scattergl(
                    x=p.coords[order, 0],
                    y=p.coords[order, 1],
                    mode="markers",
                    marker=dict(
                        size=dot_size,
                        color=p.values[order],
                        colorscale=_EXPR_SCALE,
                        cmin=0.0,
                        cmax=float(p.values.max()) or 1.0,
                        colorbar=dict(
                            title="lognorm",
                            thickness=12,
                            len=0.8,
                            x=domain[1] + 0.01,
                            xanchor="left",
                        ),
                    ),
                    hovertemplate="%{marker.color:.2f}<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
        style_scatter_axes(
            fig,
            row=1,
            col=col,
            xref="x" if col == 1 else f"x{col}",
            coords=p.coords,
            xtitle=p.axis_titles[0],
            ytitle=p.axis_titles[1],
            equal_aspect=True,
            reversed_y=p.reversed_y,
            hide_ticks=True,
        )
    return base_layout(fig, height=height, margin=dict(l=10, r=10, t=40, b=10))


def render_cell_type_maps_figure(
    adata_sc: AnnData,
    root: Path,
    k: int,
    state: int,
    *,
    spatial: SpatialContext | None = None,
    dot_size: float = 3.0,
) -> go.Figure:
    """The modal's first row: ``state`` in its colour over every other cell in
    grey on the all-gene reference UMAP and, with ``spatial``, the spots this
    mapper assigned to it on the spatial map."""
    data, spot_coords = _modal_coords(adata_sc, root, k, spatial)
    label = state_label(state, names_for(root, k))
    panels = [
        _MapPanel(
            f"Reference — {label}",
            data["coords"],
            data["states"] == int(state),
            ("UMAP1", "UMAP2"),
            False,
        )
    ]
    if spot_coords is not None:
        panels.append(
            _MapPanel(
                f"Spatial map — {label}",
                spot_coords,
                np.asarray(spatial.hard) == int(state),
                ("x", "y"),
                True,
            )
        )
    # The modal's header row: kept low so the views below get the room.
    return _map_row_figure(
        panels,
        color=hex_color(state_palette(k).get(int(state))),
        dot_size=dot_size,
        height=300,
    )


def render_gene_maps_figure(
    adata_sc: AnnData,
    root: Path,
    k: int,
    gene: str,
    *,
    spatial: SpatialContext | None = None,
    dot_size: float = 3.0,
) -> go.Figure:
    """The modal's second row: every cell coloured by its lognorm expression of
    ``gene`` on the all-gene reference UMAP and, with ``spatial``, every spot by
    its ST lognorm expression on the spatial map."""
    data, spot_coords = _modal_coords(adata_sc, root, k, spatial)
    panels = [
        _MapPanel(
            f"Reference — {gene} expression (lognorm)",
            data["coords"],
            gene_expression(adata_sc, str(root), gene),
            ("UMAP1", "UMAP2"),
            False,
        )
    ]
    st_expr = (
        spatial_gene_expression(adata_sc, spatial, gene)
        if spot_coords is not None
        else None
    )
    if st_expr is not None:
        panels.append(
            _MapPanel(
                f"Spatial — {gene} expression (lognorm)",
                spot_coords,
                st_expr,
                ("x", "y"),
                True,
            )
        )
    return _map_row_figure(panels, color=None, dot_size=dot_size)


def render_gene_distribution_figure(
    adata_sc: AnnData,
    root: Path,
    k: int,
    state: int,
    gene: str,
    *,
    spatial: SpatialContext | None = None,
) -> go.Figure:
    """Violins of ``gene``'s lognorm expression in ``state`` vs. all other cells
    of the reference and, with ``spatial``, in the spots assigned to ``state`` vs.
    all other spots — side by side, each on its own y-axis since the two
    normalisations are not on a common scale."""
    data = umap_panel_data(adata_sc, str(root), int(k), OBSM_UMAP, False)
    label = state_label(state, names_for(root, k))
    color = hex_color(state_palette(k).get(int(state)))
    rng = np.random.default_rng(0)
    st_expr = (
        spatial_gene_expression(adata_sc, spatial, gene)
        if spatial is not None
        else None
    )

    n_cols = 2 if st_expr is not None else 1
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=[
            "Distribution in scRNA reference cells",
            "Distribution in cells of spatial data",
        ][:n_cols],
        horizontal_spacing=0.1,
    )
    inside = data["states"] == int(state)
    expr = gene_expression(adata_sc, str(root), gene)
    add_violin(fig, expr[inside], label, color, rng, col=1)
    add_violin(fig, expr[~inside], "All other cells", "#9e9e9e", rng, col=1)
    fig.update_yaxes(title_text=f"{gene} (lognorm)", row=1, col=1)
    if st_expr is not None:
        on = np.asarray(spatial.hard) == int(state)
        add_violin(fig, st_expr[on], label, color, rng, col=2)
        add_violin(fig, st_expr[~on], "All other cells", "#9e9e9e", rng, col=2)
        fig.update_yaxes(title_text=f"{gene} (lognorm)", row=1, col=2)
    return base_layout(fig, height=360, margin=dict(l=10, r=10, t=40, b=10))


def render_state_confidence_figure(
    root: Path, k: int, state: int, spatial: SpatialContext
) -> go.Figure:
    """Violins of the per-spot confidence (``spatial.confidence``) among the
    spots assigned to ``state`` vs. among all spots."""
    label = state_label(state, names_for(root, k))
    color = hex_color(state_palette(k).get(int(state)))
    rng = np.random.default_rng(0)
    conf = np.asarray(spatial.confidence, dtype=float)
    on = np.asarray(spatial.hard) == int(state)
    fig = make_subplots(
        rows=1, cols=1, subplot_titles=[f"{spatial.confidence_title} per cell"]
    )
    add_violin(fig, conf[on], label, color, rng, col=1, expressing=False)
    add_violin(fig, conf, "All cells", "#9e9e9e", rng, col=1, expressing=False)
    fig.update_yaxes(title_text=spatial.confidence_title, range=[0.0, 1.0])
    return base_layout(fig, height=360, margin=dict(l=10, r=10, t=40, b=10))


@st.cache_data(show_spinner=False)
def _coherence_per_spot(root_str: str, k: int) -> np.ndarray:
    """``data_access.load_coherence_per_spot``, cached per run root and K."""
    return data_access.load_coherence_per_spot(Path(root_str), k)


def render_state_coherence_figure(
    root: Path, k: int, state: int, spatial: SpatialContext
) -> go.Figure:
    """Transcriptional coherence of ``state``: violins of each spot's share of
    ST expression neighbours mapped to its own state, for the spots assigned to
    ``state`` vs. all spots, with the state's chance level (its share of spots)
    as a line."""
    per_state = (data_access.load_data_json(root, k, "modularity_metrics.json") or {})[
        "per_state"
    ]
    chance = per_state[str(int(state))]["chance"]
    share = _coherence_per_spot(str(root), int(k))
    label = state_label(state, names_for(root, k))
    color = hex_color(state_palette(k).get(int(state)))
    rng = np.random.default_rng(0)
    on = np.asarray(spatial.hard) == int(state)
    fig = make_subplots(rows=1, cols=1)
    for vals, name, c in ((share[on], label, color), (share, "All cells", "#9e9e9e")):
        vals = vals[np.isfinite(vals)]
        add_violin(fig, vals, name, c, rng, col=1, expressing=False)
    fig.add_hline(
        y=chance,
        line_dash="dash",
        line_color="#555555",
        annotation_text=f"chance {chance:.2f}",
        annotation_position="top left",
    )
    fig.update_yaxes(
        title_text="same-type share of expression neighbours", range=[0.0, 1.0]
    )
    return base_layout(fig, height=360, margin=dict(l=10, r=10, t=20, b=10))


def render_self_enrichment_figure(
    root: Path, k: int, *, highlight: int | None = None
) -> go.Figure:
    """Self neighbourhood-enrichment z-score per computed state (how much more
    often its spots border each other than under a label shuffle), as bars in
    the per-state palette; with ``highlight`` every other state is grey. A state
    without spots, or whose z-score is undefined, has no bar."""
    topo = data_access.load_data_json(root, k, "topology_metrics.json") or {}
    per_state = topo["per_state"]
    palette = state_palette(k)
    states = list(range(k))
    z = [per_state.get(str(s), {}).get("self_zscore", float("nan")) for s in states]
    fig = go.Figure(
        go.Bar(
            x=[state_label(s, names_for(root, k)) for s in states],
            y=[v if np.isfinite(v) else None for v in z],
            marker=dict(
                color=[
                    (
                        hex_color(palette.get(s))
                        if highlight is None or s == int(highlight)
                        else BACKDROP_GREY
                    )
                    for s in states
                ]
            ),
            hovertemplate="%{x}<br>z = %{y:.1f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line_color="#555555", line_width=1)
    card_layout(fig, height=360, yaxis_title="self-enrichment z-score")
    return fig
