"""
The report cards' figures: cell-type fractions, reconstruction cosine
similarity and mapping confidence of one mapper, their Compare-tab versions
across mappers, and the reference tab's start-cluster merge bars.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from anndata import AnnData
from plotly.subplots import make_subplots

from spatialaim.adata_schema import OBS_COMPUTED_STATE, OBS_START_CLUSTER
from spatialaim.analysis.loading import (
    infer_cell_to_state_cluster,
    load_start_cluster_to_state,
)
from spatialaim.gui import data_access
from spatialaim.gui.render.common import (
    BACKDROP_GREY,
    CARD_FONT,
    base_layout,
    card_layout,
    hex_color,
    mapper_color,
    names_for,
    state_palette,
)
from spatialaim.gui.render.scatter import start_cluster_names
from spatialaim.gui.state_names import label as state_label

_CELL_COLOR = "#4c78a8"  # cell / soft

_SPOT_COLOR = "#f58518"  # spot / hard


def render_fractions_figure(
    adata_sc: AnnData,
    root: Path,
    k: int,
    hard: np.ndarray | None,
    *,
    highlight: int | None = None,
) -> go.Figure:
    """Cell fraction (reference states) and spot fraction (this mapper's
    assignment) per computed state, as one grouped-bar plot sharing a single
    y-axis. Both bars use the same per-state palette as the UMAP/spatial plots;
    the spot bars are hatched to tell them apart. Without ``hard`` only the cell
    bars; with ``highlight`` every other state is drawn in grey."""
    start_cluster_to_state = load_start_cluster_to_state(data_access.k_dir(root, k))
    infer_cell_to_state_cluster(adata_sc, start_cluster_to_state)
    cell_states = adata_sc.obs[OBS_COMPUTED_STATE].astype(int).to_numpy()

    palette = state_palette(k)
    states = list(range(k))
    x = [state_label(s, names_for(root, k)) for s in states]
    colors = [
        (
            hex_color(palette.get(s))
            if highlight is None or s == int(highlight)
            else BACKDROP_GREY
        )
        for s in states
    ]
    cell_frac = [float(np.mean(cell_states == s)) for s in states]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Reference (solid)",
            x=x,
            y=cell_frac,
            marker=dict(color=colors),
            hovertemplate="Reference · %{x}<br>%{y:.1%}<extra></extra>",
        )
    )
    if hard is not None:
        hard = np.asarray(hard).astype(int)
        fig.add_trace(
            go.Bar(
                name="Spatial (hatched)",
                x=x,
                y=[float(np.mean(hard == s)) for s in states],
                marker=dict(
                    color=colors,
                    pattern=dict(shape="/", size=6, solidity=0.55, fgcolor="white"),
                ),
                hovertemplate="Spatial · %{x}<br>%{y:.1%}<extra></extra>",
            )
        )
    card_layout(
        fig,
        height=320,
        barmode="group",
        bargap=0.25,
        bargroupgap=0.05,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        yaxis_title="fraction",
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def render_reconstruction_figure(cossim: dict[str, dict]) -> go.Figure | None:
    """Two-panel (gene-wise, spot-wise) box plot of reconstruction cosine
    similarity for the soft/hard x raw/norm combos. Returns ``None`` if no combos
    are present.

    ``cossim`` maps each label to ``{"per_gene": [...], "per_spot": [...]}``.
    """
    order = ["soft-raw", "hard-raw", "soft-norm", "hard-norm"]
    labels = [lbl for lbl in order if lbl in cossim]
    if not labels:
        return None

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Gene-wise", "Cell-wise"],
        horizontal_spacing=0.12,
    )
    for col, attr in ((1, "per_gene"), (2, "per_spot")):
        for lbl in labels:
            fig.add_trace(
                go.Box(
                    y=cossim[lbl][attr],
                    name=lbl,
                    marker_color=_CELL_COLOR if lbl.startswith("soft") else _SPOT_COLOR,
                    boxpoints=False,
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
    card_layout(fig, height=320)
    fig.update_yaxes(title_text="cosine similarity", row=1, col=1)
    fig.update_xaxes(tickangle=-30)
    return fig


def render_onehot_figure(max_prob: np.ndarray) -> go.Figure:
    """Histogram of per-spot max probability (1.0 = fully one-hot), with mean and
    median markers."""
    max_prob = np.asarray(max_prob, dtype=float)
    mean, median = float(np.mean(max_prob)), float(np.median(max_prob))

    lo, hi = float(np.min(max_prob)), float(np.max(max_prob))
    if hi - lo < 1e-9:
        # Degenerate case (e.g. a fully one-hot mapping): every spot has the
        # same max probability. A plain histogram auto-ranges to a single giant
        # bar spanning ~[0.5, 1.5]; instead draw a thin bar at the value with a
        # tight, readable x-range.
        center = hi
        half = 0.05
        fig = go.Figure(
            go.Histogram(
                x=max_prob,
                xbins=dict(start=center - half, end=center + half, size=half / 5),
                marker_color=_CELL_COLOR,
            )
        )
    else:
        fig = go.Figure(go.Histogram(x=max_prob, nbinsx=40, marker_color=_CELL_COLOR))
    fig.add_vline(
        x=mean,
        line_dash="dash",
        line_color="black",
        annotation_text=f"mean {mean:.3f}",
        annotation_position="top left",
    )
    fig.add_vline(
        x=median,
        line_dash="dot",
        line_color=_SPOT_COLOR,
        annotation_text=f"median {median:.3f}",
        annotation_position="top right",
    )
    card_layout(
        fig,
        height=300,
        xaxis_title="max probability per cell (1.0 = one-hot)",
        yaxis_title="cells",
        bargap=0.02,
    )
    if hi - lo < 1e-9:
        fig.update_xaxes(range=[center - half, center + half])
    return fig


def render_confidence_figure(confidence: np.ndarray) -> go.Figure:
    """Histogram of per-spot assignment confidence in [0, 1], with mean and
    median markers."""
    confidence = np.asarray(confidence, dtype=float)
    mean, median = float(np.mean(confidence)), float(np.median(confidence))

    fig = go.Figure(
        go.Histogram(
            x=confidence,
            xbins=dict(start=0.0, end=1.0, size=1.0 / 40),
            # Same blue as render_onehot_figure: both fill the Mapping Confidence
            # card, so the colour must not imply the panels are unrelated.
            marker_color=_CELL_COLOR,
        )
    )
    fig.add_vline(
        x=mean,
        line_dash="dash",
        line_color="black",
        annotation_text=f"mean {mean:.3f}",
        annotation_position="top left",
    )
    fig.add_vline(
        x=median,
        line_dash="dot",
        line_color=_SPOT_COLOR,
        annotation_text=f"median {median:.3f}",
        annotation_position="top right",
    )
    card_layout(
        fig,
        height=300,
        xaxis_title="assignment confidence per cell",
        yaxis_title="cells",
        bargap=0.02,
    )
    fig.update_xaxes(range=[0.0, 1.0])
    return fig


def render_compare_reconstruction_figure(cossims: dict[str, dict]) -> go.Figure | None:
    """Two-panel (gene-wise, spot-wise) box plot of reconstruction cosine
    similarity, with every selected mapper's two hard x raw/norm combos in the
    same axes (grouped by combo, coloured by mapper).

    ``cossims`` maps each mapper to ``{combo: {"per_gene": [...], "per_spot": [...]}}``.
    Returns ``None`` if no data is present.
    """
    order = ["hard-raw", "hard-norm"]
    mappers = [m for m in cossims if cossims[m]]
    if not mappers:
        return None

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Gene-wise", "Cell-wise"],
        horizontal_spacing=0.12,
    )
    for col, attr in ((1, "per_gene"), (2, "per_spot")):
        for i, m in enumerate(mappers):
            xs: list[str] = []
            ys: list[float] = []
            for combo in order:
                vals = (cossims[m].get(combo) or {}).get(attr) or []
                xs += [combo] * len(vals)
                ys += list(vals)
            if not ys:
                continue
            fig.add_trace(
                go.Box(
                    x=xs,
                    y=ys,
                    name=m,
                    legendgroup=m,
                    showlegend=(col == 1),
                    marker_color=mapper_color(i),
                    boxpoints=False,
                ),
                row=1,
                col=col,
            )
    base_layout(
        fig,
        height=340,
        font=CARD_FONT,
        boxmode="group",
        margin=dict(l=10, r=10, t=30, b=52),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
    )
    fig.update_yaxes(title_text="cosine similarity", row=1, col=1)
    fig.update_xaxes(tickangle=-30)
    return fig


def render_compare_fractions_figure(
    spot_fracs: dict[str, list[float]],
    k: int,
    *,
    height: int = 300,
    names: dict[int, str] | None = None,
) -> go.Figure:
    """Grouped bar of the spot-state fraction per state, one bar group per state
    and one bar per mapper. (Cell fractions are identical across mappers, so only
    the mapping-dependent spot fractions are compared here.)

    Takes ``names`` rather than reading them itself: it gets fractions, not a run
    root, so the caller is the one that knows the output dir.
    """
    states = list(range(k))
    x = [state_label(s, names) for s in states]
    fig = go.Figure()
    for i, (m, fr) in enumerate(spot_fracs.items()):
        fig.add_trace(go.Bar(name=m, x=x, y=fr, marker_color=mapper_color(i)))
    base_layout(
        fig,
        height=height,
        font=CARD_FONT,
        barmode="group",
        title=dict(text="Spatial cell-type fractions", font=dict(size=13)),
        yaxis_title="fraction",
        margin=dict(l=10, r=10, t=40, b=40),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def render_start_cluster_merge_figure(
    adata_sc: AnnData, root: Path, k: int
) -> go.Figure:
    """Horizontal stacked bars: one bar per computed state, segmented by the
    start clusters merged into it (segment width ∝ cell count, labelled with the
    start-cluster name)."""
    start_cluster_to_state = load_start_cluster_to_state(data_access.k_dir(root, k))
    infer_cell_to_state_cluster(adata_sc, start_cluster_to_state)
    names = start_cluster_names(adata_sc)
    state_labels = names_for(root, k)
    start_cluster = adata_sc.obs[OBS_START_CLUSTER].astype(int).to_numpy()
    cell_states = adata_sc.obs[OBS_COMPUTED_STATE].astype(int).to_numpy()
    palette = state_palette(k)
    states = sorted(np.unique(cell_states).tolist())

    # cell_states is constant within a start cluster, so each maps to one state.
    start_clusters_of_state: dict[int, list[tuple[int, int]]] = {s: [] for s in states}
    for lc in np.unique(start_cluster):
        mask = start_cluster == lc
        s = int(cell_states[mask][0])
        start_clusters_of_state[s].append((int(lc), int(mask.sum())))
    for s in states:
        start_clusters_of_state[s].sort(key=lambda t: t[1], reverse=True)

    def _row(s: int) -> str:
        n = len(start_clusters_of_state[s])
        return f"{state_label(s, state_labels)}  ({n} cluster{'s' if n != 1 else ''})"

    fig = go.Figure()
    # barmode="stack" accumulates same-row segments in trace order (largest first).
    for s in states:
        color = hex_color(palette.get(s))
        for lc, size in start_clusters_of_state[s]:
            fig.add_trace(
                go.Bar(
                    y=[_row(s)],
                    x=[size],
                    orientation="h",
                    marker=dict(color=color, line=dict(color="white", width=1.5)),
                    text=[f"L{lc}"],
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(size=9, color="black"),
                    showlegend=False,
                    hovertemplate=(
                        f"{state_label(s, state_labels)} · {names[lc]}"
                        "<br>%{x} cells<extra></extra>"
                    ),
                )
            )

    n_states = len(states)
    base_layout(
        fig,
        height=max(240, n_states * 46 + 120),
        barmode="stack",
        title="Start clusters merged per computed SpatialAIM cell type",
        xaxis_title="cells   (segment width ∝ start-cluster size)",
        margin=dict(l=10, r=10, t=50, b=40),
        bargap=0.35,
    )
    # State 0 on top (inverted y-axis).
    fig.update_yaxes(
        categoryorder="array", categoryarray=[_row(s) for s in reversed(states)]
    )
    return fig
