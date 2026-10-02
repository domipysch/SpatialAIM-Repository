"""Figure production for the GUI.

Every figure is an interactive Plotly object built here from data read straight
off disk (each K's ``analysis/data`` folder) and, for the UMAP / marker /
fractions / merge-map views, the reference scaffold. The headline UMAP(s) +
spatial map share a single state legend and are recoloured client-side; the
spatial scatter in particular is rebuilt here on every confidence-threshold
change so spots below the threshold can be drawn grey. The threshold-independent
UMAP-panel data is memoised (``_umap_panel_data``) so a threshold drag rebuilds
only the spatial panel, not the UMAP.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from anndata import AnnData
from plotly.subplots import make_subplots

from spatialaim.adata_schema import (
    OBS_COMPUTED_STATE,
    OBS_START_CLUSTER,
    UNS_START_CLUSTER_NAMES,
    OBSM_PCA,
    OBSM_PCA_SHARED_GENES,
    OBSM_UMAP,
    OBSM_UMAP_SHARED_GENES,
    UNS_SHARED_GENES,
    LAYER_LOGNORM,
)
from spatialaim.analysis.loading import (
    infer_cell_to_state_cluster,
    load_start_cluster_to_state,
)

from spatialaim.analysis.utils import to_dense
from spatialaim.metrics import kselection as scores

from . import data_access, state_names
from .state_names import label as _state_label

logger = logging.getLogger(__name__)


def _names_for(root: Path | None, k: int) -> dict[int, str]:
    """This K's user-set cell-type names, read from the output dir above ``root``.

    A run root is ``<output_dir>/<mapper>/`` and the names are shared by every
    mapper, so they sit one level up. Empty when there is no root (a figure drawn
    without a run, e.g. spatial-only) or nothing has been named.
    """
    return state_names.load(Path(root).parent, k) if root is not None else {}


# Grey (as a CSS rgba string) for spots drawn below the confidence threshold.
_GREY_RGBA = "rgba(184,184,184,0.9)"

# Matplotlib's tab20 / tab10 qualitative palettes as hex strings, inlined so the
# GUI carries no matplotlib dependency. tab20 keys the per-state colours (state
# id -> colour); tab10 distinguishes mappers in the Compare tab and the K-sweep
# lines. Values match matplotlib exactly, so figures look identical to before.
_TAB20 = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]
_TAB10 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# Shared Plotly look used by every figure: a white template, transparent
# backgrounds so it blends with the app, and the app's sans-serif font.
_FONT = dict(family="'Source Sans Pro', 'Segoe UI', sans-serif", size=13)
# The horizontal state-legend strip shared by the headline / compare / reference
# figures; each sets its own vertical ``y`` offset via ``{**_STATE_LEGEND, ...}``.
_STATE_LEGEND = dict(
    itemsizing="constant",
    title="Cell types",
    orientation="h",
    yanchor="bottom",
    xanchor="left",
    x=0,
)


def _base_layout(
    fig: go.Figure, *, height: int, font: dict = _FONT, **extra
) -> go.Figure:
    """Apply the shared template/background/font to ``fig`` (returns ``fig``).

    Every other layout knob (legend, margin, title, axis titles, barmode, …) is
    passed through ``extra`` unchanged.
    """
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=font,
        height=height,
        **extra,
    )
    return fig


def figure_to_bytes(fig: go.Figure, fmt: str = "png", *, scale: float = 2.0) -> bytes:
    """Static-export a Plotly figure to PNG / SVG / PDF bytes via kaleido.

    ``scale`` up-samples raster (png) output for crisp exports; it is ignored by
    the vector formats. Raises ``RuntimeError`` with an install hint if kaleido
    is missing. Note: the UMAP / spatial panels use WebGL (``Scattergl``) traces,
    which kaleido rasterises inside svg/pdf — png is always faithful.
    """
    try:
        return fig.to_image(format=fmt, scale=scale)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        raise RuntimeError(
            f"Could not export as {fmt.upper()}: {exc}. "
            "Static export needs the 'kaleido' package (pip install kaleido)."
        ) from exc


def state_palette(k: int) -> dict[int, str]:
    """tab20 palette keyed by state id, as ``#rrggbb`` hex strings."""
    return {s: _TAB20[s % 20] for s in range(k)}


def _hex(color: str | None) -> str:
    """Pass a hex colour through, with a grey fallback for a missing (None) state."""
    return color if color else "#b8b8b8"


def _start_cluster_names(adata_sc: AnnData) -> list[str]:
    """Display name per start cluster: the annotated cell types when the sweep
    started from an annotation, else ``cluster_<i>``.

    Falls back to positional names for a scaffold written before start-cluster names
    were stored, so older result folders still render.
    """
    names = adata_sc.uns.get(UNS_START_CLUSTER_NAMES)
    n = int(adata_sc.obs[OBS_START_CLUSTER].astype(int).max()) + 1
    if names is None or len(names) < n:
        return [f"cluster_{i}" for i in range(n)]
    return [str(name) for name in names]


def _mod_suffix(mod: dict | None, key: str) -> str:
    """`` (modularity = 0.123)`` for modularity ``key`` in ``mod`` (else ``''``),
    appended to a subplot title so each UMAP shows its own modularity."""
    v = (mod or {}).get(key)
    return f"  (modularity = {v:.3f})" if isinstance(v, (int, float)) else ""


def _padded_range(vals: np.ndarray, frac: float = 0.03) -> tuple[float, float]:
    """``(lo, hi)`` covering all of ``vals`` with a small margin.

    Used to pin axis ranges from the full data, so hiding states via the legend
    never re-autoranges (zooms) the view.
    """
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    span = hi - lo
    pad = frac * span if span > 0 else 1.0
    return lo - pad, hi + pad


def _style_scatter_axes(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    xref: str,
    coords: np.ndarray,
    xtitle: str,
    ytitle: str,
    equal_aspect: bool,
    reversed_y: bool = False,
    hide_ticks: bool = False,
) -> None:
    """Set fixed axis ranges for the subplot at (``row``, ``col``) (so state
    recolour/toggle never re-zooms), optionally flipping y and hiding tick labels.

    ``equal_aspect`` adds ``scaleanchor``/``constrain='domain'`` for a true 1:1
    ratio. ``constrain='domain'`` only ever *shrinks* a panel within its own
    subplot cell (centring it with whitespace), so it never collides and is safe
    for any grid. ``xref`` is this subplot's x-axis id (e.g. "x", "x2") that the
    equal-aspect ``scaleanchor`` must reference — taken from a trace already added
    to the subplot, so it stays correct in any multi-row/-column layout.
    """
    xr = _padded_range(coords[:, 0])
    yr = _padded_range(coords[:, 1])
    xkw = dict(title_text=xtitle, range=list(xr), autorange=False, row=row, col=col)
    ykw = dict(
        title_text=ytitle,
        range=[yr[1], yr[0]] if reversed_y else list(yr),
        autorange=False,
        row=row,
        col=col,
    )
    if hide_ticks:
        xkw["showticklabels"] = False
        ykw["showticklabels"] = False
    if equal_aspect:
        xkw["constrain"] = "domain"
        ykw["constrain"] = "domain"
        ykw["scaleanchor"] = xref
        ykw["scaleratio"] = 1
    fig.update_xaxes(**xkw)
    fig.update_yaxes(**ykw)


# --- EXPERIMENTAL: per-state centroid markers on the UMAP panels --------------
# Hand-set switch (see the question in the session that added this). When True,
# each UMAP panel gets one diamond marker per state at the state's *expression
# centroid*, projected into the displayed embedding.
SHOW_STATE_CENTROIDS: bool = True

# Each UMAP embedding paired with the PCA representation it was trained on, so a
# centroid can be projected in the same space UMAP's neighbor graph used.
_UMAP_TO_PCA: dict[str, str] = {
    OBSM_UMAP: OBSM_PCA,
    OBSM_UMAP_SHARED_GENES: OBSM_PCA_SHARED_GENES,
}


def _project_state_centroids(
    coords: np.ndarray,
    rep: np.ndarray,
    states: np.ndarray,
    *,
    n_neighbors: int = 15,
) -> dict[int, tuple[float, float]]:
    """Locate each state's expression centroid within the displayed UMAP.

    ``sc.tl.umap`` discards the fitted reducer, so there is no exact
    ``transform`` for a fresh point. Instead the centroid is taken in ``rep``
    (the PCA space the embedding was trained on) -- where, PCA being linear, the
    mean of a state's cells equals the projection of that state's gene-space
    expression centroid -- and then placed in the embedding by averaging the
    UMAP coordinates of its ``n_neighbors`` nearest cells in ``rep``. This is a
    cheap, deterministic stand-in for ``umap.transform`` that, unlike refitting,
    lands the centroid in the *same* embedding the cells are plotted in.

    Returns ``{state: (umap_x, umap_y)}``.
    """
    n = min(n_neighbors, rep.shape[0])
    out: dict[int, tuple[float, float]] = {}
    for state in sorted(np.unique(states).tolist()):
        centroid = rep[states == state].mean(axis=0)
        d2 = np.sum((rep - centroid) ** 2, axis=1)
        nn = np.argpartition(d2, n - 1)[:n]  # n nearest cells in PCA space
        out[int(state)] = (float(coords[nn, 0].mean()), float(coords[nn, 1].mean()))
    return out


def _add_state_centroid_markers(
    fig: go.Figure,
    centroids: dict[int, tuple[float, float]],
    *,
    row: int,
    col: int,
    palette: dict[int, tuple],
    dot_size: float,
    names: dict[int, str] | None = None,
) -> None:
    """Overlay one diamond per state at its (precomputed) projected expression
    centroid.

    Each marker shares the state's ``legendgroup`` (so the existing legend
    toggles it with its cells) and carries no legend entry of its own.
    ``centroids`` maps state -> (umap_x, umap_y); an empty dict adds nothing.
    """
    for state, (cx, cy) in centroids.items():
        fig.add_trace(
            go.Scattergl(
                x=[cx],
                y=[cy],
                mode="markers",
                marker=dict(
                    size=dot_size * 3.0 + 6.0,
                    color=_hex(palette.get(state)),
                    symbol="diamond",
                    line=dict(width=1.5, color="black"),
                    opacity=1.0,
                ),
                name=f"{_state_label(state, names)} centroid",
                legendgroup=f"state{state}",
                showlegend=False,
                hovertemplate=f"{_state_label(state, names)} centroid<extra></extra>",
            ),
            row=row,
            col=col,
        )


@st.cache_data(show_spinner=False)
def _umap_panel_data(
    _adata_sc, root_str: str, k: int, umap_key: str, want_centroids: bool
):
    """Threshold-independent UMAP-panel data for ``(root, k, umap_key)``, cached.

    The confidence threshold only recolours the *spatial* panel, but the headline
    figure is rebuilt on every threshold drag; without this cache each drag would
    re-run the per-cell state inference and the O(n_cells) centroid projection for
    the UMAP panel(s) too. Here that work is memoised on ``(root_str, k, umap_key,
    want_centroids)`` — the scaffold ``_adata_sc`` is passed with a leading
    underscore so ``st.cache_data`` does not try to hash it (it is the stable,
    resource-cached scaffold for this run root).

    Returns picklable arrays: per-cell ``states``/``start_cluster``, the embedding
    ``coords``, and ``centroids`` = {state: (umap_x, umap_y)} (empty if unwanted
    or the PCA rep is missing).
    """
    start_cluster_to_state = load_start_cluster_to_state(
        data_access.k_dir(Path(root_str), k)
    )
    start_cluster = _adata_sc.obs[OBS_START_CLUSTER].astype(int).to_numpy()
    states = start_cluster_to_state[start_cluster].astype(int)
    coords = np.asarray(_adata_sc.obsm[umap_key])

    centroids: dict[int, tuple[float, float]] = {}
    if want_centroids:
        pca_key = _UMAP_TO_PCA.get(umap_key)
        if pca_key is not None and pca_key in _adata_sc.obsm:
            rep = np.asarray(_adata_sc.obsm[pca_key])
            centroids = _project_state_centroids(coords, rep, states)
        else:
            logger.warning(
                "SHOW_STATE_CENTROIDS: %s missing from scaffold; skipping centroids "
                "for %s.",
                pca_key,
                umap_key,
            )
    return {
        "coords": coords,
        "start_cluster": start_cluster,
        "states": states,
        "centroids": centroids,
    }


def _add_umap_traces(
    fig: go.Figure,
    adata_sc: AnnData,
    root: Path,
    k: int,
    *,
    col: int,
    row: int = 1,
    palette: dict[int, tuple],
    legend_shown: set[int],
    dot_size: float,
    umap_key: str = OBSM_UMAP,
    equal_aspect: bool = True,
    top_genes: TopGenes | None = None,
    names: dict[int, str] | None = None,
) -> None:
    """Add a UMAP scatter (one trace per state) to subplot (``row``, ``col``).

    ``umap_key`` selects the embedding (all-gene ``X_umap`` by default, or the
    shared-gene ``X_umap_shared_genes``). This K's tree cut is applied to the
    scaffold via the cached ``_umap_panel_data`` so ``computed_state`` is correct
    for ``k`` without recomputing on unrelated (e.g. threshold) reruns. Each trace
    joins ``legendgroup`` ``state<n>`` so the interaction layer can toggle the same
    state in every subplot; a state gets a legend entry only the first time it is
    seen (tracked in ``legend_shown``).
    """
    data = _umap_panel_data(adata_sc, str(root), int(k), umap_key, SHOW_STATE_CENTROIDS)
    coords = data["coords"]
    states = data["states"]
    start_cluster = data["start_cluster"]

    for state in sorted(np.unique(states).tolist()):
        mask = states == state
        show = state not in legend_shown
        legend_shown.add(state)
        fig.add_trace(
            go.Scattergl(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker=dict(size=dot_size, color=_hex(palette.get(state)), opacity=1.0),
                name=_state_label(state, names),
                legendgroup=f"state{state}",
                showlegend=show,
                customdata=start_cluster[mask][:, None],
                hovertemplate=(
                    f"{_state_label(state, names)}"
                    "<br>Start cluster %{customdata[0]}"
                    f"{_top_genes_suffix(top_genes, state)}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    _style_scatter_axes(
        fig,
        row=row,
        col=col,
        xref=fig.data[-1].xaxis or "x",
        coords=coords,
        xtitle="UMAP1",
        ytitle="UMAP2",
        equal_aspect=equal_aspect,
        hide_ticks=True,
    )

    if SHOW_STATE_CENTROIDS:
        _add_state_centroid_markers(
            fig,
            data["centroids"],
            row=row,
            col=col,
            palette=palette,
            dot_size=dot_size,
            names=names,
        )


def _pin_spatial_axes(
    fig: go.Figure, coords: np.ndarray, *, col: int, row: int = 1, equal_aspect: bool
) -> None:
    """Style the spatial subplot at (``row``, ``col``): fixed extent, flipped y,
    and equal aspect only when ``equal_aspect`` (see ``_style_scatter_axes``).
    ``xref`` is read from the last-added trace, which belongs to this subplot."""
    _style_scatter_axes(
        fig,
        row=row,
        col=col,
        xref=fig.data[-1].xaxis or "x",
        coords=coords,
        xtitle="x",
        ytitle="y",
        equal_aspect=equal_aspect,
        reversed_y=True,
    )


def _add_spatial_traces(
    fig: go.Figure,
    coords: np.ndarray,
    hard: np.ndarray,
    confidence: np.ndarray | None,
    threshold: float,
    *,
    col: int,
    row: int = 1,
    palette: dict[int, tuple],
    legend_shown: set[int],
    dot_size: float,
    plot_confidence: bool = False,
    equal_aspect: bool = True,
    top_genes: TopGenes | None = None,
    names: dict[int, str] | None = None,
) -> None:
    """Add the ST spatial scatter to subplot (``row``, ``col``).

    Default: one trace per state (colours from ``palette``); spots below
    ``threshold`` are drawn grey. State traces share the ``state<n>`` legendgroup
    with the UMAP so one legend controls both panels.

    When ``plot_confidence`` is set and confidence exists, instead draw a single
    trace coloured by a continuous confidence colourscale (with a colourbar); the
    threshold is ignored in this mode.
    """
    if plot_confidence and confidence is not None:
        fig.add_trace(
            go.Scattergl(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker=dict(
                    size=dot_size,
                    color=confidence,
                    colorscale="Viridis",
                    cmin=0.0,
                    cmax=1.0,
                    showscale=True,
                    colorbar=dict(title="confidence", thickness=12, len=0.85, x=1.0),
                ),
                name="confidence",
                showlegend=False,
                meta="spatial",
                # One trace for every state here, so the label has to travel
                # per point: an object array keeps the name a string and the
                # confidence a float, which a column_stack would collapse.
                customdata=np.array(
                    [
                        (float(c), _state_label(s, names))
                        for c, s in zip(confidence, hard)
                    ],
                    dtype=object,
                ),
                hovertemplate=(
                    "confidence %{customdata[0]:.3f}<br>"
                    "%{customdata[1]}<br>"
                    "x %{x:.1f}  y %{y:.1f}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        _pin_spatial_axes(fig, coords, row=row, col=col, equal_aspect=equal_aspect)
        return

    if confidence is not None and threshold > 0.0:
        low = confidence < threshold
    else:
        low = np.zeros(len(hard), dtype=bool)
    keep = ~low

    n_low = int(low.sum())
    if n_low:
        # Only reached when confidence is not None (threshold > 0 required above).
        fig.add_trace(
            go.Scattergl(
                x=coords[low, 0],
                y=coords[low, 1],
                mode="markers",
                marker=dict(size=dot_size, color=_GREY_RGBA),
                name=f"below threshold ({n_low})",
                legendgroup="below_threshold",
                meta="spatial",
                # Mixed states in one trace, so the label rides along per point.
                customdata=np.array(
                    [
                        (_state_label(s, names), float(c))
                        for s, c in zip(hard[low], confidence[low])
                    ],
                    dtype=object,
                ),
                hovertemplate=(
                    "%{customdata[0]} (below)<br>"
                    "confidence %{customdata[1]:.3f}<br>"
                    "x %{x:.1f}  y %{y:.1f}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    for state in sorted(np.unique(hard[keep]).tolist()):
        mask = keep & (hard == state)
        show = state not in legend_shown
        legend_shown.add(state)
        genes = _top_genes_suffix(top_genes, state)
        head = f"{_state_label(state, names)}{genes}"
        if confidence is not None:
            customdata = confidence[mask][:, None]
            hovertemplate = (
                f"{head}<br>"
                "confidence %{customdata[0]:.3f}<br>"
                "x %{x:.1f}  y %{y:.1f}<extra></extra>"
            )
        else:
            customdata = None
            hovertemplate = f"{head}<br>x %{{x:.1f}}  y %{{y:.1f}}<extra></extra>"
        fig.add_trace(
            go.Scattergl(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker=dict(size=dot_size, color=_hex(palette.get(state)), opacity=1.0),
                name=_state_label(state, names),
                legendgroup=f"state{state}",
                showlegend=show,
                meta="spatial",
                customdata=customdata,
                hovertemplate=hovertemplate,
            ),
            row=row,
            col=col,
        )

    _pin_spatial_axes(fig, coords, row=row, col=col, equal_aspect=equal_aspect)


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
    names = _names_for(root, k)
    # Each adder pins its own subplot axes.
    if have_shared:
        _add_umap_traces(
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
        _add_umap_traces(
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
        _add_spatial_traces(
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
    return _base_layout(
        fig,
        height=720,
        legend={**_STATE_LEGEND, "y": 1.12},
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
    names = _names_for(root, k)
    if have_umap:
        _add_umap_traces(
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
        _add_spatial_traces(
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
    return _base_layout(
        fig,
        height=height,
        legend={**_STATE_LEGEND, "y": 1.02},
        margin=dict(l=10, r=10, t=60, b=10),
    )


# --------------------------------------------------------------------------- #
# Compare-tab combined report cards (all selected mappers in one plot/table)
# --------------------------------------------------------------------------- #
def _mapper_color(i: int) -> str:
    """Stable qualitative colour for the i-th mapper in a comparison."""
    return _TAB10[i % 10]


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
        subplot_titles=["Gene-wise", "Spot-wise"],
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
                    marker_color=_mapper_color(i),
                    boxpoints=False,
                ),
                row=1,
                col=col,
            )
    _base_layout(
        fig,
        height=340,
        font=_CARD_FONT,
        boxmode="group",
        margin=dict(l=10, r=10, t=30, b=52),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
    )
    fig.update_yaxes(title_text="cosine similarity", row=1, col=1)
    fig.update_xaxes(tickangle=-30)
    return fig


def render_compare_box_figure(
    data: dict[str, np.ndarray],
    *,
    title: str,
    ytitle: str,
    yrange: tuple[float, float] | None = None,
    height: int = 300,
) -> go.Figure:
    """One box per mapper of the per-spot distribution in ``data`` (mapper ->
    values). Used to compare mapping sharpness (max probability) and confidence."""
    fig = go.Figure()
    for i, (m, vals) in enumerate(data.items()):
        fig.add_trace(
            go.Box(
                y=np.asarray(vals, dtype=float),
                name=m,
                marker_color=_mapper_color(i),
                boxpoints=False,
                showlegend=False,
            )
        )
    _base_layout(
        fig,
        height=height,
        font=_CARD_FONT,
        title=dict(text=title, font=dict(size=13)),
        yaxis_title=ytitle,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    if yrange is not None:
        fig.update_yaxes(range=list(yrange))
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
    x = [_state_label(s, names) for s in states]
    fig = go.Figure()
    for i, (m, fr) in enumerate(spot_fracs.items()):
        fig.add_trace(go.Bar(name=m, x=x, y=fr, marker_color=_mapper_color(i)))
    _base_layout(
        fig,
        height=height,
        font=_CARD_FONT,
        barmode="group",
        title=dict(text="Spot cell-type fractions", font=dict(size=13)),
        yaxis_title="fraction",
        margin=dict(l=10, r=10, t=40, b=40),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


# --------------------------------------------------------------------------- #
# Single-cell reference tab (clustering-side, mapper-independent) figures
# --------------------------------------------------------------------------- #
def _add_start_cluster_umap_traces(
    fig: go.Figure,
    adata_sc: AnnData,
    start_cluster_to_state: np.ndarray,
    *,
    col: int,
    row: int = 1,
    dot_size: float,
    equal_aspect: bool = True,
    state_labels: dict[int, str] | None = None,
) -> None:
    """Add the start-cluster UMAP (one trace per start cluster) to subplot
    (``row``, ``col``).

    ``state_labels`` names the state each start cluster merges into, for the
    hover's "-> <cell type>" line; the traces themselves are start clusters and
    keep their own names.

    Each start cluster gets a distinct qualitative colour but is tagged with the
    ``state<n>`` legendgroup of the state it merges into, so a single legend click
    greys its cells here too; the start clusters keep out of the legend (identified
    via hover) to avoid duplicating the shared state legend.
    """
    coords = np.asarray(adata_sc.obsm[OBSM_UMAP])
    names = _start_cluster_names(adata_sc)
    start_cluster = adata_sc.obs[OBS_START_CLUSTER].astype(int).to_numpy()
    for i, lc in enumerate(sorted(np.unique(start_cluster).tolist())):
        mask = start_cluster == lc
        s = int(start_cluster_to_state[lc])
        target = _state_label(s, state_labels)
        fig.add_trace(
            go.Scattergl(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker=dict(size=dot_size, color=_TAB20[i % 20], opacity=1.0),
                name=names[lc],
                legendgroup=f"state{s}",
                showlegend=False,
                hovertemplate=(f"{names[lc]}<br>→ {target}<extra></extra>"),
            ),
            row=row,
            col=col,
        )
    _style_scatter_axes(
        fig,
        row=row,
        col=col,
        xref=fig.data[-1].xaxis or "x",
        coords=coords,
        xtitle="UMAP1",
        ytitle="UMAP2",
        equal_aspect=equal_aspect,
        hide_ticks=True,
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
    names = _names_for(root, k)
    # Left panel shows start clusters, not cell types, so it gets no top genes —
    # only the names, for the "-> <cell type>" line in its hover.
    _add_start_cluster_umap_traces(
        fig,
        adata_sc,
        start_cluster_to_state,
        col=1,
        dot_size=dot_size_umap,
        state_labels=names,
    )
    _add_umap_traces(
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
        _add_umap_traces(
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

    return _base_layout(
        fig,
        height=520,
        legend={**_STATE_LEGEND, "y": 1.12},
        margin=dict(l=10, r=10, t=70, b=10),
    )


def render_start_cluster_merge_figure(
    adata_sc: AnnData, root: Path, k: int
) -> go.Figure:
    """Horizontal stacked bars: one bar per computed state, segmented by the
    start clusters merged into it (segment width ∝ cell count, labelled with the
    start-cluster name)."""
    start_cluster_to_state = load_start_cluster_to_state(data_access.k_dir(root, k))
    infer_cell_to_state_cluster(adata_sc, start_cluster_to_state)
    names = _start_cluster_names(adata_sc)
    state_labels = _names_for(root, k)
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
        return f"{_state_label(s, state_labels)}  ({n} cluster{'s' if n != 1 else ''})"

    fig = go.Figure()
    # barmode="stack" accumulates same-row segments in trace order (largest first).
    for s in states:
        color = _hex(palette.get(s))
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
                        f"{_state_label(s, state_labels)} · {names[lc]}"
                        "<br>%{x} cells<extra></extra>"
                    ),
                )
            )

    n_states = len(states)
    _base_layout(
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


def _rank_markers(X, labels: np.ndarray, genes: list[str]) -> dict | None:
    """Marker genes per label: scanpy ``rank_genes_groups`` (Wilcoxon, each label
    vs. the rest) over the columns of ``X`` (lognorm, observations x ``genes``)
    — the scanpy clustering tutorial's recipe.

    Returns ``{"genes", "states", "ranked", "score", "lfc", "padj", "mean",
    "frac"}`` or ``None`` when fewer than two testable labels (>= 2 observations
    each) exist. ``ranked`` maps each tested label to gene indices by decreasing
    score; ``score``/``lfc``/``padj`` are (labels x genes), NaN for an untested
    label; ``mean`` (mean lognorm, zeros included) and ``frac`` (fraction > 0)
    feed the dotplot.
    """
    import pandas as pd
    import scanpy as sc

    labels = np.asarray(labels).astype(int)
    states = sorted(np.unique(labels).tolist())
    counts = {s: int((labels == s).sum()) for s in states}
    # rank_genes_groups needs >= 2 observations per tested group and >= 2 groups.
    tested = [s for s in states if counts[s] >= 2]
    if len(tested) < 2:
        return None

    adata = AnnData(
        X=X.copy(),
        obs=pd.DataFrame({"state": pd.Categorical([str(s) for s in labels])}),
    )
    adata.obs_names = [str(i) for i in range(adata.n_obs)]
    adata.var_names = genes
    adata.uns["log1p"] = {"base": None}  # natural-log lognorm, for the fold changes
    sc.tl.rank_genes_groups(
        adata,
        groupby="state",
        groups=[str(s) for s in tested],
        reference="rest",
        method="wilcoxon",
        n_genes=len(genes),
    )

    gene_idx = {g: j for j, g in enumerate(genes)}
    shape = (len(states), len(genes))
    score, lfc, padj = (np.full(shape, np.nan) for _ in range(3))
    ranked: dict[int, list[int]] = {}
    for i, s in enumerate(states):
        if s not in tested:
            continue
        df = sc.get.rank_genes_groups_df(adata, group=str(s))
        cols = df["names"].map(gene_idx).to_numpy()
        score[i, cols] = df["scores"].to_numpy()
        lfc[i, cols] = df["logfoldchanges"].to_numpy()
        padj[i, cols] = df["pvals_adj"].to_numpy()
        ranked[s] = cols.tolist()  # already sorted by decreasing score

    mean = np.zeros(shape)
    frac = np.zeros(shape)
    for i, s in enumerate(states):
        block = X[labels == s]
        mean[i] = np.asarray(block.mean(axis=0)).ravel()
        frac[i] = np.asarray((block > 0).mean(axis=0)).ravel()

    return {
        "genes": genes,
        "states": states,
        "ranked": ranked,
        "score": score,
        "lfc": lfc,
        "padj": padj,
        "mean": mean,
        "frac": frac,
    }


def _shared_genes(adata_sc: AnnData) -> list[str]:
    """The scaffold's shared genes (scRNA ∩ ST), in their stored order."""
    shared = list(adata_sc.uns.get(UNS_SHARED_GENES, []))
    return [g for g in shared if g in adata_sc.var_names]


@st.cache_data(show_spinner="Ranking marker genes…")
def _state_markers(_adata_sc, root_str: str, k: int):
    """Reference marker genes per computed cell type (``_rank_markers``) over the
    shared genes, on the all-gene lognorm layer. ``None`` when fewer than two
    shared genes or testable cell types exist. Cached on ``(root, k)`` because
    the dotplot, the modal and every panel's hover read it; ``_adata_sc`` leads
    with an underscore so ``st.cache_data`` does not hash the scaffold (it is the
    stable, resource-cached one for this run root).
    """
    start_cluster_to_state = load_start_cluster_to_state(
        data_access.k_dir(Path(root_str), k)
    )
    infer_cell_to_state_cluster(_adata_sc, start_cluster_to_state)
    cell_states = _adata_sc.obs[OBS_COMPUTED_STATE].astype(int).to_numpy()
    genes = _shared_genes(_adata_sc)
    if len(genes) < 2:
        return None
    return _rank_markers(_adata_sc[:, genes].layers[LAYER_LOGNORM], cell_states, genes)


class SpatialContext(NamedTuple):
    """What the spatial side of the hover / cell-type modal needs: the ST file
    (for its expression), this mapper's per-spot hard assignment at this K, and
    the spot coordinates (``None`` when the ST file has none)."""

    st_path: str
    hard: np.ndarray
    coords: np.ndarray | None


@st.cache_resource(show_spinner="Loading spatial expression…")
def _spatial_lognorm(st_path_str: str, genes: tuple[str, ...]):
    """ST lognorm over ``genes`` (see ``data_access.load_spatial_lognorm``),
    loaded once per ST file and gene set."""
    return data_access.load_spatial_lognorm(Path(st_path_str), list(genes))


@st.cache_data(show_spinner="Ranking spatial marker genes…")
def _spatial_state_markers(_adata_sc, _hard, root_str: str, k: int, st_path_str: str):
    """Spatial marker genes per *assigned* cell type (``_rank_markers`` on the ST
    lognorm, grouped by this mapper's hard assignment) over the shared genes.
    ``None`` when the ranking is impossible or the assignment does not match the
    ST file's spots. Cached on ``(root, k, st_path)``, which fix ``_hard``.
    """
    genes = _shared_genes(_adata_sc)
    if len(genes) < 2:
        return None
    X = _spatial_lognorm(st_path_str, tuple(genes))
    if X.shape[0] != len(_hard):
        logger.warning(
            "Spatial markers skipped: %d spots in %s but %d assignments.",
            X.shape[0],
            st_path_str,
            len(_hard),
        )
        return None
    return _rank_markers(X, _hard, genes)


def _markers(
    adata_sc: AnnData, root: Path, k: int, spatial: SpatialContext | None
) -> dict | None:
    """The reference ranking, or the spatial one when ``spatial`` is given."""
    if spatial is None:
        return _state_markers(adata_sc, str(root), int(k))
    return _spatial_state_markers(
        adata_sc, spatial.hard, str(root), int(k), spatial.st_path
    )


def _up_down(mk: dict, state: int, n: int) -> tuple[list[int], list[int]]:
    """Gene indices of ``state``'s ``n`` most up- (positive score, highest first)
    and downregulated (negative score, lowest first) genes in ranking ``mk``."""
    cols = mk["ranked"][state]
    row = mk["score"][mk["states"].index(state)]
    up = [j for j in cols if row[j] > 0][:n]
    down = [j for j in reversed(cols) if row[j] < 0][:n]
    return up, down


# Per state: its top-ranked marker genes for the hover box, keyed "up"/"down"
# (reference) and "up_st"/"down_st" (spatial, mapper tabs only).
TopGenes = dict[int, dict[str, list[str]]]


def top_genes_per_state(
    adata_sc: AnnData | None,
    root: Path | None,
    k: int,
    n: int = 3,
    *,
    spatial: SpatialContext | None = None,
) -> TopGenes:
    """Per state, the ``n`` most up- and downregulated marker genes (Wilcoxon vs.
    rest) in the reference and, with ``spatial``, among the ST spots assigned to
    it: ``{state: {"up", "down"[, "up_st", "down_st"]: [...]}}``.

    Up genes need a positive score, down genes a negative one, most extreme
    first. Empty when the scaffold or the ranking is unavailable — callers then
    simply show no genes; a state too small to test is left out.
    """
    if adata_sc is None or root is None:
        return {}
    out: TopGenes = {}
    sources = [("", None)] + ([("_st", spatial)] if spatial is not None else [])
    for suffix, ctx in sources:
        mk = _markers(adata_sc, root, k, ctx)
        if mk is None:
            continue
        for s in mk["ranked"]:
            up, down = _up_down(mk, s, n)
            hits = out.setdefault(int(s), {})
            hits["up" + suffix] = [mk["genes"][j] for j in up]
            hits["down" + suffix] = [mk["genes"][j] for j in down]
    return out


def _top_genes_suffix(top_genes: TopGenes | None, state: int) -> str:
    """The hover box's marker lines for ``state``: ``↑ A, B, C`` / ``↓ D, E, F``
    from the reference, or — when the figure carries spatial rankings — four
    lines ``↑ scRNA``, ``↑ ST``, ``↓ scRNA``, ``↓ ST``. Empty lists are left out;
    '' when the state is unknown."""
    top_genes = top_genes or {}
    hits = top_genes.get(int(state)) or {}
    has_st = any("up_st" in v for v in top_genes.values())
    lines = (
        (
            ("up", "↑ scRNA:"),
            ("up_st", "↑ ST:"),
            ("down", "↓ scRNA:"),
            ("down_st", "↓ ST:"),
        )
        if has_st
        else (("up", "↑"), ("down", "↓"))
    )
    return "".join(
        f"<br>{label} {', '.join(hits[key])}" for key, label in lines if hits.get(key)
    )


# Largest dot diameter (px) of the marker dotplot; a dot's *area* is proportional
# to the fraction of the row's cells expressing the gene, as in scanpy's dotplot.
_DOT_MAX_PX = 16.0
# Colourbar length (px) and the margins of the dotplot; the size legend is placed
# below the colourbar in pixels, so the two never overlap whatever the height.
_COLORBAR_PX = 150
# Vertical room (px) for the size legend: its title plus five entries.
_SIZE_LEGEND_PX = 150
_DOTPLOT_MARGIN = dict(l=10, r=10, t=24, b=90)


def render_state_markers_figure(
    adata_sc: AnnData, root: Path, k: int, *, n_genes: int = 5
) -> go.Figure | None:
    """Dotplot of each computed cell type's top ``n_genes`` marker genes
    (``rank_genes_groups``, Wilcoxon vs. rest, shared genes), in the style of
    scanpy's ``rank_genes_groups_dotplot(standard_scale="var")``.

    Rows are cell types; columns are each tested type's markers in turn (a gene
    can recur under several types), grouped under a bar in that type's colour.
    Dot area = fraction of the row's cells expressing the gene; dot colour = the
    row's mean lognorm expression, min-max scaled per gene across rows. Returns
    ``None`` when the ranking is unavailable (see ``_state_markers``).
    """
    mk = _state_markers(adata_sc, str(root), int(k))
    if mk is None:
        return None
    states, genes = mk["states"], mk["genes"]
    names = _names_for(root, k)
    palette = state_palette(k)

    # Columns: per tested state (in state order) its top n genes.
    col_genes: list[int] = []
    blocks: list[tuple[int, int, int]] = []  # (state, first col, last col)
    for s in states:
        cols = mk["ranked"].get(s, [])[:n_genes]
        if cols:
            blocks.append((s, len(col_genes), len(col_genes) + len(cols) - 1))
            col_genes += cols
    gcols = np.asarray(col_genes)

    mean = mk["mean"][:, gcols]
    lo, hi = mean.min(axis=0), mean.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    scaled = (mean - lo) / span
    frac = mk["frac"][:, gcols]

    row_labels = [
        f'<span style="color:{_hex(palette.get(s))};font-size:22px">●</span>'
        f" {_state_label(s, names)}"
        for s in states
    ]
    n_rows, n_cols = len(states), len(col_genes)
    xs = np.tile(np.arange(n_cols), n_rows)
    ys = np.repeat(np.arange(n_rows), n_cols)
    # Per-dot hover fields; object dtype keeps the labels as strings.
    custom = np.empty((n_rows * n_cols, 7), dtype=object)
    custom[:, 0] = np.repeat([_state_label(s, names) for s in states], n_cols)
    custom[:, 1] = np.tile([genes[j] for j in col_genes], n_rows)
    custom[:, 2] = frac.ravel()
    custom[:, 3] = mean.ravel()
    custom[:, 4] = mk["score"][:, gcols].ravel()
    custom[:, 5] = mk["lfc"][:, gcols].ravel()
    custom[:, 6] = mk["padj"][:, gcols].ravel()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                # Diameter ~ sqrt(frac) so the dot *area* tracks the fraction.
                size=_DOT_MAX_PX * np.sqrt(frac.ravel()),
                color=scaled.ravel(),
                colorscale="Reds",
                cmin=0.0,
                cmax=1.0,
                line=dict(width=0.5, color="rgba(80,80,80,0.6)"),
                colorbar=dict(
                    title=dict(text="Mean expr.<br>(scaled)", side="top"),
                    thickness=12,
                    lenmode="pixels",
                    len=_COLORBAR_PX,
                    y=1.0,
                    yanchor="top",
                ),
            ),
            customdata=custom,
            hovertemplate=(
                "%{customdata[0]}<br><b>%{customdata[1]}</b>"
                "<br>expressed in %{customdata[2]:.0%} of cells"
                "<br>mean lognorm %{customdata[3]:.2f}"
                "<br>score %{customdata[4]:.1f} · logFC %{customdata[5]:.2f}"
                " · p_adj %{customdata[6]:.1e}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    # Size key: invisible points that only draw legend entries.
    for f in (0.2, 0.4, 0.6, 0.8, 1.0):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=_DOT_MAX_PX * np.sqrt(f), color="#888", line=dict(width=0)
                ),
                name=f"{f:.0%}",
                hoverinfo="skip",
            )
        )
    # One bar per state's marker block, in the state's colour, above the dots.
    for s, a, b in blocks:
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=a - 0.35,
            x1=b + 0.35,
            y0=1.03,
            y1=1.03,
            line=dict(color=_hex(palette.get(s)), width=5),
        )

    margins = _DOTPLOT_MARGIN["t"] + _DOTPLOT_MARGIN["b"]
    height = max(n_rows * 30 + 190, margins + _COLORBAR_PX + 30 + _SIZE_LEGEND_PX)
    plot_px = height - _DOTPLOT_MARGIN["t"] - _DOTPLOT_MARGIN["b"]
    _base_layout(
        fig,
        height=height,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(n_cols)),
            ticktext=[genes[j] for j in col_genes],
            tickangle=-90,
            range=[-0.7, n_cols - 0.3],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_rows)),
            ticktext=row_labels,
            # State 0 on top.
            range=[n_rows - 0.5, -0.5],
            showgrid=False,
            zeroline=False,
            title="Computed cell type",
        ),
        legend=dict(
            title=dict(text="Fraction of cells<br>in group"),
            itemsizing="trace",
            x=1.02,
            xanchor="left",
            y=1.0 - (_COLORBAR_PX + 30) / plot_px,
            yanchor="top",
        ),
        # No title: the card header above the figure already names it.
        margin=_DOTPLOT_MARGIN,
    )
    return fig


# --------------------------------------------------------------------------- #
# Cell-type modal (right click on a state in the headline / reference UMAPs)
# --------------------------------------------------------------------------- #
def state_marker_table(
    adata_sc: AnnData,
    root: Path,
    k: int,
    state: int,
    n: int = 10,
    *,
    spatial: SpatialContext | None = None,
) -> dict[str, list[dict]] | None:
    """The ``n`` most up- and downregulated marker genes of ``state``, with their
    Wilcoxon statistics: ``{"up": [...], "down": [...]}``, each row a dict of
    ``gene``/``score``/``lfc``/``padj``/``frac`` (fraction of the state's cells
    — or, with ``spatial``, its assigned spots — expressing the gene). From the
    reference ranking, or the spatial one when ``spatial`` is given. ``None``
    when the state was not ranked there.
    """
    mk = _markers(adata_sc, root, k, spatial)
    if mk is None or int(state) not in mk["ranked"]:
        return None
    i = mk["states"].index(int(state))
    up, down = _up_down(mk, int(state), n)

    def row(j: int) -> dict:
        return {
            "gene": mk["genes"][j],
            "score": float(mk["score"][i, j]),
            "lfc": float(mk["lfc"][i, j]),
            "padj": float(mk["padj"][i, j]),
            "frac": float(mk["frac"][i, j]),
        }

    return {"up": [row(j) for j in up], "down": [row(j) for j in down]}


@st.cache_data(show_spinner=False)
def _gene_expression(_adata_sc, root_str: str, gene: str) -> np.ndarray:
    """Per-cell lognorm expression of ``gene`` (all-gene layer), cached per run
    root (one scaffold each) and gene."""
    return to_dense(_adata_sc[:, gene].layers[LAYER_LOGNORM]).ravel()


def _spatial_gene_expression(
    adata_sc: AnnData, spatial: SpatialContext, gene: str
) -> np.ndarray | None:
    """Per-spot ST lognorm expression of ``gene``, or ``None`` if it is not a
    shared gene."""
    genes = _shared_genes(adata_sc)
    if gene not in genes:
        return None
    X = _spatial_lognorm(spatial.st_path, tuple(genes))
    return to_dense(X[:, genes.index(gene)]).ravel()


# The "every other point" backdrop of the modal's highlight panels: lighter than
# ``_GREY_RGBA`` so even the pale tab20 state colours stand out against it.
_BACKDROP_GREY = "#e4e4e4"
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
    data = _umap_panel_data(adata_sc, str(root), int(k), OBSM_UMAP, False)
    spot_coords = (
        np.asarray(spatial.coords)
        if spatial is not None and spatial.coords is not None
        else None
    )
    return data, spot_coords


def _map_row_figure(
    panels: list[_MapPanel], *, color: str | None, dot_size: float
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
            for mask, c in ((~p.values, _BACKDROP_GREY), (p.values, color)):
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
        _style_scatter_axes(
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
    return _base_layout(fig, height=430, margin=dict(l=10, r=10, t=40, b=10))


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
    label = _state_label(state, _names_for(root, k))
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
    return _map_row_figure(
        panels, color=_hex(state_palette(k).get(int(state))), dot_size=dot_size
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
            _gene_expression(adata_sc, str(root), gene),
            ("UMAP1", "UMAP2"),
            False,
        )
    ]
    st_expr = (
        _spatial_gene_expression(adata_sc, spatial, gene)
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


# Cells per group handed to the browser for the violin's density estimate; a
# larger group is subsampled (fixed seed), which leaves the shape unchanged.
_VIOLIN_MAX_CELLS = 20000


def _add_violin(
    fig: go.Figure, vals: np.ndarray, name: str, color: str, rng, *, col: int
) -> None:
    """One violin (box + mean line) of ``vals`` in subplot column ``col``, with
    the group's size and fraction of non-zero values under its label."""
    if vals.size == 0:
        return
    frac = float((vals > 0).mean())
    shown = (
        rng.choice(vals, _VIOLIN_MAX_CELLS, replace=False)
        if vals.size > _VIOLIN_MAX_CELLS
        else vals
    )
    fig.add_trace(
        go.Violin(
            y=shown,
            name=f"{name}<br>n = {vals.size:,} · {frac:.0%} expressing",
            line_color=color,
            fillcolor=color,
            opacity=0.75,
            box=dict(
                visible=True,
                width=0.12,
                fillcolor="white",
                line=dict(color="#333333", width=1.5),
            ),
            meanline=dict(visible=True, color="#333333", width=1.5),
            points=False,
            spanmode="hard",
            showlegend=False,
            hoverinfo="y",
        ),
        row=1,
        col=col,
    )


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
    data = _umap_panel_data(adata_sc, str(root), int(k), OBSM_UMAP, False)
    label = _state_label(state, _names_for(root, k))
    color = _hex(state_palette(k).get(int(state)))
    rng = np.random.default_rng(0)
    st_expr = (
        _spatial_gene_expression(adata_sc, spatial, gene)
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
    expr = _gene_expression(adata_sc, str(root), gene)
    _add_violin(fig, expr[inside], label, color, rng, col=1)
    _add_violin(fig, expr[~inside], "All other cells", "#9e9e9e", rng, col=1)
    fig.update_yaxes(title_text=f"{gene} (lognorm)", row=1, col=1)
    if st_expr is not None:
        on = np.asarray(spatial.hard) == int(state)
        _add_violin(fig, st_expr[on], label, color, rng, col=2)
        _add_violin(fig, st_expr[~on], "All other spots", "#9e9e9e", rng, col=2)
        fig.update_yaxes(title_text=f"{gene} (lognorm)", row=1, col=2)
    return _base_layout(fig, height=360, margin=dict(l=10, r=10, t=40, b=10))


# --------------------------------------------------------------------------- #
# Per-mapper report dashboard figures (interactive Plotly cards)
# --------------------------------------------------------------------------- #
_CARD_FONT = dict(family="'Source Sans Pro', 'Segoe UI', sans-serif", size=12)
_CELL_COLOR = "#4c78a8"  # cell / soft
_SPOT_COLOR = "#f58518"  # spot / hard


def _card_layout(fig: go.Figure, *, height: int, **extra) -> go.Figure:
    """Apply the shared compact look used by every report-dashboard card."""
    extra.setdefault("margin", dict(l=10, r=10, t=30, b=10))
    return _base_layout(fig, height=height, font=_CARD_FONT, **extra)


# --------------------------------------------------------------------------- #
# K-sweep ("Comparing K") figures
# --------------------------------------------------------------------------- #
# Two rows of three cards, all linked client-side by ``widgets.linked_plot``: one
# line card per criterion (its two measured curves plus their harmonic mean, see
# spatialaim/metrics/kselection.py) and one scatter card per criterion pair (one dot per K, at its
# two harmonic means). Every K-indexed trace carries ``customdata[i][0] = K`` --
# that is what the linked-plot component keys hovering, highlighting and clicking
# on; the scatter cards additionally stash their null-crosshair coordinates and
# axis ranges in ``layout.meta["spatialaim"]``.

# The harmonic-mean curve and the shuffle-null crosshair are derived, not
# measured, so they stay off the categorical palette: light grey dashed for the
# combined score, slightly darker grey dotted for the null crosshair (they never
# share a plot). ``_RING_COLOR`` outlines whichever point the linked hover picks.
_COMBINED_COLOR = "#aeb4bb"
_NULL_COLOR = "#7c828a"
_RING_COLOR = "#22262b"
# Marks the best overall K in every criterion card: amber rather than pure yellow,
# which is unreadable on white. Public because the GUI tints the matching ★ in the
# "Best K" buttons with it.
BEST_K_COLOR = "#e0a800"
_MARKER_SIZE = 7
_SCATTER_MARKER_SIZE = 11


def _k_customdata(*columns) -> list[list[float]]:
    """Per-point ``customdata`` rows whose first column is K (the linking key).

    Returned as plain nested lists on purpose: plotly.py serialises ndarrays as
    base64 typed arrays, which only Plotly.js itself decodes -- the linked-plot
    component reads ``customdata`` out of the figure JSON directly.
    """
    stacked = [np.asarray(c, dtype=float) for c in columns]
    return [[float(v) for v in row] for row in zip(*stacked)]


def _finite_or_none(value: float) -> float | None:
    """``None`` for non-finite values, so they survive JSON as ``null``."""
    v = float(value)
    return v if np.isfinite(v) else None


def _null_text(value: float) -> str:
    """A null baseline formatted for a hover label ("n/a" when not computed)."""
    v = float(value)
    return f"{v:.3f}" if np.isfinite(v) else "n/a"


def render_ksweep_criterion_figure(
    df, criterion, *, index: int = 0, height: int = 300, best_k: int | None = None
) -> go.Figure:
    """The line card for one criterion: its two curves over K plus their harmonic
    mean (the score the scatter cards below plot against each other).

    A criterion whose two curves live on different scales (the spatial z-scores)
    is drawn *scaled* -- each curve divided by its own maximum over the sweep --
    so both curves and their mean share one axis; the hover still reports the raw
    value. ``index`` picks the criterion's colour pair out of the shared palette.

    ``best_k`` marks one K with an amber dashed line and a star on the x-axis (the
    best overall K, so the same K is marked in all three cards). It is drawn as a
    shape plus an annotation, not a trace, so it stays out of the legend and out of
    the linked-plot component's K bookkeeping.
    """
    k = df["k"].to_numpy()
    raw_a, raw_b, scaled_a, scaled_b = scores.scaled_curves(df, criterion)
    palette = (_mapper_color(2 * index), _mapper_color(2 * index + 1))

    fig = go.Figure()
    for column, label, raw, scaled, color in zip(
        criterion.columns,
        criterion.curve_labels,
        (raw_a, raw_b),
        (scaled_a, scaled_b),
        palette,
    ):
        if column not in df.columns:
            continue
        if criterion.scale_to_max:
            hover = (
                f"{label}<br>K %{{customdata[0]:.0f}}<br>"
                "scaled %{y:.3f}  (raw %{customdata[1]:.2f})<extra></extra>"
            )
        else:
            hover = f"{label}<br>K %{{customdata[0]:.0f}}<br>%{{y:.3f}}<extra></extra>"
        fig.add_trace(
            go.Scatter(
                x=k,
                y=scaled,
                customdata=_k_customdata(k, raw),  # [K, raw value]
                mode="lines+markers",
                name=label,
                line=dict(color=color),
                marker=dict(
                    color=color,
                    size=_MARKER_SIZE,
                    line=dict(width=0, color=_RING_COLOR),
                ),
                hovertemplate=hover,
            )
        )

    combined = scores.harmonic_mean(scaled_a, scaled_b)
    fig.add_trace(
        go.Scatter(
            x=k,
            y=combined,
            customdata=_k_customdata(k),
            mode="lines+markers",
            name="harmonic mean",
            line=dict(color=_COMBINED_COLOR, dash="dash", width=2),
            # The mean is undefined wherever either curve is (a K with no
            # neighbourhood-enrichment z, say); bridge those Ks rather than break
            # the line, the missing markers already show where they are.
            connectgaps=True,
            marker=dict(
                color=_COMBINED_COLOR,
                size=_MARKER_SIZE,
                symbol="diamond",
                line=dict(width=0, color=_RING_COLOR),
            ),
            hovertemplate=(
                "harmonic mean<br>K %{customdata[0]:.0f}<br>%{y:.3f}<extra></extra>"
            ),
        )
    )

    extra_layout: dict = {}
    if best_k is not None:
        # The line stops above the star instead of running through it: a dashed
        # line ending inside the marker reads as a smudge.
        fig.add_shape(
            type="line",
            xref="x",
            x0=float(best_k),
            x1=float(best_k),
            yref="paper",
            y0=0.1,
            y1=1.0,
            line=dict(color=BEST_K_COLOR, dash="dash", width=2),
            layer="below",
        )
        # A marker, not a text annotation: an SVG star symbol is centred exactly on
        # its coordinate, whereas a "★" glyph is centred by the font's side
        # bearings and drifts off the line. It rides an invisible 0..1 overlay
        # y-axis so pinning it to the bottom cannot stretch the real y range.
        fig.add_trace(
            go.Scatter(
                x=[float(best_k)],
                y=[0.0],
                yaxis="y2",
                mode="markers",
                marker=dict(symbol="star", size=13, color=BEST_K_COLOR),
                showlegend=False,
                cliponaxis=False,
                hoverinfo="text",
                hovertext=f"best overall K = {int(best_k)}",
            )
        )
        extra_layout["yaxis2"] = dict(
            overlaying="y", range=[0.0, 1.0], visible=False, fixedrange=True
        )

    return _base_layout(
        fig,
        height=height,
        font=_CARD_FONT,
        title=dict(text=f"{criterion.label} (higher is better)", font=dict(size=13)),
        xaxis_title="K",
        yaxis_title=criterion.unit,
        margin=dict(l=10, r=10, t=40, b=52),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
        **extra_layout,
    )


def _score_axis_range(
    *value_arrays, pad_lo: float = 0.06, pad_hi: float = 0.06, from_zero: bool = False
) -> list[float] | None:
    """A padded [lo, hi] range covering every finite value, or ``None``.

    The scatter axes are pinned (rather than autoranged) so the client-side null
    crosshair can span the full axis and so hovering never re-zooms the view; any
    null baseline is included, which is why a null far from the observed values
    compresses the dots — the plots stay drag-zoomable. The two paddings are
    separate fractions of the span because each dot carries its K as a label above
    it, which needs more headroom at the top than anywhere else. ``from_zero`` pins
    the low end at 0 (no bottom pad) so a criterion is read in absolute terms.
    """
    values = np.concatenate([np.asarray(a, dtype=float).ravel() for a in value_arrays])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    lo, hi = float(finite.min()), float(finite.max())
    if from_zero:
        lo = min(lo, 0.0)
    span = (hi - lo) if hi > lo else max(abs(hi), 1.0)
    lo_out = 0.0 if from_zero and lo >= 0.0 else lo - pad_lo * span
    return [lo_out, hi + pad_hi * span]


def render_ksweep_scatter_figure(
    score_table, x_criterion, y_criterion, *, mask=None, height: int = 320
) -> go.Figure:
    """One criterion's combined score against another's -- one dot per K.

    The dots are joined in K order so the sweep's trajectory is readable. A grey
    line marks an axis's label-shuffle null where one exists (only reconstruction
    does; spatial's is always 0 and coherence has none, so those axes get no
    line). The lines start hidden: the linked-plot component moves them to the
    hovered K's null (its per-K coordinates travel in ``layout.meta["spatialaim"]["nulls"]``,
    keyed by K, ``null`` where absent) and hides them again on unhover.

    ``mask`` (a boolean array over the table's rows) restricts the plot to a
    subset of K -- the Pareto-optimal ones, say. The colour scale still spans the
    full sweep, so a K keeps its colour whether or not the rest is shown.
    """
    full_k = score_table["k"].to_numpy(dtype=int)
    if mask is not None:
        score_table = score_table[np.asarray(mask, dtype=bool)]
    k = score_table["k"].to_numpy(dtype=int)
    xs = score_table[x_criterion.key].to_numpy(dtype=float)
    ys = score_table[y_criterion.key].to_numpy(dtype=float)
    x_null = score_table[f"{x_criterion.key}{scores.NULL_SUFFIX}"].to_numpy(dtype=float)
    y_null = score_table[f"{y_criterion.key}{scores.NULL_SUFFIX}"].to_numpy(dtype=float)

    # Extra room where the K labels go (above each dot) and to either side, so a
    # dot at the edge of the data still has its label fully inside the card.
    x_range = _score_axis_range(
        xs, x_null, pad_lo=0.10, pad_hi=0.10, from_zero=x_criterion.axis_from_zero
    )
    y_range = _score_axis_range(
        ys, y_null, pad_lo=0.06, pad_hi=0.18, from_zero=y_criterion.axis_from_zero
    )

    fig = go.Figure()
    # The K-ordered path, drawn under the dots.
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="rgba(150,150,150,0.5)", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # Null crosshair: positioned client-side, spanning the (fixed) axis ranges.
    # Kept out of the legend -- the card's caption explains the grey lines, and a
    # legend entry blinking in and out on every hover is more distracting.
    for tag in ("nullV", "nullH"):
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                meta=tag,
                line=dict(color=_NULL_COLOR, width=1.5, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
                visible=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            # [K, x null, y null] -- the nulls pre-formatted so an uncomputed one
            # reads "n/a" in the hover instead of an empty number.
            customdata=[
                [float(kk), _null_text(xn), _null_text(yn)]
                for kk, xn, yn in zip(k, x_null, y_null)
            ],
            mode="markers+text",
            text=[str(v) for v in k],
            textposition="top center",
            textfont=dict(size=9, color="#6b7280"),
            # Draw dot and label whole even when they reach past the axis — the
            # padding above keeps them off the title, and a half-cropped K label is
            # worse than one overhanging the plot area.
            cliponaxis=False,
            name="K",
            showlegend=False,
            marker=dict(
                size=_SCATTER_MARKER_SIZE,
                color=k,
                colorscale="Viridis",
                cmin=float(full_k.min()) if full_k.size else 0.0,
                cmax=float(full_k.max()) if full_k.size else 1.0,
                showscale=False,
                line=dict(width=0, color=_RING_COLOR),
            ),
            hovertemplate=(
                "K %{customdata[0]:.0f}<br>"
                f"{x_criterion.short} %{{x:.3f}}  (null %{{customdata[1]}})<br>"
                f"{y_criterion.short} %{{y:.3f}}  (null %{{customdata[2]}})"
                "<extra></extra>"
            ),
        )
    )

    _base_layout(
        fig,
        height=height,
        font=_CARD_FONT,
        title=dict(
            text=f"{y_criterion.short} vs {x_criterion.short}", font=dict(size=13)
        ),
        xaxis=dict(title=x_criterion.label, range=x_range),
        yaxis=dict(title=y_criterion.label, range=y_range),
        margin=dict(l=10, r=10, t=40, b=52),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
        # Read by widgets.linked_plot (not by Plotly itself).
        meta=dict(
            spatialaim=dict(
                nulls={
                    str(int(kk)): [_finite_or_none(xn), _finite_or_none(yn)]
                    for kk, xn, yn in zip(k, x_null, y_null)
                },
                ranges=dict(x=x_range, y=y_range),
            )
        ),
    )
    return fig


def render_fractions_figure(
    adata_sc: AnnData, root: Path, k: int, hard: np.ndarray
) -> go.Figure:
    """Cell fraction (reference states) and spot fraction (this mapper's
    assignment) per computed state, as one grouped-bar plot sharing a single
    y-axis. Both bars use the same per-state palette as the UMAP/spatial plots;
    the spot bars are hatched to tell them apart."""
    start_cluster_to_state = load_start_cluster_to_state(data_access.k_dir(root, k))
    infer_cell_to_state_cluster(adata_sc, start_cluster_to_state)
    cell_states = adata_sc.obs[OBS_COMPUTED_STATE].astype(int).to_numpy()
    hard = np.asarray(hard).astype(int)

    palette = state_palette(k)
    states = list(range(k))
    x = [_state_label(s, _names_for(root, k)) for s in states]
    colors = [_hex(palette.get(s)) for s in states]
    cell_frac = [float(np.mean(cell_states == s)) for s in states]
    spot_frac = [float(np.mean(hard == s)) for s in states]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Cell (solid)",
            x=x,
            y=cell_frac,
            marker=dict(color=colors),
            hovertemplate="Cell · %{x}<br>%{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Spot (hatched)",
            x=x,
            y=spot_frac,
            marker=dict(
                color=colors,
                pattern=dict(shape="/", size=6, solidity=0.55, fgcolor="white"),
            ),
            hovertemplate="Spot · %{x}<br>%{y:.1%}<extra></extra>",
        )
    )
    _card_layout(
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
        subplot_titles=["Gene-wise", "Spot-wise"],
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
    _card_layout(fig, height=320)
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
    _card_layout(
        fig,
        height=300,
        xaxis_title="max probability per spot (1.0 = one-hot)",
        yaxis_title="spots",
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
    _card_layout(
        fig,
        height=300,
        xaxis_title="assignment confidence per spot",
        yaxis_title="spots",
        bargap=0.02,
    )
    fig.update_xaxes(range=[0.0, 1.0])
    return fig
