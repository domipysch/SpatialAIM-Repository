"""
The scatter traces the multi-panel figures are built from: one trace per cell
type on a reference UMAP or the spatial map (sharing a ``state<n>`` legendgroup,
so one legend click toggles a type in every panel), the start-cluster UMAP, and
the optional per-type centroid markers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from anndata import AnnData

from spatialaim.adata_schema import (
    OBS_START_CLUSTER,
    OBSM_PCA,
    OBSM_PCA_SHARED_GENES,
    OBSM_UMAP,
    OBSM_UMAP_SHARED_GENES,
    UNS_START_CLUSTER_NAMES,
)
from spatialaim.analysis.loading import load_start_cluster_to_state
from spatialaim.gui import data_access
from spatialaim.gui.render.common import (
    TAB20,
    hex_color,
    style_scatter_axes,
)
from spatialaim.gui.render.markers import TopGenes, top_genes_suffix

from spatialaim.gui.state_names import label as state_label

logger = logging.getLogger(__name__)

# Grey (as a CSS rgba string) for spots drawn below the confidence threshold.
_GREY_RGBA = "rgba(184,184,184,0.9)"

# The spatial panel draws each state in up to _MAX_CHUNKS interleaved chunks (see
# add_spatial_traces), capped at _MAX_STATE_TRACES traces in all: 20 chunks for a
# handful of states is all but a random order, while a K of 50 still gets 4
# without the panel turning into thousands of WebGL traces.
_MAX_CHUNKS = 20
_MAX_STATE_TRACES = 200

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
                    color=hex_color(palette.get(state)),
                    symbol="diamond",
                    line=dict(width=1.5, color="black"),
                    opacity=1.0,
                ),
                name=f"{state_label(state, names)} centroid",
                legendgroup=f"state{state}",
                showlegend=False,
                hovertemplate=f"{state_label(state, names)} centroid<extra></extra>",
            ),
            row=row,
            col=col,
        )


@st.cache_data(show_spinner=False)
def umap_panel_data(
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


def add_umap_traces(
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
    scaffold via the cached ``umap_panel_data`` so ``computed_state`` is correct
    for ``k`` without recomputing on unrelated (e.g. threshold) reruns. Each trace
    joins ``legendgroup`` ``state<n>`` so the interaction layer can toggle the same
    state in every subplot; a state gets a legend entry only the first time it is
    seen (tracked in ``legend_shown``).
    """
    data = umap_panel_data(adata_sc, str(root), int(k), umap_key, SHOW_STATE_CENTROIDS)
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
                marker=dict(
                    size=dot_size, color=hex_color(palette.get(state)), opacity=1.0
                ),
                name=state_label(state, names),
                legendgroup=f"state{state}",
                showlegend=show,
                customdata=start_cluster[mask][:, None],
                hovertemplate=(
                    f"{state_label(state, names)}"
                    "<br>Start cluster %{customdata[0]}"
                    f"{top_genes_suffix(top_genes, state)}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    style_scatter_axes(
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
    and equal aspect only when ``equal_aspect`` (see ``style_scatter_axes``).
    ``xref`` is read from the last-added trace, which belongs to this subplot."""
    style_scatter_axes(
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


def add_spatial_traces(
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
                        (float(c), state_label(s, names))
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
                        (state_label(s, names), float(c))
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

    # Draw order decides which colour wins where spots overlap: one trace per
    # state, stacked by state id, lets the last states cover the others however
    # rare they are. So each state is split into random chunks and the chunks are
    # drawn round-robin (state 0 chunk 0, state 1 chunk 0, ..., state 0 chunk 1,
    # ...), which comes close to a random order spot by spot. Every chunk is still
    # a trace in its state's legendgroup, so the headline widget's toggling,
    # clicks and hover work unchanged; only the first chunk carries the legend
    # entry. The seed is fixed, so a rerun draws the same picture.
    states = sorted(np.unique(hard[keep]).tolist())
    n_chunks = max(1, min(_MAX_CHUNKS, _MAX_STATE_TRACES // max(1, len(states))))
    chunk = np.random.default_rng(0).integers(n_chunks, size=len(hard))
    for c in range(n_chunks):
        for state in states:
            mask = keep & (hard == state) & (chunk == c)
            if not mask.any():
                continue
            show = state not in legend_shown
            legend_shown.add(state)
            genes = top_genes_suffix(top_genes, state)
            head = f"{state_label(state, names)}{genes}"
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
                    marker=dict(
                        size=dot_size, color=hex_color(palette.get(state)), opacity=1.0
                    ),
                    name=state_label(state, names),
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


def start_cluster_names(adata_sc: AnnData) -> list[str]:
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


def add_start_cluster_umap_traces(
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
    names = start_cluster_names(adata_sc)
    start_cluster = adata_sc.obs[OBS_START_CLUSTER].astype(int).to_numpy()
    for i, lc in enumerate(sorted(np.unique(start_cluster).tolist())):
        mask = start_cluster == lc
        s = int(start_cluster_to_state[lc])
        target = state_label(s, state_labels)
        fig.add_trace(
            go.Scattergl(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker=dict(size=dot_size, color=TAB20[i % 20], opacity=1.0),
                name=names[lc],
                legendgroup=f"state{s}",
                showlegend=False,
                hovertemplate=(f"{names[lc]}<br>→ {target}<extra></extra>"),
            ),
            row=row,
            col=col,
        )
    style_scatter_axes(
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
