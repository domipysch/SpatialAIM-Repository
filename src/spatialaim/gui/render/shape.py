"""
Cell shape features from the segmentation mask (area, eccentricity, solidity):
per mapped cell type for the report card, and one type vs. all cells for the
cell-type modal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from spatialaim.gui.render.common import (
    VIOLIN_MAX_CELLS,
    add_violin,
    base_layout,
    card_layout,
    hex_color,
    names_for,
    state_palette,
)
from spatialaim.gui.state_names import label as state_label


def _shape_panels(
    features: pd.DataFrame, n_spots: int
) -> tuple[tuple[tuple[np.ndarray, str], ...], np.ndarray]:
    """The three shape panels as ``(values, axis title)`` -- log10 area (um^2
    when the mask's pixel size is known, else pixels), eccentricity, solidity --
    and the raw area. ``features`` is ``data_access.load_shape_features`` (spot
    order)."""
    if len(features) != n_spots:
        raise ValueError(
            f"cell_shape_features.csv has {len(features)} cells, the mapping {n_spots}"
        )
    use_um = bool(features["area_um2"].notna().any())
    area = features["area_um2" if use_um else "area_px"].to_numpy(dtype=float)
    panels = (
        (np.log10(area), f"area [{'µm²' if use_um else 'px'}]"),
        (features["eccentricity"].to_numpy(dtype=float), "eccentricity"),
        (features["solidity"].to_numpy(dtype=float), "solidity"),
    )
    return panels, area


def _log_area_ticks(fig: go.Figure, area: np.ndarray, *, row: int, col: int) -> None:
    """Area is plotted as log10 so the densities are estimated on the log scale;
    label that axis's ticks (1-2-5 steps) with the area itself."""
    finite = np.log10(area[np.isfinite(area) & (area > 0)])
    if not finite.size:
        return
    lo, hi = np.floor(finite.min()), np.ceil(finite.max())
    ticks = [
        t
        for e in np.arange(lo, hi + 1)
        for t in (e, e + np.log10(2), e + np.log10(5))
        if finite.min() - 0.2 <= t <= finite.max() + 0.2
    ]
    fig.update_yaxes(
        tickvals=ticks, ticktext=[f"{10 ** t:,.0f}" for t in ticks], row=row, col=col
    )


def render_shape_features_figure(
    root: Path, k: int, hard: np.ndarray, features: pd.DataFrame
) -> go.Figure:
    """Per-cell shape features by mapped cell type: one panel each for area
    (log scale), eccentricity and solidity, stacked over a shared x-axis of the
    cell types. Each cell type with assigned, segmented cells gets a violin with
    a box inside, in the per-state palette."""
    hard = np.asarray(hard).astype(int)
    panels, area = _shape_panels(features, len(hard))
    names = names_for(root, k)
    palette = state_palette(k)
    states = [s for s in range(k) if np.any((hard == s) & np.isfinite(area))]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    rng = np.random.default_rng(0)
    for row, (vals, title) in enumerate(panels, start=1):
        for s in states:
            v = vals[(hard == s) & np.isfinite(vals)]
            if v.size == 0:
                continue
            if v.size > VIOLIN_MAX_CELLS:
                v = rng.choice(v, VIOLIN_MAX_CELLS, replace=False)
            color = hex_color(palette.get(s))
            label = state_label(s, names)
            fig.add_trace(
                go.Violin(
                    x=np.full(v.size, label),
                    y=v,
                    name=label,
                    line_color=color,
                    fillcolor=color,
                    opacity=0.75,
                    box=dict(
                        visible=True,
                        width=0.15,
                        fillcolor="white",
                        line=dict(color="#333333", width=1),
                    ),
                    points=False,
                    spanmode="hard",
                    scalemode="width",
                    showlegend=False,
                    hoverinfo="name+y",
                ),
                row=row,
                col=1,
            )
        fig.update_yaxes(title_text=title, row=row, col=1)
    _log_area_ticks(fig, area, row=1, col=1)
    fig.update_xaxes(tickangle=-30, row=3, col=1)
    # Narrower gaps than Plotly's default 0.3, so each violin gets more width.
    return card_layout(fig, height=640, violinmode="overlay", violingap=0.1)


# What the modal's axis titles add to the feature names.
_MODAL_AXIS_NOTES = {
    "eccentricity": " (elongation)",
    "solidity": " (area / area of convex hull)",
}


def render_state_shape_figure(
    root: Path, k: int, state: int, hard: np.ndarray, features: pd.DataFrame
) -> go.Figure:
    """Violins of area (log scale), eccentricity and solidity among the
    segmented spots assigned to ``state`` vs. among all segmented spots, side
    by side. The axis titles say what eccentricity and solidity measure."""
    hard = np.asarray(hard).astype(int)
    panels, area = _shape_panels(features, len(hard))
    label = state_label(state, names_for(root, k))
    color = hex_color(state_palette(k).get(int(state)))
    rng = np.random.default_rng(0)
    on = hard == int(state)
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08)
    for col, (vals, title) in enumerate(panels, start=1):
        ok = np.isfinite(vals)
        add_violin(fig, vals[on & ok], label, color, rng, col=col, expressing=False)
        add_violin(
            fig, vals[ok], "All cells", "#9e9e9e", rng, col=col, expressing=False
        )
        fig.update_yaxes(
            title_text=title + _MODAL_AXIS_NOTES.get(title, ""), row=1, col=col
        )
    _log_area_ticks(fig, area, row=1, col=1)
    return base_layout(fig, height=360, margin=dict(l=10, r=10, t=20, b=10))
