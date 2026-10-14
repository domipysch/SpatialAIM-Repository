"""
Shared figure plumbing: the palettes, the common Plotly layout, cell-type
labels, fixed scatter axes, the violin trace, and the static export.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import plotly.graph_objects as go

from spatialaim.gui import state_names


def names_for(root: Path | None, k: int) -> dict[int, str]:
    """This K's user-set cell-type names, read from the output dir above ``root``.

    A run root is ``<output_dir>/<mapper>/`` and the names are shared by every
    mapper, so they sit one level up. Empty when there is no root (a figure drawn
    without a run, e.g. spatial-only) or nothing has been named.
    """
    return state_names.load(Path(root).parent, k) if root is not None else {}


# Matplotlib's tab20 / tab10 qualitative palettes as hex strings, inlined so the
# GUI carries no matplotlib dependency. tab20 keys the per-state colours (state
# id -> colour); tab10 distinguishes mappers in the Compare tab and the K-sweep
# lines. Values match matplotlib exactly, so figures look identical to before.
TAB20 = [
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

TAB10 = [
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
FONT = dict(family="'Source Sans Pro', 'Segoe UI', sans-serif", size=13)

CARD_FONT = dict(family="'Source Sans Pro', 'Segoe UI', sans-serif", size=12)

# The horizontal state-legend strip shared by the headline / compare / reference
# figures; each sets its own vertical ``y`` offset via ``{**STATE_LEGEND, ...}``.
STATE_LEGEND = dict(
    itemsizing="constant",
    title="Cell types",
    orientation="h",
    yanchor="bottom",
    xanchor="left",
    x=0,
)

# The "every other point" backdrop of the modal's highlight panels: lighter than
# ``scatter._GREY_RGBA`` so even the pale tab20 state colours stand out against it.
BACKDROP_GREY = "#e4e4e4"


def base_layout(
    fig: go.Figure, *, height: int, font: dict = FONT, **extra
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


def card_layout(fig: go.Figure, *, height: int, **extra) -> go.Figure:
    """Apply the shared compact look used by every report-dashboard card."""
    extra.setdefault("margin", dict(l=10, r=10, t=30, b=10))
    return base_layout(fig, height=height, font=CARD_FONT, **extra)


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
    return {s: TAB20[s % 20] for s in range(k)}


def hex_color(color: str | None) -> str:
    """Pass a hex colour through, with a grey fallback for a missing (None) state."""
    return color if color else "#b8b8b8"


def mapper_color(i: int) -> str:
    """Stable qualitative colour for the i-th mapper in a comparison."""
    return TAB10[i % 10]


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


def style_scatter_axes(
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


class SpatialContext(NamedTuple):
    """What the spatial side of the hover / cell-type modal needs: the ST file
    (for its expression), this mapper's per-spot hard assignment at this K, and
    the spot coordinates (``None`` when the ST file has none), plus the per-spot
    confidence the modal's "General" view plots (``None`` = not shown) and its
    axis title."""

    st_path: str
    hard: np.ndarray
    coords: np.ndarray | None
    confidence: np.ndarray | None = None
    confidence_title: str = "Confidence"


# Cells per group handed to the browser for the violin's density estimate; a
# larger group is subsampled (fixed seed), which leaves the shape unchanged.
VIOLIN_MAX_CELLS = 20000


def add_violin(
    fig: go.Figure,
    vals: np.ndarray,
    name: str,
    color: str,
    rng,
    *,
    col: int,
    expressing: bool = True,
) -> None:
    """One violin (box + mean line) of ``vals`` in subplot column ``col``, with
    the group's size and (``expressing``: for expression values) fraction of
    non-zero values under its label."""
    if vals.size == 0:
        return
    sub = f"n = {vals.size:,}"
    if expressing:
        sub += f" · {float((vals > 0).mean()):.0%} expressing"
    shown = (
        rng.choice(vals, VIOLIN_MAX_CELLS, replace=False)
        if vals.size > VIOLIN_MAX_CELLS
        else vals
    )
    fig.add_trace(
        go.Violin(
            y=shown,
            name=f"{name}<br>{sub}",
            line_color=color,
            fillcolor=color,
            opacity=0.75,
            box=dict(
                visible=True,
                width=0.35,
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
