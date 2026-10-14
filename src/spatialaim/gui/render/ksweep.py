"""
The "Choose K" figures: one line card per criterion (its two curves over K plus
their harmonic mean, see ``spatialaim/analysis/compare_ks/scoring.py``) and one
scatter per criterion pair (one dot per K, at its two harmonic means).

All are linked client-side by ``widgets.linked_plot``: every K-indexed trace
carries ``customdata[i][0] = K`` -- that is what the component keys hovering,
highlighting and clicking on; the scatters additionally stash their
null-crosshair coordinates and axis ranges in ``layout.meta["spatialaim"]``.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from spatialaim.analysis import compare_ks as scores
from spatialaim.gui.render.common import CARD_FONT, base_layout, mapper_color

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
    palette = (mapper_color(2 * index), mapper_color(2 * index + 1))

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

    return base_layout(
        fig,
        height=height,
        font=CARD_FONT,
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

    base_layout(
        fig,
        height=height,
        font=CARD_FONT,
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
