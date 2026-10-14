"""
Marker genes per computed cell type (Wilcoxon, each type vs. the rest, over the
shared genes) in the reference and, per mapper, among the ST spots assigned to
it; plus the per-gene expression lookups the cell-type modal plots.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scanpy as sc
import streamlit as st
from anndata import AnnData

from spatialaim.adata_schema import LAYER_LOGNORM, OBS_COMPUTED_STATE, UNS_SHARED_GENES
from spatialaim.analysis.loading import (
    infer_cell_to_state_cluster,
    load_start_cluster_to_state,
)
from spatialaim.gui import data_access
from spatialaim.gui.render.common import (
    SpatialContext,
    base_layout,
    hex_color,
    names_for,
    state_palette,
)
from spatialaim.gui.state_names import label as state_label
from spatialaim.utils import to_dense

logger = logging.getLogger(__name__)


def _rank_markers(X, labels: np.ndarray, genes: list[str]) -> dict | None:
    """Marker genes per label: scanpy ``rank_genes_groups`` (Wilcoxon, each label
    vs. the rest) over the columns of ``X`` (lognorm, observations x ``genes``)
    — the scanpy clustering tutorial's recipe.

    Returns ``{"genes", "states", "ranked", "score", "padj", "mean",
    "frac"}`` or ``None`` when fewer than two testable labels (>= 2 observations
    each) exist. ``ranked`` maps each tested label to gene indices by decreasing
    score; ``score``/``padj`` are (labels x genes), NaN for an untested
    label; ``mean`` (mean lognorm, zeros included) and ``frac`` (fraction > 0)
    feed the dotplot.
    """
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
    adata.uns["log1p"] = {"base": None}  # tells scanpy the data is natural-log lognorm
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
    score, padj = (np.full(shape, np.nan) for _ in range(2))
    ranked: dict[int, list[int]] = {}
    for i, s in enumerate(states):
        if s not in tested:
            continue
        df = sc.get.rank_genes_groups_df(adata, group=str(s))
        cols = df["names"].map(gene_idx).to_numpy()
        score[i, cols] = df["scores"].to_numpy()
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


def top_genes_suffix(top_genes: TopGenes | None, state: int) -> str:
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
    ``gene``/``score``/``padj``/``frac`` (fraction of the state's cells
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
            "padj": float(mk["padj"][i, j]),
            "frac": float(mk["frac"][i, j]),
        }

    return {"up": [row(j) for j in up], "down": [row(j) for j in down]}


@st.cache_data(show_spinner=False)
def gene_expression(_adata_sc, root_str: str, gene: str) -> np.ndarray:
    """Per-cell lognorm expression of ``gene`` (all-gene layer), cached per run
    root (one scaffold each) and gene."""
    return to_dense(_adata_sc[:, gene].layers[LAYER_LOGNORM]).ravel()


def spatial_gene_expression(
    adata_sc: AnnData, spatial: SpatialContext, gene: str
) -> np.ndarray | None:
    """Per-spot ST lognorm expression of ``gene``, or ``None`` if it is not a
    shared gene."""
    genes = _shared_genes(adata_sc)
    if gene not in genes:
        return None
    X = _spatial_lognorm(spatial.st_path, tuple(genes))
    return to_dense(X[:, genes.index(gene)]).ravel()


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
    names = names_for(root, k)
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
        f'<span style="color:{hex_color(palette.get(s))};font-size:22px">●</span>'
        f" {state_label(s, names)}"
        for s in states
    ]
    n_rows, n_cols = len(states), len(col_genes)
    xs = np.tile(np.arange(n_cols), n_rows)
    ys = np.repeat(np.arange(n_rows), n_cols)
    # Per-dot hover fields; object dtype keeps the labels as strings.
    custom = np.empty((n_rows * n_cols, 6), dtype=object)
    custom[:, 0] = np.repeat([state_label(s, names) for s in states], n_cols)
    custom[:, 1] = np.tile([genes[j] for j in col_genes], n_rows)
    custom[:, 2] = frac.ravel()
    custom[:, 3] = mean.ravel()
    custom[:, 4] = mk["score"][:, gcols].ravel()
    custom[:, 5] = mk["padj"][:, gcols].ravel()

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
                "<br>score %{customdata[4]:.1f} · p_adj %{customdata[5]:.1e}"
                "<extra></extra>"
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
            line=dict(color=hex_color(palette.get(s)), width=5),
        )

    margins = _DOTPLOT_MARGIN["t"] + _DOTPLOT_MARGIN["b"]
    height = max(n_rows * 30 + 190, margins + _COLORBAR_PX + 30 + _SIZE_LEGEND_PX)
    plot_px = height - _DOTPLOT_MARGIN["t"] - _DOTPLOT_MARGIN["b"]
    base_layout(
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
