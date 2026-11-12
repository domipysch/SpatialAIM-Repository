"""
Marker genes per computed cell type (Wilcoxon, each type vs. the rest, over the
shared genes) in the reference and, per mapper, among the ST spots assigned to
it; plus the per-gene expression lookups the cell-type modal plots.
"""

from __future__ import annotations

import logging
import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable, Collection
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scanpy as sc
import streamlit as st
from anndata import AnnData

from spatialaim.adata_schema import LAYER_LOGNORM, OBS_START_CLUSTER, UNS_SHARED_GENES
from spatialaim.analysis.loading import load_start_cluster_to_state
from spatialaim.gui import data_access, demo
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

# scanpy's rank_genes_groups fills its result table column by column, which
# makes pandas warn about fragmentation once per tested group, every ranking.
# Harmless and not ours to fix; set once here (filters are process-wide, so
# the worker thread's rankings are covered too).
warnings.filterwarnings(
    "ignore",
    category=pd.errors.PerformanceWarning,
    module=r"scanpy\.tools\._rank_genes_groups",
)


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

    # String indices up front: AnnData would otherwise convert them and warn.
    adata = AnnData(
        X=X.copy(),
        obs=pd.DataFrame(
            {"state": pd.Categorical([str(s) for s in labels])},
            index=[str(i) for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=list(genes)),
    )
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


def shared_genes(adata_sc: AnnData) -> list[str]:
    """The scaffold's shared genes (scRNA-seq ∩ ST), in their stored order."""
    shared = list(adata_sc.uns.get(UNS_SHARED_GENES, []))
    return [g for g in shared if g in adata_sc.var_names]


# -- The marker store ---------------------------------------------------------
#
# Ranking marker genes (Wilcoxon over every cell or spot) is by far the slowest
# step of showing a new K: seconds on a slice with a few 100k spots. So the
# rankings live in a process-wide store fed by one background worker: the plots
# ask without waiting (``top_genes_per_state(wait=False)``), draw "Computing
# marker genes…" in their hover meanwhile, and ``marker_poller`` reruns the app
# once a ranking this session waits for is done. Views that need a ranking to
# draw at all (the dotplots, the cell-type modal) wait for it; one not started
# yet is then computed in the caller's thread right away.
#
# The worker takes the most recently requested ranking first, so sliding through
# K does not queue every K passed on the way in front of the one on show.


def _cap(demo_n: int, local_n: int) -> int:
    """Entry cap of a process-wide store: ``demo_n`` in the online demo,
    ``local_n`` on a local GUI (bounded too, unlike Streamlit's caches there,
    since these hold whole rankings and expression matrices)."""
    return demo.cache_entries(demo_n) or local_n


_cond = threading.Condition()
_results: OrderedDict[tuple, dict | None] = OrderedDict()
_running: set[tuple] = set()
_stack: list[tuple[tuple, Callable[[], dict | None]]] = []
_worker: threading.Thread | None = None
_lognorm: OrderedDict[tuple, object] = OrderedDict()
_lognorm_lock = threading.Lock()
_WAITING = "_markers_waiting"  # session key: rankings this session waits for


def _store(key: tuple, result: dict | None) -> None:
    """Keep ``result`` (call with ``_cond`` held), dropping the oldest beyond
    the cap."""
    _results[key] = result
    _results.move_to_end(key)
    while len(_results) > _cap(64, 512):
        _results.popitem(last=False)


def _run(key: tuple, fn: Callable[[], dict | None]) -> None:
    """Compute one ranking (``key`` already marked running) and publish it."""
    try:
        result = fn()
    except Exception:  # noqa: BLE001 - a failed ranking just shows no genes
        logger.exception("Marker ranking %s failed", key)
        result = None
    with _cond:
        _store(key, result)
        _running.discard(key)
        _cond.notify_all()


def _work() -> None:
    while True:
        with _cond:
            while not _stack:
                _cond.wait()
            key, fn = _stack.pop()
            _running.add(key)
        _run(key, fn)


def _request(key: tuple, fn: Callable[[], dict | None]) -> tuple[bool, dict | None]:
    """``(ready, ranking)`` without waiting; queues the ranking (on top) if it
    is neither done nor computing."""
    global _worker
    with _cond:
        if key in _results:
            _results.move_to_end(key)
            return True, _results[key]
        if key not in _running:
            _stack[:] = [job for job in _stack if job[0] != key]
            _stack.append((key, fn))
            if _worker is None or not _worker.is_alive():
                _worker = threading.Thread(
                    target=_work, name="marker-ranking", daemon=True
                )
                _worker.start()
            _cond.notify_all()
    return False, None


def _get(key: tuple, fn: Callable[[], dict | None]) -> dict | None:
    """The ranking, waiting for it: one computing is awaited, one queued or not
    asked for yet is computed here."""
    with _cond:
        if key in _results:
            return _results[key]
        if key in _running:
            while key not in _results:
                _cond.wait()
            return _results[key]
        _stack[:] = [job for job in _stack if job[0] != key]
        _running.add(key)
    _run(key, fn)
    with _cond:
        return _results.get(key)


def _spatial_lognorm(st_path_str: str, genes: tuple[str, ...]):
    """ST lognorm over ``genes`` (see ``data_access.load_spatial_lognorm``),
    loaded once per ST file and gene set; safe from the worker thread."""
    key = (st_path_str, genes)
    with _lognorm_lock:
        if key not in _lognorm:
            _lognorm[key] = data_access.load_spatial_lognorm(
                Path(st_path_str), list(genes)
            )
            while len(_lognorm) > _cap(2, 4):
                _lognorm.popitem(last=False)
        return _lognorm[key]


_ref_cache: dict[int, tuple[AnnData, tuple[str, ...], object]] = {}


def _ref_lognorm(adata_sc: AnnData, genes: list[str]):
    """The scaffold's lognorm layer over ``genes``, sliced once per scaffold (in
    the calling thread), so asking for a cached ranking stays cheap."""
    hit = _ref_cache.get(id(adata_sc))
    if hit is None or hit[0] is not adata_sc or hit[1] != tuple(genes):
        if len(_ref_cache) >= _cap(4, 8):
            _ref_cache.clear()
        hit = (adata_sc, tuple(genes), adata_sc[:, genes].layers[LAYER_LOGNORM])
        _ref_cache[id(adata_sc)] = hit
    return hit[2]


def _job(
    adata_sc: AnnData, root: Path, k: int, spatial: SpatialContext | None
) -> tuple[tuple, Callable[[], dict | None]]:
    """The store key and the ranking function of one ranking: the reference one
    (per computed cell type, over the shared genes on the all-gene lognorm
    layer), or with ``spatial`` the ST one (per *assigned* cell type, on the ST
    lognorm). Everything the function reads from the scaffold is taken here, in
    the calling thread, so the worker never touches the shared scaffold."""
    genes = shared_genes(adata_sc)
    if spatial is None:
        key = ("ref", str(root), int(k))
        cut = load_start_cluster_to_state(data_access.k_dir(Path(root), k))
        states = cut[adata_sc.obs[OBS_START_CLUSTER].astype(int).to_numpy()]
        X = _ref_lognorm(adata_sc, genes) if len(genes) >= 2 else None

        def fn() -> dict | None:
            if X is None:
                return None
            return _rank_markers(X, states.astype(int), genes)

        return key, fn

    key = ("st", str(root), int(k), spatial.st_path)
    hard = np.asarray(spatial.hard)

    def fn() -> dict | None:
        if len(genes) < 2:
            return None
        X = _spatial_lognorm(spatial.st_path, tuple(genes))
        if X.shape[0] != len(hard):
            logger.warning(
                "Spatial markers skipped: %d spots in %s but %d assignments.",
                X.shape[0],
                spatial.st_path,
                len(hard),
            )
            return None
        return _rank_markers(X, hard, genes)

    return key, fn


def _markers(
    adata_sc: AnnData, root: Path, k: int, spatial: SpatialContext | None
) -> dict | None:
    """The reference ranking, or the spatial one when ``spatial`` is given
    (``None`` when impossible); waits for it."""
    with st.spinner("Ranking marker genes…"):
        return _get(*_job(adata_sc, root, k, spatial))


def _markers_now(
    adata_sc: AnnData, root: Path, k: int, spatial: SpatialContext | None
) -> tuple[bool, dict | None]:
    """``(ready, ranking)`` without waiting. A ranking not ready yet is queued
    and remembered as awaited by this session, so ``marker_poller`` reruns the
    app when it is done."""
    key, fn = _job(adata_sc, root, k, spatial)
    ready, mk = _request(key, fn)
    if not ready:
        try:
            st.session_state.setdefault(_WAITING, set()).add(key)
        except Exception:  # noqa: BLE001 - no session (e.g. a script): no poll
            pass
    return ready, mk


def _awaited_done() -> bool:
    """Whether a ranking this session waits for has finished (then forgotten)."""
    waiting: set = st.session_state.get(_WAITING) or set()
    with _cond:
        done = {key for key in waiting if key in _results}
    waiting -= done
    return bool(done)


@st.fragment(run_every=0.75)
def _poll() -> None:
    if _awaited_done():
        st.rerun(scope="app")


def marker_poller() -> None:
    """Rerun the app as soon as a marker ranking a plot is waiting for is done,
    so its hover gains the genes. Call once per page, after its plots; draws
    nothing and polls only while a ranking is awaited."""
    if st.session_state.get(_WAITING):
        _poll()


def _up_down(mk: dict, state: int, n: int) -> tuple[list[int], list[int]]:
    """Gene indices of ``state``'s ``n`` most up- (positive score, highest first)
    and downregulated (negative score, lowest first) genes in ranking ``mk``."""
    cols = mk["ranked"][state]
    row = mk["score"][mk["states"].index(state)]
    up = [j for j in cols if row[j] > 0][:n]
    down = [j for j in reversed(cols) if row[j] < 0][:n]
    return up, down


# Per state: its top-ranked marker genes for the hover box, keyed "up"/"down"
# (reference) and "up_st"/"down_st" (spatial, mapper tabs only). The entry
# under PENDING lists the rankings still computing ("scRNA-seq", "ST").
TopGenes = dict[int, dict[str, list[str]]]
PENDING = -1


def top_genes_per_state(
    adata_sc: AnnData | None,
    root: Path | None,
    k: int,
    n: int = 3,
    *,
    spatial: SpatialContext | None = None,
    wait: bool = False,
) -> TopGenes:
    """Per state, the ``n`` most up- and downregulated marker genes (Wilcoxon vs.
    rest) in the reference and, with ``spatial``, among the ST spots assigned to
    it: ``{state: {"up", "down"[, "up_st", "down_st"]: [...]}}``.

    Up genes need a positive score, down genes a negative one, most extreme
    first. Empty when the scaffold or the ranking is unavailable — callers then
    simply show no genes; a state too small to test is left out.

    Without ``wait`` a ranking that is not ready yet is queued in the background
    and named under ``PENDING`` instead (see ``marker_poller``).
    """
    if adata_sc is None or root is None:
        return {}
    out: TopGenes = {}
    sources = [("", None, "scRNA-seq")] + (
        [("_st", spatial, "ST")] if spatial is not None else []
    )
    for suffix, ctx, name in sources:
        if wait:
            mk = _markers(adata_sc, root, k, ctx)
        else:
            ready, mk = _markers_now(adata_sc, root, k, ctx)
            if not ready:
                out.setdefault(PENDING, {}).setdefault("pending", []).append(name)
                continue
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
    lines ``↑ scRNA-seq``, ``↑ ST``, ``↓ scRNA-seq``, ``↓ ST``. Empty lists are left out;
    '' when the state is unknown. A ranking still computing adds an italic
    "Computing … marker genes…" line."""
    top_genes = top_genes or {}
    hits = top_genes.get(int(state)) or {}
    pending = (top_genes.get(PENDING) or {}).get("pending", [])
    has_st = any("up_st" in v for v in top_genes.values())
    lines = (
        (
            ("up", "↑ scRNA-seq:"),
            ("up_st", "↑ ST:"),
            ("down", "↓ scRNA-seq:"),
            ("down_st", "↓ ST:"),
        )
        if has_st
        else (("up", "↑"), ("down", "↓"))
    )
    text = "".join(
        f"<br>{label} {', '.join(hits[key])}" for key, label in lines if hits.get(key)
    )
    if pending:
        which = " and ".join(pending)
        text += f"<br><i>Computing {which} marker genes…</i>"
    return text


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


@st.cache_data(show_spinner=False, max_entries=demo.cache_entries(256))
def gene_expression(_adata_sc, root_str: str, gene: str) -> np.ndarray:
    """Per-cell lognorm expression of ``gene`` (all-gene layer), cached per run
    root (one scaffold each) and gene."""
    return to_dense(_adata_sc[:, gene].layers[LAYER_LOGNORM]).ravel()


def spatial_gene_expression(
    adata_sc: AnnData, spatial: SpatialContext, gene: str
) -> np.ndarray | None:
    """Per-spot ST lognorm expression of ``gene``, or ``None`` if it is not a
    shared gene."""
    genes = shared_genes(adata_sc)
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
    adata_sc: AnnData,
    root: Path,
    k: int,
    *,
    n_genes: int = 5,
    spatial: SpatialContext | None = None,
    marker_types: Collection[int] | None = None,
) -> go.Figure | None:
    """Dotplot of each computed cell type's top ``n_genes`` marker genes
    (``rank_genes_groups``, Wilcoxon vs. rest, shared genes), in the style of
    scanpy's ``rank_genes_groups_dotplot(standard_scale="var")``.

    Rows are cell types; columns are each tested type's markers in turn (a gene
    can recur under several types), grouped under a bar in that type's colour.
    Dot area = fraction of the row's cells expressing the gene; dot colour = the
    row's mean lognorm expression, min-max scaled per gene across rows. With
    ``spatial`` the same plot is drawn for the ST spots, grouped by this mapper's
    hard assignment (the ST ranking of ``_job``). ``marker_types`` keeps only
    those types' rows and marker columns; the colour stays scaled across all
    types, so a lone row keeps its contrast. Returns ``None`` when the ranking
    is unavailable (see ``_job``) or no column is left.
    """
    mk = _markers(adata_sc, root, k, spatial)
    if mk is None:
        return None
    obs_word = "cells" if spatial is None else "spots"
    states, genes = mk["states"], mk["genes"]
    names = names_for(root, k)
    palette = state_palette(k)

    # Columns: per tested state (in state order) its top n genes.
    col_genes: list[int] = []
    blocks: list[tuple[int, int, int]] = []  # (state, first col, last col)
    for s in states:
        if marker_types is not None and s not in marker_types:
            continue
        cols = mk["ranked"].get(s, [])[:n_genes]
        if cols:
            blocks.append((s, len(col_genes), len(col_genes) + len(cols) - 1))
            col_genes += cols
    if not col_genes:
        return None
    gcols = np.asarray(col_genes)

    mean = mk["mean"][:, gcols]
    lo, hi = mean.min(axis=0), mean.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    scaled = (mean - lo) / span
    frac = mk["frac"][:, gcols]
    score = mk["score"][:, gcols]
    padj = mk["padj"][:, gcols]

    # Rows: the shown types only (scaled above over all of them).
    keep = np.asarray(
        [marker_types is None or s in marker_types for s in states], dtype=bool
    )
    states = [s for s, kept in zip(states, keep) if kept]
    mean, scaled, frac = mean[keep], scaled[keep], frac[keep]
    score, padj = score[keep], padj[keep]

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
    custom[:, 4] = score.ravel()
    custom[:, 5] = padj.ravel()

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
                f"<br>expressed in %{{customdata[2]:.0%}} of {obs_word}"
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
            title="Computed cell type" if spatial is None else "Assigned cell type",
        ),
        legend=dict(
            title=dict(text=f"Fraction of {obs_word}<br>in group"),
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
