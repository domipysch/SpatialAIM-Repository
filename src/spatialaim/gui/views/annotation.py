"""
The "Show annotation" option of the mapper and reference views: a toggle with an
annotation picker, and a K-independent row of plots showing that reference
annotation on the UMAP(s) and, for a mapper, its annotation start's spatial
mapping.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import streamlit as st
from anndata import AnnData

from spatialaim.adata_schema import OBS_START_CLUSTER
from spatialaim.analysis.loading import load_start_cluster_to_state
from spatialaim.gui import compute, data_access
from spatialaim.gui.loaders import annotation_labels, load_soft, obs_columns
from spatialaim.gui.render.common import EXTRA_COLORS, TAB20, state_palette
from spatialaim.gui.render.headline import render_annotation_figure
from spatialaim.gui.views.controls import toggle_button
from spatialaim.io import start_folder_name


def annotation_picker(
    args: argparse.Namespace, key_prefix: str, *, help: str
) -> str | None:
    """The "Show annotation" toggle and, while on, the annotation dropdown.
    Returns the chosen ``obs`` column, or ``None`` (off, or no annotation)."""
    annotations = obs_columns(str(args.scdata))
    if not annotations:
        return None
    if not toggle_button("Show annotation", f"{key_prefix}_show_ann", help=help):
        return None
    return st.selectbox(
        "Annotation",
        annotations,
        key=f"{key_prefix}_ann_col",
        label_visibility="collapsed",
        width=220,
    )


def _annotation_mapping(
    pair_dir: Path, mapper: str, column: str, types: list[str], n_spots: int
) -> np.ndarray | None:
    """Each spot's annotated type as ``mapper`` mapped it when starting from
    annotation ``column`` (its top K, where the tree cut is the identity), or
    ``None`` when that start has not been computed. ``types`` are the annotated
    types in sorted order, i.e. the annotation start's start-cluster order."""
    if not compute.start_complete(pair_dir, mapper, column):
        return None
    root = data_access.run_root(Path(pair_dir) / start_folder_name(column), mapper)
    ks = data_access.list_ks(root)
    if not ks:
        return None
    t = max(ks)
    _P, hard, _conf = load_soft(str(root), t)
    if len(hard) != n_spots:
        return None
    # Start cluster c is the c-th type in sorted order; invert the cut.
    state_to_type = {
        int(s): types[c]
        for c, s in enumerate(load_start_cluster_to_state(data_access.k_dir(root, t)))
        if c < len(types)
    }
    return np.array([state_to_type.get(int(h), "unmapped") for h in hard], dtype=object)


def _dominant_state_colors(
    adata_sc: AnnData, labels: np.ndarray, state_root: Path, k: int
) -> dict[str, str] | None:
    """Each annotated type's colour: that of the computed cell type at ``k``
    (run root ``state_root``) holding most of its cells, so the annotation row
    reads against the plots above. When several types share a dominant cell
    type (small K), only the one with most cells in it keeps that colour; the
    others get fresh colours that no cell type at ``k`` uses. ``None`` if this
    K's tree cut is unreadable."""
    try:
        cut = load_start_cluster_to_state(data_access.k_dir(state_root, k))
    except FileNotFoundError:
        return None
    states = cut[adata_sc.obs[OBS_START_CLUSTER].astype(int).to_numpy()].astype(int)
    palette = state_palette(k)

    # (type, dominant state, its cell count there), largest count first.
    dominant: list[tuple[str, int, int]] = []
    for t in np.unique(labels):
        ids, counts = np.unique(states[labels == t], return_counts=True)
        i = int(np.argmax(counts))
        dominant.append((str(t), int(ids[i]), int(counts[i])))
    dominant.sort(key=lambda d: (-d[2], d[0]))

    used = {c.lower() for c in palette.values()}
    spare = iter(
        dict.fromkeys(c for c in TAB20 + EXTRA_COLORS if c.lower() not in used)
    )
    colors: dict[str, str] = {}
    claimed: set[int] = set()
    for t, state, _n in dominant:
        if state not in claimed:
            claimed.add(state)
            colors[t] = palette.get(state)
        else:
            colors[t] = next(spare, None)  # None -> grey once all are spent
    return colors


def annotation_plots(
    args: argparse.Namespace,
    adata_sc: AnnData,
    column: str,
    key_prefix: str,
    *,
    mapper: str | None = None,
    coords: np.ndarray | None = None,
    show_shared_umap: bool = False,
    reference_layout: bool = False,
    state_root: Path | None = None,
    k: int | None = None,
) -> None:
    """The annotation row: ``column`` on the UMAP(s), plus (given ``mapper`` and
    ``coords``) the spatial map of that mapper's annotation start if it has been
    computed. The plotted labels do not depend on K; with ``state_root`` and
    ``k`` each type takes the colour of its dominant computed cell type there."""
    labels = annotation_labels(str(args.scdata), column)
    per_cell = labels.reindex(adata_sc.obs_names)
    if per_cell.isna().any():
        st.info(f"Annotation '{column}' does not cover the reference scaffold's cells.")
        return
    types = sorted(set(per_cell.tolist()))
    cell_labels = per_cell.to_numpy(dtype=object)
    colors = (
        _dominant_state_colors(adata_sc, cell_labels, state_root, k)
        if state_root is not None and k is not None
        else None
    )
    st_labels = (
        _annotation_mapping(args.pair_dir, mapper, column, types, len(coords))
        if mapper is not None and coords is not None
        else None
    )
    fig = render_annotation_figure(
        adata_sc,
        cell_labels,
        column,
        coords=coords if st_labels is not None else None,
        st_labels=st_labels,
        show_shared_umap=show_shared_umap,
        reference_layout=reference_layout,
        colors=colors,
    )
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_ann_plot")
