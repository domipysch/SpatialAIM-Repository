"""
Pure on-disk readers for SpatialAIM sweep outputs.

Everything here is cheap and side-effect-free (reads only); ``loaders`` wraps
the expensive ones with ``st.cache_data``.

A sweep writes one *pair output* (the sidebar's output directory) holding one
*start folder* per start clusters (``leiden/``, ``annotation_<obs column>/``, see
``list_starts``). A start folder holds one *run root* per mapper
(``<start folder>/<mapper>/``) with ``config.yaml``, ``k_comparison.csv`` and
one ``k_<kkk>/`` folder per swept K, plus the ``start_clustering.h5ad`` its
mappers share. The result tabs show one start folder at a time, so their
``output_dir`` is a start folder. Outputs written before the start folders
existed hold the run roots directly in the pair output, which then serves as
its own single start folder.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import yaml

# MAPPING_CHOICES is the authoritative list of valid mapper names.
from spatialaim import MAPPING_CHOICES
from spatialaim.analysis.reconstruction import (
    COSSIM_BOXPLOTS_JSON_LEGACY,
    COSSIM_METRICS_JSON,
)
from spatialaim.analysis.shape_consistency import SHAPE_FEATURES_CSV
from spatialaim.io import (
    ANNOTATION_START_PREFIX,
    LEIDEN_START_FOLDER,
    START_CLUSTERING_FILENAME,
    start_folder_name,
)
from spatialaim.utils import box_stats

_K_DIR_RE = re.compile(r"^k_(\d{3})$")


def run_root(output_dir: Path, mapper: str) -> Path:
    """The run-root folder for one mapper under a start folder."""
    return Path(output_dir) / mapper


class Start(NamedTuple):
    """One start clusters' results in a pair output: its folder and the
    annotation it starts from (``None`` = the Leiden over-clustering)."""

    folder: Path
    annotation: str | None

    @property
    def label(self) -> str:
        return start_label(self.annotation)


def start_label(annotation: str | None) -> str:
    """Display name of a start: ``Leiden`` or the annotation's obs column."""
    return "Leiden" if annotation is None else str(annotation)


def is_legacy_layout(pair_dir: Path) -> bool:
    """Whether ``pair_dir`` holds run roots directly (an output written before the
    start folders existed)."""
    return bool(list_mappers(pair_dir))


def list_starts(pair_dir: Path, planned: list[str | None] | None = None) -> list[Start]:
    """The starts of a pair output, the Leiden start first, then the annotations
    in name order.

    A start counts once its folder holds a run root; ``planned`` adds the starts a
    queued or running sweep is about to write (``None`` = Leiden), so their tabs can
    show progress before the folder exists. A legacy output is its own single
    start, read from its configs.
    """
    pair_dir = Path(pair_dir)
    if is_legacy_layout(pair_dir):
        return [Start(pair_dir, start_from_annotation_from_config(pair_dir))]

    found: dict[Path, str | None] = {}
    if pair_dir.is_dir():
        for folder in sorted(p for p in pair_dir.iterdir() if p.is_dir()):
            if folder.name != LEIDEN_START_FOLDER and not folder.name.startswith(
                ANNOTATION_START_PREFIX
            ):
                continue
            if not list_mappers(folder):
                continue
            found[folder] = start_from_annotation_from_config(folder)
    for annotation in planned or []:
        found.setdefault(pair_dir / start_folder_name(annotation), annotation)
    starts = [Start(folder, annotation) for folder, annotation in found.items()]
    return sorted(starts, key=lambda s: (s.annotation is not None, s.label.lower()))


def list_pair_mappers(pair_dir: Path) -> list[str]:
    """Mappers with a run root in any start of a pair output (MAPPING_CHOICES
    order)."""
    present = {m for s in list_starts(pair_dir) for m in list_mappers(s.folder)}
    return [m for m in MAPPING_CHOICES if m in present]


def is_run_root(path: Path) -> bool:
    """A folder counts as a finished/started run root if it has a config or any K folder."""
    path = Path(path)
    if (path / "config.yaml").exists():
        return True
    return any(_K_DIR_RE.match(p.name) for p in path.glob("k_*") if p.is_dir())


def list_mappers(output_dir: Path) -> list[str]:
    """Mapper names (in MAPPING_CHOICES order) that already have a run root on disk."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    present = {p.name for p in output_dir.iterdir() if p.is_dir() and is_run_root(p)}
    return [m for m in MAPPING_CHOICES if m in present]


def k_dir(root: Path, k: int) -> Path:
    return Path(root) / f"k_{k:03d}"


def list_ks(root: Path) -> list[int]:
    """Sorted (ascending) list of K values that have a folder under ``root``."""
    root = Path(root)
    if not root.exists():
        return []
    ks: list[int] = []
    for p in root.glob("k_*"):
        m = _K_DIR_RE.match(p.name)
        if p.is_dir() and m:
            ks.append(int(m.group(1)))
    return sorted(ks)


def data_dir(root: Path, k: int) -> Path:
    """One K's analysis metrics: ``k_<kkk>/analysis/``, or ``analysis/data/``
    in outputs written before that subfolder was dropped."""
    new = k_dir(root, k) / "analysis"
    old = new / "data"
    return old if old.is_dir() else new


def load_soft(root: Path, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return ``(P, hard, confidence)`` for one K.

    ``P`` is the S x K soft matrix, ``hard`` its per-spot argmax, and
    ``confidence`` the per-spot value in [0, 1] or ``None`` when the mapper wrote
    none (tangram / tacco / dot).
    """
    path = k_dir(root, k) / "spot_to_state_mapping_soft.h5ad"
    if not path.exists():
        raise FileNotFoundError(path)
    adata = ad.read_h5ad(path)
    P = np.asarray(adata.X, dtype=np.float64)
    hard = P.argmax(axis=1)
    confidence = None
    if "mapping_confidence" in adata.obs:
        confidence = adata.obs["mapping_confidence"].to_numpy(dtype=np.float64)
    return P, hard, confidence


def load_data_json(root: Path, k: int, name: str) -> dict | None:
    path = data_dir(root, k) / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def shape_features_path(root: Path) -> Path | None:
    """The ST slice's ``cell_shape_features.csv`` for a run root: in the pair
    output, two levels up, or in the run root itself in older outputs. ``None``
    when the sweep ran without a segmentation mask."""
    for folder in (Path(root).parent.parent, Path(root)):
        path = folder / SHAPE_FEATURES_CSV
        if path.exists():
            return path
    return None


def load_shape_features(root: Path) -> pd.DataFrame | None:
    """Per-cell shape features of the ST slice (``shape_features_path``; in spot
    order: ``area_px``, ``area_um2``, ``eccentricity``, ``solidity``), or ``None``
    when the sweep ran without a segmentation mask."""
    path = shape_features_path(root)
    return None if path is None else pd.read_csv(path, index_col=0)


def load_cossim_boxplots(root: Path, k: int) -> dict[str, dict[str, dict]]:
    """Per-combo box-plot statistics of the reconstruction cosine similarity
    for one K: ``{label: {"per_gene": stats, "per_spot": stats}}`` for the
    soft/hard x raw/norm combos (``stats`` as ``spatialaim.utils.box_stats``;
    empty dict if none). Hard-raw also carries its ``"null"`` medians.

    Read from ``cossim_metrics.json`` (``cossim_boxplots.json`` in older
    outputs); the oldest stored every value under ``cossim/``, and the
    statistics are computed from those instead.
    """
    for name in (COSSIM_METRICS_JSON, COSSIM_BOXPLOTS_JSON_LEGACY):
        path = data_dir(root, k) / name
        if path.exists():
            with open(path) as f:
                return json.load(f)

    cdir = data_dir(root, k) / "cossim"
    out: dict[str, dict[str, dict]] = {}
    for label in ("soft-raw", "hard-raw", "soft-norm", "hard-norm"):
        gene_path = cdir / f"cossim-per-gene-{label}.json"
        spot_path = cdir / f"cossim-per-spot-{label}.json"
        if not (gene_path.exists() and spot_path.exists()):
            continue
        with open(gene_path) as f:
            per_gene = list(json.load(f)["values"].values())
        with open(spot_path) as f:
            per_spot = list(json.load(f)["values"].values())
        out[label] = {"per_gene": box_stats(per_gene), "per_spot": box_stats(per_spot)}
    return out


def ksweep_table(root: Path) -> pd.DataFrame | None:
    """The run root's ``k_comparison.csv``, or ``None`` if it does not exist."""
    path = Path(root) / "k_comparison.csv"
    return pd.read_csv(path) if path.exists() else None


def list_obs_columns(sc_path: Path) -> list[str]:
    """The ``obs`` columns of an h5ad that can serve as start clusters: every
    column that labels each cell and is not a float (a continuous value, not a
    cell type). Read in backed mode -- cheap even for a multi-GB reference."""
    adata = ad.read_h5ad(sc_path, backed="r")
    try:
        obs = adata.obs
        return [
            str(c)
            for c in obs.columns
            if not pd.api.types.is_float_dtype(obs[c]) and not obs[c].isna().any()
        ]
    finally:
        if adata.isbacked:
            adata.file.close()


def load_obs_column(sc_path: Path, column: str) -> pd.Series:
    """One ``obs`` column of an h5ad as strings, indexed by cell (backed mode)."""
    adata = ad.read_h5ad(sc_path, backed="r")
    try:
        return adata.obs[column].astype(str).copy()
    finally:
        if adata.isbacked:
            adata.file.close()


def load_spatial_coords(st_path: Path) -> np.ndarray | None:
    """Spatial coordinates (n_spots x 2) from the ST h5ad, or None if absent."""
    adata = ad.read_h5ad(st_path, backed="r")
    try:
        if "spatial" not in adata.obsm:
            return None
        return np.asarray(adata.obsm["spatial"])
    finally:
        if adata.isbacked:
            adata.file.close()


def load_spatial_lognorm(st_path: Path, genes: list[str]):
    """ST expression over ``genes`` (in that order) as a CSR float32 matrix
    (n_spots x len(genes)), normalised the way the sweep builds the all-gene
    lognorm layer: ``normalize_total(1e4)`` over all genes, then ``log1p``."""
    adata = ad.read_h5ad(st_path)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return sp.csr_matrix(adata[:, genes].X, dtype=np.float32)


def start_clustering_path(root: Path) -> Path | None:
    """The ``start_clustering.h5ad`` of the run root's start (in the start folder,
    one level up, or in the run root itself in older outputs), or ``None`` if
    there is none."""
    for folder in (Path(root).parent, Path(root)):
        path = folder / START_CLUSTERING_FILENAME
        if path.exists():
            return path
    return None


def load_start_cluster_labels(root: Path) -> np.ndarray | None:
    """Each reference cell's start cluster (the index used in
    ``start_cluster_to_state.csv``), or ``None`` without ``start_clustering.h5ad``."""
    path = start_clustering_path(root)
    if path is None:
        return None
    return ad.read_h5ad(path).obs["start_cluster"].to_numpy(dtype=int)


def load_start_cluster_names(root: Path) -> list[str] | None:
    """The name of every start cluster (by index), or ``None`` without
    ``start_clustering.h5ad``."""
    path = start_clustering_path(root)
    if path is None:
        return None
    obs = ad.read_h5ad(path).obs
    labels = obs["start_cluster"].to_numpy(dtype=int)
    names = obs["start_cluster_name"].astype(str).to_numpy()
    out = [""] * (int(labels.max()) + 1)
    for label, name in zip(labels, names):
        out[label] = name
    return out


def _config_value(output_dir: Path, field: str):
    """Best-effort read of ``field`` from any mapper's config.yaml under the output
    dir. Returns ``None`` if no config records it (configs are advisory only)."""
    for cfg_path in Path(output_dir).glob("*/config.yaml"):
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict) and cfg.get(field) is not None:
                return cfg[field]
        except Exception:  # noqa: BLE001 - config is advisory only
            continue
    return None


def leiden_resolution_from_config(output_dir: Path) -> float | None:
    """Best-effort read of the leiden resolution any mapper's config.yaml recorded."""
    value = _config_value(output_dir, "leiden_resolution")
    return float(value) if value is not None else None


def start_from_annotation_from_config(output_dir: Path) -> str | None:
    """Best-effort read of the start-cluster annotation column any mapper's
    config.yaml recorded, so re-opening an output dir rebuilds its scaffold the way
    the sweep built it. ``None`` means Leiden over-clustering."""
    value = _config_value(output_dir, "start_from_annotation")
    return str(value) if value is not None else None


def linkage_method_from_config(output_dir: Path) -> str | None:
    """Best-effort read of the agglomeration linkage any mapper's config.yaml
    recorded. ``None`` when no config records it (an older run root). Falls back to
    ``agglo_tree_method``, the key's name before it was renamed."""
    value = _config_value(output_dir, "linkage_method")
    if value is None:
        value = _config_value(output_dir, "agglo_tree_method")
    return str(value) if value is not None else None
