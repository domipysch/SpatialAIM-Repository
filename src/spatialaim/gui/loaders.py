"""
The app's cached reads: a K's mapping, the ST coordinates, the per-cell shape
features, the scRNA annotation columns and the reference scaffold. Each is cached
by Streamlit, so a rerun does not read the file again.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import streamlit as st
from anndata import AnnData

from spatialaim.analysis.shape_consistency import SHAPE_FEATURES_CSV
from spatialaim.gui import compute, data_access, scaffold


@st.cache_data(show_spinner=False)
def load_soft(root_str: str, k: int):
    return data_access.load_soft(Path(root_str), k)


@st.cache_data(show_spinner=False)
def _cached_shape_features(root_str: str, _mtime: float) -> pd.DataFrame | None:
    return data_access.load_shape_features(Path(root_str))


def shape_features(root: Path) -> pd.DataFrame | None:
    """Per-cell shape features of a run root; ``None`` without a segmentation.
    Keyed on the file's mtime too, so a sweep rerun in the same session shows."""
    path = root / SHAPE_FEATURES_CSV
    mtime = path.stat().st_mtime if path.exists() else -1.0
    return _cached_shape_features(str(root), mtime)


@st.cache_data(show_spinner=False)
def spatial_coords(st_path_str: str):
    return data_access.load_spatial_coords(Path(st_path_str))


@st.cache_resource(show_spinner=False)
def _scaffold_sc(
    sc_path_str: str,
    st_path_str: str,
    out_str: str,
    resolution: float,
    start_from_annotation: str | None,
):
    return scaffold.load_or_build_sc(
        Path(sc_path_str),
        Path(st_path_str),
        Path(out_str),
        resolution,
        start_from_annotation,
    )


def _resolution(output_dir: Path) -> float:
    return (
        data_access.leiden_resolution_from_config(output_dir)
        or compute.DEFAULT_LEIDEN_RESOLUTION
    )


@st.cache_data(show_spinner=False)
def obs_columns(sc_path_str: str) -> list[str]:
    """Annotation columns offered as start clusters, read cheaply (backed mode)."""
    try:
        return data_access.list_obs_columns(Path(sc_path_str))
    except Exception:  # noqa: BLE001 - an unreadable/absent file just offers nothing
        return []


def load_scaffold(
    args: argparse.Namespace, *, warn: str | None = None
) -> AnnData | None:
    """Return the cached reference scaffold, or ``None`` if it fails to build.

    On failure, surface ``warn`` (with the exception appended) as an
    ``st.warning`` when given, else fail silently — callers that can render
    without the scaffold simply skip it.
    """
    try:
        return _scaffold_sc(
            str(args.scdata),
            str(args.stdata),
            str(args.output_dir),
            _resolution(args.output_dir),
            getattr(args, "start_from_annotation", None),
        )
    except Exception as exc:  # noqa: BLE001
        if warn:
            st.warning(f"{warn}: {exc}")
        return None
