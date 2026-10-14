"""
Validate one ST slice ``.h5ad`` and, if given, its segmentation mask.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from .common import (
    check_count_matrix,
    check_gene_names_unique,
    check_gene_names_uppercase,
    check_index_counts,
    load_X,
)
from .validate_segmentation import validate_segmentation

__all__ = ["validate_st", "segmentation_path_of"]

SEGMENTATION_DIR_NAME = "Segmentation"


def validate_st(
    h5ad_path: Path,
    index_row: pd.Series | None = None,
    label: str = "st",
    segmentation_path: Path | None = None,
) -> tuple[list[str], list[str], ad.AnnData | None]:
    """``(errors, warns, adata)`` for the slice; ``adata`` is ``None`` if the
    file cannot be read.

    Checks ``obsm["spatial"]``, gene names (uppercase, unique) and that ``X``
    holds raw counts. With an ``ST/index.csv`` row, also its counts. The mask
    is ``segmentation_path``, or else the one the index row names.
    """
    errors: list[str] = []
    warns: list[str] = []

    h5ad_path = Path(h5ad_path)
    if not h5ad_path.exists():
        return [f"{label}: missing file {h5ad_path}"], warns, None

    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as e:
        return [f"{label}: cannot read {h5ad_path} ({e})"], warns, None

    errors.extend(_check_spatial(adata, label))
    errors.extend(check_gene_names_uppercase(adata, label))
    errors.extend(check_gene_names_unique(adata, label))

    errs, wrns = check_count_matrix(load_X(adata), label)
    errors.extend(errs)
    warns.extend(wrns)

    warns.extend(
        check_index_counts(
            index_row,
            label,
            [
                ("SpotCount", "spots", adata.n_obs),
                ("GeneCount", "genes", adata.n_vars),
            ],
        )
    )

    if segmentation_path is None:
        segmentation_path = segmentation_path_of(h5ad_path, index_row)
    if segmentation_path is not None:
        errs, wrns = validate_segmentation(segmentation_path, adata, label)
        errors.extend(errs)
        warns.extend(wrns)

    return errors, warns, adata


def segmentation_path_of(h5ad_path: Path, index_row: pd.Series | None) -> Path | None:
    """The mask named in the index row's ``Segmentation`` column (in
    ``Segmentation/`` next to the h5ad), or ``None``."""
    if index_row is None:
        return None
    name = str(index_row.get("Segmentation", "")).strip()
    if not name:
        return None
    return Path(h5ad_path).parent / SEGMENTATION_DIR_NAME / name


def _check_spatial(adata: ad.AnnData, label: str) -> list[str]:
    """Error unless ``obsm["spatial"]`` exists with shape (spots, 2)."""
    if "spatial" not in adata.obsm:
        return [f"{label}: obsm['spatial'] is missing"]
    spatial = adata.obsm["spatial"]
    if spatial.shape != (adata.n_obs, 2):
        return [f"{label}: obsm['spatial'] shape {spatial.shape} != ({adata.n_obs}, 2)"]
    return []
