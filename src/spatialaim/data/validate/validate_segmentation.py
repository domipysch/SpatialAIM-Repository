"""Validate the optional cell segmentation mask of an ST slice.

Contract (see ``CLAUDE.md``):

* the mask is a 2D ``.tif``/``.tiff`` of dtype ``uint32``; ``0`` is background,
  the cells carry the labels ``1..max`` - every value ``0..max`` occurs
* the ST h5ad carries the integer obs column ``segmentation_id``: the mask label
  of each cell, ``0`` = no segmentation; non-zero ids are unique and occur in the
  mask. Mask labels without a cell in the h5ad are allowed (filtered cells).
* ``uns["segmentation_pixel_size_um"]`` (um per pixel) is optional

The mask is read in bands of rows, so a mask of several GB is never held in
memory at once.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

__all__ = ["validate_segmentation", "SEGMENTATION_ID", "PIXEL_SIZE_KEY"]

SEGMENTATION_ID = "segmentation_id"
PIXEL_SIZE_KEY = "segmentation_pixel_size_um"
_SUFFIXES = (".tif", ".tiff")
_BAND = 1024  # rows per read


def validate_segmentation(
    mask_path: Path,
    adata_st: "ad.AnnData",
    label: str = "st",
) -> tuple[list[str], list[str]]:
    """Return ``(errors, warns)`` for the mask and its link to ``adata_st``."""
    mask_path = Path(mask_path)
    errors: list[str] = []
    warns: list[str] = []

    labels, mask_errs = _mask_labels(mask_path, label)
    errors.extend(mask_errs)

    errs, wrns = _check_segmentation_id(adata_st, labels, label)
    errors.extend(errs)
    warns.extend(wrns)

    errs, wrns = _check_pixel_size(adata_st, label)
    errors.extend(errs)
    warns.extend(wrns)

    return errors, warns


def _mask_labels(mask_path: Path, label: str) -> tuple["np.ndarray | None", list[str]]:
    """The sorted unique labels of the mask, or None if it cannot be checked."""
    import tifffile
    import zarr

    if not mask_path.exists():
        return None, [f"{label}: segmentation mask not found: {mask_path}"]
    if mask_path.suffix.lower() not in _SUFFIXES:
        return None, [
            f"{label}: segmentation mask {mask_path.name} is not a .tif/.tiff"
        ]

    try:
        with tifffile.TiffFile(mask_path) as tif:
            series = tif.series[0]
            if len(series.shape) != 2:
                return None, [
                    f"{label}: segmentation mask has shape {series.shape} - expected 2D"
                ]
            if series.dtype != np.uint32:
                return None, [
                    f"{label}: segmentation mask dtype {series.dtype} - expected uint32"
                ]
            mask = zarr.open_array(series.aszarr(), mode="r")
            labels = np.zeros(0, dtype=np.uint32)
            for row in range(0, series.shape[0], _BAND):
                band = np.asarray(mask[row : row + _BAND])
                labels = np.union1d(labels, pd.unique(band.ravel()))
    except Exception as e:
        return None, [f"{label}: cannot read segmentation mask {mask_path} ({e})"]

    errors: list[str] = []
    if labels.size == 0 or labels[0] != 0:
        errors.append(f"{label}: segmentation mask has no background (label 0)")
    cells = labels[labels > 0]
    if cells.size and cells.size != int(cells[-1]):
        missing = np.setdiff1d(np.arange(1, int(cells[-1]) + 1), cells)
        errors.append(
            f"{label}: segmentation mask labels are not 1..{cells[-1]} without gaps - "
            f"{missing.size} label(s) missing (e.g. {missing[:3].tolist()})"
        )
    return labels, errors


def _check_segmentation_id(
    adata_st: "ad.AnnData",
    labels: "np.ndarray | None",
    label: str,
) -> tuple[list[str], list[str]]:
    """``obs["segmentation_id"]``: integer, >= 0, non-zero ids unique and in the mask."""
    if SEGMENTATION_ID not in adata_st.obs:
        return [
            f"{label}: a segmentation mask is given but obs['{SEGMENTATION_ID}'] is missing"
        ], []
    col = adata_st.obs[SEGMENTATION_ID]
    if not pd.api.types.is_integer_dtype(col):
        return [
            f"{label}: obs['{SEGMENTATION_ID}'] has dtype {col.dtype} - expected integer"
        ], []

    ids = col.to_numpy(dtype=np.int64)
    errors: list[str] = []
    warns: list[str] = []
    if (ids < 0).any():
        errors.append(
            f"{label}: obs['{SEGMENTATION_ID}'] has {(ids < 0).sum()} negative value(s)"
        )

    nonzero = ids[ids > 0]
    uniq, counts = np.unique(nonzero, return_counts=True)
    if (counts > 1).any():
        dups = uniq[counts > 1]
        errors.append(
            f"{label}: {dups.size} non-zero {SEGMENTATION_ID}(s) shared by several "
            f"cells (e.g. {dups[:3].tolist()})"
        )
    if labels is not None:
        absent = uniq[~np.isin(uniq, labels)]
        if absent.size:
            errors.append(
                f"{label}: {absent.size} {SEGMENTATION_ID}(s) do not occur in the "
                f"segmentation mask (e.g. {absent[:3].tolist()})"
            )

    n_zero = int((ids == 0).sum())
    if n_zero:
        warns.append(
            f"{label}: {n_zero} of {ids.size} cells ({n_zero / ids.size:.1%}) have "
            f"{SEGMENTATION_ID} 0 (no segmentation)"
        )
    return errors, warns


def _check_pixel_size(
    adata_st: "ad.AnnData", label: str
) -> tuple[list[str], list[str]]:
    """``uns["segmentation_pixel_size_um"]`` is optional; if set, a positive number."""
    if PIXEL_SIZE_KEY not in adata_st.uns:
        return [], [
            f"{label}: uns['{PIXEL_SIZE_KEY}'] not set - cell sizes are shown in "
            f"pixels only"
        ]
    try:
        value = float(np.asarray(adata_st.uns[PIXEL_SIZE_KEY]).item())
    except Exception:
        return [f"{label}: uns['{PIXEL_SIZE_KEY}'] is not a number"], []
    if not np.isfinite(value) or value <= 0:
        return [
            f"{label}: uns['{PIXEL_SIZE_KEY}']={value} - expected a positive number"
        ], []
    return [], []
