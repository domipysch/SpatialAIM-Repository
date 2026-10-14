"""
Shared helpers for the curation scripts that ship a cell segmentation mask with
their ST slice (02, 03, 04, 05, 09).

The mask follows the contract in the repo ``CLAUDE.md``: a 2D uint32 ``.tif`` in
``ST/Segmentation/<st_name>.tif``, 0 = background, cells labelled 1..max without
gaps; the ST h5ad gets ``obs["segmentation_id"]`` (0 = no segmentation) and, if
the pixel size is known, ``uns["segmentation_pixel_size_um"]``.

Where the source ships cell outlines as polygons (02, 03, 04, 05), they are
rasterised here: each cell of the h5ad that has an outline gets the next label
in obs order. A cell left without a single pixel (no outline in the source, or
entirely covered by later cells) gets ``segmentation_id`` 0 and the labels are
compacted, so 1..max stays gap-free. Cells of the source that are not in the
h5ad (filtered out at curation) are not drawn, i.e. they are background.
"""

from __future__ import annotations

import io
import logging
import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import anndata as ad
import numpy as np

logger = logging.getLogger(__name__)

SEGMENTATION_DIR = "Segmentation"
SEGMENTATION_ID = "segmentation_id"
PIXEL_SIZE_KEY = "segmentation_pixel_size_um"
TILE = 1024
_USER_AGENT = "spatialaim-curate/1.0 (academic dataset reproduction)"


def mask_path(st_dir: Path, st_name: str) -> Path:
    """Where the mask of ``st_name`` lives: ``ST/Segmentation/<st_name>.tif``."""
    return st_dir / SEGMENTATION_DIR / f"{st_name}.tif"


# --------------------------------------------------------------------------- #
# Rasterising polygon outlines
# --------------------------------------------------------------------------- #
def rasterize(
    outlines: Sequence[list[np.ndarray] | None],
    step: float,
    margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one outline per h5ad cell into a fresh label mask.

    ``outlines[i]`` belongs to obs row i: a list of rings (N x 2, x/y in source
    units) whose first ring is the outer boundary and any further rings are
    holes, or None for a cell without an outline. ``step`` is the pixel size in
    source units; the grid covers all outlines plus ``margin``.

    Returns ``(mask, segmentation_id)``, the latter aligned with obs.
    """
    from skimage.draw import polygon as draw_polygon

    rings_of = {i: rings for i, rings in enumerate(outlines) if rings is not None}
    drawn = list(rings_of)
    if not drawn:
        raise ValueError("no cell has an outline")
    lo = np.min([rings_of[i][0].min(0) for i in drawn], axis=0) - margin
    hi = np.max([rings_of[i][0].max(0) for i in drawn], axis=0) + margin
    x0, y0 = np.floor(lo / step) * step
    shape = (int(np.ceil((hi[1] - y0) / step)), int(np.ceil((hi[0] - x0) / step)))
    logger.info("  mask    %d x %d px", *shape)

    def _pixels(ring: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # pixel (row, col) has its centre at (x0 + (col + .5) step, y0 + (row + .5) step)
        return draw_polygon(
            (ring[:, 1] - y0) / step - 0.5, (ring[:, 0] - x0) / step - 0.5, shape=shape
        )

    mask = np.zeros(shape, dtype=np.uint32)
    label = np.zeros(len(outlines), dtype=np.int64)
    label[drawn] = np.arange(1, len(drawn) + 1)
    for i in drawn:  # later cells overwrite earlier ones where outlines overlap
        rr, cc = _pixels(rings_of[i][0])
        mask[rr, cc] = label[i]
    for i in drawn:  # holes are cut out of their own cell only
        for hole in rings_of[i][1:]:
            rr, cc = _pixels(hole)
            own = mask[rr, cc] == label[i]
            mask[rr[own], cc[own]] = 0
    return _compact(mask, label)


def _compact(mask: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Renumber the labels that kept pixels to 1..max; the others become 0."""
    present = np.bincount(mask.ravel(), minlength=int(label.max()) + 1) > 0
    present[0] = True
    lut = (np.cumsum(present) - 1).astype(np.uint32)
    n_lost = int((~present[1:]).sum())
    if n_lost:
        logger.info("  %d cell(s) without a single pixel -> segmentation_id 0", n_lost)
        mask = lut[mask]
    seg_id = np.where(present[label] & (label > 0), lut[label], 0).astype(np.int64)
    without = int((seg_id == 0).sum())
    logger.info(
        "  cells   %d with a mask label, %d without", len(seg_id) - without, without
    )
    return mask, seg_id


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def write_mask(
    path: Path,
    mask: np.ndarray | None = None,
    *,
    tiles: Iterator[np.ndarray] | None = None,
    shape: tuple[int, int] | None = None,
) -> None:
    """Tiled, zlib-compressed BigTIFF via a .part file. Either a full ``mask``,
    or ``tiles`` (TILE x TILE uint32 tiles in row-major order) plus ``shape``."""
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tif.part")
    if tiles is not None:
        tifffile.imwrite(
            tmp,
            tiles,
            shape=shape,
            dtype=np.uint32,
            bigtiff=True,
            tile=(TILE, TILE),
            compression="zlib",
        )
    else:
        tifffile.imwrite(
            tmp,
            np.ascontiguousarray(mask, dtype=np.uint32),
            bigtiff=True,
            tile=(TILE, TILE),
            compression="zlib",
        )
    shutil.move(str(tmp), str(path))
    logger.info("  wrote   %s (%.1f MB)", path.name, path.stat().st_size / 1e6)


def annotate(
    adata: ad.AnnData, seg_id: Iterable[int], pixel_size_um: float | None
) -> None:
    """Add ``obs["segmentation_id"]`` and, if known, the mask's pixel size."""
    adata.obs[SEGMENTATION_ID] = np.asarray(seg_id, dtype=np.int64)
    if pixel_size_um is not None:
        adata.uns[PIXEL_SIZE_KEY] = float(pixel_size_um)


# --------------------------------------------------------------------------- #
# Vizgen run exports (03, 04: Liu et al. 2023 on figshare)
# --------------------------------------------------------------------------- #
VIZGEN_EXPORT_MEMBERS = [
    "cell_boundaries.geojson",
    "cell_metadata.csv",
    "micron_to_mosaic_pixel_transform.csv",
]


def vizgen_export_outlines(
    url: str, dest_dir: Path, obs
) -> tuple[list[list[np.ndarray]], float]:
    """Cell outlines from a Vizgen run export zip, for the cells of ``obs``.

    The export's ``cell_boundaries.geojson`` holds one Polygon per cell without
    an id: feature i is row i of its ``cell_metadata.csv``, whose row number is
    the obs_name. Outlines, like ``obsm["spatial"]`` and obs min/max_x/y, are in
    mosaic pixels; each polygon's bounding box is checked against obs
    min/max_x/y. Returns the outlines (outer ring + holes, in obs order) and the
    mosaic pixels per um from ``micron_to_mosaic_pixel_transform.csv``.
    """
    import json

    import pandas as pd

    files = extract_remote_zip_members(url, VIZGEN_EXPORT_MEMBERS, dest_dir)
    transform = np.loadtxt(files["micron_to_mosaic_pixel_transform.csv"])
    sx, sy = float(transform[0, 0]), float(transform[1, 1])
    if abs(sx - sy) > 1e-3 * sx:
        raise ValueError(f"mosaic pixels are not square ({sx} x {sy} per um)")

    n_meta = len(pd.read_csv(files["cell_metadata.csv"], index_col=0))
    with open(files["cell_boundaries.geojson"], encoding="utf-8") as handle:
        features = json.load(handle)["features"]
    if len(features) != n_meta:
        raise ValueError(
            f"cell_boundaries.geojson has {len(features)} features, "
            f"cell_metadata.csv {n_meta} rows"
        )

    rows = obs.index.astype(np.int64).to_numpy()
    outlines = []
    for row in rows:
        geometry = features[row]["geometry"]
        if geometry["type"] != "Polygon":
            raise ValueError(f"cell {row}: {geometry['type']}, expected a Polygon")
        outlines.append(
            [np.asarray(r, dtype=np.float64) for r in geometry["coordinates"]]
        )
    bbox = np.array(
        [
            [o[0][:, 0].min(), o[0][:, 1].min(), o[0][:, 0].max(), o[0][:, 1].max()]
            for o in outlines
        ]
    )
    if np.abs(bbox - obs[["min_x", "min_y", "max_x", "max_y"]].to_numpy()).max() > 1e-3:
        raise ValueError("the polygons no longer match the cells' bounding boxes")
    logger.info(
        "  outline %d cells (%.4f mosaic px per um)", len(outlines), (sx + sy) / 2
    )
    return outlines, (sx + sy) / 2


# --------------------------------------------------------------------------- #
# Reading single members out of a large remote zip
# --------------------------------------------------------------------------- #
class _HttpFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests."""

    def __init__(self, url: str):
        self.source = url
        self._resolve()
        self.pos = 0

    def _resolve(self) -> None:
        """Follow the redirect; figshare hands out signed links that expire."""
        request = urllib.request.Request(
            self.source, headers={"User-Agent": _USER_AGENT, "Range": "bytes=0-0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            self.url = response.url
            self.size = int(response.headers["Content-Range"].split("/")[1])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        for attempt in range(3):
            request = urllib.request.Request(
                self.url,
                headers={"User-Agent": _USER_AGENT, "Range": f"bytes={self.pos}-{end}"},
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    data = response.read()
                break
            except urllib.error.HTTPError as err:
                if err.code != 403 or attempt == 2:
                    raise
                self._resolve()  # the signed link expired: get a fresh one
        self.pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def extract_remote_zip_members(
    url: str, names: list[str], dest_dir: Path
) -> dict[str, Path]:
    """Copy the members called ``names`` (matched by file name) out of the remote
    zip at ``url`` into ``dest_dir``, without downloading the whole archive.
    Members already in ``dest_dir`` are reused."""
    out = {name: dest_dir / name for name in names}
    todo = [name for name in names if not out[name].exists()]
    for name in names:
        if name not in todo:
            logger.info("  cached  %s", name)
    if not todo:
        return out
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(io.BufferedReader(_HttpFile(url), buffer_size=1 << 22))
    for name in todo:
        info = next(
            i
            for i in archive.infolist()
            if i.filename.endswith("/" + name) and "__MACOSX" not in i.filename
        )
        logger.info(
            "  extract %s (%.0f MB compressed) from %s",
            name,
            info.compress_size / 1e6,
            url,
        )
        tmp = out[name].with_suffix(out[name].suffix + ".part")
        with archive.open(info) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 22)
        tmp.replace(out[name])
    return out
