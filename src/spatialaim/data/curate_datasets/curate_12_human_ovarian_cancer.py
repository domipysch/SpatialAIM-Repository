"""
Build dataset 12_human-ovarian-cancer: one 10x Genomics pair, a Flex reference +
one Xenium Prime 5K slice of the same ovarian tumour, for the GUI use case
(Results section 5). Not part of the Experiment 1 corpus, so it is written to
Data/03_GUI by default.

Pair (same_paper):
    12_human-ovarian-cancer <-> 12_human-ovarian-cancer

Source: 10x Genomics public datasets (CC BY 4.0), one FFPE block of an ovarian
papillary serous carcinoma (Discovery Life Sciences, stage III-B); the Flex
sections were cut adjacent to the Xenium section.
  * "Human Ovarian Cancer FFPE Single Cell Gene Expression Flex" (17k cells)
  * "FFPE Human Ovarian Cancer with 5K Human Pan Tissue and Pathways Panel plus
    100 Custom Genes" (Xenium Prime, Xenium Ranger 3.1 relabel)

------------------------------------------------------------------------------
scRNA  12_human-ovarian-cancer.h5ad   —  Chromium Flex
------------------------------------------------------------------------------
  counts      .../cell-exp/8.0.1/17k_Ovarian_Cancer_scFFPE/
              17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5
              (aggregate of two Flex runs, 17553 cells)
  annotation  .../supp/cell-exp/FLEX_Ovarian_Barcode_Cluster_Annotation.csv
              (17050 annotated barcodes, 16 types)

Curation:
  * cells = the annotated barcodes, in annotation order; obs index = barcode
  * obs["cellType"] = ``Cell Annotation`` verbatim
  * X    = counts, float32; var = gene symbols uppercased, duplicate symbols
           summed onto the first occurrence

------------------------------------------------------------------------------
ST  12_human-ovarian-cancer.h5ad   —  Xenium Prime, 5101 genes
------------------------------------------------------------------------------
  .../xenium/3.0.0/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun/
    ..._xe_outs.zip (10 GB; only cell_feature_matrix.zarr.zip and
                     cells.zarr.zip are read out of it, by range requests)
    ..._cell_groups.csv (10x's own Xenium cell-type labels)

Curation:
  * cells in the zarr's own order; obs index = Xenium ``cell_id`` string
  * crop: only cells with centroid in x 5000-8000, y 2500-5500 um (a 3 x 3 mm
           tumour / stroma region, ~11 % of the 398k cells); the whole slide
           makes the per-K analysis take ~18 min
  * cells with fewer than 10 transcripts are dropped (as for 11_human-crc)
  * X    = the 5101 ``gene`` features; codewords, controls and the aggregate
           feature are dropped; var = names uppercased
  * obsm["spatial"] = cell centroids (um) relative to the crop corner, float32
  * obs["cellType"] = 10x's Xenium group (``Unassigned`` kept as is); for
           comparison only, SpatialAIM never reads it
  * obs["segmentation_id"], uns["segmentation_pixel_size_um"] = 0.2125 and the
           mask ST/Segmentation/<name>.tif: the crop of cells.zarr.zip masks/1,
           labels as is; label = zarr cell_id row + 1 (the dropped cells'
           labels stay)

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python -m spatialaim.data.curate_datasets.curate_12_human_ovarian_cancer
    python -m spatialaim.data.curate_datasets.curate_12_human_ovarian_cancer --out-dir <root>

Set SPATIALAIM_KEEP_DOWNLOADS=1 to keep (and reuse) the sources in
<out-dir>/_downloads/12_human-ovarian-cancer.
"""

import logging
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from . import _common, _segmentation
from .curate_09_human_breast_cancer import (
    MASK_PIXEL_UM,
    _check_shape,
    _fold_duplicates,
    _read_10x_h5,
)

logger = logging.getLogger(__name__)

NAME = "12_human-ovarian-cancer"
OUT_ROOT = _common.DATA_ROOT / "03_GUI"

FLEX = "https://cf.10xgenomics.com/samples/cell-exp/8.0.1/17k_Ovarian_Cancer_scFFPE"
SC_COUNTS_URL = f"{FLEX}/17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5"
SC_COUNTS_BYTES = 29_956_782
SC_ANNOT_URL = "https://cf.10xgenomics.com/supp/cell-exp/FLEX_Ovarian_Barcode_Cluster_Annotation.csv"
SC_ANNOT_BYTES = 821_443

XENIUM = (
    "https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun/"
    "Xenium_Prime_Ovarian_Cancer_FFPE_XRrun"
)
ST_OUTS_URL = f"{XENIUM}_xe_outs.zip"
ST_GROUPS_URL = f"{XENIUM}_cell_groups.csv"
ST_GROUPS_BYTES = 16_487_299

SC_EXPECTED = (17050, 18079)
ST_GENES = 5101
N_SPOTS = 407124
N_SPOTS_KEPT = 44406
MIN_TRANSCRIPTS = 10
# (x0, x1, y0, y1) um: a 3 x 3 mm tumour / stroma region; the per-K analysis
# of all 398k cells x 5101 genes takes ~18 min, this crop ~2 min
CROP_UM = (5000, 8000, 2500, 5500)


# --------------------------------------------------------------------------- #
# scRNA
# --------------------------------------------------------------------------- #
def build_scrna(tmp_dir: Path, out_path: Path) -> None:
    logger.info("scRNA %s — Chromium Flex, 17k ovarian cancer", NAME)
    counts_file = _common.download(
        SC_COUNTS_URL,
        tmp_dir / Path(SC_COUNTS_URL).name,
        expected_bytes=SC_COUNTS_BYTES,
    )
    annot_file = _common.download(
        SC_ANNOT_URL, tmp_dir / Path(SC_ANNOT_URL).name, expected_bytes=SC_ANNOT_BYTES
    )
    annot = pd.read_csv(annot_file)
    matrix, barcodes, names, _ = _read_10x_h5(counts_file)
    logger.info("  read    matrix (%d x %d), %d annotated", *matrix.shape, len(annot))

    position = pd.Index(barcodes).get_indexer(annot["Barcode"].to_numpy())
    if (position < 0).any():
        raise KeyError(
            f"{int((position < 0).sum())} annotated barcodes not in the matrix"
        )
    X = matrix[:, position].T.tocsr().astype(np.float32)
    X, genes = _fold_duplicates(X, np.asarray(_common.uppercase(names)))
    logger.info("  genes   %d -> %d (duplicate symbols summed)", len(names), len(genes))

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "cellType": pd.Categorical(
                    annot["Cell Annotation"].astype(str).to_numpy()
                )
            },
            index=pd.Index(annot["Barcode"].to_numpy(), name=None),
        ),
        var=pd.DataFrame(index=pd.Index(genes, name=None)),
    )
    _check_shape(adata, SC_EXPECTED, "scRNA")
    logger.info("  cellType: %d types", adata.obs["cellType"].cat.categories.size)
    _common.write(adata, out_path)


# --------------------------------------------------------------------------- #
# ST
# --------------------------------------------------------------------------- #
def read_xenium_zarr(
    matrix_zip: Path, cells_zip: Path
) -> tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Xenium >= 3 zarr outputs as (cells x genes CSR, cell ids, gene names, centroids um)."""
    import zarr

    # zarr types item access as Array | Group; the layout here is fixed.
    features: Any = zarr.open_group(
        zarr.storage.ZipStore(str(matrix_zip), mode="r"), mode="r"
    )["cell_features"]
    attrs = features.attrs
    n_cells, n_features = int(attrs["number_cells"]), int(attrs["number_features"])
    csc = features["csc"]  # per cell (column) the feature indices
    X = csr_matrix(
        (
            np.asarray(csc["data"][:], dtype=np.float32),
            np.asarray(csc["indices"][:], dtype=np.int64),
            np.asarray(csc["indptr"][:], dtype=np.int64),
        ),
        shape=(n_cells, n_features),
    )
    genes = np.asarray(attrs["feature_types"]) == "gene"
    names = np.asarray(attrs["feature_keys"])[genes]
    ids = np.asarray(features["cell_id"][:])

    cells: Any = zarr.open_group(
        zarr.storage.ZipStore(str(cells_zip), mode="r"), mode="r"
    )
    if not np.array_equal(np.asarray(cells["cell_id"][:]), ids):
        raise ValueError("cells.zarr and cell_feature_matrix.zarr differ in cell order")
    summary = cells["cell_summary"]
    columns = list(summary.attrs["column_names"])
    table = np.asarray(summary[:])
    xy = table[:, [columns.index("cell_centroid_x"), columns.index("cell_centroid_y")]]
    return X[:, np.flatnonzero(genes)].tocsr(), _id_strings(ids), names, xy


def _id_strings(ids: np.ndarray) -> np.ndarray:
    """Xenium (prefix, suffix) ids -> 'prefix-suffix', the prefix in base 16 as a..p."""
    letters = str.maketrans("0123456789abcdef", "abcdefghijklmnop")
    return np.array([f"{p:08x}".translate(letters) + f"-{s}" for p, s in ids])


def build_st(tmp_dir: Path, st_dir: Path) -> None:
    logger.info("ST %s — Xenium Prime 5K ovarian cancer", NAME)
    members = _segmentation.extract_remote_zip_members(
        ST_OUTS_URL, ["cell_feature_matrix.zarr.zip", "cells.zarr.zip"], tmp_dir
    )
    groups_file = _common.download(
        ST_GROUPS_URL,
        tmp_dir / Path(ST_GROUPS_URL).name,
        expected_bytes=ST_GROUPS_BYTES,
    )
    masks_file = members["cells.zarr.zip"]
    X, ids, names, xy = read_xenium_zarr(
        members["cell_feature_matrix.zarr.zip"], masks_file
    )
    logger.info("  read    matrix (%d x %d genes)", *X.shape)

    groups = pd.read_csv(groups_file).set_index("cell_id")["group"]
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"cellType": pd.Categorical(groups.reindex(ids).astype(str).to_numpy())},
            index=pd.Index(ids, name=None),
        ),
        var=pd.DataFrame(index=pd.Index(_common.uppercase(names), name=None)),
    )
    adata.obsm["spatial"] = xy.astype(np.float32)
    _check_shape(adata, (N_SPOTS, ST_GENES), "ST")
    _segmentation.annotate(adata, np.arange(1, adata.n_obs + 1), MASK_PIXEL_UM)

    x0, x1, y0, y1 = CROP_UM
    inside = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
    keep = inside & (np.asarray(adata.X.sum(axis=1)).ravel() >= MIN_TRANSCRIPTS)
    adata = adata[keep].copy()
    adata.obsm["spatial"] -= np.array([x0, y0], dtype=np.float32)
    logger.info(
        "  cells   %d -> %d in the crop -> %d (>= %d transcripts)",
        len(keep),
        int(inside.sum()),
        adata.n_obs,
        MIN_TRANSCRIPTS,
    )
    if N_SPOTS_KEPT:
        _check_shape(adata, (N_SPOTS_KEPT, ST_GENES), "ST")

    # the crop holds only some labels: renumber the present ones to 1..n
    mask = _cropped_mask(masks_file)
    label = adata.obs[_segmentation.SEGMENTATION_ID].to_numpy()
    present = np.zeros(max(int(mask.max()), int(label.max())) + 1, dtype=bool)
    present[np.unique(mask)] = True
    present[0] = True
    lut = (np.cumsum(present) - 1).astype(np.uint32)
    seg_id = np.where(present[label], lut[label], 0)
    logger.info(
        "  labels  %d in the crop, %d kept cells without a pixel",
        int(lut[-1]),
        int((seg_id == 0).sum()),
    )
    _segmentation.annotate(adata, seg_id, MASK_PIXEL_UM)
    _segmentation.write_mask(_segmentation.mask_path(st_dir, NAME), lut[mask])
    _common.write(adata, st_dir / f"{NAME}.h5ad")


def _cropped_mask(zarr_zip: Path) -> np.ndarray:
    """masks/1 inside CROP_UM, labels as in the full slide."""
    import zarr

    group: Any = zarr.open_group(
        zarr.storage.ZipStore(str(zarr_zip), mode="r"), mode="r"
    )
    scale = float(np.asarray(group["masks/homogeneous_transform"][:])[0, 0])
    if abs(scale - 1 / MASK_PIXEL_UM) > 1e-4:
        raise ValueError(f"mask scale {scale} px per um, expected {1 / MASK_PIXEL_UM}")
    x0, x1, y0, y1 = (round(v * scale) for v in CROP_UM)
    mask = np.asarray(group["masks/1"][y0:y1, x0:x1])
    logger.info("  mask    %d x %d px crop of masks/1", *mask.shape)
    return mask


# --------------------------------------------------------------------------- #
def main() -> None:
    _common.setup_logging()
    parser = _common.build_parser(__doc__)
    parser.set_defaults(out_dir=OUT_ROOT)
    args = parser.parse_args()
    sc_dir, st_dir = _common.prepare_dirs(args.out_dir)
    with _common.staging(args.out_dir, NAME) as tmp_dir:
        build_scrna(tmp_dir, sc_dir / f"{NAME}.h5ad")
        build_st(tmp_dir, st_dir)
    logger.info("%s done", NAME)


if __name__ == "__main__":
    main()
