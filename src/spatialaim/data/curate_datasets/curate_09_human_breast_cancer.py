"""
Rebuild dataset 09_human-breast-cancer (scRNA reference + its one ST slice).

Pairs covered (pairs.csv):
    09_human-breast-cancer <-> 09_human-breast-cancer  (ExperimentMatch: same_paper)

Paper: Janesick et al., Nature Communications 2023, "High resolution mapping of
the tumor microenvironment using integrated single-cell, spatial and in situ
analysis" (https://www.nature.com/articles/s41467-023-43458-x). Both sides come
from that study: the scFFPE-seq reference from its GEO series, the Xenium slice
from the 10x Genomics preview dataset the paper is built on.

Three downloads (~180 MB). Both sides reproduce essentially exactly; see the
note on the ST cell order below.

------------------------------------------------------------------------------
scRNA  09_human-breast-cancer.h5ad   —  27472 x 37133, Chromium Flex scFFPE-seq
------------------------------------------------------------------------------
  counts      https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243275
              GSM7782698_count_raw_feature_bc_matrix.h5 (142 MB, 37143 x 708252)
  annotation  GSE243275_Barcode_Cell_Type_Matrices.xlsx, sheet "scFFPE-Seq"

The GEO matrix is the **raw** (all-barcode) one — no filtered matrix is
published for this sample — so the workbook's 27472 annotated barcodes are what
selects the cells, in the workbook's order.

Curation:
  * obs  index = the workbook's ``Barcode``, ``cellType`` = its ``Annotation``
           (19 types)
  * X    = the raw matrix's columns for those barcodes, float32
  * var  index = the matrix's gene symbols uppercased. Ten symbols occur twice;
           **the duplicate columns are summed onto the first occurrence**,
           37143 -> 37133. Keeping only the first occurrence instead leaves
           8016 nonzeros missing, so this is not a cosmetic choice.

------------------------------------------------------------------------------
ST  09_human-breast-cancer.h5ad   —  167780 x 313, Xenium
------------------------------------------------------------------------------
  https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast
    Xenium_FFPE_Human_Breast_Cancer_Rep1_cell_feature_matrix.h5  (12 MB)
    Xenium_FFPE_Human_Breast_Cancer_Rep1_cells.csv.gz            (8 MB)
  annotation  GSE243275_Barcode_Cell_Type_Matrices.xlsx,
              sheet "Xenium R1 Fig1-5 (supervised)"

Curation:
  * cells are taken in the Xenium output's own order (cell_id 1..167780).
           01_Datasets_Used holds the same 167780 cells in a different, opaque
           permutation that matches neither the Xenium order nor the workbook's;
           since the contents are identical cell for cell, the source order is
           used here.
  * X    = the 313 ``Gene Expression`` features only; the 228 blank and negative
           control codewords in the file are dropped
  * obs["cellType"] = the workbook's ``Cluster``, joined on cell id (20 types,
           including "Unlabeled")
  * var  index = the feature names uppercased
  * obsm["spatial"] = ``x_centroid``, ``y_centroid`` from cells.csv.gz as
           float32. 01_Datasets_Used's values differ by at most 2.4e-4 um,
           which is float32 rounding at these magnitudes.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python -m spatialaim.data.curate_datasets.curate_09_human_breast_cancer
    python -m spatialaim.data.curate_datasets.curate_09_human_breast_cancer --out-dir <root>

Sources are downloaded to a temporary folder below --out-dir and deleted again.
Set SPATIALAIM_KEEP_DOWNLOADS=1 to keep (and on a re-run reuse) them in
<out-dir>/_downloads/09_human-breast-cancer instead. Needs ~4 GB of free RAM.
"""

import logging
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix

from . import _common

logger = logging.getLogger(__name__)

NAME = "09_human-breast-cancer"

GEO = "https://ftp.ncbi.nlm.nih.gov/geo"
WORKBOOK_URL = (
    f"{GEO}/series/GSE243nnn/GSE243275/suppl/GSE243275_Barcode_Cell_Type_Matrices.xlsx"
)
WORKBOOK_BYTES = 9_409_803
SC_COUNTS_URL = f"{GEO}/samples/GSM7782nnn/GSM7782698/suppl/GSM7782698_count_raw_feature_bc_matrix.h5"
SC_COUNTS_BYTES = 148_916_296

XENIUM = (
    "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1"
    "/Xenium_FFPE_Human_Breast_Cancer_Rep1"
)
ST_COUNTS_URL = f"{XENIUM}_cell_feature_matrix.h5"
ST_COUNTS_BYTES = 12_148_885
ST_CELLS_URL = f"{XENIUM}_cells.csv.gz"
ST_CELLS_BYTES = 7_930_261

SC_SHEET = "scFFPE-Seq"
ST_SHEET = "Xenium R1 Fig1-5 (supervised)"
SC_EXPECTED = (27472, 37133)
ST_EXPECTED = (167780, 313)
GENE_EXPRESSION = "Gene Expression"


def _check_shape(adata: ad.AnnData, expected: tuple[int, int], what: str) -> None:
    if adata.shape != expected:
        raise ValueError(
            f"{what}: expected {expected}, got {adata.shape} — the source has changed"
        )


def _read_10x_h5(path: Path) -> tuple[csc_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """A CellRanger .h5 as (features x cells CSC, barcodes, feature names, feature types)."""
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        n_features, n_cells = group["shape"][:]
        matrix = csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=(int(n_features), int(n_cells)),
        )
        barcodes = np.array([b.decode() for b in group["barcodes"][:]])
        names = np.array([n.decode() for n in group["features/name"][:]])
        types = np.array([t.decode() for t in group["features/feature_type"][:]])
    return matrix, barcodes, names, types


# --------------------------------------------------------------------------- #
# scRNA
# --------------------------------------------------------------------------- #
def _fold_duplicates(X: csr_matrix, names: np.ndarray) -> tuple[csr_matrix, np.ndarray]:
    """Sum columns that share a gene symbol onto the symbol's first occurrence."""
    first: dict[str, int] = {}
    slot = np.empty(len(names), dtype=np.int64)
    for i, name in enumerate(names):
        if name not in first:
            first[name] = len(first)
        slot[i] = first[name]
    fold = csr_matrix(
        (np.ones(len(names), dtype=np.float32), (np.arange(len(names)), slot)),
        shape=(len(names), len(first)),
    )
    kept = np.empty(len(first), dtype=object)
    for name, index in first.items():
        kept[index] = name
    return (X @ fold).tocsr(), kept.astype(str)


def build_scrna(tmp_dir: Path, workbook: Path, out_path: Path) -> None:
    logger.info("scRNA %s — Chromium Flex scFFPE-seq (GSE243275)", NAME)
    counts_file = _common.download(
        SC_COUNTS_URL,
        tmp_dir / "GSM7782698_count_raw_feature_bc_matrix.h5",
        expected_bytes=SC_COUNTS_BYTES,
    )

    sheet = pd.read_excel(workbook, sheet_name=SC_SHEET)
    logger.info("  read    sheet %r (%d annotated cells)", SC_SHEET, len(sheet))

    matrix, barcodes, names, _ = _read_10x_h5(counts_file)
    logger.info("  read    raw matrix (%d x %d)", *matrix.shape)

    position = pd.Index(barcodes).get_indexer(sheet["Barcode"].to_numpy())
    if (position < 0).any():
        missing = sheet["Barcode"].to_numpy()[position < 0][:5]
        raise KeyError(
            f"{int((position < 0).sum())} annotated barcodes are not in the matrix, "
            f"e.g. {list(missing)}"
        )

    X = matrix[:, position].T.tocsr().astype(np.float32)
    X, genes = _fold_duplicates(X, np.asarray(_common.uppercase(names)))
    logger.info("  genes   %d -> %d (duplicate symbols summed)", len(names), len(genes))

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"cellType": pd.Categorical(sheet["Annotation"].astype(str).to_numpy())},
            index=pd.Index(sheet["Barcode"].to_numpy(), name=None),
        ),
        var=pd.DataFrame(index=pd.Index(genes, name=None)),
    )
    _check_shape(adata, SC_EXPECTED, "scRNA")
    logger.info("  cellType: %d types", adata.obs["cellType"].cat.categories.size)
    _common.write(adata, out_path)


# --------------------------------------------------------------------------- #
# ST
# --------------------------------------------------------------------------- #
def build_st(tmp_dir: Path, workbook: Path, out_path: Path) -> None:
    logger.info("ST %s — Xenium human breast cancer, replicate 1", NAME)
    counts_file = _common.download(
        ST_COUNTS_URL,
        tmp_dir / "Xenium_Rep1_cell_feature_matrix.h5",
        expected_bytes=ST_COUNTS_BYTES,
    )
    cells_file = _common.download(
        ST_CELLS_URL,
        tmp_dir / "Xenium_Rep1_cells.csv.gz",
        expected_bytes=ST_CELLS_BYTES,
    )

    matrix, barcodes, names, types = _read_10x_h5(counts_file)
    genes = types == GENE_EXPRESSION
    logger.info(
        "  read    matrix (%d x %d, %d gene features)", *matrix.shape, int(genes.sum())
    )

    cells = pd.read_csv(cells_file)
    if not np.array_equal(cells["cell_id"].astype(str).to_numpy(), barcodes):
        raise ValueError("cells.csv.gz and the matrix no longer share a cell order")

    sheet = pd.read_excel(workbook, sheet_name=ST_SHEET)
    labels = sheet.set_index(sheet["Barcode"].astype(str))["Cluster"]
    missing = [b for b in barcodes if b not in labels.index]
    if missing:
        raise KeyError(f"{len(missing)} cells have no annotation, e.g. {missing[:5]}")

    adata = ad.AnnData(
        X=matrix[genes].T.tocsr().astype(np.float32),
        obs=pd.DataFrame(
            {"cellType": pd.Categorical(labels.loc[barcodes].astype(str).to_numpy())},
            index=pd.Index(barcodes, name=None),
        ),
        var=pd.DataFrame(index=pd.Index(_common.uppercase(names[genes]), name=None)),
    )
    adata.obsm["spatial"] = np.column_stack(
        [cells["x_centroid"].to_numpy(), cells["y_centroid"].to_numpy()]
    ).astype(np.float32)
    _check_shape(adata, ST_EXPECTED, "ST")
    logger.info("  cellType: %d types", adata.obs["cellType"].cat.categories.size)
    _common.write(adata, out_path)


# --------------------------------------------------------------------------- #
def main() -> None:
    _common.setup_logging()
    args = _common.build_parser(__doc__).parse_args()
    sc_dir, st_dir = _common.prepare_dirs(args.out_dir)

    with _common.staging(args.out_dir, NAME) as tmp_dir:
        workbook = _common.download(
            WORKBOOK_URL,
            tmp_dir / "GSE243275_Barcode_Cell_Type_Matrices.xlsx",
            expected_bytes=WORKBOOK_BYTES,
        )
        build_scrna(tmp_dir, workbook, sc_dir / f"{NAME}.h5ad")
        build_st(tmp_dir, workbook, st_dir / f"{NAME}.h5ad")

    logger.info("%s done", NAME)


if __name__ == "__main__":
    main()
