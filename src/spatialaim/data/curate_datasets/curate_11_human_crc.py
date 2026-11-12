"""
Build dataset 11_human-crc: the patient-matched pair of patient P2, a Flex
reference + one Xenium slice, for the GUI use case (Results section 5). Not part
of the Experiment 1 corpus, so it is written to Data/03_GUI by default, not to
01_Datasets_Paper.

Pair covered (same_paper):
    11_human-crc-p2 <-> 11_human-crc-p2

Paper: Oliveira et al., Nature Genetics 2025, "High-definition spatial
transcriptomic profiling of immune cell populations in colorectal cancer"
(https://doi.org/10.1038/s41588-025-02193-3; GEO SuperSeries GSE280318). Both
sides come from one FFPE block: the Flex reference was dissociated from
curls serial to the sections the spatial assays used.

Downloads: four Flex libraries, the authors' metadata and three Xenium files
(~460 MB; the segmentation is most of it).

------------------------------------------------------------------------------
scRNA  11_human-crc-p2.h5ad   —  Chromium Flex
------------------------------------------------------------------------------
  counts      GSE280311 (Flex). Each patient is one probe barcode in four
              libraries: P2CRC = BC1 in pool 1 (GSM8594533..36, *_P1L{1..4}_*);
              files *_count_filtered_feature_bc_matrix.h5
  annotation  github.com/10XGenomics/HumanColonCancer_VisiumHD,
              MetaData/SingleCell_MetaData.csv.gz (all 8 Flex samples)

The authors' barcodes carry the cellranger-aggr index as suffix (pool 1: two
samples per library), so P2CRC's libraries are aggr samples 1, 3, 5, 7. Its
barcodes match the four matrices one to one; the authors' QC
(``QCFilter == "Keep"``) selects the cells, kept in metadata order:
41933 -> 35673.

Curation:
  * obs  index = the authors' ``Barcode``; ``Level1`` and ``Level2`` verbatim.
           Level2 was sub-clustered within Level1 on a 15 % sketch and then
           extended to all cells, so it is only mostly nested (3.8 % of cells
           sit outside their Level2's main Level1); kept as published.
  * X    = the four matrices' columns for those barcodes, float32
  * var  index = gene symbols uppercased; three symbols occur twice and are
           summed onto the first occurrence, 18082 -> 18079

------------------------------------------------------------------------------
ST  11_human-crc-p2.h5ad   —  Xenium, 422 genes (colon panel + 100 add-on)
------------------------------------------------------------------------------
  https://cf.10xgenomics.com/samples/xenium/2.0.0/
  Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE/  (a GSE280314 sample)
    ..._cell_feature_matrix.h5, ..._cells.csv.gz, ..._cells.zarr.zip

The full outs bundles (14-16 GB) are unnecessary. The authors' repo lists ~1.1M
cells per slide for a larger Xenium region; the public 10x release used here
has 340837 cells and carries no cell-type labels.

Curation:
  * cells in the Xenium output's own order; obs index = Xenium ``cell_id``
  * cells with fewer than 10 transcripts over the 422 genes are dropped
           (340837 -> 331383): mapped, they form
           spurious states of near-empty cells
  * X    = the 422 ``Gene Expression`` features; unassigned / negative control
           codewords and probes are dropped
  * var  index = the feature names uppercased
  * obsm["spatial"] = ``x_centroid``, ``y_centroid`` (um), float32
  * no cell-type column: the paper annotated Visium HD bins, not Xenium cells
  * obs["segmentation_id"], uns["segmentation_pixel_size_um"] = 0.2125 and the
           mask ST/Segmentation/<name>.tif: cells.zarr.zip masks/1, copied as
           is. Its labels are the row of the zarr ``cell_id`` array plus one,
           which is the cells.csv order, so segmentation_id = that row + 1
           (the dropped cells' labels stay in the mask). Xenium 2.0
           multimodal segmentation (boundary stain, then interior stain, then
           5 um nucleus expansion).

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python -m spatialaim.data.curate_datasets.curate_11_human_crc
    python -m spatialaim.data.curate_datasets.curate_11_human_crc --out-dir <root>

Sources are downloaded to a temporary folder below --out-dir and deleted again.
Set SPATIALAIM_KEEP_DOWNLOADS=1 to keep (and on a re-run reuse) them in
<out-dir>/_downloads/11_human-crc-p2 instead. The ~5 GB mask is copied band
by band, never held whole.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import hstack

from . import _common, _segmentation
from .curate_09_human_breast_cancer import (
    GENE_EXPRESSION,
    MASK_PIXEL_UM,
    _check_shape,
    _copy_cell_mask,
    _fold_duplicates,
    _read_10x_h5,
)

logger = logging.getLogger(__name__)

OUT_ROOT = _common.DATA_ROOT / "03_GUI"

GEO_SAMPLES = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8594nnn"
REPO = "https://raw.githubusercontent.com/10XGenomics/HumanColonCancer_VisiumHD/main"
SC_META_URL = f"{REPO}/MetaData/SingleCell_MetaData.csv.gz"
SC_META_BYTES = 7_288_481
XENIUM = "https://cf.10xgenomics.com/samples/xenium/2.0.0"
SC_GENES = 18079
ST_GENES = 422
MIN_TRANSCRIPTS = 10


@dataclass(frozen=True)
class Patient:
    patient: str  # the metadata's Patient value, e.g. "P2CRC"
    probe_barcode: str
    # (GSM, library, aggr index of this patient in the authors' barcodes, bytes)
    libraries: tuple[tuple[str, str, int, int], ...]
    n_cells: int
    n_spots: int  # all Xenium cells
    n_spots_kept: int  # with >= MIN_TRANSCRIPTS
    # bytes of cell_feature_matrix.h5, cells.csv.gz, cells.zarr.zip
    xenium_bytes: tuple[int, int, int]

    @property
    def name(self) -> str:
        return f"11_human-crc-{self.patient[:2].lower()}"

    @property
    def xenium_prefix(self) -> str:
        sample = f"Xenium_V1_Human_Colon_Cancer_{self.patient[:2]}_CRC_Add_on_FFPE"
        return f"{XENIUM}/{sample}/{sample}"


PATIENT = Patient(
    patient="P2CRC",
    probe_barcode="BC1",
    libraries=(
        ("GSM8594533", "P1L1", 1, 26_874_789),
        ("GSM8594534", "P1L2", 3, 27_306_204),
        ("GSM8594535", "P1L3", 5, 27_294_024),
        ("GSM8594536", "P1L4", 7, 26_361_551),
    ),
    n_cells=35673,
    n_spots=340837,
    n_spots_kept=331383,
    xenium_bytes=(21_438_841, 18_562_765, 291_928_061),
)


# --------------------------------------------------------------------------- #
# scRNA
# --------------------------------------------------------------------------- #
def build_scrna(p: Patient, tmp_dir: Path, out_path: Path) -> None:
    logger.info("scRNA %s — Chromium Flex, patient %s (GSE280311)", p.name, p.patient)
    meta_file = _common.download(
        SC_META_URL,
        tmp_dir / "SingleCell_MetaData.csv.gz",
        expected_bytes=SC_META_BYTES,
    )
    meta = pd.read_csv(meta_file)
    meta = meta[meta["Patient"] == p.patient]
    logger.info("  read    metadata (%d %s barcodes)", len(meta), p.patient)

    blocks = []
    barcodes: list[str] = []
    names: np.ndarray | None = None
    for gsm, library, aggr, size in p.libraries:
        file_name = (
            f"{gsm}_{library}_{p.patient}_{p.probe_barcode}"
            "_count_filtered_feature_bc_matrix.h5"
        )
        path = _common.download(
            f"{GEO_SAMPLES}/{gsm}/suppl/{file_name}",
            tmp_dir / file_name,
            expected_bytes=size,
        )
        matrix, library_barcodes, library_names, _ = _read_10x_h5(path)
        if names is not None and not np.array_equal(names, library_names):
            raise ValueError(f"{file_name}: gene list differs from the first library")
        names = library_names
        blocks.append(matrix)
        barcodes.extend(b.split("-")[0] + f"-{aggr}" for b in library_barcodes)
    if names is None:
        raise ValueError(f"{p.patient}: no libraries listed")
    matrix = hstack(blocks).tocsc()
    logger.info("  read    %d libraries (%d x %d)", len(blocks), *matrix.shape)

    if set(barcodes) != set(meta["Barcode"]):
        raise ValueError("the four libraries no longer match the metadata's barcodes")
    meta = meta[meta["QCFilter"] == "Keep"]
    position = pd.Index(barcodes).get_indexer(meta["Barcode"].to_numpy())

    X = matrix[:, position].T.tocsr().astype(np.float32)
    X, genes = _fold_duplicates(X, np.asarray(_common.uppercase(names)))
    logger.info("  genes   %d -> %d (duplicate symbols summed)", len(names), len(genes))

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                level: pd.Categorical(meta[level].astype(str).to_numpy())
                for level in ("Level1", "Level2")
            },
            index=pd.Index(meta["Barcode"].to_numpy(), name=None),
        ),
        var=pd.DataFrame(index=pd.Index(genes, name=None)),
    )
    _check_shape(adata, (p.n_cells, SC_GENES), "scRNA")
    for level in ("Level1", "Level2"):
        logger.info("  %s: %d types", level, adata.obs[level].cat.categories.size)
    _common.write(adata, out_path)


# --------------------------------------------------------------------------- #
# ST
# --------------------------------------------------------------------------- #
def build_st(p: Patient, tmp_dir: Path, st_dir: Path) -> None:
    logger.info(
        "ST %s — Xenium human colon cancer %s (GSE280314)", p.name, p.patient[:2]
    )
    tag = f"Xenium_{p.patient[:2]}"
    counts_bytes, cells_bytes, masks_bytes = p.xenium_bytes
    counts_file = _common.download(
        f"{p.xenium_prefix}_cell_feature_matrix.h5",
        tmp_dir / f"{tag}_cell_feature_matrix.h5",
        expected_bytes=counts_bytes,
    )
    cells_file = _common.download(
        f"{p.xenium_prefix}_cells.csv.gz",
        tmp_dir / f"{tag}_cells.csv.gz",
        expected_bytes=cells_bytes,
    )
    masks_file = _common.download(
        f"{p.xenium_prefix}_cells.zarr.zip",
        tmp_dir / f"{tag}_cells.zarr.zip",
        expected_bytes=masks_bytes,
    )

    matrix, barcodes, names, types = _read_10x_h5(counts_file)
    genes = types == GENE_EXPRESSION
    logger.info(
        "  read    matrix (%d x %d, %d gene features)", *matrix.shape, int(genes.sum())
    )

    cells = pd.read_csv(cells_file)
    if not np.array_equal(cells["cell_id"].astype(str).to_numpy(), barcodes):
        raise ValueError("cells.csv.gz and the matrix no longer share a cell order")

    adata = ad.AnnData(
        X=matrix[genes].T.tocsr().astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(barcodes, name=None)),
        var=pd.DataFrame(index=pd.Index(_common.uppercase(names[genes]), name=None)),
    )
    adata.obsm["spatial"] = np.column_stack(
        [cells["x_centroid"].to_numpy(), cells["y_centroid"].to_numpy()]
    ).astype(np.float32)
    _check_shape(adata, (p.n_spots, ST_GENES), "ST")

    _check_mask_order(masks_file, barcodes)
    _copy_cell_mask(masks_file, _segmentation.mask_path(st_dir, p.name), adata.n_obs)
    _segmentation.annotate(adata, np.arange(1, adata.n_obs + 1), MASK_PIXEL_UM)

    keep = np.asarray(adata.X.sum(axis=1)).ravel() >= MIN_TRANSCRIPTS
    adata = adata[keep].copy()
    logger.info(
        "  cells   %d -> %d (>= %d transcripts)",
        len(keep),
        adata.n_obs,
        MIN_TRANSCRIPTS,
    )
    _check_shape(adata, (p.n_spots_kept, ST_GENES), "ST")
    _common.write(adata, st_dir / f"{p.name}.h5ad")


def _check_mask_order(zarr_zip: Path, barcodes: np.ndarray) -> None:
    """Mask label i+1 is the zarr's cell_id row i; assert that row order = ours."""
    import zarr

    group: Any = zarr.open_group(
        zarr.storage.ZipStore(str(zarr_zip), mode="r"), mode="r"
    )
    rows = np.asarray(group["cell_id"][:])
    # Xenium string ids: the prefix integer in base 16 written with letters a..p
    decoded = np.array(
        [
            [int(prefix.translate(_HEX_LETTERS), 16), int(suffix)]
            for prefix, suffix in (b.split("-") for b in barcodes)
        ],
        dtype=rows.dtype,
    )
    if not np.array_equal(decoded, rows):
        raise ValueError("the mask's cell_id rows no longer follow the cells.csv order")


_HEX_LETTERS = str.maketrans("abcdefghijklmnop", "0123456789abcdef")


# --------------------------------------------------------------------------- #
def main() -> None:
    _common.setup_logging()
    parser = _common.build_parser(__doc__)
    parser.set_defaults(out_dir=OUT_ROOT)
    args = parser.parse_args()
    sc_dir, st_dir = _common.prepare_dirs(args.out_dir)

    p = PATIENT
    with _common.staging(args.out_dir, p.name) as tmp_dir:
        build_scrna(p, tmp_dir, sc_dir / f"{p.name}.h5ad")
        build_st(p, tmp_dir, st_dir)
    logger.info("%s done", p.name)


if __name__ == "__main__":
    main()
