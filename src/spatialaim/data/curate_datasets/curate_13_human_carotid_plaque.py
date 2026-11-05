"""
Build dataset 13_human-carotid-plaque: human carotid atherosclerotic plaques, a
scRNA-seq reference + one Xenium slice, for the GUI use case (Results section
5). Not part of the Experiment 1 corpus, so it is written to Data/03_GUI by
default.

Pair (same_paper; reference and slice come from different patients):
    13_human-carotid-plaque <-> 13_human-carotid-plaque

Paper: Single cell spatial transcriptomics integration deciphers the
morphological heterogeneity of atherosclerotic carotid arteries, Nat Commun
2025 (https://doi.org/10.1038/s41467-025-67679-4).

------------------------------------------------------------------------------
scRNA  13_human-carotid-plaque.h5ad   —  10x scRNA-seq, 17 patients
------------------------------------------------------------------------------
  Zenodo 17520898, scRNA_final_dataset.zip:
    GRCh38-p14-Gencode_v44_with_intron_final_data.h5ad (14641 cells, plaque and
    adjacent control tissue)

Curation:
  * cells as published; obs index kept
  * obs["final_low_level_celltype"] (26 types, e.g. Mac_C1Q / Mac_HMOX1 /
           Mac_TREM2hi, VSMC_contr / VSMC_modul / VSMC_fb-like) and
           obs["final_high_level_celltype"] (10 types) verbatim (the paper's
           own data dictionary swaps the two descriptions; low = finer)
  * X    = layers["raw"] (integer counts), float32
  * var  index = ``HGNC_NAME`` uppercased, duplicates summed

------------------------------------------------------------------------------
ST  13_human-carotid-plaque.h5ad   —  Xenium, custom panel 1
------------------------------------------------------------------------------
  GEO GSE294466, one plaque with panel 1 (the cell-typing panel). The outs zip
  is read by range requests; only cell_feature_matrix.h5, cells.csv.gz and
  cells.zarr.zip are fetched.

Curation:
  * cells in the Xenium output's own order; obs index = Xenium ``cell_id``
  * cells with fewer than 10 transcripts are dropped (as for 11_human-crc)
  * X    = the ``Gene Expression`` features; var = names uppercased
  * obsm["spatial"] = ``x_centroid``, ``y_centroid`` (um), float32
  * no cell-type column (the paper labels its Baysor re-segmentation)
  * obs["segmentation_id"], uns["segmentation_pixel_size_um"] = 0.2125 and the
           mask ST/Segmentation/<name>.tif: the 10x segmentation, cells.zarr.zip
           masks/1 copied as is; label = cells.csv row + 1

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python -m spatialaim.data.curate_datasets.curate_13_human_carotid_plaque
    python -m spatialaim.data.curate_datasets.curate_13_human_carotid_plaque --out-dir <root>

Set SPATIALAIM_KEEP_DOWNLOADS=1 to keep (and reuse) the sources in
<out-dir>/_downloads/13_human-carotid-plaque.
"""

import logging
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from . import _common, _segmentation
from .curate_09_human_breast_cancer import (
    GENE_EXPRESSION,
    MASK_PIXEL_UM,
    _check_shape,
    _copy_cell_mask,
    _fold_duplicates,
    _read_10x_h5,
)
from .curate_11_human_crc import _check_mask_order

logger = logging.getLogger(__name__)

NAME = "13_human-carotid-plaque"
OUT_ROOT = _common.DATA_ROOT / "03_GUI"

SC_ZIP_URL = "https://zenodo.org/records/17520898/files/scRNA_final_dataset.zip"
SC_ZIP_BYTES = 458_601_777
SC_H5AD = "GRCh38-p14-Gencode_v44_with_intron_final_data.h5ad"
LEVELS = ("final_low_level_celltype", "final_high_level_celltype")

ST_SAMPLE = "Patient4_Plaque_Panel1"
ST_ZIP_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8907nnn/GSM8907614/suppl/"
    "GSM8907614_output-XETG00050__0003370__P4_D__20230808__140759.zip"
)
ST_MEMBERS = ["cell_feature_matrix.h5", "cells.csv.gz", "cells.zarr.zip"]

SC_EXPECTED = (14641, 28816)
ST_GENES = 314
N_SPOTS = 26134
N_SPOTS_KEPT = 20493
MIN_TRANSCRIPTS = 10


# --------------------------------------------------------------------------- #
# scRNA
# --------------------------------------------------------------------------- #
def build_scrna(tmp_dir: Path, out_path: Path) -> None:
    logger.info("scRNA %s — 10x scRNA-seq carotid plaques (Zenodo 17520898)", NAME)
    archive = _common.download(
        SC_ZIP_URL, tmp_dir / Path(SC_ZIP_URL).name, expected_bytes=SC_ZIP_BYTES
    )
    h5ad = tmp_dir / SC_H5AD
    if not h5ad.exists():
        with zipfile.ZipFile(archive) as z:
            member = next(
                n
                for n in z.namelist()
                if n.endswith("/" + SC_H5AD) and "__MACOSX" not in n
            )
            with z.open(member) as src, open(h5ad, "wb") as dst:
                dst.write(src.read())
    source = ad.read_h5ad(h5ad)
    logger.info("  read    %s (%d x %d)", SC_H5AD, *source.shape)

    raw = csr_matrix(source.layers["raw"]).astype(np.float32)
    if raw.data.size and not np.array_equal(raw.data, np.round(raw.data)):
        raise ValueError("layers['raw'] is not integer counts")
    names = np.asarray(_common.uppercase(source.var["HGNC_NAME"].astype(str)))
    X, genes = _fold_duplicates(raw, names)
    logger.info("  genes   %d -> %d (duplicate symbols summed)", len(names), len(genes))

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                level: pd.Categorical(source.obs[level].astype(str).to_numpy())
                for level in LEVELS
            },
            index=pd.Index(source.obs_names.to_numpy(), name=None),
        ),
        var=pd.DataFrame(index=pd.Index(genes, name=None)),
    )
    if SC_EXPECTED[1]:
        _check_shape(adata, SC_EXPECTED, "scRNA")
    for level in LEVELS:
        logger.info("  %s: %d types", level, adata.obs[level].cat.categories.size)
    _common.write(adata, out_path)


# --------------------------------------------------------------------------- #
# ST
# --------------------------------------------------------------------------- #
def build_st(tmp_dir: Path, st_dir: Path) -> None:
    logger.info("ST %s — Xenium carotid plaque %s (GSE294466)", NAME, ST_SAMPLE)
    files = _segmentation.extract_remote_zip_members(ST_ZIP_URL, ST_MEMBERS, tmp_dir)
    matrix, barcodes, names, types = _read_10x_h5(files["cell_feature_matrix.h5"])
    genes = types == GENE_EXPRESSION
    logger.info(
        "  read    matrix (%d x %d, %d gene features)", *matrix.shape, int(genes.sum())
    )

    cells = pd.read_csv(files["cells.csv.gz"])
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
    if N_SPOTS:
        _check_shape(adata, (N_SPOTS, ST_GENES), "ST")

    masks_file = files["cells.zarr.zip"]
    _check_mask_order(masks_file, barcodes)
    _copy_cell_mask(masks_file, _segmentation.mask_path(st_dir, NAME), adata.n_obs)
    _segmentation.annotate(adata, np.arange(1, adata.n_obs + 1), MASK_PIXEL_UM)

    keep = np.asarray(adata.X.sum(axis=1)).ravel() >= MIN_TRANSCRIPTS
    adata = adata[keep].copy()
    logger.info(
        "  cells   %d -> %d (>= %d transcripts)",
        len(keep),
        adata.n_obs,
        MIN_TRANSCRIPTS,
    )
    if N_SPOTS_KEPT:
        _check_shape(adata, (N_SPOTS_KEPT, ST_GENES), "ST")
    _common.write(adata, st_dir / f"{NAME}.h5ad")


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
