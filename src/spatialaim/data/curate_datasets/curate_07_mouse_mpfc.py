"""
Rebuild dataset 07_mouse-mpfc (scRNA reference + its ST slices) from source.

Pairs covered (pairs.csv): one per ST slice, ExperimentMatch "-".

  scRNA  Bhattacherjee et al., Nature Communications 2019, "Cell type-specific
         transcriptional programs in mouse prefrontal cortex during adolescence
         and addiction" (https://www.nature.com/articles/s41467-019-12054-3)
  ST     Wang et al., Science 2018, "Three-dimensional intact-tissue sequencing
         of single-cell transcriptional states" (STARmap)
         (https://doi.org/10.1126/science.aat5691)

------------------------------------------------------------------------------
Three ST slices, not four — the STARmap original is gone
------------------------------------------------------------------------------
STARmap's own distribution is no longer reachable: starmapresources.org is down
("Website is under maintenance") and the Dropbox folder it pointed at has been
deleted. The github.com/weallen/STARmap repository holds code only.

The mPFC sections are still available through a third-party figshare mirror,
which reproduces the shipped files faithfully — X bit-identical, gene names
identical, coordinates equal to 5e-11 — but carries only three of the four
sections. 01_Datasets_Used's 1247-cell section has no public source, so this
script produces three slices and they are numbered 07_1..07_3:

    07_1  20180417_BZ5_control   1049 cells   (01_Datasets_Used's 07_1)
    07_2  20180419_BZ9_control   1053 cells   (01_Datasets_Used's 07_2)
    07_3  20180424_BZ14_control  1088 cells   (01_Datasets_Used's 07_4)
                                 1247 cells   (01_Datasets_Used's 07_3) — dropped

------------------------------------------------------------------------------
scRNA  07_mouse-mpfc.h5ad   —  24822 x 19517
------------------------------------------------------------------------------
  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE124952
    GSE124952_meta_data.csv.gz          cell metadata, 35360 rows
    GSE124952_expression_matrix.csv.gz  genes x cells, 21000 x 35360

Both are series-level supplementary files. They are **not** in the series'
filelist.txt, which lists only the per-sample cellranger tar — they are only
visible in the suppl directory listing.

Curation:
  * cells = ``DevStage == "Adult"`` -> 24822 of 35360, in the metadata's order;
           the P21 adolescent cells are dropped
  * X    = the expression matrix's columns for those cells, float32
  * var  index = the matrix's gene symbols uppercased, with ``n_cells``
  * genes are filtered to ``n_cells >= 3``, 21000 -> 19517. **n_cells counts
           over all 35360 cells**, i.e. the filter was applied before the Adult
           subset — counting on the Adult cells alone leaves 19162 genes and
           does not reproduce 01_Datasets_Used.
  * obs  the metadata's ten columns verbatim (nGene, nUMI, percent.mito,
           Sample, treatment, Period, stage, L2_clusters, DevStage) plus
           ``celltype``. 01_Datasets_Used also has ``CellType``, which is a
           byte-for-byte duplicate of ``celltype``; the duplicate is dropped.

------------------------------------------------------------------------------
ST  07_1..07_3_mouse-mpfc.h5ad   —  166 genes each
------------------------------------------------------------------------------
  https://figshare.com/articles/dataset/STARmap_datasets/22565200

Curation, per section:
  * X    = the source X verbatim (dense float32 upstream), as sparse CSR
  * var  index = the source gene symbols uppercased
  * obs  ``ct`` (the paper's 15 fine types) and ``region`` verbatim, plus
           ``celltype`` = ``ct`` coarsened to the 6 types 01_Datasets_Used uses
           (CELL_TYPES). The source's ``Region`` is dropped: it is a
           byte-for-byte duplicate of ``region``.
  * obsm["spatial"] = the source's ``obsm["spatial"]``
  * obs_names are the mirror's "<row>x<col>" ids; 01_Datasets_Used used the
           STARmap cell indices, which the mirror does not carry.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python -m spatialaim.data.curate_datasets.curate_07_mouse_mpfc
    python -m spatialaim.data.curate_datasets.curate_07_mouse_mpfc --out-dir <root>

Sources are downloaded to a temporary folder below --out-dir and deleted again.
Set SPATIALAIM_KEEP_DOWNLOADS=1 to keep (and on a re-run reuse) them in
<out-dir>/_downloads/07_mouse-mpfc instead. Needs ~2 GB of free RAM.
"""

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack

from . import _common

logger = logging.getLogger(__name__)

NAME = "07_mouse-mpfc"

GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE124nnn/GSE124952/suppl"
METADATA_URL = f"{GEO}/GSE124952_meta_data.csv.gz"
METADATA_BYTES = 881_812
MATRIX_URL = f"{GEO}/GSE124952_expression_matrix.csv.gz"
MATRIX_BYTES = 92_321_939

FIGSHARE = "https://ndownloader.figshare.com/files"
# (slice number, figshare file id, section, expected cells)
ST_SECTIONS = [
    (1, "40038637", "20180417_BZ5_control", 1049, 798_672),
    (2, "40038640", "20180419_BZ9_control", 1053, 801_456),
    (3, "40038643", "20180424_BZ14_control", 1088, 825_816),
]
ST_GENES = 166

SC_EXPECTED = (24822, 19517)
MIN_CELLS = 3
CHUNK_ROWS = 1000

# The metadata columns carried over, in 01_Datasets_Used's order. CellType is
# omitted: it is rebuilt below as obs["celltype"], and 01_Datasets_Used holds
# the two as identical copies.
SC_OBS_COLUMNS = [
    "nGene",
    "nUMI",
    "percent.mito",
    "Sample",
    "treatment",
    "Period",
    "stage",
    "L2_clusters",
    "DevStage",
]
ADULT = "Adult"

# STARmap's 15 fine types -> the 6 types 01_Datasets_Used uses.
CELL_TYPES = {
    "Astro": "Astro",
    "Endo": "Endo",
    "Oligo": "Oligo",
    "Smc": "Smc",
    "L5-1": "Excitatory",
    "eL2/3": "Excitatory",
    "eL5-2": "Excitatory",
    "eL5-3": "Excitatory",
    "eL6-1": "Excitatory",
    "eL6-2": "Excitatory",
    "Lhx6": "Inhibitory",
    "NPY": "Inhibitory",
    "Reln": "Inhibitory",
    "SST": "Inhibitory",
    "VIP": "Inhibitory",
}


def _check_shape(adata: ad.AnnData, expected: tuple[int, int], what: str) -> None:
    if adata.shape != expected:
        raise ValueError(
            f"{what}: expected {expected}, got {adata.shape} — the source has changed"
        )


# --------------------------------------------------------------------------- #
# scRNA
# --------------------------------------------------------------------------- #
def build_scrna(tmp_dir: Path, out_path: Path) -> None:
    logger.info("scRNA %s — GSE124952, adult prefrontal cortex", NAME)
    metadata_file = _common.download(
        METADATA_URL,
        tmp_dir / "GSE124952_meta_data.csv.gz",
        expected_bytes=METADATA_BYTES,
    )
    matrix_file = _common.download(
        MATRIX_URL,
        tmp_dir / "GSE124952_expression_matrix.csv.gz",
        expected_bytes=MATRIX_BYTES,
    )

    metadata = pd.read_csv(metadata_file, index_col=0)
    adult = (metadata["DevStage"] == ADULT).to_numpy()
    logger.info(
        "  read    metadata (%d cells, %d adult)", len(metadata), int(adult.sum())
    )

    # Every column is read: n_cells is counted over all cells, because the gene
    # filter was applied before the adult subset. Only the adult columns are kept.
    blocks: list[csr_matrix] = []
    genes: list[str] = []
    counts: list[np.ndarray] = []
    for chunk in pd.read_csv(matrix_file, index_col=0, chunksize=CHUNK_ROWS):
        if len(genes) == 0 and list(chunk.columns) != list(metadata.index):
            raise ValueError("the matrix's columns no longer match the metadata's rows")
        genes.extend(chunk.index.tolist())
        values = chunk.to_numpy(dtype=np.float32)
        counts.append((values != 0).sum(1))
        blocks.append(csr_matrix(values[:, adult]))
    X = vstack(blocks).T.tocsr()
    n_cells = np.concatenate(counts)
    logger.info(
        "  read    matrix (%d genes x %d cells, %d nonzeros)",
        len(genes),
        X.shape[0],
        X.nnz,
    )

    keep = n_cells >= MIN_CELLS
    logger.info(
        "  kept    %d of %d genes (n_cells >= %d)",
        int(keep.sum()),
        len(genes),
        MIN_CELLS,
    )

    obs = metadata.loc[adult, SC_OBS_COLUMNS].copy()
    obs["celltype"] = pd.Categorical(
        metadata.loc[adult, "CellType"].astype(str).to_numpy()
    )

    adata = ad.AnnData(
        X=X[:, keep].astype(np.float32),
        obs=obs,
        var=pd.DataFrame(
            {"n_cells": n_cells[keep].astype(np.int32)},
            index=pd.Index(_common.uppercase(np.asarray(genes)[keep]), name=None),
        ),
    )
    _check_shape(adata, SC_EXPECTED, "scRNA")
    logger.info("  celltype: %d types", adata.obs["celltype"].cat.categories.size)
    _common.write(adata, out_path)


# --------------------------------------------------------------------------- #
# ST
# --------------------------------------------------------------------------- #
def build_st(tmp_dir: Path, st_dir: Path) -> None:
    logger.info("ST %s — STARmap mPFC, via the figshare mirror", NAME)
    for number, file_id, section, cells, size in ST_SECTIONS:
        path = _common.download(
            f"{FIGSHARE}/{file_id}", tmp_dir / f"{section}.h5ad", expected_bytes=size
        )
        source = ad.read_h5ad(path)

        fine = source.obs["ct"].astype(str).to_numpy()
        unmapped = sorted(set(fine) - set(CELL_TYPES))
        if unmapped:
            raise KeyError(f"CELL_TYPES does not cover {unmapped}")

        obs = pd.DataFrame(index=source.obs_names)
        obs["ct"] = pd.Categorical(fine)
        obs["region"] = source.obs["region"].to_numpy()
        obs["celltype"] = pd.Categorical([CELL_TYPES[c] for c in fine])

        adata = ad.AnnData(
            X=csr_matrix(np.asarray(source.X), dtype=np.float32),
            obs=obs,
            var=pd.DataFrame(
                index=pd.Index(_common.uppercase(source.var_names), name=None)
            ),
        )
        adata.obsm["spatial"] = np.asarray(source.obsm["spatial"])

        slice_name = f"07_{number}_{NAME.split('_', 1)[1]}"
        _check_shape(adata, (cells, ST_GENES), slice_name)
        _common.write(adata, st_dir / f"{slice_name}.h5ad")


# --------------------------------------------------------------------------- #
def main() -> None:
    _common.setup_logging()
    args = _common.build_parser(__doc__).parse_args()
    sc_dir, st_dir = _common.prepare_dirs(args.out_dir)

    with _common.staging(args.out_dir, NAME) as tmp_dir:
        build_scrna(tmp_dir, sc_dir / f"{NAME}.h5ad")
        build_st(tmp_dir, st_dir)

    logger.info("%s done", NAME)


if __name__ == "__main__":
    main()
