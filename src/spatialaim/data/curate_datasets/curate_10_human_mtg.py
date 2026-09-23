"""
Rebuild dataset 10_human-mtg (scRNA reference + its one ST slice) from source.

Pairs covered (pairs.csv):
    10_human-mtg <-> 10_human-mtg  (ExperimentMatch: same_consortium)

  scRNA  Hodge et al., Nature 2019, "Conserved cell types with divergent
         features in human versus mouse cortex"
         (https://www.nature.com/articles/s41586-019-1506-7), released as the
         Allen Institute's human MTG SMART-seq dataset
  ST     SpaceTx consortium ISS of human MTG, as used by Li et al.,
         Nature Methods 2022 (https://www.nature.com/articles/s41592-022-01480-9)

Both sides reproduce bit-exactly. Two downloads (~295 MB).

------------------------------------------------------------------------------
scRNA  10_human-mtg.h5ad   —  15928 x 50281
------------------------------------------------------------------------------
  https://brain-map.org/atlases-and-data/rnaseq/human-mtg-smart-seq
  human_MTG_gene_expression_matrices_2018-06-14.zip (295 MB), containing
  exon-matrix.csv, intron-matrix.csv, genes-rows.csv and samples-columns.csv

Curation:
  * all 15928 nuclei and all 50281 genes are kept — no filtering at all
  * X    = **exon + intron** read counts, summed, float32. Exon alone gives
           87.7M nonzeros against the shipped 148.6M, so the intron matrix is
           not optional here. (Dataset 08, the Allen *mouse* release, uses exon
           counts only — the two Allen datasets differ in this.)
  * obs  index = "MTG_" + the sample name, e.g. MTG_F1S4_160106_001_B01;
           ``cellType`` = the sample table's ``cluster`` (76 types),
           ``region`` = its ``brain_region`` (constant "MTG")
  * var  index = the gene symbols uppercased. They are **not** put through R's
           make.names here — the shipped file keeps names like "3.8-1.2" and
           "5-HT3C2" intact, unlike dataset 08.

------------------------------------------------------------------------------
ST  10_human-mtg.h5ad   —  22398 x 120
------------------------------------------------------------------------------
  https://github.com/spacetx-spacejam/data  ->  the ISS row of its data table,
  "cellxgene Human MTG 1 (FOVs 540)", a Google Drive CSV published by the
  consortium. Of the four human MTG cell-by-gene tables listed there this is
  the one with 22398 rows and matching cell ids; the others have 23659, 20309
  and 38982.

Curation:
  * X    = the table's 120 gene columns, float32
  * obs  index = "iss_" + ``CellID``; no obs columns (the table carries no
           annotation, and none is needed)
  * var  index = the gene column names uppercased
  * obsm["spatial"] = the table's ``centroidX``, ``centroidY`` as float32

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python -m spatialaim.data.curate_datasets.curate_10_human_mtg
    python -m spatialaim.data.curate_datasets.curate_10_human_mtg --out-dir <root>

Sources are downloaded to a temporary folder below --out-dir and deleted again.
Set SPATIALAIM_KEEP_DOWNLOADS=1 to keep (and on a re-run reuse) them in
<out-dir>/_downloads/10_human-mtg instead. Needs ~6 GB of free RAM.
"""

import logging
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack

from . import _common

logger = logging.getLogger(__name__)

NAME = "10_human-mtg"

ALLEN_URL = "https://celltypes.brain-map.org/api/v2/well_known_file_download/694416044"
ALLEN_BYTES = 308_844_265
# "cellxgene Human MTG 1 (FOVs 540)" from the SpaceTx data table.
ISS_URL = (
    "https://drive.google.com/uc?export=download&id=1VjGa0QGHFRMRLQRAP1PGCkMjEVFVB_g5"
)
ISS_BYTES = 6_095_585

EXON_MEMBER = "human_MTG_2018-06-14_exon-matrix.csv"
INTRON_MEMBER = "human_MTG_2018-06-14_intron-matrix.csv"
GENES_MEMBER = "human_MTG_2018-06-14_genes-rows.csv"
SAMPLES_MEMBER = "human_MTG_2018-06-14_samples-columns.csv"

SC_EXPECTED = (15928, 50281)
ST_EXPECTED = (22398, 120)
CHUNK_ROWS = 2000
COORDINATES = ["centroidX", "centroidY"]


def _check_shape(adata: ad.AnnData, expected: tuple[int, int], what: str) -> None:
    if adata.shape != expected:
        raise ValueError(
            f"{what}: expected {expected}, got {adata.shape} — the source has changed"
        )


# --------------------------------------------------------------------------- #
# scRNA
# --------------------------------------------------------------------------- #
def _read_matrix(
    archive: zipfile.ZipFile, member: str, genes: pd.DataFrame, samples: pd.DataFrame
) -> csr_matrix:
    """One genes x cells CSV from the release as a cells x genes CSR matrix."""
    with archive.open(member) as handle:
        line = handle.readline().decode().rstrip("\r\n")
    header = [c.strip().strip('"') for c in line.split(",")][1:]
    if header != samples["sample_name"].tolist():
        raise ValueError(
            f"{member}: the column order no longer matches {SAMPLES_MEMBER}"
        )

    blocks: list[csr_matrix] = []
    keys: list[int] = []
    with archive.open(member) as handle:
        for chunk in pd.read_csv(handle, index_col=0, chunksize=CHUNK_ROWS):
            keys.extend(chunk.index.tolist())
            blocks.append(csr_matrix(chunk.to_numpy(dtype=np.float32)))
    if keys != genes["entrez_id"].tolist():
        raise ValueError(f"{member}: the row order no longer matches {GENES_MEMBER}")
    matrix = vstack(blocks).T.tocsr()
    logger.info(
        "  read    %s (%d x %d, %d nonzeros)", member, *matrix.shape, matrix.nnz
    )
    return matrix


def build_scrna(tmp_dir: Path, out_path: Path) -> None:
    logger.info("scRNA %s — Allen human MTG SMART-seq (Hodge et al. 2019)", NAME)
    archive_path = _common.download(
        ALLEN_URL,
        tmp_dir / "human_MTG_gene_expression_matrices_2018-06-14.zip",
        expected_bytes=ALLEN_BYTES,
    )

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(GENES_MEMBER) as handle:
            genes = pd.read_csv(handle)
        with archive.open(SAMPLES_MEMBER) as handle:
            samples = pd.read_csv(handle)
        logger.info("  read    %d genes, %d nuclei", len(genes), len(samples))
        # The shipped counts are exon + intron; exon alone is ~40% short.
        X = (
            _read_matrix(archive, EXON_MEMBER, genes, samples)
            + _read_matrix(archive, INTRON_MEMBER, genes, samples)
        ).tocsr()

    obs = pd.DataFrame(
        index=pd.Index([f"MTG_{n}" for n in samples["sample_name"]], name=None)
    )
    obs["cellType"] = pd.Categorical(samples["cluster"].astype(str).to_numpy())
    obs["region"] = pd.Categorical(samples["brain_region"].astype(str).to_numpy())

    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(_common.uppercase(genes["gene"]), name=None)),
    )
    _check_shape(adata, SC_EXPECTED, "scRNA")
    logger.info("  cellType: %d types", adata.obs["cellType"].cat.categories.size)
    _common.write(adata, out_path)


# --------------------------------------------------------------------------- #
# ST
# --------------------------------------------------------------------------- #
def build_st(tmp_dir: Path, out_path: Path) -> None:
    logger.info("ST %s — SpaceTx ISS human MTG (FOVs 540)", NAME)
    table_path = _common.download(
        ISS_URL, tmp_dir / "ISS_human_MTG_1_cellxgene.csv", expected_bytes=ISS_BYTES
    )
    table = pd.read_csv(table_path)
    logger.info("  read    cell-by-gene table (%d x %d)", *table.shape)

    missing = [c for c in ["CellID", *COORDINATES] if c not in table.columns]
    if missing:
        raise KeyError(f"the table has no {missing} column(s) — the source has changed")
    genes = [c for c in table.columns if c not in {"CellID", *COORDINATES}]

    adata = ad.AnnData(
        X=csr_matrix(table[genes].to_numpy(dtype=np.float32)),
        obs=pd.DataFrame(
            index=pd.Index([f"iss_{i}" for i in table["CellID"]], name=None)
        ),
        var=pd.DataFrame(index=pd.Index(_common.uppercase(genes), name=None)),
    )
    adata.obsm["spatial"] = table[COORDINATES].to_numpy(dtype=np.float32)
    _check_shape(adata, ST_EXPECTED, "ST")
    _common.write(adata, out_path)


# --------------------------------------------------------------------------- #
def main() -> None:
    _common.setup_logging()
    args = _common.build_parser(__doc__).parse_args()
    sc_dir, st_dir = _common.prepare_dirs(args.out_dir)

    with _common.staging(args.out_dir, NAME) as tmp_dir:
        build_scrna(tmp_dir, sc_dir / f"{NAME}.h5ad")
        build_st(tmp_dir, st_dir / f"{NAME}.h5ad")

    logger.info("%s done", NAME)


if __name__ == "__main__":
    main()
