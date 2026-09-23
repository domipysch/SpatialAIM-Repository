"""
Rebuild dataset 02_colorectal-cancer (scRNA reference + its one ST slice) from source.

Pairs covered (pairs.csv):
    PairID 2   02_colorectal-cancer  <->  02_colorectal-cancer

------------------------------------------------------------------------------
Source: one archive covers both sides
------------------------------------------------------------------------------
Both files come from the CytoSPACE example bundle for single-cell-resolution ST,
published with the CytoSPACE paper (Vahid et al., Nat Biotechnol 2023):

  readme https://github.com/digitalcytometry/cytospace/tree/main?tab=readme-ov-file#running-cytospace-on-single-cell-st-data
  zip    https://drive.google.com/file/d/1odOcIfY3oqvLCNdXHLRaSmTraRxqnHLp/view
         -> CytoSPACE_example_colon_cancer_merscope.zip  (8.5 MB, 5 TSVs)

Upstream provenance of the two halves:
  scRNA  human colorectal cancer, 10x Chromium, Lee et al., Nat Genet 2020
         (https://pubmed.ncbi.nlm.nih.gov/32451460/, GEO GSE132465) — the bundle
         ships a 496-gene / 21917-cell subset of it, already annotated
  ST     Vizgen MERSCOPE FFPE human colon cancer showcase, 500-gene panel
         (https://info.vizgen.com/ffpe-showcase) — the bundle ships a
         52235-cell crop spanning x 5000-8000, y 1500-4500 um

The bundle is the actual download: the raw GEO/Vizgen releases are neither
gene-subset nor cropped, so they cannot reproduce these files on their own.

------------------------------------------------------------------------------
Curation — both sides reproduce bit-exactly, no filtering, no reordering
------------------------------------------------------------------------------
scRNA 02_colorectal-cancer.h5ad   (21917 x 496)
  * X    = <P>_scRNA_expressions_cytospace.tsv transposed to cells x genes, float32
  * obs  index = the TSV's column order;
           obs["cellType"] = <P>_scRNA_annotations_cytospace.tsv (10 types,
           already in the same row order)
  * var  index = the TSV's row order, uppercased (already uppercase at source)

ST 02_colorectal-cancer.h5ad      (52235 x 500)
  * X    = <P>_ST_expressions_cytospace.tsv transposed to cells x genes, float32
  * obs  index = the TSV's column order; no obs columns
  * var  index = the TSV's row order, uppercased
  * obsm["spatial"] = <P>_ST_coordinates_cytospace.tsv X/Y, float32, verbatim

Two things in the sources are deliberately not carried over, matching
01_Datasets_Used:
  * <P>_ST_celltypes_cytospace.tsv — a 10-type annotation for every one of the
    52235 ST cells. The shipped ST h5ad has no obs columns; ST annotations are
    dropped project-wide (dataset 01 drops the osmFISH ClusterName the same way).
  * obsm["X_pca"] / obsm["X_umap"] on the scRNA side — locally computed with
    scanpy, hence not reproducible bit-exactly; SpatialAIM computes its own.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python -m spatialaim.data.curate_datasets.curate_02_colorectal_cancer
    python -m spatialaim.data.curate_datasets.curate_02_colorectal_cancer --out-dir <root>

Sources are downloaded to a temporary folder below --out-dir and deleted again.
Set SPATIALAIM_KEEP_DOWNLOADS=1 to keep (and on a re-run reuse) them in
<out-dir>/_downloads/02_colorectal-cancer instead. Needs ~1 GB of free RAM.
"""

import logging
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from . import _common

logger = logging.getLogger(__name__)

NAME = "02_colorectal-cancer"

ZIP_URL = (
    "https://drive.google.com/uc?export=download&id=1odOcIfY3oqvLCNdXHLRaSmTraRxqnHLp"
)
ZIP_NAME = "CytoSPACE_example_colon_cancer_merscope.zip"
ZIP_BYTES = 8_955_055

PREFIX = "HumanColonCancerPatient2"
SC_EXPECTED = (21917, 496)
ST_EXPECTED = (52235, 500)


def _find(root: Path, filename: str) -> Path:
    """Locate a bundle member regardless of the folder nesting inside the zip."""
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"{filename} not in the bundle; it contains: "
            f"{sorted(p.name for p in root.rglob('*') if p.is_file())}"
        )
    return matches[0]


def fetch_bundle(tmp_dir: Path) -> Path:
    """Download and unpack the CytoSPACE example archive; return its folder."""
    archive = _common.download(ZIP_URL, tmp_dir / ZIP_NAME, expected_bytes=ZIP_BYTES)
    unpacked = tmp_dir / "bundle"
    if not unpacked.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(unpacked)
        logger.info("  unpacked %s", ZIP_NAME)
    return unpacked


def _read_expressions(path: Path) -> pd.DataFrame:
    """The bundle's expression TSVs are genes x cells with a GENES index column."""
    frame = pd.read_csv(path, sep="\t", index_col=0)
    logger.info("  read    %s (%d genes x %d cells)", path.name, *frame.shape)
    return frame


def _counts(frame: pd.DataFrame) -> np.ndarray:
    """genes x cells DataFrame -> contiguous cells x genes float32 array."""
    return np.ascontiguousarray(frame.to_numpy(dtype=np.float32).T)


# --------------------------------------------------------------------------- #
# scRNA
# --------------------------------------------------------------------------- #
def build_scrna(bundle: Path, out_path: Path) -> None:
    logger.info("scRNA %s — CytoSPACE bundle, Lee et al. subset", NAME)
    expressions = _read_expressions(
        _find(bundle, f"{PREFIX}_scRNA_expressions_cytospace.tsv")
    )
    annotations = pd.read_csv(
        _find(bundle, f"{PREFIX}_scRNA_annotations_cytospace.tsv"),
        sep="\t",
        index_col=0,
    )

    cells = expressions.columns.to_numpy()
    missing = set(cells) - set(annotations.index)
    if missing:
        raise ValueError(
            f"{len(missing)} cells have no annotation, e.g. {sorted(missing)[:5]}"
        )

    adata = ad.AnnData(
        X=_counts(expressions),
        obs=pd.DataFrame(
            {"cellType": pd.Categorical(annotations.loc[cells, "CellType"].to_numpy())},
            index=pd.Index(cells, name=None),
        ),
        var=pd.DataFrame(
            index=pd.Index(_common.uppercase(expressions.index), name=None)
        ),
    )
    if adata.shape != SC_EXPECTED:
        raise ValueError(
            f"expected {SC_EXPECTED}, got {adata.shape} — the bundle has changed"
        )
    logger.info("  cellType: %d categories", adata.obs["cellType"].cat.categories.size)
    _common.write(adata, out_path)


# --------------------------------------------------------------------------- #
# ST
# --------------------------------------------------------------------------- #
def build_st(bundle: Path, out_path: Path) -> None:
    logger.info("ST %s — Vizgen MERSCOPE colon crop", NAME)
    expressions = _read_expressions(
        _find(bundle, f"{PREFIX}_ST_expressions_cytospace.tsv")
    )
    coordinates = pd.read_csv(
        _find(bundle, f"{PREFIX}_ST_coordinates_cytospace.tsv"), sep="\t", index_col=0
    )

    cells = expressions.columns.to_numpy()
    missing = set(cells) - set(coordinates.index)
    if missing:
        raise ValueError(
            f"{len(missing)} cells have no coordinates, e.g. {sorted(missing)[:5]}"
        )

    adata = ad.AnnData(
        X=_counts(expressions),
        obs=pd.DataFrame(index=pd.Index(cells, name=None)),
        var=pd.DataFrame(
            index=pd.Index(_common.uppercase(expressions.index), name=None)
        ),
    )
    adata.obsm["spatial"] = (
        coordinates.loc[cells, ["X", "Y"]].to_numpy().astype(np.float32)
    )
    if adata.shape != ST_EXPECTED:
        raise ValueError(
            f"expected {ST_EXPECTED}, got {adata.shape} — the bundle has changed"
        )
    extent = adata.obsm["spatial"]
    logger.info(
        "  spatial extent x %.0f-%.0f, y %.0f-%.0f um",
        extent[:, 0].min(),
        extent[:, 0].max(),
        extent[:, 1].min(),
        extent[:, 1].max(),
    )
    _common.write(adata, out_path)


# --------------------------------------------------------------------------- #
def main() -> None:
    _common.setup_logging()
    args = _common.build_parser(__doc__).parse_args()
    sc_dir, st_dir = _common.prepare_dirs(args.out_dir)

    with _common.staging(args.out_dir, NAME) as tmp_dir:
        bundle = fetch_bundle(tmp_dir)
        build_scrna(bundle, sc_dir / f"{NAME}.h5ad")
        build_st(bundle, st_dir / f"{NAME}.h5ad")

    logger.info("%s done", NAME)


if __name__ == "__main__":
    main()
