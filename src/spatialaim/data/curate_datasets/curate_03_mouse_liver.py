"""
Rebuild dataset 03_mouse-liver (scRNA reference + its one ST slice) from source.

Pairs covered (pairs.csv):
    PairID 3   03_mouse-liver  <->  03_mouse-liver   (ExperimentMatch: same_paper)

Paper: Liu et al., Life Science Alliance 2023, "Concordance of MERFISH spatial
transcriptomics with bulk and single-cell RNA sequencing"
(https://www.life-science-alliance.org/content/6/1/e202201701). Its data lives in
the figshare project "MERFISH mouse comparison study"
(https://figshare.com/projects/MERFISH_mouse_comparison_study/134213) — the
source named in the STAMapper data-availability statement, used here in place of
the STAMapper Google Drive copy.

Four downloads (~440 MB); both sides then reproduce bit-exactly.

------------------------------------------------------------------------------
scRNA  03_mouse-liver.h5ad   —  2321 x 20138, Tabula Muris Senis droplet liver
------------------------------------------------------------------------------
Counts and genes come from Tabula Muris Senis, not from the MERFISH project:
the paper reuses TMS as its 10x reference and only re-annotates it.

  tms   https://figshare.com/articles/dataset/Tabula_Muris_Senis_Data_Objects/12654728
        tabula-muris-senis-droplet-processed-official-annotations-Liver.h5ad (294 MB)
  cells https://figshare.com/articles/dataset/SingleCellData_annotated_/19310672
        symphony.zip -> symphony/MACAliver_10x_metadata.csv (2321 rows)

Curation:
  * X    = the TMS object's ``raw.X`` (raw counts, sparse CSR float32), rows
           subset to the 2321 cells named in the metadata CSV's ``cell`` column,
           **in that CSV's order** (2321 of the TMS file's 7294 liver cells)
  * var  index = ``raw.var`` gene symbols uppercased, keeping ``n_cells``
           (which counts over the whole droplet atlas, not just liver)
  * obs  index = ``cell``; the 12 TMS-native columns, except that
           ``cell_ontology_class`` is taken from the metadata CSV — Liu et al.
           relabelled the TMS classes into zonated types ("pericentral
           hepatocyte", "periportal endothelial cell", ...), and the TMS file
           still holds the original labels
  * obs["celltype"] = ``cell_ontology_class`` with the four immune classes
           (B cell, NK cell, myeloid leukocyte, plasmacytoid dendritic cell)
           merged into "immune cell" -> 12 classes become 9

The TMS h5ad is in the legacy AnnData layout (compound obs/var dtypes,
``raw.X``/``raw.var`` as top-level groups); anndata reads it with FutureWarnings
about moving ``uns['neighbors']`` into ``obsp``, which are harmless here.

------------------------------------------------------------------------------
ST  03_mouse-liver.h5ad   —  34217 x 307, MERFISH mouse liver
------------------------------------------------------------------------------
Assembled from two files of the same figshare project — the raw one carries the
counts, coordinates and segmentation metadata, the annotated one the cell types:

  raw   https://figshare.com/articles/dataset/SingleCellData_raw_/19310675
        MsLiver_Cellbound_VZG116_V1_JH_09-18-2021_FilteredSingleCellCounts.h5ad
  ann   https://figshare.com/articles/dataset/SingleCellData_annotated_/19310672
        MERFISH_liver_object.h5ad

Curation:
  * X    = the raw file's X verbatim (sparse CSR float32 counts)
  * obs  index and the 11 Vizgen columns (fov, volume, center_x/y, min/max_x/y,
           barcodeCount, average_DAPI_score, area) verbatim from the raw file
  * obs["celltype"] = the annotated object's ``free_annotation`` (8 types); the
           two files share one cell order, which is asserted before use
  * obs["celtype"]  = a coarser 6-type grouping of the above (the misspelling is
           in 01_Datasets_Used and is kept): bile duct epithelial -> duct
           epithelial, peri{central,portal} endothelial -> endothelial,
           peri{central,portal} hepatocyte -> hepatocyte
  * var  index = the **annotated** object's gene symbols, uppercased. Both files
           list the same 307 genes in the same order, but the raw file uses
           newer symbols for five of them (ADGRL4/MIR205HG/ACKR1/CAVIN2/JCHAIN
           vs ELTD1/4631405K08RIK/DARC/SDPR/IGJ); 01_Datasets_Used has the older
           set, so the annotated file is the one to follow.
  * obsm["spatial"] = the raw file's ``obsm["spatial"]``, float64, verbatim

The annotated object's own X (log-normalised) and spatial (rescaled) are *not*
used — only its labels and gene names.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python -m spatialaim.data.curate_datasets.curate_03_mouse_liver
    python -m spatialaim.data.curate_datasets.curate_03_mouse_liver --out-dir <root>

Sources are downloaded to a temporary folder below --out-dir and deleted again.
Set SPATIALAIM_KEEP_DOWNLOADS=1 to keep (and on a re-run reuse) them in
<out-dir>/_downloads/03_mouse-liver instead. Needs ~1 GB of free disk and ~3 GB
of free RAM.
"""

import logging
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from . import _common

logger = logging.getLogger(__name__)

NAME = "03_mouse-liver"

FIGSHARE = "https://ndownloader.figshare.com/files"
TMS_URL = f"{FIGSHARE}/23872763"  # tabula-muris-senis-droplet-...-Liver.h5ad
TMS_BYTES = 308_116_386
SYMPHONY_URL = f"{FIGSHARE}/37019608"  # symphony.zip
SYMPHONY_BYTES = 58_421_311
ST_RAW_URL = f"{FIGSHARE}/36879147"  # MsLiver_..._FilteredSingleCellCounts.h5ad
ST_RAW_BYTES = 19_213_228
ST_ANN_URL = f"{FIGSHARE}/36879168"  # MERFISH_liver_object.h5ad
ST_ANN_BYTES = 53_756_018

METADATA_MEMBER = "symphony/MACAliver_10x_metadata.csv"

SC_EXPECTED = (2321, 20138)
ST_EXPECTED = (34217, 307)

# The 12 TMS-native obs columns 01_Datasets_Used keeps, in its order.
SC_OBS_COLUMNS = [
    "age",
    "cell",
    "cell_ontology_class",
    "cell_ontology_id",
    "free_annotation",
    "method",
    "mouse.id",
    "n_genes",
    "sex",
    "subtissue",
    "tissue",
    "tissue_free_annotation",
]
# cell_ontology_class values folded into "immune cell" to form obs["celltype"].
SC_IMMUNE_CLASSES = frozenset(
    {"B cell", "NK cell", "myeloid leukocyte", "plasmacytoid dendritic cell"}
)
# celltype -> celtype; anything absent keeps its name.
ST_COARSE_TYPES = {
    "bile duct epithelial cell": "duct epithelial cell",
    "pericentral endothelial cell": "endothelial cell",
    "periportal endothelial cell": "endothelial cell",
    "pericentral hepatocyte": "hepatocyte",
    "periportal hepatocyte": "hepatocyte",
}


def _upper_index(names) -> pd.Index:
    return pd.Index(_common.uppercase(names), name=None)


def _check_shape(adata: ad.AnnData, expected: tuple[int, int], what: str) -> None:
    if adata.shape != expected:
        raise ValueError(
            f"{what}: expected {expected}, got {adata.shape} — the source has changed"
        )


# --------------------------------------------------------------------------- #
# scRNA
# --------------------------------------------------------------------------- #
def build_scrna(tmp_dir: Path, out_path: Path) -> None:
    logger.info(
        "scRNA %s — Tabula Muris Senis droplet liver, relabelled by Liu et al.", NAME
    )
    tms_file = _common.download(
        TMS_URL,
        tmp_dir / "tabula-muris-senis-droplet-Liver.h5ad",
        expected_bytes=TMS_BYTES,
    )
    symphony_zip = _common.download(
        SYMPHONY_URL, tmp_dir / "symphony.zip", expected_bytes=SYMPHONY_BYTES
    )

    with zipfile.ZipFile(symphony_zip) as zf:
        with zf.open(METADATA_MEMBER) as handle:
            # keep_default_na: cell_ontology_id uses the literal string "NA",
            # and several columns carry a literal "nan" category
            metadata = pd.read_csv(handle, index_col=0, keep_default_na=False)
    logger.info("  read    %s (%d cells)", METADATA_MEMBER, len(metadata))

    tms = ad.read_h5ad(tms_file)
    if tms.raw is None:
        raise ValueError("the TMS object has no .raw — raw counts are required")
    logger.info(
        "  read    TMS liver object (%d x %d, raw %d x %d)", *tms.shape, *tms.raw.shape
    )

    wanted = metadata["cell"].to_numpy()
    position = pd.Index(tms.obs["cell"].to_numpy()).get_indexer(wanted)
    if (position < 0).any():
        missing = wanted[position < 0][:5]
        raise KeyError(
            f"{int((position < 0).sum())} of {len(wanted)} cells are not in the TMS "
            f"object, e.g. {list(missing)}"
        )
    logger.info("  matched %d of the TMS file's %d liver cells", len(wanted), tms.n_obs)

    obs = tms.obs.iloc[position][SC_OBS_COLUMNS].copy()
    obs.index = pd.Index(wanted, name=None)
    # Liu et al. relabelled the TMS classes into zonated types; the TMS file
    # still carries the originals, so this column comes from the metadata CSV.
    relabelled = metadata["cell_ontology_class"].to_numpy()
    obs["cell_ontology_class"] = pd.Categorical(relabelled)
    obs["celltype"] = pd.Categorical(
        ["immune cell" if c in SC_IMMUNE_CLASSES else c for c in relabelled]
    )

    adata = ad.AnnData(
        X=tms.raw.X[position].astype(np.float32),
        obs=obs,
        var=pd.DataFrame(
            {"n_cells": tms.raw.var["n_cells"].to_numpy()},
            index=_upper_index(tms.raw.var_names),
        ),
    )
    _check_shape(adata, SC_EXPECTED, "scRNA")
    logger.info(
        "  cell_ontology_class: %d classes -> celltype: %d",
        adata.obs["cell_ontology_class"].cat.categories.size,
        adata.obs["celltype"].cat.categories.size,
    )
    _common.write(adata, out_path)


# --------------------------------------------------------------------------- #
# ST
# --------------------------------------------------------------------------- #
def build_st(tmp_dir: Path, out_path: Path) -> None:
    logger.info("ST %s — MERFISH mouse liver", NAME)
    raw_file = _common.download(
        ST_RAW_URL,
        tmp_dir / "MsLiver_FilteredSingleCellCounts.h5ad",
        expected_bytes=ST_RAW_BYTES,
    )
    ann_file = _common.download(
        ST_ANN_URL, tmp_dir / "MERFISH_liver_object.h5ad", expected_bytes=ST_ANN_BYTES
    )

    raw = ad.read_h5ad(raw_file)
    annotated = ad.read_h5ad(ann_file)
    logger.info("  read    raw %s, annotated %s", raw.shape, annotated.shape)
    if not raw.obs_names.equals(annotated.obs_names):
        raise ValueError(
            "the raw and annotated MERFISH liver files no longer share a cell order; "
            "the labels can not be transferred positionally"
        )

    fine = annotated.obs["free_annotation"].astype(str).to_numpy()
    obs = raw.obs.copy()
    obs["celtype"] = pd.Categorical([ST_COARSE_TYPES.get(c, c) for c in fine])
    obs["celltype"] = pd.Categorical(fine)

    adata = ad.AnnData(
        X=raw.X.astype(np.float32),
        obs=obs,
        # the raw file carries newer symbols for five of the 307 genes
        var=pd.DataFrame(index=_upper_index(annotated.var_names)),
    )
    adata.obsm["spatial"] = np.asarray(raw.obsm["spatial"])
    _check_shape(adata, ST_EXPECTED, "ST")
    logger.info(
        "  celltype: %d types -> celtype: %d",
        adata.obs["celltype"].cat.categories.size,
        adata.obs["celtype"].cat.categories.size,
    )
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
