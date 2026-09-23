"""
Rebuild dataset 04_mouse-kidney (scRNA reference + its one ST slice) from source.

Pairs covered (pairs.csv):
    PairID 4   04_mouse-kidney  <->  04_mouse-kidney   (ExperimentMatch: same_paper)

Paper: Liu et al., Life Science Alliance 2023, "Concordance of MERFISH spatial
transcriptomics with bulk and single-cell RNA sequencing"
(https://www.life-science-alliance.org/content/6/1/e202201701). Its data lives in
the figshare project "MERFISH mouse comparison study"
(https://figshare.com/projects/MERFISH_mouse_comparison_study/134213) — the
source named in the STAMapper data-availability statement, used here in place of
the STAMapper Google Drive copy.

The kidney counterpart of curate_03_mouse_liver: same project, same 307-gene
MERFISH panel, same Tabula Muris Senis reference. Five downloads (~1.1 GB); both
sides then reproduce bit-exactly.

------------------------------------------------------------------------------
scRNA  04_mouse-kidney.h5ad   —  2327 x 20138, Tabula Muris Senis droplet kidney
------------------------------------------------------------------------------
Counts and genes come from Tabula Muris Senis, not from the MERFISH project:
the paper reuses TMS as its 10x reference and only re-annotates it.

  tms     https://figshare.com/articles/dataset/Tabula_Muris_Senis_Data_Objects/12654728
          tabula-muris-senis-droplet-processed-official-annotations-Kidney.h5ad (741 MB)
  cells   https://figshare.com/articles/dataset/SingleCellData_annotated_/19310672
          symphony.zip -> symphony/MACAkidney_10x_metadata.csv (2330 rows)
  relabel https://figshare.com/articles/dataset/Mouse_kidney_cell_type_relabeling_map/20775424
          mousekidney_celltype_relabeling.csv (the paper's own MACA -> ontology map)

Curation:
  * obs  comes wholly from the metadata CSV (unlike 03, where it came from TMS):
           its 12 columns minus ``n_counts``, index = ``cell``. Its ``n_genes``
           column is empty throughout, which is why 01_Datasets_Used carries an
           all-NaN float64 ``n_genes`` — reproduced by coercing the empty strings.
  * obs["celltype"] = the CSV's 22 MACA classes pushed through the published
           relabeling map and then coarsened once more (the map's own targets
           are finer than the shipped file): lymphocyte / macrophage /
           leukocyte / plasma cell -> "immune cell", glomerular capillary
           endothelial cell / kidney blood vessel cell -> "endothelial cell".
           22 classes become 9.
           01_Datasets_Used also carries a ``cell_ontology_class`` column that
           is byte-for-byte identical to ``celltype``; the duplicate is dropped.
  * the 3 cells the map leaves as "kidney cell" (an unusable catch-all) are
           dropped, which is exactly the CSV's 2330 -> the shipped 2327.
  * X    = the TMS object's ``raw.X`` (raw counts, sparse CSR float32), rows
           subset to those 2327 cells **in the metadata CSV's order**
  * var  index = ``raw.var`` gene symbols uppercased, keeping ``n_cells``
           (which counts over the whole droplet atlas, not just kidney)

The TMS h5ad is in the legacy AnnData layout (compound obs/var dtypes,
``raw.X``/``raw.var`` as top-level groups); anndata reads it with FutureWarnings
about moving ``uns['neighbors']`` into ``obsp``, which are harmless here.

------------------------------------------------------------------------------
ST  04_mouse-kidney.h5ad   —  126547 x 307, MERFISH mouse kidney
------------------------------------------------------------------------------
Assembled from two files of the same figshare project — the raw one carries the
counts, coordinates and segmentation metadata, the annotated one the cell types:

  raw   https://figshare.com/articles/dataset/SingleCellData_raw_/19310675
        MsKidney_CellBoundary_VZG116_111921_FilteredSingleCellCounts.h5ad
  ann   https://figshare.com/articles/dataset/SingleCellData_annotated_/19310672
        MERFISH_kidney_object.h5ad

Curation:
  * X    = the raw file's X verbatim (sparse CSR float32 counts)
  * obs  index and the 11 Vizgen columns (fov, volume, center_x/y, min/max_x/y,
           barcodeCount, average_DAPI_score, area) verbatim from the raw file
  * obs["celltype"] = the annotated object's ``free_annotation`` (8 types,
           already the final names); the two files share one cell order, which
           is asserted before use. There is no second, coarser column here —
           01_Datasets_Used has no ``celtype`` for kidney.
  * var  index = the **annotated** object's gene symbols, uppercased. Both files
           list the same 307 genes in the same order, but the raw file uses
           newer symbols for five of them (ADGRL4/MIR205HG/ACKR1/CAVIN2/JCHAIN
           vs ELTD1/4631405K08RIK/DARC/SDPR/IGJ); 01_Datasets_Used has the older
           set, so the annotated file is the one to follow (same five as 03).
  * obsm["spatial"] = the raw file's ``obsm["spatial"]``, float64, verbatim

The annotated object's own X (log-normalised), embeddings and spatial (rescaled)
are *not* used — only its labels and gene names.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python -m spatialaim.data.curate_datasets.curate_04_mouse_kidney
    python -m spatialaim.data.curate_datasets.curate_04_mouse_kidney --out-dir <root>

Sources are downloaded to a temporary folder below --out-dir and deleted again.
Set SPATIALAIM_KEEP_DOWNLOADS=1 to keep (and on a re-run reuse) them in
<out-dir>/_downloads/04_mouse-kidney instead. Needs ~1.5 GB of free disk and
~6 GB of free RAM (the TMS kidney object is the largest source in the set).
"""

import logging
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from . import _common

logger = logging.getLogger(__name__)

NAME = "04_mouse-kidney"

FIGSHARE = "https://ndownloader.figshare.com/files"
TMS_URL = f"{FIGSHARE}/23873024"  # tabula-muris-senis-droplet-...-Kidney.h5ad
TMS_BYTES = 776_671_418
SYMPHONY_URL = f"{FIGSHARE}/37019608"  # symphony.zip
SYMPHONY_BYTES = 58_421_311
RELABEL_URL = f"{FIGSHARE}/37020649"  # mousekidney_celltype_relabeling.csv
RELABEL_BYTES = 1_171
ST_RAW_URL = f"{FIGSHARE}/36879141"  # MsKidney_..._FilteredSingleCellCounts.h5ad
ST_RAW_BYTES = 70_053_108
ST_ANN_URL = f"{FIGSHARE}/36879162"  # MERFISH_kidney_object.h5ad
ST_ANN_BYTES = 195_688_374

METADATA_MEMBER = "symphony/MACAkidney_10x_metadata.csv"
RELABEL_COLUMN = "Cell ontology relabel"

SC_EXPECTED = (2327, 20138)
ST_EXPECTED = (126547, 307)

# The metadata columns carried over, in 01_Datasets_Used's order. The CSV's
# n_counts is dropped, and so is cell_ontology_class — it is rebuilt below as
# obs["celltype"], and 01_Datasets_Used holds the two as identical copies.
SC_OBS_COLUMNS = [
    "age",
    "cell",
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
# Applied on top of the published relabeling map, whose targets are still finer
# than the nine classes 01_Datasets_Used carries.
SC_COARSE_TYPES = {
    "lymphocyte": "immune cell",
    "macrophage": "immune cell",
    "leukocyte": "immune cell",
    "plasma cell": "immune cell",
    "glomerular capillary endothelial cell": "endothelial cell",
    "kidney blood vessel cell": "endothelial cell",
}
# The map's catch-all target; those cells are not in 01_Datasets_Used.
SC_DROPPED_TYPE = "kidney cell"


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
        "scRNA %s — Tabula Muris Senis droplet kidney, relabelled by Liu et al.", NAME
    )
    tms_file = _common.download(
        TMS_URL,
        tmp_dir / "tabula-muris-senis-droplet-Kidney.h5ad",
        expected_bytes=TMS_BYTES,
    )
    symphony_zip = _common.download(
        SYMPHONY_URL, tmp_dir / "symphony.zip", expected_bytes=SYMPHONY_BYTES
    )
    relabel_file = _common.download(
        RELABEL_URL,
        tmp_dir / "mousekidney_celltype_relabeling.csv",
        expected_bytes=RELABEL_BYTES,
    )

    with zipfile.ZipFile(symphony_zip) as zf:
        with zf.open(METADATA_MEMBER) as handle:
            # keep_default_na: cell_ontology_id uses the literal string "NA",
            # and several columns carry a literal "nan" category
            metadata = pd.read_csv(handle, index_col=0, keep_default_na=False)
    logger.info("  read    %s (%d cells)", METADATA_MEMBER, len(metadata))

    relabel = pd.read_csv(relabel_file, index_col=0)[RELABEL_COLUMN].to_dict()
    classes = metadata["cell_ontology_class"].to_numpy()
    unmapped = sorted({c for c in classes if c not in relabel})
    if unmapped:
        raise KeyError(f"the relabeling map no longer covers {unmapped}")
    labels = np.array([SC_COARSE_TYPES.get(relabel[c], relabel[c]) for c in classes])

    keep = labels != SC_DROPPED_TYPE
    logger.info("  dropped %d %r cells", int((~keep).sum()), SC_DROPPED_TYPE)
    metadata, labels = metadata[keep], labels[keep]

    tms = ad.read_h5ad(tms_file)
    if tms.raw is None:
        raise ValueError("the TMS object has no .raw — raw counts are required")
    logger.info(
        "  read    TMS kidney object (%d x %d, raw %d x %d)", *tms.shape, *tms.raw.shape
    )

    wanted = metadata["cell"].to_numpy()
    position = pd.Index(tms.obs["cell"].to_numpy()).get_indexer(wanted)
    if (position < 0).any():
        missing = wanted[position < 0][:5]
        raise KeyError(
            f"{int((position < 0).sum())} of {len(wanted)} cells are not in the TMS "
            f"object, e.g. {list(missing)}"
        )
    logger.info(
        "  matched %d of the TMS file's %d kidney cells", len(wanted), tms.n_obs
    )

    obs = metadata[SC_OBS_COLUMNS].copy()
    obs.index = pd.Index(wanted, name=None)
    # the CSV's n_genes is empty throughout; 01_Datasets_Used keeps it as NaN
    obs["n_genes"] = pd.to_numeric(metadata["n_genes"], errors="coerce").astype(
        "float64"
    )
    obs["celltype"] = pd.Categorical(labels)

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
        "  %d MACA classes -> %d cell_ontology_class / celltype",
        len(set(classes)),
        adata.obs["celltype"].cat.categories.size,
    )
    _common.write(adata, out_path)


# --------------------------------------------------------------------------- #
# ST
# --------------------------------------------------------------------------- #
def build_st(tmp_dir: Path, out_path: Path) -> None:
    logger.info("ST %s — MERFISH mouse kidney", NAME)
    raw_file = _common.download(
        ST_RAW_URL,
        tmp_dir / "MsKidney_FilteredSingleCellCounts.h5ad",
        expected_bytes=ST_RAW_BYTES,
    )
    ann_file = _common.download(
        ST_ANN_URL, tmp_dir / "MERFISH_kidney_object.h5ad", expected_bytes=ST_ANN_BYTES
    )

    raw = ad.read_h5ad(raw_file)
    annotated = ad.read_h5ad(ann_file)
    logger.info("  read    raw %s, annotated %s", raw.shape, annotated.shape)
    if not raw.obs_names.equals(annotated.obs_names):
        raise ValueError(
            "the raw and annotated MERFISH kidney files no longer share a cell order; "
            "the labels can not be transferred positionally"
        )

    obs = raw.obs.copy()
    obs["celltype"] = pd.Categorical(
        annotated.obs["free_annotation"].astype(str).to_numpy()
    )

    adata = ad.AnnData(
        X=raw.X.astype(np.float32),
        obs=obs,
        # the raw file carries newer symbols for five of the 307 genes
        var=pd.DataFrame(index=_upper_index(annotated.var_names)),
    )
    adata.obsm["spatial"] = np.asarray(raw.obsm["spatial"])
    _check_shape(adata, ST_EXPECTED, "ST")
    logger.info("  celltype: %d types", adata.obs["celltype"].cat.categories.size)
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
