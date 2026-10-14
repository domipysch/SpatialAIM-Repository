"""
Rebuild dataset 02_colorectal-cancer (scRNA reference + its one ST slice) from source.

Pairs covered (pairs.csv):
    PairID 2   02_colorectal-cancer  <->  02_colorectal-cancer

------------------------------------------------------------------------------
Sources: one per side
------------------------------------------------------------------------------
scRNA  the CytoSPACE example bundle for single-cell-resolution ST, published
       with the CytoSPACE paper (Vahid et al., Nat Biotechnol 2023):

  readme https://github.com/digitalcytometry/cytospace/tree/main?tab=readme-ov-file#running-cytospace-on-single-cell-st-data
  zip    https://drive.google.com/file/d/1odOcIfY3oqvLCNdXHLRaSmTraRxqnHLp/view
         -> CytoSPACE_example_colon_cancer_merscope.zip  (8.5 MB, 5 TSVs)

       Human colorectal cancer, 10x Chromium, Lee et al., Nat Genet 2020
       (https://pubmed.ncbi.nlm.nih.gov/32451460/, GEO GSE132465) — the bundle
       ships a 496-gene / 21917-cell subset of it, already annotated.

ST     the raw Vizgen release of the same sample the bundle was cropped from:
       MERSCOPE FFPE Human Immuno-Oncology showcase, HumanColonCancerPatient2,
       500-gene panel + 50 blanks, 817588 cells
       (https://info.vizgen.com/ffpe-showcase -> Colon cancer 2 -> "Access Data Set").
       The bucket gs://vz-ffpe-showcase needs a Google sign-in, so the files
       cannot be fetched by the script: it prints what to download and asks for
       the folder (or takes it from --vizgen-dir). Needed:
         HumanColonCancerPatient2_cell_by_gene.csv  (1.8 GB)
         cell_metadata.csv                          (114 MB)
         cell_boundaries/                           (17.5 GB, 2382 hdf5 files;
                                                     only the crop's FOVs are read)

Why not the bundle's ST half: its cells, coordinates and gene order are exactly
those of the Vizgen release, but CytoSPACE zeroed 1.69 M (cell, gene) entries
(per cell a median 21 % of the counts) by an undocumented rule; every other
entry is identical. Taking the raw release removes that step and keeps the
cell IDs linkable to the Vizgen segmentation (cell_boundaries/).

------------------------------------------------------------------------------
Curation
------------------------------------------------------------------------------
scRNA 02_colorectal-cancer.h5ad   (21917 x 496) — bit-exact, no filtering, no reordering
  * X    = <P>_scRNA_expressions_cytospace.tsv transposed to cells x genes, float32
  * obs  index = the TSV's column order;
           obs["cellType"] = <P>_scRNA_annotations_cytospace.tsv (10 types,
           already in the same row order)
  * var  index = the TSV's row order, uppercased (already uppercase at source)

ST 02_colorectal-cancer.h5ad      (cells x 500)
  * cells = the CytoSPACE crop, x 5000-8000, y 1500-4500 um (bounds inclusive,
           on cell_metadata center_x/center_y; 69276 cells), keeping cells with
           more than MIN_COUNTS total gene counts (CytoSPACE kept > 100: 52235)
  * X    = cell_by_gene.csv counts, blanks dropped, float32
  * obs  index = "cell_<EntityID>" (the bundle's naming), ascending EntityID;
           no obs columns
  * var  index = the panel's column order without the 50 blanks, uppercased
           (identical to the bundle's gene order)
  * obsm["spatial"] = cell_metadata center_x/center_y, float32
  * obs["segmentation_id"], uns["segmentation_pixel_size_um"] = 0.25 and the
           mask ST/Segmentation/02_colorectal-cancer.tif: each cell's boundary
           polygon (um) from cell_boundaries/feature_data_<fov>.hdf5, group
           featuredata/<EntityID>, rasterised at 0.25 um per pixel. Vizgen
           stores one polygon per z-plane (zIndex_0..6), but in this release
           all seven are the same 2D outline (asserted); zIndex_0 is used. A
           cell split into several polygons keeps its largest one.

Two things in the sources are deliberately not carried over, matching
01_Datasets_Used:
  * <P>_ST_celltypes_cytospace.tsv — the bundle's ST cell types. ST annotations
    are dropped project-wide (dataset 01 drops the osmFISH ClusterName the same way).
  * obsm["X_pca"] / obsm["X_umap"] on the scRNA side — locally computed with
    scanpy, hence not reproducible bit-exactly; SpatialAIM computes its own.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python -m spatialaim.data.curate_datasets.curate_02_colorectal_cancer
    python -m spatialaim.data.curate_datasets.curate_02_colorectal_cancer --out-dir <root>
    python -m spatialaim.data.curate_datasets.curate_02_colorectal_cancer --vizgen-dir <folder>

The bundle is downloaded to a temporary folder below --out-dir and deleted again.
Set SPATIALAIM_KEEP_DOWNLOADS=1 to keep (and on a re-run reuse) it in
<out-dir>/_downloads/02_colorectal-cancer instead. The Vizgen folder is only
read, never modified. Needs ~3 GB of free RAM.
"""

import logging
import zipfile
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

from . import _common, _segmentation

logger = logging.getLogger(__name__)

NAME = "02_colorectal-cancer"

ZIP_URL = (
    "https://drive.google.com/uc?export=download&id=1odOcIfY3oqvLCNdXHLRaSmTraRxqnHLp"
)
ZIP_NAME = "CytoSPACE_example_colon_cancer_merscope.zip"
ZIP_BYTES = 8_955_055

PREFIX = "HumanColonCancerPatient2"
SC_EXPECTED = (21917, 496)

VIZGEN_PAGE = "https://info.vizgen.com/ffpe-showcase"
VIZGEN_BUCKET = (
    "https://console.cloud.google.com/storage/browser/vz-ffpe-showcase/" + PREFIX
)
VIZGEN_GS = f"gs://vz-ffpe-showcase/{PREFIX}"
CELL_BY_GENE = f"{PREFIX}_cell_by_gene.csv"
CELL_METADATA = "cell_metadata.csv"
CELL_BOUNDARIES = "cell_boundaries"
VIZGEN_BYTES = {CELL_BY_GENE: 1_810_825_493, CELL_METADATA: 114_163_220}
VIZGEN_CELLS = 817_588
N_BLANKS = 50

# the CytoSPACE crop, in um (inclusive)
CROP_X = (5000.0, 8000.0)
CROP_Y = (1500.0, 4500.0)
CROP_CELLS = 69_276
MIN_COUNTS = 5  # keep cells with more than this many gene counts
N_GENES = 500
MASK_PIXEL_UM = 0.25


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
def _vizgen_problems(folder: Path) -> list[str]:
    """What is missing or wrong in a candidate Vizgen folder (empty = usable)."""
    if not folder.is_dir():
        return [f"{folder} is not a folder"]
    problems = []
    for name, size in VIZGEN_BYTES.items():
        path = folder / name
        if not path.is_file():
            problems.append(f"{name} not found in {folder}")
        elif path.stat().st_size != size:
            problems.append(
                f"{name} has {path.stat().st_size} bytes, expected {size} "
                "(download incomplete, or a different release)"
            )
    if not (folder / CELL_BOUNDARIES).is_dir():
        problems.append(f"{CELL_BOUNDARIES}/ not found in {folder}")
    return problems


def ask_vizgen_dir() -> Path:
    """Tell the user what to download from the sign-in-only bucket, then ask where it is."""
    print(f"""
============================================================================
 {NAME}: the ST counts come from the raw Vizgen release, which needs a
 Google sign-in and therefore cannot be downloaded by this script.

 1. Open (any Google account; accept the Cloud terms if asked):
      {VIZGEN_BUCKET}
    (also linked from {VIZGEN_PAGE} -> "Colon cancer 2" -> "Access Data Set")

 2. Download these two files into one folder:
      {CELL_BY_GENE:<45s} {VIZGEN_BYTES[CELL_BY_GENE]:>15,d} bytes
      {CELL_METADATA:<45s} {VIZGEN_BYTES[CELL_METADATA]:>15,d} bytes
    Or with the Google Cloud CLI, after `gcloud auth login`:
      gcloud storage cp {VIZGEN_GS}/{CELL_BY_GENE} {VIZGEN_GS}/{CELL_METADATA} <folder>

    And the folder with the cell segmentation (2382 files, 17.5 GB; only the
    files of the cropped FOVs are read, so a partial copy holding those works):
      {CELL_BOUNDARIES}/
    With the Google Cloud CLI:
      gcloud storage cp -r {VIZGEN_GS}/{CELL_BOUNDARIES} <folder>

    Not needed here: detected_transcripts.csv, images/.

 3. Paste the folder path below and press Enter (empty input aborts).
============================================================================""")
    while True:
        answer = input("Vizgen folder: ").strip().strip('"').strip("'")
        if not answer:
            raise SystemExit("aborted — no Vizgen folder given")
        folder = Path(answer).expanduser()
        problems = _vizgen_problems(folder)
        if not problems:
            return folder
        for problem in problems:
            print(f"  ! {problem}")
        print("  Fix this and paste the folder again.")


def _read_cell_by_gene(path: Path, keep: set[int]) -> pd.DataFrame:
    """Rows of ``keep`` from the 1.8 GB cell x gene CSV, read in chunks to bound RAM."""
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {column: np.float32 for column in header[1:]} | {header[0]: np.int64}
    parts, seen = [], 0
    for chunk in pd.read_csv(path, dtype=dtypes, chunksize=100_000):
        seen += len(chunk)
        parts.append(chunk[chunk[header[0]].isin(keep)])
    if seen != VIZGEN_CELLS:
        raise ValueError(f"{path.name}: {seen} cells, expected {VIZGEN_CELLS}")
    frame = pd.concat(parts).set_index(header[0])
    logger.info("  read    %s (%d of %d cells kept)", path.name, len(frame), seen)
    return frame


def _area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _outlines(vizgen_dir: Path, meta: pd.DataFrame) -> list[list[np.ndarray] | None]:
    """Each cell's boundary polygon (um), in meta's row order."""
    folder = vizgen_dir / CELL_BOUNDARIES
    groups = meta.groupby("fov").groups
    missing = [f for f in groups if not (folder / f"feature_data_{f}.hdf5").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} FOV file(s) of the crop are missing in {folder}, "
            f"e.g. feature_data_{missing[0]}.hdf5"
        )
    row = {entity: i for i, entity in enumerate(meta.index)}
    outlines: list[list[np.ndarray] | None] = [None] * len(meta)
    n_split = 0
    for fov, entities in groups.items():
        with h5py.File(folder / f"feature_data_{fov}.hdf5", "r") as handle:
            features = handle["featuredata"]
            for entity in entities:
                if str(entity) not in features:
                    continue
                cell = features[str(entity)]
                planes = [
                    [
                        np.asarray(part["coordinates"][0], dtype=np.float64)
                        for part in cell[key].values()
                    ]
                    for key in sorted(k for k in cell if k.startswith("zIndex_"))
                ]
                for plane in planes[1:]:
                    if len(plane) != len(planes[0]) or not all(
                        a.shape == b.shape and np.array_equal(a, b)
                        for a, b in zip(plane, planes[0])
                    ):
                        raise ValueError(
                            f"cell {entity}: the z-planes hold different outlines - "
                            "this release is not the 2D segmentation the script expects"
                        )
                parts = [ring for ring in planes[0] if len(ring) >= 3]
                if not parts:
                    continue
                n_split += len(parts) > 1
                outlines[row[entity]] = [max(parts, key=_area)]
    logger.info(
        "  outline %d of %d cells (%d split into several polygons: largest kept)",
        sum(o is not None for o in outlines),
        len(meta),
        n_split,
    )
    return outlines


def build_st(vizgen_dir: Path, st_dir: Path) -> None:
    logger.info("ST %s — Vizgen MERSCOPE %s, CytoSPACE crop", NAME, PREFIX)
    problems = _vizgen_problems(vizgen_dir)
    if problems:
        raise FileNotFoundError("; ".join(problems))

    meta = pd.read_csv(vizgen_dir / CELL_METADATA, index_col=0)
    in_crop = meta["center_x"].between(*CROP_X) & meta["center_y"].between(*CROP_Y)
    meta = meta[in_crop].sort_index()
    if len(meta) != CROP_CELLS:
        raise ValueError(f"crop holds {len(meta)} cells, expected {CROP_CELLS}")

    counts = _read_cell_by_gene(vizgen_dir / CELL_BY_GENE, set(meta.index))
    blanks = [g for g in counts.columns if g.lower().startswith("blank")]
    genes = [g for g in counts.columns if g not in blanks]
    if len(blanks) != N_BLANKS or len(genes) != N_GENES:
        raise ValueError(
            f"expected {N_GENES} genes + {N_BLANKS} blanks, got "
            f"{len(genes)} + {len(blanks)}"
        )
    counts = counts.loc[meta.index, genes]

    totals = counts.sum(axis=1)
    keep = (totals > MIN_COUNTS).to_numpy()
    logger.info(
        "  crop    %d cells, %d with > %d counts", len(meta), keep.sum(), MIN_COUNTS
    )
    meta, counts = meta[keep], counts[keep]

    adata = ad.AnnData(
        X=np.ascontiguousarray(counts.to_numpy(dtype=np.float32)),
        obs=pd.DataFrame(index=pd.Index([f"cell_{i}" for i in meta.index], name=None)),
        var=pd.DataFrame(index=pd.Index(_common.uppercase(genes), name=None)),
    )
    adata.obsm["spatial"] = meta[["center_x", "center_y"]].to_numpy(np.float32)
    extent = adata.obsm["spatial"]
    logger.info(
        "  spatial extent x %.0f-%.0f, y %.0f-%.0f um",
        extent[:, 0].min(),
        extent[:, 0].max(),
        extent[:, 1].min(),
        extent[:, 1].max(),
    )

    mask, seg_id = _segmentation.rasterize(
        _outlines(vizgen_dir, meta), MASK_PIXEL_UM, margin=5.0
    )
    _segmentation.write_mask(_segmentation.mask_path(st_dir, NAME), mask)
    _segmentation.annotate(adata, seg_id, MASK_PIXEL_UM)
    _common.write(adata, st_dir / f"{NAME}.h5ad")


# --------------------------------------------------------------------------- #
def main() -> None:
    _common.setup_logging()
    parser = _common.build_parser(__doc__)
    parser.add_argument(
        "--vizgen-dir",
        type=Path,
        default=None,
        help=f"Folder with the downloaded {CELL_BY_GENE}, {CELL_METADATA} and "
        f"{CELL_BOUNDARIES}/ (asked interactively if omitted)",
    )
    args = parser.parse_args()
    vizgen_dir = args.vizgen_dir or ask_vizgen_dir()
    sc_dir, st_dir = _common.prepare_dirs(args.out_dir)

    with _common.staging(args.out_dir, NAME) as tmp_dir:
        bundle = fetch_bundle(tmp_dir)
        build_scrna(bundle, sc_dir / f"{NAME}.h5ad")
    build_st(vizgen_dir, st_dir)

    logger.info("%s done", NAME)


if __name__ == "__main__":
    main()
