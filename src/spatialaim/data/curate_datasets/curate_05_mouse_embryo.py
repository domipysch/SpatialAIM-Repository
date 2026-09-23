"""
Rebuild dataset 05_mouse-embryo (scRNA reference + its six ST slices) from source.

Pairs covered (pairs.csv):
    PairID  5   05_mouse-embryo  <->  05_1_mouse-embryo   (seqFISH, embryo1)
    PairID  6   05_mouse-embryo  <->  05_2_mouse-embryo   (seqFISH, embryo2)
    PairID  7   05_mouse-embryo  <->  05_3_mouse-embryo   (seqFISH, embryo3)
    PairID  8   05_mouse-embryo  <->  05_4_mouse-embryo   (smFISH,  embryo1)
    PairID  9   05_mouse-embryo  <->  05_5_mouse-embryo   (smFISH,  embryo2)
    PairID 10   05_mouse-embryo  <->  05_6_mouse-embryo   (smFISH,  embryo3)
    (all ExperimentMatch: same_lab)

Two Marioni-lab releases, used here in place of the STAMapper Google Drive copy:

  atlas   Pijuan-Sala et al., Nature 2019, "A single-cell molecular map of mouse
          gastrulation and early organogenesis"
          https://content.cruk.cam.ac.uk/jmlab/atlas_data/
  spatial Lohoff et al., Nature Biotechnology 2022, "Integration of spatial and
          single-cell transcriptomic data elucidates mouse organogenesis"
          https://content.cruk.cam.ac.uk/jmlab/SpatialMouseAtlas2020/

Seven downloads (~1.6 GB, almost all of it the atlas count matrix).

All three spatial files are R ``.Rds`` objects — the Lohoff release ships nothing
else (its two HDF5 files hold the imputed transcriptome, not these counts). They
are read with the pure-Python ``rdata`` package, so no R installation is needed;
it comes with the ``data`` extra (``pip install 'spatial-aim[data]'``) and is in
environment.yml. This is the only curation script that needs it.

------------------------------------------------------------------------------
Reproduction fidelity
------------------------------------------------------------------------------
Counts, cell and gene identities, and every label reproduce bit-exactly. Four
floating-point columns do not, because the public releases are not the binary
objects the shipped files were built from; the differences are 1-10 ulps and
have no effect on any analysis:

    scRNA obs["sizeFactor"]      <= 2.3e-16   (sizefactors.tab.gz is a text export)
    scRNA obs["doub.density"]    <= 1.0e-16   (meta.tab.gz is a text export)
    ST    obsm["spatial"]        <= 5.4e-15   (coordinates in mm)

------------------------------------------------------------------------------
scRNA  05_mouse-embryo.h5ad   —  16861 x 19362, mouse gastrulation atlas, E8.5
------------------------------------------------------------------------------
  counts  raw_counts.mtx.gz     (1.4 GB, 29452 genes x 139331 cells, MatrixMarket)
  meta    meta.tab.gz           (cell metadata; also holds the atlas UMAP)
  genes   genes.tsv.gz          (ENSEMBL id + symbol per matrix row)
  sf      sizefactors.tab.gz    (one size factor per cell)

Curation:
  * cells = ``stage == "E8.5"`` and not ``doublet`` and not ``stripped`` and a
           ``celltype`` that Lohoff et al. carried over (see ATLAS_TO_LOHOFF).
           That last condition drops 48 cells of five types absent from the
           spatial data (PGC, Visceral endoderm, Rostral neurectoderm, Parietal
           endoderm, Notochord): 139331 -> 16909 -> 16861.
  * obs["celltype"] = the atlas label mapped through ATLAS_TO_LOHOFF, which
           merges Erythroid1/2/3, Blood progenitors 1/2 and renames five more —
           24 atlas types become the 21 the spatial data uses.
  * obs   also carries the 15 other meta.tab columns, ``sizeFactor``, and three
           Seurat-derived ones: ``orig.ident`` (constant 0),
           ``nCount_originalexp`` and ``nFeature_originalexp`` (column sums and
           nonzero counts of the *unfiltered* matrix — gene filtering only drops
           near-empty genes, so these are the same either way).
  * var   ENSEMBL, SYMBOL, SymbolUniq (R's ``make.unique``: duplicates get
           ".1", 41 of them), plus ``n_cells``
  * genes are filtered to ``n_cells >= 3`` (29452 -> 19362), then the index is
           the uppercased SymbolUniq with "_" replaced by "-". The substitution
           matters for exactly one gene, RMST_1 -> RMST-1: the shipped file went
           through Seurat, which forbids underscores in feature names.
  * X     float64 raw counts (that dtype is what 01_Datasets_Used carries here)

------------------------------------------------------------------------------
ST  05_1..3 (seqFISH, 351 genes) and 05_4..6 (smFISH, 36 genes)
------------------------------------------------------------------------------
  metadata.Rds        cell metadata: coordinates, embryo, celltype_mapped_refined
  counts.Rds          seqFISH counts, dgCMatrix, 351 genes x 57536 cells
  smFISH_counts.Rds   smFISH counts, dense, 36 genes x 59615 cells

Curation (both panels):
  * cells = those whose ``celltype_mapped_refined`` is not "Low quality"
           (57536 -> 52568), split by embryo. The smFISH matrix covers slightly
           fewer cells, so those slices are the intersection.
  * cell order differs between the two panels, and both orders are reproduced:
           the seqFISH slices keep metadata.Rds order, the smFISH slices are
           sorted by cell id.
  * obs["orig.ident"] = the embryo index (0/1/2), not a constant
  * obs["nCount_RNA"], obs["nFeature_RNA"] = row sums and nonzero counts
  * obs["celltype"] = celltype_mapped_refined (23 types), obs["embryo"]
  * var   index = uppercased gene symbols, ``features`` = the originals
  * obsm["spatial"] = x_global_affine, y_global_affine (mm, see fidelity note)

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python -m spatialaim.data.curate_datasets.curate_05_mouse_embryo
    python -m spatialaim.data.curate_datasets.curate_05_mouse_embryo --out-dir <root>

Sources are downloaded to a temporary folder below --out-dir and deleted again.
Set SPATIALAIM_KEEP_DOWNLOADS=1 to keep (and on a re-run reuse) them in
<out-dir>/_downloads/05_mouse-embryo instead. Needs ~3 GB of free disk and ~8 GB
of free RAM.
"""

import gzip
import logging
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from . import _common

logger = logging.getLogger(__name__)

NAME = "05_mouse-embryo"

ATLAS = "https://content.cruk.cam.ac.uk/jmlab/atlas_data"
SPATIAL = "https://content.cruk.cam.ac.uk/jmlab/SpatialMouseAtlas2020"

COUNTS_URL = f"{ATLAS}/raw_counts.mtx.gz"
COUNTS_BYTES = 1_527_653_430
META_URL = f"{ATLAS}/meta.tab.gz"
META_BYTES = 4_941_002
GENES_URL = f"{ATLAS}/genes.tsv.gz"
GENES_BYTES = 228_898
SIZEFACTORS_URL = f"{ATLAS}/sizefactors.tab.gz"
SIZEFACTORS_BYTES = 1_117_878

ST_META_URL = f"{SPATIAL}/metadata.Rds"
ST_META_BYTES = 26_924_289
SEQFISH_URL = f"{SPATIAL}/counts.Rds"
SEQFISH_BYTES = 8_290_362
SMFISH_URL = f"{SPATIAL}/smFISH_counts.Rds"
SMFISH_BYTES = 1_114_194

SC_EXPECTED = (16861, 19362)
ST_EXPECTED = {
    "05_1_mouse-embryo": (17806, 351),
    "05_2_mouse-embryo": (14185, 351),
    "05_3_mouse-embryo": (20577, 351),
    "05_4_mouse-embryo": (17758, 36),
    "05_5_mouse-embryo": (14127, 36),
    "05_6_mouse-embryo": (20544, 36),
}
EMBRYOS = ["embryo1", "embryo2", "embryo3"]
LOW_QUALITY = "Low quality"
MIN_CELLS = 3

# Atlas celltype -> the label Lohoff et al. use. Types absent from this map are
# not in the spatial data and their cells are dropped.
ATLAS_TO_LOHOFF = {
    "Allantois": "Allantois",
    "Blood progenitors 1": "Blood progenitors",
    "Blood progenitors 2": "Blood progenitors",
    "Cardiomyocytes": "Cardiomyocytes",
    "Caudal Mesoderm": "Caudal Mesoderm",
    "Def. endoderm": "Definitive endoderm",
    "Endothelium": "Endothelium",
    "Erythroid1": "Erythroid",
    "Erythroid2": "Erythroid",
    "Erythroid3": "Erythroid",
    "ExE endoderm": "ExE endoderm",
    "ExE mesoderm": "Lateral plate mesoderm",
    "Forebrain/Midbrain/Hindbrain": "Forebrain/Midbrain/Hindbrain",
    "Gut": "Gut tube",
    "Haematoendothelial progenitors": "Haematoendothelial progenitors",
    "Intermediate mesoderm": "Intermediate mesoderm",
    "Mesenchyme": "Mesenchyme",
    "NMP": "NMP",
    "Neural crest": "Neural crest",
    "Paraxial mesoderm": "Paraxial mesoderm",
    "Pharyngeal mesoderm": "Splanchnic mesoderm",
    "Somitic mesoderm": "Presomitic mesoderm",
    "Spinal cord": "Spinal cord",
    "Surface ectoderm": "Surface ectoderm",
}
# meta.tab columns 01_Datasets_Used keeps, in its order; sizeFactor is appended
# from the separate file and the three Seurat columns are prepended.
SC_META_COLUMNS = [
    "cell",
    "barcode",
    "sample",
    "pool",
    "stage",
    "sequencing.batch",
    "theiler",
    "doub.density",
    "doublet",
    "cluster",
    "cluster.sub",
    "cluster.stage",
    "cluster.theiler",
    "stripped",
    "celltype",
    "colour",
]
SC_INT_COLUMNS = [
    "doublet",
    "stripped",
    "cluster",
    "cluster.sub",
    "cluster.stage",
    "cluster.theiler",
]


def _check_shape(adata: ad.AnnData, expected: tuple[int, int], what: str) -> None:
    if adata.shape != expected:
        raise ValueError(
            f"{what}: expected {expected}, got {adata.shape} — the source has changed"
        )


def _make_unique(names) -> list[str]:
    """R's make.unique: repeats get ".1", ".2", ...; the first keeps its name."""
    seen: dict[str, int] = {}
    out = []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}.{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


def _read_rds(path: Path):
    try:
        import rdata
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "curate_05_mouse_embryo needs the 'rdata' package to read the Lohoff .Rds "
            "files; install it with `pip install 'spatial-aim[data]'` or `pip install rdata`"
        ) from exc

    with warnings.catch_warnings():
        # Expected for these files: the extension is .Rds rather than .rds, and
        # rdata has no constructor for R's dgCMatrix / AsIs — it hands back the
        # raw slots, which is exactly what the callers below want.
        warnings.filterwarnings("ignore", message=".*extension.*", category=UserWarning)
        warnings.filterwarnings(
            "ignore", message=".*Unknown file type.*", category=UserWarning
        )
        warnings.filterwarnings(
            "ignore", message=".*Missing constructor.*", category=UserWarning
        )
        return rdata.conversion.convert(rdata.parser.parse_file(path))


# --------------------------------------------------------------------------- #
# scRNA
# --------------------------------------------------------------------------- #
def _read_selected_columns(path: Path, wanted: np.ndarray) -> csr_matrix:
    """Stream a gzipped MatrixMarket file (genes x cells), keeping `wanted` cells.

    The atlas matrix is 1.4 GB compressed and holds every stage; materialising it
    whole would need tens of GB, so the triplets are filtered chunk by chunk and
    only the selected columns are ever kept.
    """
    with gzip.open(path, "rt") as handle:
        line = handle.readline()
        while line.startswith("%"):
            line = handle.readline()
        n_genes, n_cells, _ = (int(value) for value in line.split())
        lookup = np.full(n_cells + 1, -1, dtype=np.int32)  # 1-based column -> row
        lookup[wanted + 1] = np.arange(len(wanted), dtype=np.int32)

        rows, cols, values = [], [], []
        for chunk in pd.read_csv(
            handle,
            sep=" ",
            header=None,
            dtype=np.int32,
            names=["gene", "cell", "count"],
            chunksize=20_000_000,
        ):
            row = lookup[chunk["cell"].to_numpy()]
            hit = row >= 0
            rows.append(row[hit])
            cols.append(chunk["gene"].to_numpy()[hit] - 1)
            values.append(chunk["count"].to_numpy()[hit])

    matrix = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(wanted), n_genes),
    ).tocsr()
    logger.info(
        "  read    %s (%d x %d, %d nonzeros)", path.name, *matrix.shape, matrix.nnz
    )
    return matrix


def build_scrna(tmp_dir: Path, out_path: Path) -> None:
    logger.info("scRNA %s — mouse gastrulation atlas, E8.5", NAME)
    meta_file = _common.download(
        META_URL, tmp_dir / "meta.tab.gz", expected_bytes=META_BYTES
    )
    genes_file = _common.download(
        GENES_URL, tmp_dir / "genes.tsv.gz", expected_bytes=GENES_BYTES
    )
    sf_file = _common.download(
        SIZEFACTORS_URL,
        tmp_dir / "sizefactors.tab.gz",
        expected_bytes=SIZEFACTORS_BYTES,
    )
    counts_file = _common.download(
        COUNTS_URL, tmp_dir / "raw_counts.mtx.gz", expected_bytes=COUNTS_BYTES
    )

    meta = pd.read_csv(meta_file, sep="\t")
    sizefactor = pd.read_csv(sf_file, header=None)[0].to_numpy()
    genes = pd.read_csv(genes_file, sep="\t", header=None, names=["ENSEMBL", "SYMBOL"])
    logger.info(
        "  read    meta.tab (%d cells), genes.tsv (%d genes)", len(meta), len(genes)
    )

    selected = (
        (meta["stage"] == "E8.5")
        & ~meta["doublet"]
        & ~meta["stripped"]
        & meta["celltype"].isin(ATLAS_TO_LOHOFF)
    ).to_numpy()
    logger.info(
        "  selected %d cells (E8.5, not doublet/stripped, celltype kept by Lohoff et al.)",
        int(selected.sum()),
    )

    X = _read_selected_columns(counts_file, np.flatnonzero(selected))

    obs = meta.loc[selected, SC_META_COLUMNS].copy()
    obs.index = pd.Index(obs["cell"].to_numpy(), name=None)
    obs.insert(0, "orig.ident", np.zeros(X.shape[0], dtype=np.int32))
    obs.insert(1, "nCount_originalexp", np.asarray(X.sum(1)).ravel().astype(np.float64))
    obs.insert(2, "nFeature_originalexp", np.diff(X.indptr).astype(np.int32))
    for column in SC_INT_COLUMNS:
        obs[column] = obs[column].to_numpy().astype(np.int32)
    obs["celltype"] = pd.Categorical([ATLAS_TO_LOHOFF[c] for c in obs["celltype"]])
    obs["sizeFactor"] = sizefactor[selected]

    n_cells = np.diff(X.tocsc().indptr)
    keep = n_cells >= MIN_CELLS
    logger.info(
        "  kept    %d of %d genes (n_cells >= %d)",
        int(keep.sum()),
        len(genes),
        MIN_CELLS,
    )
    var = pd.DataFrame(
        {
            "ENSEMBL": genes["ENSEMBL"].to_numpy()[keep],
            "SYMBOL": pd.Categorical(genes["SYMBOL"].to_numpy()[keep]),
            "SymbolUniq": np.asarray(_make_unique(genes["SYMBOL"].to_numpy()))[keep],
            "n_cells": n_cells[keep].astype(np.int32),
        },
        # "_" -> "-" because the shipped file went through Seurat, which rejects
        # underscores in feature names; this affects RMST_1 alone.
        index=pd.Index(
            [
                g.replace("_", "-")
                for g in _common.uppercase(
                    np.asarray(_make_unique(genes["SYMBOL"].to_numpy()))[keep]
                )
            ],
            name=None,
        ),
    )

    adata = ad.AnnData(X=X[:, keep].astype(np.float64), obs=obs, var=var)
    _check_shape(adata, SC_EXPECTED, "scRNA")
    logger.info("  celltype: %d types", adata.obs["celltype"].cat.categories.size)
    _common.write(adata, out_path)


# --------------------------------------------------------------------------- #
# ST
# --------------------------------------------------------------------------- #
def _slice(
    meta: pd.DataFrame, names: np.ndarray, counts, genes: np.ndarray
) -> ad.AnnData:
    X = csr_matrix(counts).astype(np.float64)
    rows = meta.loc[names]
    obs = pd.DataFrame(index=pd.Index(names, name=None))
    obs["orig.ident"] = np.asarray(
        pd.Categorical(rows["embryo"].astype(str), categories=EMBRYOS).codes,
        dtype=np.int32,
    )
    obs["nCount_RNA"] = np.asarray(X.sum(1)).ravel().astype(np.float64)
    obs["nFeature_RNA"] = np.diff(X.indptr).astype(np.int32)
    obs["celltype"] = pd.Categorical(
        rows["celltype_mapped_refined"].astype(str).to_numpy()
    )
    obs["embryo"] = pd.Categorical(rows["embryo"].astype(str).to_numpy())

    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(
            {"features": genes}, index=pd.Index(_common.uppercase(genes), name=None)
        ),
    )
    adata.obsm["spatial"] = np.column_stack(
        [rows["x_global_affine"].to_numpy(), rows["y_global_affine"].to_numpy()]
    )
    return adata


def build_st(tmp_dir: Path, st_dir: Path) -> None:
    logger.info("ST %s — seqFISH and smFISH mouse embryos", NAME)
    meta_file = _common.download(
        ST_META_URL, tmp_dir / "metadata.Rds", expected_bytes=ST_META_BYTES
    )
    seqfish_file = _common.download(
        SEQFISH_URL, tmp_dir / "counts.Rds", expected_bytes=SEQFISH_BYTES
    )
    smfish_file = _common.download(
        SMFISH_URL, tmp_dir / "smFISH_counts.Rds", expected_bytes=SMFISH_BYTES
    )

    meta = _read_rds(meta_file)
    logger.info("  read    metadata.Rds (%d cells)", len(meta))

    dgc = _read_rds(seqfish_file)  # dgCMatrix, genes x cells
    seqfish_genes = np.asarray(dgc.Dimnames[0])
    seqfish_cells = np.asarray(dgc.Dimnames[1])
    from scipy.sparse import csc_matrix

    seqfish = csc_matrix((dgc.x, dgc.i, dgc.p), shape=tuple(dgc.Dim)).T.tocsr()
    if not np.array_equal(seqfish_cells, meta.index.to_numpy()):
        raise ValueError("counts.Rds and metadata.Rds no longer share a cell order")

    dense = _read_rds(smfish_file)  # plain matrix, genes x cells
    smfish_genes = np.asarray(dense.coords["dim_0"])
    smfish_cells = np.asarray(dense.coords["dim_1"])
    smfish = np.asarray(dense.data).T
    logger.info("  read    seqFISH %s, smFISH %s", seqfish.shape, smfish.shape)

    good = meta["celltype_mapped_refined"].astype(str).to_numpy() != LOW_QUALITY
    embryo = meta["embryo"].astype(str).to_numpy()
    logger.info("  kept    %d cells (dropped %r)", int(good.sum()), LOW_QUALITY)

    # seqFISH slices: metadata order
    for number, name in enumerate(EMBRYOS, start=1):
        mask = good & (embryo == name)
        slice_name = f"05_{number}_{NAME.split('_', 1)[1]}"
        adata = _slice(meta, meta.index.to_numpy()[mask], seqfish[mask], seqfish_genes)
        _check_shape(adata, ST_EXPECTED[slice_name], slice_name)
        _common.write(adata, st_dir / f"{slice_name}.h5ad")

    # smFISH slices: sorted by cell id, and only cells the smFISH panel covers
    kept_ids = set(meta.index.to_numpy()[good])
    position = pd.Index(smfish_cells)
    for number, name in enumerate(EMBRYOS, start=4):
        names = np.sort(
            np.array(
                [c for c in smfish_cells if c in kept_ids and c.startswith(f"{name}_")]
            )
        )
        slice_name = f"05_{number}_{NAME.split('_', 1)[1]}"
        adata = _slice(meta, names, smfish[position.get_indexer(names)], smfish_genes)
        _check_shape(adata, ST_EXPECTED[slice_name], slice_name)
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
