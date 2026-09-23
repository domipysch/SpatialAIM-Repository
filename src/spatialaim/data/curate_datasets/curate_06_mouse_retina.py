"""
Rebuild dataset 06_mouse-retina (scRNA reference + its 72 ST slices) from source.

Pairs covered (pairs.csv): one per ST slice, 72 in total, ExperimentMatch "-".

  scRNA  Hoang et al., Science 2020, "Gene regulatory networks controlling
         vertebrate retinal regeneration" (https://doi.org/10.1126/science.abb8598)
  ST     Choi et al., Nature Communications 2023
         (https://www.nature.com/articles/s41467-023-40674-3)

------------------------------------------------------------------------------
Two deliberate departures from 01_Datasets_Used
------------------------------------------------------------------------------
1. The ST slices are split by ``sampleid`` x ``region`` as the Zenodo record
   defines them, not by eye from the coordinates. That yields **72** slices
   rather than 71: the old split had merged DV1's regions 6 and 7 into one.
   Slice n of sample s is ``06_<s>_<n>``, with n = region + 1 (region is
   0-based upstream) and s numbered in the order below.

2. The scRNA counts come from scRetinaDB and **differ from the shipped file**.
   Hoang et al.'s data-availability statement names three sources, and none of
   them serves the raw counts: GEO GSE135406 holds only bulk RNA-seq and ATAC;
   the authors' GitHub repository publishes the cell and gene feature tables but
   states the count matrix "will be recovered soon"; ProteinPaint serves an FPKM
   matrix. The only public counts are scRetinaDB's re-processed Seurat object,
   which has 18528 genes (not 27998) and drops 284 of the 15256 cells, and whose
   counts differ from the shipped file for 27 genes. Accepted by decision.

------------------------------------------------------------------------------
scRNA  06_mouse-retina.h5ad   —  14972 x 18528
------------------------------------------------------------------------------
  counts   https://casapp.dnayun.com/scretina/  (Download page, accession
           GitHub-jiewwwang) -> PMID33004674_LD.qs.gz, 69 MB
  features https://github.com/jiewwwang/Single-cell-retinal-regeneration
           -> Mouse_LD_cell_features.tsv

The counts are an R `qs`-serialised Seurat v5 object. `qs` has no Python reader,
so that one step runs in R via dump_retina_counts.R in a small conda env (see
environment_retina_qs.yml); everything else is Python.

Curation:
  * cells = the object's ``status == "mmP"`` — the four undamaged P60 replicates
           (14972 of its 40169 cells); the light-damage time course is dropped
  * X     = the object's ``assays$RNA@layers$counts``, cells x genes, float32
  * var   index = its gene symbols uppercased (unique as they stand, no suffixes)
  * obs   the object carries no cell-type annotation at all, so the labels come
           from the GitHub feature table, joined on the barcode:
             Sample              the replicate, "P60 R1".."P60 R4"
             Cell.type           the paper's 17 types
             cell_ontology_class 12 types, mapped from Cell.type (CELL_ONTOLOGY)
             celltype            11 types, the above with the glial classes merged

------------------------------------------------------------------------------
ST  06_<sample>_<slice>_mouse-retina.h5ad   —  72 slices
------------------------------------------------------------------------------
  https://zenodo.org/records/8144355
    VA45_integrated.h5ad             500 genes, samples DV1-3 and TN1-3
    VZA105a_integrated_368genes.h5ad 368 genes, samples VZG105a_WT1-4

Curation, per (sampleid, region) group:
  * X     = the source X verbatim (dense float32 in the release), as sparse CSR
  * obs   the source's own columns verbatim — the two releases differ (the 500-
           gene one also has dataset, barcodeCount and batch) — plus
           ``celltype``, mapped from ``majorclass`` (MAJOR_CLASS, 7 types)
  * var   index = the source gene symbols uppercased
  * obsm["spatial"] = ``center_x``, ``center_y``

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python -m spatialaim.data.curate_datasets.curate_06_mouse_retina
    python -m spatialaim.data.curate_datasets.curate_06_mouse_retina --out-dir <root>

Needs the R env once:  conda env create -f environment_retina_qs.yml
Sources are downloaded to a temporary folder below --out-dir and deleted again.
Set SPATIALAIM_KEEP_DOWNLOADS=1 to keep (and on a re-run reuse) them in
<out-dir>/_downloads/06_mouse-retina instead.
"""

import gzip
import logging
import os
import shutil
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix

from . import _common

logger = logging.getLogger(__name__)

NAME = "06_mouse-retina"

QS_URL = (
    "https://casapp.dnayun.com/scretina/api/download_rds"
    "?species=mouse&file=PMID33004674_LD.qs.gz&platform=10X%20Genomics%20scRNA"
)
QS_BYTES = 72_407_277
FEATURES_URL = (
    "https://raw.githubusercontent.com/jiewwwang/Single-cell-retinal-regeneration"
    "/master/Mouse_LD_cell_features.tsv"
)
FEATURES_BYTES = 4_173_331

ZENODO = "https://zenodo.org/records/8144355/files"
VA45_URL = f"{ZENODO}/VA45_integrated.h5ad?download=1"
VA45_BYTES = 28_661_380
VZA105A_URL = f"{ZENODO}/VZA105a_integrated_368genes.h5ad?download=1"
VZA105A_BYTES = 10_262_251

R_ENV = "retina_qs_env"
R_SCRIPT = "dump_retina_counts.R"

SC_EXPECTED = (14972, 18528)
VA45_EXPECTED = (253_984, 500)
VZA105A_EXPECTED = (113_385, 368)

# ST sample number -> sampleid; VA45's six first, then VZA105a's four.
ST_SAMPLES = [
    "DV1",
    "DV2",
    "DV3",
    "TN1",
    "TN2",
    "TN3",
    "VZG105a_WT1",
    "VZG105a_WT2",
    "VZG105a_WT3",
    "VZG105a_WT4",
]

# Hoang et al.'s 17 types -> cell ontology (12).
CELL_ONTOLOGY = {
    "Activated MG": "macroglial cell",
    "Astrocytes": "macroglial cell",
    "Cones": "retinal cone cell",
    "GABAergic AC": "amacrine cell",
    "Glycinergic AC": "amacrine cell",
    "HC": "retina horizontal cell",
    "Microglia": "microglial cell",
    "OFF cone BC": "retinal bipolar neuron",
    "ON cone BC": "retinal bipolar neuron",
    "Pericytes": "pericyte cell",
    "RGC": "retinal ganglion cell",
    "RPE": "retinal pigment epithelial cell",
    "Resting MG": "macroglial cell",
    "Reticulocytes": "reticulocyte",
    "Rod BC": "retinal bipolar neuron",
    "Rods": "retinal rod cell",
    "V/E cells": "blood vessel endothelial cell",
}
# ...and the two glial ontology classes merged, giving obs["celltype"] (11).
GLIAL_CLASSES = frozenset({"macroglial cell", "microglial cell"})

# ST majorclass -> celltype.
MAJOR_CLASS = {
    "AC": "amacrine cell",
    "BC": "retinal bipolar neuron",
    "Cone": "retinal cone cell",
    "HC": "retina horizontal cell",
    "MG": "glial cell",
    "RGC": "retinal ganglion cell",
    "Rod": "retinal rod cell",
}

# Cells per slice, so a changed upstream fails loudly.
ST_EXPECTED_CELLS = {
    "06_1_1_mouse-retina": 341,
    "06_1_2_mouse-retina": 1912,
    "06_1_3_mouse-retina": 1067,
    "06_1_4_mouse-retina": 2083,
    "06_1_5_mouse-retina": 1204,
    "06_1_6_mouse-retina": 2163,
    "06_1_7_mouse-retina": 434,
    "06_1_8_mouse-retina": 1638,
    "06_2_1_mouse-retina": 7865,
    "06_2_2_mouse-retina": 8903,
    "06_2_3_mouse-retina": 5469,
    "06_2_4_mouse-retina": 7838,
    "06_2_5_mouse-retina": 8909,
    "06_2_6_mouse-retina": 8472,
    "06_2_7_mouse-retina": 7193,
    "06_3_1_mouse-retina": 7586,
    "06_3_2_mouse-retina": 7657,
    "06_3_3_mouse-retina": 6232,
    "06_3_4_mouse-retina": 6139,
    "06_3_5_mouse-retina": 7794,
    "06_3_6_mouse-retina": 9938,
    "06_3_7_mouse-retina": 4295,
    "06_4_1_mouse-retina": 2981,
    "06_4_2_mouse-retina": 3126,
    "06_4_3_mouse-retina": 2954,
    "06_4_4_mouse-retina": 8545,
    "06_4_5_mouse-retina": 1283,
    "06_4_6_mouse-retina": 1434,
    "06_4_7_mouse-retina": 248,
    "06_5_1_mouse-retina": 7488,
    "06_5_2_mouse-retina": 3252,
    "06_5_3_mouse-retina": 3642,
    "06_5_4_mouse-retina": 6200,
    "06_5_5_mouse-retina": 5960,
    "06_5_6_mouse-retina": 4989,
    "06_5_7_mouse-retina": 4628,
    "06_6_1_mouse-retina": 12163,
    "06_6_2_mouse-retina": 9672,
    "06_6_3_mouse-retina": 12304,
    "06_6_4_mouse-retina": 12611,
    "06_6_5_mouse-retina": 10929,
    "06_6_6_mouse-retina": 12591,
    "06_6_7_mouse-retina": 11852,
    "06_7_1_mouse-retina": 2674,
    "06_7_2_mouse-retina": 143,
    "06_7_3_mouse-retina": 2215,
    "06_7_4_mouse-retina": 1702,
    "06_7_5_mouse-retina": 4269,
    "06_8_1_mouse-retina": 4379,
    "06_8_2_mouse-retina": 6398,
    "06_8_3_mouse-retina": 6111,
    "06_8_4_mouse-retina": 6895,
    "06_8_5_mouse-retina": 6020,
    "06_8_6_mouse-retina": 9050,
    "06_8_7_mouse-retina": 6849,
    "06_8_8_mouse-retina": 6678,
    "06_8_9_mouse-retina": 4166,
    "06_9_1_mouse-retina": 4499,
    "06_9_2_mouse-retina": 4849,
    "06_9_3_mouse-retina": 2129,
    "06_9_4_mouse-retina": 1408,
    "06_9_5_mouse-retina": 762,
    "06_9_6_mouse-retina": 3050,
    "06_9_7_mouse-retina": 2429,
    "06_10_1_mouse-retina": 638,
    "06_10_2_mouse-retina": 6510,
    "06_10_3_mouse-retina": 301,
    "06_10_4_mouse-retina": 4941,
    "06_10_5_mouse-retina": 4527,
    "06_10_6_mouse-retina": 2380,
    "06_10_7_mouse-retina": 5123,
    "06_10_8_mouse-retina": 2290,
}


def _check_shape(adata: ad.AnnData, expected: tuple[int, int], what: str) -> None:
    if adata.shape != expected:
        raise ValueError(
            f"{what}: expected {expected}, got {adata.shape} — the source has changed"
        )


# --------------------------------------------------------------------------- #
# the one R step
# --------------------------------------------------------------------------- #
def _conda_env_prefix(name: str) -> Path | None:
    """Prefix of the conda env called ``name``, or None if it is not installed."""
    from ...reference_aligners.registry import conda_exe

    try:
        exe = conda_exe()
    except RuntimeError:
        return None
    try:
        proc = subprocess.run([exe, "env", "list", "--json"], capture_output=True)
    except OSError:
        return None
    import json

    try:
        data = json.loads((proc.stdout or b"").decode("utf-8", errors="replace"))
    except ValueError:
        return None
    for path in data.get("envs", []):
        prefix = Path(path)
        if prefix.name == name:
            return prefix
    return None


def _rscript() -> tuple[str, dict[str, str]]:
    """The Rscript to call and the environment to call it with.

    Prefers the dedicated ``retina_qs_env``. Its R needs the env's own DLL
    directories on PATH — ``conda run`` is not used because it fails on some
    Windows installs. Falls back to an ``Rscript`` already on PATH."""
    prefix = _conda_env_prefix(R_ENV)
    if prefix is not None:
        if os.name == "nt":
            exe = prefix / "lib" / "R" / "bin" / "x64" / "Rscript.exe"
            dirs = [
                prefix,
                prefix / "Library" / "bin",
                prefix / "Library" / "mingw-w64" / "bin",
                exe.parent,
            ]
        else:
            exe = prefix / "lib" / "R" / "bin" / "Rscript"
            dirs = [prefix / "bin", exe.parent]
        if exe.is_file():
            env = dict(os.environ)
            env["PATH"] = os.pathsep.join(
                [str(d) for d in dirs] + [env.get("PATH", "")]
            )
            return str(exe), env

    fallback = shutil.which("Rscript")
    if fallback:
        logger.warning("conda env %r not found — using %s from PATH", R_ENV, fallback)
        return fallback, dict(os.environ)

    raise RuntimeError(
        f"no R available: create the env with `conda env create -f environment_retina_qs.yml` "
        f"(next to this script), or put an Rscript with the 'qs' package on PATH"
    )


def _dump_counts(
    qs_file: Path, work_dir: Path
) -> tuple[csr_matrix, list[str], list[str]]:
    """Run the R dumper and read back the counts it wrote."""
    exe, env = _rscript()
    script = Path(__file__).resolve().parent / R_SCRIPT
    logger.info("  R       %s", exe)
    proc = subprocess.run(
        [exe, str(script), str(qs_file), str(work_dir)],
        capture_output=True,
        text=True,
        env=env,
    )
    for line in (proc.stdout or "").splitlines():
        logger.info("  R       %s", line.rstrip())
    if proc.returncode != 0:
        raise RuntimeError(
            f"{R_SCRIPT} failed (exit {proc.returncode}):\n{proc.stderr}"
        )

    cells = (work_dir / "cells.txt").read_text().split()
    genes = (work_dir / "genes.txt").read_text().split()
    counts = csr_matrix(mmread(work_dir / "counts.mtx").T).astype(np.float32)
    return counts, cells, genes


# --------------------------------------------------------------------------- #
# scRNA
# --------------------------------------------------------------------------- #
def build_scrna(tmp_dir: Path, out_path: Path) -> None:
    logger.info("scRNA %s — Hoang et al. undamaged P60 retina, via scRetinaDB", NAME)
    archive = _common.download(
        QS_URL, tmp_dir / "PMID33004674_LD.qs.gz", expected_bytes=QS_BYTES
    )
    features_file = _common.download(
        FEATURES_URL,
        tmp_dir / "Mouse_LD_cell_features.tsv",
        expected_bytes=FEATURES_BYTES,
    )

    qs_file = tmp_dir / "PMID33004674_LD.qs"
    if not qs_file.exists():
        logger.info("  unzip   %s", archive.name)
        with gzip.open(archive, "rb") as src, open(qs_file, "wb") as dst:
            shutil.copyfileobj(src, dst)

    X, cells, genes = _dump_counts(qs_file, tmp_dir)

    features = pd.read_csv(features_file, sep="\t")
    features = features.rename(columns={features.columns[0]: "Barcode"}).set_index(
        "Barcode"
    )
    missing = [c for c in cells if c not in features.index]
    if missing:
        raise KeyError(
            f"{len(missing)} of {len(cells)} cells are not in the feature table, "
            f"e.g. {missing[:5]}"
        )
    rows = features.loc[cells]

    cell_type = rows["Cell.type"].astype(str).to_numpy()
    unmapped = sorted(set(cell_type) - set(CELL_ONTOLOGY))
    if unmapped:
        raise KeyError(f"CELL_ONTOLOGY does not cover {unmapped}")
    ontology = np.array([CELL_ONTOLOGY[c] for c in cell_type])

    obs = pd.DataFrame(index=pd.Index(cells, name=None))
    obs["Sample"] = pd.Categorical(rows["Sample"].astype(str).to_numpy())
    obs["Cell.type"] = pd.Categorical(cell_type)
    obs["cell_ontology_class"] = pd.Categorical(ontology)
    obs["celltype"] = pd.Categorical(
        ["glial cell" if o in GLIAL_CLASSES else o for o in ontology]
    )

    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=_common.uppercase(genes)))
    adata.var_names.name = None
    _check_shape(adata, SC_EXPECTED, "scRNA")
    logger.info(
        "  Cell.type: %d -> cell_ontology_class: %d -> celltype: %d",
        adata.obs["Cell.type"].cat.categories.size,
        adata.obs["cell_ontology_class"].cat.categories.size,
        adata.obs["celltype"].cat.categories.size,
    )
    _common.write(adata, out_path)


# --------------------------------------------------------------------------- #
# ST
# --------------------------------------------------------------------------- #
def _split_slices(source: ad.AnnData, st_dir: Path) -> int:
    sampleid = source.obs["sampleid"].astype(str).to_numpy()
    region = source.obs["region"].to_numpy().astype(int)
    genes = pd.Index(_common.uppercase(source.var_names), name=None)

    written = 0
    for sid in source.obs["sampleid"].cat.categories:
        if sid not in ST_SAMPLES:
            raise KeyError(f"unknown sampleid {sid!r} — the release has changed")
        number = ST_SAMPLES.index(sid) + 1
        for value in sorted(set(region[sampleid == sid].tolist())):
            mask = (sampleid == sid) & (region == value)
            subset = source[mask]
            # region is 0-based upstream; slices are numbered from 1
            slice_name = f"06_{number}_{value + 1}_{NAME.split('_', 1)[1]}"

            major = subset.obs["majorclass"].astype(str).to_numpy()
            unmapped = sorted(set(major) - set(MAJOR_CLASS))
            if unmapped:
                raise KeyError(f"MAJOR_CLASS does not cover {unmapped}")
            obs = subset.obs.copy()
            obs["celltype"] = pd.Categorical([MAJOR_CLASS[m] for m in major])

            adata = ad.AnnData(
                X=csr_matrix(np.asarray(subset.X), dtype=np.float32),
                obs=obs,
                var=pd.DataFrame(index=genes),
            )
            adata.obsm["spatial"] = np.column_stack(
                [subset.obs["center_x"].to_numpy(), subset.obs["center_y"].to_numpy()]
            )
            expected = ST_EXPECTED_CELLS.get(slice_name)
            if expected is None:
                raise KeyError(
                    f"unexpected slice {slice_name!r} — the release has changed"
                )
            _check_shape(adata, (expected, source.n_vars), slice_name)
            _common.write(adata, st_dir / f"{slice_name}.h5ad")
            written += 1
    return written


def build_st(tmp_dir: Path, st_dir: Path) -> None:
    logger.info("ST %s — MERFISH mouse retina, split by sampleid x region", NAME)
    va45 = _common.download(
        VA45_URL, tmp_dir / "VA45_integrated.h5ad", expected_bytes=VA45_BYTES
    )
    vza105a = _common.download(
        VZA105A_URL,
        tmp_dir / "VZA105a_integrated_368genes.h5ad",
        expected_bytes=VZA105A_BYTES,
    )

    written = 0
    for path, expected in ((va45, VA45_EXPECTED), (vza105a, VZA105A_EXPECTED)):
        source = ad.read_h5ad(path)
        _check_shape(source, expected, path.name)
        logger.info("  read    %s (%d x %d)", path.name, *source.shape)
        written += _split_slices(source, st_dir)

    if written != len(ST_EXPECTED_CELLS):
        raise ValueError(f"wrote {written} slices, expected {len(ST_EXPECTED_CELLS)}")
    logger.info("  wrote   %d slices", written)


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
