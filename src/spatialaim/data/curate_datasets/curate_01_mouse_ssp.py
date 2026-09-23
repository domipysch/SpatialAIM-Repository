"""
Rebuild dataset 01_mouse-ssp (scRNA reference + its one ST slice) from source.

Pairs covered (pairs.csv):
    PairID 0   01_mouse-ssp  <->  01_mouse-ssp

------------------------------------------------------------------------------
scRNA  01_mouse-ssp.h5ad   —  mouse primary somatosensory cortex, SMART-seq v4
------------------------------------------------------------------------------
Allen Institute "Mouse Whole Cortex and Hippocampus" taxonomy, Yao et al.,
Cell 2021 (https://pubmed.ncbi.nlm.nih.gov/34004146/), deposited as GEO
GSE185862. This is the reference the DOT / TACCO benchmarks pair with osmFISH.

  series   https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE185862
  matrix   https://ftp.ncbi.nlm.nih.gov/geo/series/GSE185nnn/GSE185862/suppl/GSE185862_expression_matrix_SSv4.hdf5
  metadata https://ftp.ncbi.nlm.nih.gov/geo/series/GSE185nnn/GSE185862/suppl/GSE185862_metadata_ssv4.csv.gz

(The FTP paths are used instead of the GEO web page, which is behind a bot check.)

Curation:
  * keep the cells with ``region_label == "SSp"``   -> 5665 of 73347 cells
  * X = raw SSv4 read counts, dense float32, 45768 genes
  * obs index = ``sample_name``
    obs["cellType"]      = ``class_label``     (3 categories)
    obs["cellTypeMinor"] = ``subclass_label``  (22 categories)
  * var index = gene symbols, uppercased

``01_Datasets_Used`` additionally carries obsm["X_pca"]/["X_umap"]. They were
computed locally with scanpy (they are *not* GSE185862_umap2d_ssv4.csv.gz —
that file uses different sample names and a global embedding), so they cannot
be reproduced bit-exactly. They are deliberately dropped here; SpatialAIM
computes its own embeddings.

------------------------------------------------------------------------------
ST  01_mouse-ssp.h5ad   —  osmFISH, mouse somatosensory cortex
------------------------------------------------------------------------------
Codeluppi et al., Nat Methods 2018 (https://www.nature.com/articles/s41592-018-0175-z),
Linnarsson lab. Reproduces bit-exactly.

  page   http://linnarssonlab.org/osmFISH/availability/
  loom   http://linnarssonlab.org/osmFISH/osmFISH_SScortex_mouse_all_cells.loom
  mirror https://github.com/drieslab/spatial-datasets/tree/master/data/2018_osmFISH_SScortex/raw_data

Curation:
  * keep the cells with ``Valid == 1``  -> 4839 of 6471 (equivalently
    Region != "Excluded"); loom column order is preserved
  * X = ``matrix`` transposed to cells x genes, float32 raw counts
  * obs index = ``CellID``; no obs columns (ClusterName/Region are dropped,
    the ST side carries no annotation)
  * var index = ``row_attrs/Gene``, uppercased
  * obsm["spatial"] = ``col_attrs/X``, ``col_attrs/Y`` verbatim, no shift
    (the valid subset already starts at 0/0)

Note: 476 of the 4839 cells share their coordinates with another cell. That is
a property of the source loom and is kept.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python -m spatialaim.data.curate_datasets.curate_01_mouse_ssp
    python -m spatialaim.data.curate_datasets.curate_01_mouse_ssp --out-dir <root>

Sources are downloaded to a temporary folder below --out-dir and deleted again;
only scRNA/01_mouse-ssp.h5ad and ST/01_mouse-ssp.h5ad remain. Needs ~4.4 GB of
free disk and ~2 GB of free RAM while the scRNA side is built.

Set SPATIALAIM_KEEP_DOWNLOADS=1 to keep (and on a re-run reuse) the sources in
<out-dir>/_downloads/01_mouse-ssp instead — for debugging without refetching.
"""

import logging
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

from . import _common

logger = logging.getLogger(__name__)

NAME = "01_mouse-ssp"

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE185nnn/GSE185862/suppl"
SC_MATRIX_URL = f"{GEO_BASE}/GSE185862_expression_matrix_SSv4.hdf5"
SC_METADATA_URL = f"{GEO_BASE}/GSE185862_metadata_ssv4.csv.gz"
SC_MATRIX_BYTES = 3_506_000_000  # ~3.3 GB, only a fallback if HEAD reports nothing
SC_METADATA_BYTES = 2_250_150

ST_LOOM_URL = "http://linnarssonlab.org/osmFISH/osmFISH_SScortex_mouse_all_cells.loom"
ST_LOOM_BYTES = 1_120_947

SC_REGION = "SSp"
SC_EXPECTED_CELLS = 5665
SC_EXPECTED_GENES = 45768
ST_EXPECTED_CELLS = 4839
ST_EXPECTED_GENES = 33

# Log progress every N cells while gathering the sparse columns.
_PROGRESS_EVERY = 1000


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _decode(array: np.ndarray) -> np.ndarray:
    """HDF5 byte strings -> python str."""
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in array])


def _describe(handle: h5py.File) -> str:
    """Every dataset in the file with its shape/dtype — for error messages."""
    lines: list[str] = []
    handle.visititems(
        lambda name, obj: lines.append(
            f"    {name}: {obj.shape} {obj.dtype}"
            if isinstance(obj, h5py.Dataset)
            else f"    {name}/ (group)"
        )
    )
    return "\n".join(lines)


def _locate(handle: h5py.File, candidates: tuple[str, ...], what: str) -> str:
    """First candidate path that exists and is a dataset."""
    for key in candidates:
        if key in handle and isinstance(handle[key], h5py.Dataset):
            return key
    raise KeyError(
        f"could not find the {what} dataset (tried {candidates!r}). File contains:\n"
        + _describe(handle)
    )


def _sample_major_group(handle: h5py.File, n_samples: int) -> h5py.Group:
    """
    The dgCMatrix group whose *columns are samples*.

    GSE185862 stores the SSv4 counts as an R dgCMatrix (`i`/`p`/`x`/`dims`) in
    both orientations: `data/exon` is column-per-gene, `data/t_exon`
    column-per-sample. Only the latter lets 5665 cells be read as 5665
    contiguous slices instead of scanning all 45768 gene columns, so it is
    identified by its `p` length rather than by name.
    """
    for key in ("data/t_exon", "data/exon"):
        group = handle.get(key)
        if (
            isinstance(group, h5py.Group)
            and "p" in group
            and group["p"].shape[0] == n_samples + 1
        ):
            return group
    raise KeyError(
        f"no dgCMatrix group with one column per sample (p of length {n_samples + 1}). "
        "File contains:\n" + _describe(handle)
    )


def _read_cells(group: h5py.Group, columns: np.ndarray, n_genes: int) -> np.ndarray:
    """Densify the given dgCMatrix columns into a (len(columns), n_genes) float32 array."""
    indptr = group["p"][:].astype(np.int64)
    indices, values = group["i"], group["x"]
    nnz = indices.shape[0]
    if indptr[0] != 0 or indptr[-1] != nnz:
        raise ValueError(
            f"p is not a valid CSC pointer: p[0]={indptr[0]}, p[-1]={indptr[-1]}, nnz={nnz}"
        )

    out = np.zeros((len(columns), n_genes), dtype=np.float32)
    for row, column in enumerate(columns):
        start, stop = indptr[column], indptr[column + 1]
        if stop > start:
            gene_rows = indices[start:stop].astype(np.int64)
            # R's dgCMatrix stores 0-based indices; a 1-based file would overflow here
            if gene_rows.min() < 0 or gene_rows.max() >= n_genes:
                raise ValueError(
                    f"gene index out of range in column {column}: "
                    f"[{gene_rows.min()}, {gene_rows.max()}] for {n_genes} genes"
                )
            out[row, gene_rows] = values[start:stop]
        if (row + 1) % _PROGRESS_EVERY == 0:
            logger.info("          cells %d / %d", row + 1, len(columns))
    return out


# --------------------------------------------------------------------------- #
# scRNA
# --------------------------------------------------------------------------- #
def build_scrna(tmp_dir: Path, out_path: Path) -> None:
    logger.info("scRNA %s — GEO GSE185862 (Allen SMART-seq v4)", NAME)
    matrix_file = _common.download(
        SC_MATRIX_URL,
        tmp_dir / "GSE185862_expression_matrix_SSv4.hdf5",
        expected_bytes=SC_MATRIX_BYTES,
    )
    metadata_file = _common.download(
        SC_METADATA_URL,
        tmp_dir / "GSE185862_metadata_ssv4.csv.gz",
        expected_bytes=SC_METADATA_BYTES,
    )

    metadata = pd.read_csv(metadata_file, low_memory=False)
    selected = metadata.loc[metadata["region_label"] == SC_REGION].copy()
    logger.info("  %s cells with region_label == %r", len(selected), SC_REGION)
    if len(selected) != SC_EXPECTED_CELLS:
        raise ValueError(
            f"expected {SC_EXPECTED_CELLS} {SC_REGION} cells, got {len(selected)} — "
            "GSE185862 metadata has changed"
        )
    wanted = selected["sample_name"].to_numpy()

    with h5py.File(matrix_file, "r") as handle:
        gene_key = _locate(handle, ("gene_names", "data/gene", "genes"), "gene name")
        sample_key = _locate(
            handle, ("sample_names", "data/samples", "samples"), "sample name"
        )
        genes = _decode(handle[gene_key][:])
        samples = _decode(handle[sample_key][:])
        n_genes, n_samples = len(genes), len(samples)
        logger.info("  matrix holds %d genes x %d samples", n_genes, n_samples)
        if n_genes != SC_EXPECTED_GENES:
            raise ValueError(f"expected {SC_EXPECTED_GENES} genes, got {n_genes}")

        # The matrix is keyed by `sample_name`; `exp_component_name` is the only
        # other identifier in the metadata that could plausibly be used instead.
        matrix_index = pd.Index(samples)
        for column in ("sample_name", "exp_component_name"):
            if column not in selected.columns:
                continue
            position = matrix_index.get_indexer(selected[column].to_numpy())
            if not (position < 0).any():
                logger.info("  matrix columns keyed by %r", column)
                break
        else:
            raise KeyError(
                "none of the metadata identifiers match the matrix column names.\n"
                f"    matrix sample_names: {list(samples[:3])}\n"
                f"    metadata sample_name: {list(wanted[:3])}"
            )

        group = _sample_major_group(handle, n_samples)
        logger.info("  reading %s (%d nonzeros total)", group.name, group["i"].shape[0])

        # Read the columns in ascending file order — sequential slices keep the
        # HDF5 chunk cache useful — then restore the metadata order.
        order = np.argsort(position, kind="stable")
        restore = np.argsort(order, kind="stable")
        X = _read_cells(group, position[order], n_genes)[restore]

        totals = handle.get("data/total_exon_counts")
        if totals is not None:
            expected = totals[:][position]
            agreement = np.isclose(X.sum(axis=1), expected).mean()
            if agreement < 1.0:
                logger.warning(
                    "  row sums match data/total_exon_counts for only %.1f%% of cells",
                    100 * agreement,
                )
            else:
                logger.info("  row sums match data/total_exon_counts for all cells")

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "cellType": pd.Categorical(selected["class_label"].to_numpy()),
                "cellTypeMinor": pd.Categorical(selected["subclass_label"].to_numpy()),
            },
            index=pd.Index(wanted, name=None),
        ),
        var=pd.DataFrame(index=pd.Index(_common.uppercase(genes), name=None)),
    )
    logger.info(
        "  cellType: %d categories, cellTypeMinor: %d categories",
        adata.obs["cellType"].cat.categories.size,
        adata.obs["cellTypeMinor"].cat.categories.size,
    )
    _common.write(adata, out_path)


# --------------------------------------------------------------------------- #
# ST
# --------------------------------------------------------------------------- #
def build_st(tmp_dir: Path, out_path: Path) -> None:
    logger.info("ST %s — osmFISH SScortex (Linnarsson lab)", NAME)
    loom_file = _common.download(
        ST_LOOM_URL,
        tmp_dir / "osmFISH_SScortex_mouse_all_cells.loom",
        expected_bytes=ST_LOOM_BYTES,
    )

    with h5py.File(loom_file, "r") as handle:
        genes = _decode(handle["row_attrs/Gene"][:])
        cell_id = _decode(handle["col_attrs/CellID"][:])
        valid = handle["col_attrs/Valid"][:] == 1
        x_coord = handle["col_attrs/X"][:]
        y_coord = handle["col_attrs/Y"][:]
        counts = handle["matrix"][:]  # (genes, cells), ~1 MB

    logger.info("  %d of %d cells with Valid == 1", int(valid.sum()), valid.size)
    if int(valid.sum()) != ST_EXPECTED_CELLS or len(genes) != ST_EXPECTED_GENES:
        raise ValueError(
            f"expected {ST_EXPECTED_CELLS} x {ST_EXPECTED_GENES}, "
            f"got {int(valid.sum())} x {len(genes)} — the loom has changed"
        )

    adata = ad.AnnData(
        X=counts[:, valid].T.astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(cell_id[valid], name=None)),
        var=pd.DataFrame(index=pd.Index(_common.uppercase(genes), name=None)),
    )
    adata.obsm["spatial"] = np.column_stack([x_coord[valid], y_coord[valid]]).astype(
        np.float32
    )

    duplicates = pd.DataFrame(adata.obsm["spatial"]).duplicated().sum()
    logger.info(
        "  %d cells share coordinates with another cell (kept, as in the source)",
        duplicates,
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
