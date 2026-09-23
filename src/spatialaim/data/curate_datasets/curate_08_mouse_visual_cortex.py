"""
Rebuild dataset 08_mouse-visual-cortex (scRNA reference + its one ST slice).

Pairs covered (pairs.csv): 08_mouse-visual-cortex <-> 08_mouse-visual-cortex ("-").

  scRNA  Tasic et al., Nature 2018, "Shared and distinct transcriptomic cell
         types across neocortical areas"
         (https://www.nature.com/articles/s41586-018-0654-5)
  ST     Wang et al., Science 2018, "Three-dimensional intact-tissue sequencing
         of single-cell transcriptional states" (STARmap)
         (https://doi.org/10.1126/science.aat5691)

Note on the reference: 01_Datasets_Used's index credits this dataset to Hodge et
al. (s41586-019-1506-7), which is the **human MTG** paper behind dataset 10. The
h5ad is unambiguously the Allen mouse V1/ALM SMART-seq data of Tasic et al.
2018 — its cell ids are Allen SMART-seq names and its 23 subclasses are Tasic's.

------------------------------------------------------------------------------
One ST slice, not five — the STARmap original is gone
------------------------------------------------------------------------------
STARmap's own distribution is unreachable: starmapresources.org is down and the
Dropbox folder it names has been deleted; github.com/weallen/STARmap is code
only. Of 01_Datasets_Used's five slices (three 160-gene, one 1020-gene, one
882-gene) the only one with a public source is the 1020-gene experiment, through
a third-party Zenodo copy. Its gene panel is exactly the shipped 08_4's 1020
genes (and the 882 of 08_5 are a subset of them), but it holds the unfiltered
1207 cells rather than 930, so it is not the shipped slice — accepted by
decision, as for dataset 06.

------------------------------------------------------------------------------
scRNA  08_mouse-visual-cortex.h5ad   —  14249 x 34041
------------------------------------------------------------------------------
  counts   https://brain-map.org/atlases-and-data/rnaseq/mouse-v1-and-alm-smart-seq
           mouse_VISp_gene_expression_matrices_2018-06-14.zip (292 MB)
  metadata https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115746
           GSE115746_complete_metadata_28706-cells.csv.gz

The counts come from the Allen release rather than GEO: GEO's
``GSE115746_cells_exon_counts.csv.gz`` covers only 23178 of the 28706 cells and
is missing 663 of the ones needed here.

Curation:
  * cells = ``dissected_region == "VISp"`` whose ``cell_subclass`` is one of the
           23 real subclasses, **sorted by name**. That drops 1403 of the 15652
           VISp cells — the QC categories (No Class, Low Quality, High Intronic,
           Batch Grouping, Doublet and blank) — leaving 14249.
  * X    = the release's **exon** matrix (introns are not used), float32
  * obs["subclass"] = ``cell_subclass`` with "/" -> "and" and spaces removed,
           so "L2/3 IT" becomes "L2and3IT" (23 subclasses)
  * obs["celltype"] = those 23 mapped onto 12 (SUBCLASS_TO_CELLTYPE)
  * var  index = the release's gene symbols through R's ``make.names`` and then
           uppercased. make.names matters: 1404 symbols start with a digit and
           gain an "X" prefix, and 853 contain a character it rewrites to "."
           (``Actg-ps1`` -> ``ACTG.PS1``).
  * genes are filtered to ``n_cells >= 10`` over these 14249 cells, 45768 ->
           34041

------------------------------------------------------------------------------
ST  08_mouse-visual-cortex.h5ad   —  1207 x 1020
------------------------------------------------------------------------------
  https://zenodo.org/records/10698912  ->  STARmap_mouse_visual_cortex.zip
  containing STARmap_20180505_BY3_1k.h5ad

Curation:
  * X    = the source X verbatim (integer counts) as sparse CSR float32
  * var  index = the source gene symbols uppercased
  * obs["label"] = the source's spatial domain (L1-L6, CC, HPC). No cell-type
           column is produced: this dataset does not need one, and the source
           carries none. The source's X/Y (identical to the coordinates) and
           Total_counts (the row sums) are dropped as redundant.
  * obsm["spatial"] = the source's ``obsm["spatial"]``

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python -m spatialaim.data.curate_datasets.curate_08_mouse_visual_cortex
    python -m spatialaim.data.curate_datasets.curate_08_mouse_visual_cortex --out-dir <root>

Sources are downloaded to a temporary folder below --out-dir and deleted again.
Set SPATIALAIM_KEEP_DOWNLOADS=1 to keep (and on a re-run reuse) them in
<out-dir>/_downloads/08_mouse-visual-cortex instead. Needs ~2 GB of free RAM.
"""

import logging
import re
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack

from . import _common

logger = logging.getLogger(__name__)

NAME = "08_mouse-visual-cortex"

ALLEN_URL = "https://celltypes.brain-map.org/api/v2/well_known_file_download/694413985"
ALLEN_BYTES = 305_703_326
METADATA_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115746/suppl"
    "/GSE115746_complete_metadata_28706-cells.csv.gz"
)
METADATA_BYTES = 501_614
ST_URL = "https://zenodo.org/api/records/10698912/files/STARmap_mouse_visual_cortex.zip/content"
ST_BYTES = 471_002

EXON_MEMBER = "mouse_VISp_2018-06-14_exon-matrix.csv"
GENES_MEMBER = "mouse_VISp_2018-06-14_genes-rows.csv"
ST_MEMBER = "STARmap_mouse_visual_cortex/STARmap_20180505_BY3_1k.h5ad"

SC_EXPECTED = (14249, 34041)
ST_EXPECTED = (1207, 1020)
MIN_CELLS = 10
CHUNK_ROWS = 2000
REGION = "VISp"

# Tasic's 23 subclasses (after normalisation) -> the 12 types of
# 01_Datasets_Used. Cells whose cell_subclass is not a key here are the QC
# categories and are dropped.
SUBCLASS_TO_CELLTYPE = {
    "Astro": "Astro",
    "CR": "CR",
    "Endo": "Endo",
    "L2and3IT": "ExcitatoryL2and3",
    "L4": "ExcitatoryL4",
    "L5IT": "ExcitatoryL5and6",
    "L5PT": "ExcitatoryL5and6",
    "L6CT": "ExcitatoryL5and6",
    "L6IT": "ExcitatoryL5and6",
    "L6b": "ExcitatoryL5and6",
    "NP": "ExcitatoryL5and6",
    "Lamp5": "Inhibitory",
    "Meis2": "Inhibitory",
    "Pvalb": "Inhibitory",
    "Serpinf1": "Inhibitory",
    "Sncg": "Inhibitory",
    "Sst": "Inhibitory",
    "Vip": "Inhibitory",
    "Macrophage": "Micro",
    "Oligo": "Oligo",
    "Peri": "Peri",
    "SMC": "Smc",
    "VLMC": "VLMC",
}


def _check_shape(adata: ad.AnnData, expected: tuple[int, int], what: str) -> None:
    if adata.shape != expected:
        raise ValueError(
            f"{what}: expected {expected}, got {adata.shape} — the source has changed"
        )


def _make_names(name: str) -> str:
    """R's ``make.names``: rewrite invalid characters, X-prefix a leading digit."""
    out = re.sub(r"[^A-Za-z0-9._]", ".", str(name))
    return f"X{out}" if out[:1].isdigit() else out


def _normalise_subclass(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.replace("/", "and", regex=False)
        .str.replace(" ", "", regex=False)
    )


# --------------------------------------------------------------------------- #
# scRNA
# --------------------------------------------------------------------------- #
def build_scrna(tmp_dir: Path, out_path: Path) -> None:
    logger.info("scRNA %s — Allen mouse VISp SMART-seq (Tasic et al. 2018)", NAME)
    metadata_file = _common.download(
        METADATA_URL,
        tmp_dir / "GSE115746_complete_metadata.csv.gz",
        expected_bytes=METADATA_BYTES,
    )
    allen_zip = _common.download(
        ALLEN_URL,
        tmp_dir / "mouse_VISp_gene_expression_matrices_2018-06-14.zip",
        expected_bytes=ALLEN_BYTES,
    )

    metadata = pd.read_csv(metadata_file, index_col=0, low_memory=False)
    subclass = _normalise_subclass(metadata["cell_subclass"])
    selected = (metadata["dissected_region"] == REGION) & subclass.isin(
        SUBCLASS_TO_CELLTYPE
    )
    wanted = sorted(metadata.index[selected])
    logger.info(
        "  selected %d of the %d %s cells (dropping the QC subclasses)",
        len(wanted),
        int((metadata["dissected_region"] == REGION).sum()),
        REGION,
    )

    with zipfile.ZipFile(allen_zip) as zf:
        with zf.open(GENES_MEMBER) as handle:
            genes = pd.read_csv(handle)
        with zf.open(EXON_MEMBER) as handle:
            header = [
                c.strip('"') for c in handle.readline().decode().rstrip("\n").split(",")
            ][1:]
        position = pd.Index(header).get_indexer(wanted)
        if (position < 0).any():
            missing = np.asarray(wanted)[position < 0][:5]
            raise KeyError(
                f"{int((position < 0).sum())} cells are not in {EXON_MEMBER}, e.g. {list(missing)}"
            )

        blocks: list[csr_matrix] = []
        keys: list[int] = []
        counts: list[np.ndarray] = []
        with zf.open(EXON_MEMBER) as handle:
            for chunk in pd.read_csv(handle, index_col=0, chunksize=CHUNK_ROWS):
                keys.extend(chunk.index.tolist())
                values = chunk.to_numpy(dtype=np.float32)[:, position]
                counts.append((values != 0).sum(1))
                blocks.append(csr_matrix(values))

    if keys != genes["gene_entrez_id"].tolist():
        raise ValueError(f"{EXON_MEMBER}'s rows are not in {GENES_MEMBER}'s order")
    X = vstack(blocks).T.tocsr()
    n_cells = np.concatenate(counts)
    logger.info("  read    exon matrix (%d x %d, %d nonzeros)", *X.shape, X.nnz)

    keep = n_cells >= MIN_CELLS
    logger.info(
        "  kept    %d of %d genes (n_cells >= %d)",
        int(keep.sum()),
        len(keys),
        MIN_CELLS,
    )

    chosen = subclass.loc[wanted]
    obs = pd.DataFrame(index=pd.Index(wanted, name=None))
    obs["subclass"] = pd.Categorical(chosen.to_numpy())
    obs["celltype"] = pd.Categorical([SUBCLASS_TO_CELLTYPE[s] for s in chosen])

    adata = ad.AnnData(
        X=X[:, keep].astype(np.float32),
        obs=obs,
        var=pd.DataFrame(
            index=pd.Index(
                _common.uppercase(
                    _make_names(s) for s in genes["gene_symbol"].to_numpy()[keep]
                ),
                name=None,
            )
        ),
    )
    _check_shape(adata, SC_EXPECTED, "scRNA")
    logger.info(
        "  subclass: %d -> celltype: %d",
        adata.obs["subclass"].cat.categories.size,
        adata.obs["celltype"].cat.categories.size,
    )
    _common.write(adata, out_path)


# --------------------------------------------------------------------------- #
# ST
# --------------------------------------------------------------------------- #
def build_st(tmp_dir: Path, out_path: Path) -> None:
    logger.info("ST %s — STARmap 1020-gene visual cortex, via the Zenodo copy", NAME)
    archive = _common.download(
        ST_URL, tmp_dir / "STARmap_mouse_visual_cortex.zip", expected_bytes=ST_BYTES
    )
    with zipfile.ZipFile(archive) as zf:
        zf.extract(ST_MEMBER, tmp_dir)
    source = ad.read_h5ad(tmp_dir / ST_MEMBER)
    logger.info("  read    %s (%d x %d)", Path(ST_MEMBER).name, *source.shape)

    values = np.asarray(source.X)
    if not np.array_equal(values, np.round(values)):
        raise ValueError("the source X is not integral — it is not a raw count matrix")

    obs = pd.DataFrame(index=source.obs_names)
    # X/Y duplicate obsm["spatial"] and Total_counts is the row sums, so only the
    # spatial domain is carried over. No cell types: this dataset needs none.
    obs["label"] = pd.Categorical(source.obs["label"].astype(str).to_numpy())

    adata = ad.AnnData(
        X=csr_matrix(values, dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(
            index=pd.Index(_common.uppercase(source.var_names), name=None)
        ),
    )
    adata.obsm["spatial"] = np.asarray(source.obsm["spatial"])
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
