"""
Build dataset 14_mouse-kidney-iri: mouse kidney 14 days after ischaemia-
reperfusion injury (IRI), an snRNA-seq reference + one Xenium slice, for the GUI
use case (Results section 5). Not part of the Experiment 1 corpus, so it is
written to Data/03_GUI by default.

Pair (same_lab; same model and time point, different mice):
    14_mouse-kidney-iri <-> 14_mouse-kidney-iri

------------------------------------------------------------------------------
scRNA  14_mouse-kidney-iri.h5ad   —  snRNA-seq, IRI day 14
------------------------------------------------------------------------------
  Kirita et al., PNAS 2020, "Cell profiling of acute kidney injury reveals
  conserved cellular responses to injury" (GEO GSE139107, Humphreys lab)
    GSE139107_MouseIRI_14days.dge.txt.gz   genes x nuclei counts (4 libraries)
    GSE139107_MouseIRI.metadata.txt.gz     per nucleus Group / Replicates / celltype

Curation:
  * nuclei = the 14days columns, in file order; obs index = the nucleus name
  * obs["celltype"] = the metadata's ``celltype`` verbatim (26 types, incl.
           NewPT1 = injured PT and NewPT2 = failed-repair PT, "FR-PTC")
  * X    = counts, float32; var = gene symbols uppercased, duplicates summed

------------------------------------------------------------------------------
ST  14_mouse-kidney-iri.h5ad   —  Xenium, 300-gene custom panel
------------------------------------------------------------------------------
  Humphreys lab, Nat Commun 2025, "Multimodal spatial transcriptomic
  characterization of mouse kidney injury and repair"
  (https://doi.org/10.1038/s41467-025-62599-9; GEO GSE269719)
    GSM8325623 Xenium_iri_14days_rep1: ..._xenium_day14L_male_output.tar.gz
    (5.5 GB; streamed, only cell_feature_matrix.h5, cells.csv.gz and
    cells.zarr.zip are kept)

Curation:
  * cells in the Xenium output's own order; obs index = Xenium ``cell_id``
  * cells with fewer than 10 transcripts are dropped (as for 11_human-crc)
  * X    = the ``Gene Expression`` features; var = names uppercased
  * obsm["spatial"] = ``x_centroid``, ``y_centroid`` (um), float32
  * no cell-type column (the paper's labels are on its Baysor re-segmentation)
  * obs["segmentation_id"], uns["segmentation_pixel_size_um"] = 0.2125 and the
           mask ST/Segmentation/<name>.tif: the 10x segmentation, cells.zarr.zip
           masks/1 copied as is; label = cells.csv row + 1

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python -m spatialaim.data.curate_datasets.curate_14_mouse_kidney_iri
    python -m spatialaim.data.curate_datasets.curate_14_mouse_kidney_iri --out-dir <root>

Set SPATIALAIM_KEEP_DOWNLOADS=1 to keep (and reuse) the sources in
<out-dir>/_downloads/14_mouse-kidney-iri.
"""

import logging
import shutil
import tarfile
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from . import _common, _segmentation
from .curate_09_human_breast_cancer import (
    GENE_EXPRESSION,
    MASK_PIXEL_UM,
    _check_shape,
    _copy_cell_mask,
    _fold_duplicates,
    _read_10x_h5,
)
from .curate_11_human_crc import _check_mask_order

logger = logging.getLogger(__name__)

NAME = "14_mouse-kidney-iri"
OUT_ROOT = _common.DATA_ROOT / "03_GUI"

KIRITA = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE139nnn/GSE139107/suppl"
SC_COUNTS_URL = f"{KIRITA}/GSE139107_MouseIRI_14days.dge.txt.gz"
SC_COUNTS_BYTES = 22_422_753
SC_META_URL = f"{KIRITA}/GSE139107_MouseIRI.metadata.txt.gz"
SC_META_BYTES = 785_612
SC_GROUP = "14days"

ST_TAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8325nnn/GSM8325623/suppl/"
    "GSM8325623_xenium_day14L_male_output.tar.gz"
)
ST_MEMBERS = ("cell_feature_matrix.h5", "cells.csv.gz", "cells.zarr.zip")

SC_EXPECTED = (20702, 27133)
ST_GENES = 300
N_SPOTS = 152532
N_SPOTS_KEPT = 152366
MIN_TRANSCRIPTS = 10


# --------------------------------------------------------------------------- #
# scRNA
# --------------------------------------------------------------------------- #
def build_scrna(tmp_dir: Path, out_path: Path) -> None:
    logger.info("scRNA %s — snRNA-seq IRI %s (GSE139107)", NAME, SC_GROUP)
    counts_file = _common.download(
        SC_COUNTS_URL,
        tmp_dir / Path(SC_COUNTS_URL).name,
        expected_bytes=SC_COUNTS_BYTES,
    )
    meta_file = _common.download(
        SC_META_URL, tmp_dir / Path(SC_META_URL).name, expected_bytes=SC_META_BYTES
    )
    meta = pd.read_csv(meta_file, sep="\t", index_col=0, encoding="utf-8")
    meta = meta[meta["Group"] == SC_GROUP]

    table = pd.read_csv(counts_file, sep="\t", index_col=0)
    logger.info("  read    dge (%d genes x %d nuclei)", *table.shape)
    if set(table.columns) != set(meta.index):
        raise ValueError(
            f"the {SC_GROUP} dge and metadata no longer share their nuclei"
        )

    X = csr_matrix(table.to_numpy(dtype=np.float32).T)
    X, genes = _fold_duplicates(X, np.asarray(_common.uppercase(table.index)))
    logger.info(
        "  genes   %d -> %d (duplicate symbols summed)", table.shape[0], len(genes)
    )
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "celltype": pd.Categorical(
                    meta.loc[table.columns, "celltype"].astype(str)
                )
            },
            index=pd.Index(table.columns.to_numpy(), name=None),
        ),
        var=pd.DataFrame(index=pd.Index(genes, name=None)),
    )
    if SC_EXPECTED[1]:
        _check_shape(adata, SC_EXPECTED, "scRNA")
    logger.info("  celltype: %d types", adata.obs["celltype"].cat.categories.size)
    _common.write(adata, out_path)


# --------------------------------------------------------------------------- #
# ST
# --------------------------------------------------------------------------- #
def _stream_members(
    url: str, names: tuple[str, ...], dest_dir: Path
) -> dict[str, Path]:
    """Keep only ``names`` (matched by file name) while streaming the remote tar.gz."""
    out = {name: dest_dir / name for name in names}
    todo = {name for name in names if not out[name].exists()}
    if not todo:
        logger.info("  cached  %s", ", ".join(names))
        return out
    logger.info("  stream  %s", url)
    request = urllib.request.Request(
        url, headers={"User-Agent": "spatialaim-curate/1.0"}
    )
    with (
        urllib.request.urlopen(request, timeout=600) as response,
        tarfile.open(fileobj=response, mode="r|gz") as archive,
    ):
        for member in archive:
            name = member.name.rsplit("/", 1)[-1]
            if not member.isfile() or name not in todo:
                continue
            tmp = out[name].with_suffix(out[name].suffix + ".part")
            src = archive.extractfile(member)
            if src is None:  # not reached: member.isfile() was checked
                raise ValueError(f"cannot extract {member.name}")
            with src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)
            tmp.replace(out[name])
            logger.info("  extract %s (%.0f MB)", name, member.size / 1e6)
            todo.discard(name)
            if not todo:
                break
    if todo:
        raise FileNotFoundError(f"not in the archive: {sorted(todo)}")
    return out


def build_st(tmp_dir: Path, st_dir: Path) -> None:
    logger.info("ST %s — Xenium mouse kidney IRI day 14, rep 1 (GSE269719)", NAME)
    files = _stream_members(ST_TAR_URL, ST_MEMBERS, tmp_dir)
    matrix, barcodes, names, types = _read_10x_h5(files["cell_feature_matrix.h5"])
    genes = types == GENE_EXPRESSION
    logger.info(
        "  read    matrix (%d x %d, %d gene features)", *matrix.shape, int(genes.sum())
    )

    cells = pd.read_csv(files["cells.csv.gz"])
    if not np.array_equal(cells["cell_id"].astype(str).to_numpy(), barcodes):
        raise ValueError("cells.csv.gz and the matrix no longer share a cell order")
    adata = ad.AnnData(
        X=matrix[genes].T.tocsr().astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(barcodes, name=None)),
        var=pd.DataFrame(index=pd.Index(_common.uppercase(names[genes]), name=None)),
    )
    adata.obsm["spatial"] = np.column_stack(
        [cells["x_centroid"].to_numpy(), cells["y_centroid"].to_numpy()]
    ).astype(np.float32)
    if N_SPOTS:
        _check_shape(adata, (N_SPOTS, ST_GENES), "ST")

    masks_file = files["cells.zarr.zip"]
    _check_mask_order(masks_file, barcodes)
    _copy_cell_mask(masks_file, _segmentation.mask_path(st_dir, NAME), adata.n_obs)
    _segmentation.annotate(adata, np.arange(1, adata.n_obs + 1), MASK_PIXEL_UM)

    keep = np.asarray(adata.X.sum(axis=1)).ravel() >= MIN_TRANSCRIPTS
    adata = adata[keep].copy()
    logger.info(
        "  cells   %d -> %d (>= %d transcripts)",
        len(keep),
        adata.n_obs,
        MIN_TRANSCRIPTS,
    )
    if N_SPOTS_KEPT:
        _check_shape(adata, (N_SPOTS_KEPT, ST_GENES), "ST")
    _common.write(adata, st_dir / f"{NAME}.h5ad")


# --------------------------------------------------------------------------- #
def main() -> None:
    _common.setup_logging()
    parser = _common.build_parser(__doc__)
    parser.set_defaults(out_dir=OUT_ROOT)
    args = parser.parse_args()
    sc_dir, st_dir = _common.prepare_dirs(args.out_dir)
    with _common.staging(args.out_dir, NAME) as tmp_dir:
        build_scrna(tmp_dir, sc_dir / f"{NAME}.h5ad")
        build_st(tmp_dir, st_dir)
    logger.info("%s done", NAME)


if __name__ == "__main__":
    main()
