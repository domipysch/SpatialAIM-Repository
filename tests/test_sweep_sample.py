"""End-to-end SpatialAIM sweep on the bundled sample dataset.

Marked ``slow``: these run the full sweep (scanpy/squidpy) and so only run where
``spatialaim_env`` and the sample data are present — the CI integration job, not the
fast unit run. Heavy imports happen inside the test bodies, so importing this
module during collection stays light.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[1]
SC = REPO_ROOT / "sample_dataset" / "scRNA" / "sample_sc.h5ad"
ST = REPO_ROOT / "sample_dataset" / "ST" / "sample_st.h5ad"

_needs_sample = pytest.mark.skipif(
    not (SC.exists() and ST.exists()), reason="sample dataset not available"
)


def _assert_run_layout(mapping_dir):
    """The on-disk layout every sweep writes, whatever the start clusters were."""
    import pandas as pd

    assert (mapping_dir / "config.yaml").is_file()
    assert (mapping_dir.parent / "start_clustering.h5ad").is_file()
    assert (mapping_dir / "k_comparison.csv").is_file()
    # The spatial criterion is the chance-corrected assortativity, bounded by 1.
    table = pd.read_csv(mapping_dir / "k_comparison.csv")
    for column in ("spatial_assort_global", "spatial_assort_celltype"):
        values = table.loc[table["k"] >= 2, column]
        assert values.notna().all() and (values <= 1).all()

    # Per-K output folders (k_NNN); exclude the k_comparison.csv summary file.
    k_dirs = sorted(p for p in mapping_dir.glob("k_[0-9]*") if p.is_dir())
    assert k_dirs, "no per-K output folders were written"
    for k_dir in k_dirs:
        assert (k_dir / "start_cluster_to_state.csv").is_file()
        assert (k_dir / "spot_to_state_mapping_soft.h5ad").is_file()
        analysis = k_dir / "analysis"
        assert (analysis / "cossim_metrics.json").is_file()
        assert not (analysis / "data").exists() and not (analysis / "cossim").exists()
    return k_dirs


@_needs_sample
def test_sweep_writes_expected_outputs(tmp_path):
    """Run a small nearest_centroid sweep and check the on-disk layout."""
    from spatialaim.config import SpatialAIMConfig
    from spatialaim.cli import run_one_pair

    out = tmp_path / "run"
    run_one_pair(
        SC, ST, out, SpatialAIMConfig(mapping="nearest_centroid", k_min=1, k_max=2)
    )

    _assert_run_layout(out / "leiden" / "nearest_centroid")
    assert (out / "reference_scaffold.h5ad").is_file()
    # No annotation given: the Leiden start only.
    assert [p.name for p in out.iterdir() if p.is_dir()] == ["leiden"]


@_needs_sample
def test_sweep_with_annotation_runs_both_starts(tmp_path):
    """--annotation adds an annotation start next to the Leiden one.

    Both share one reference scaffold. The annotation start's clusters are exactly
    the annotated ``cellType`` values -- one per type, named after it -- and each
    start's config.yaml records which start it is.
    """
    import anndata as ad
    import yaml

    from spatialaim.config import SpatialAIMConfig
    from spatialaim.cli import run_one_pair

    cell_types = sorted(set(ad.read_h5ad(SC).obs["cellType"].astype(str)))

    out = tmp_path / "run"
    run_one_pair(
        SC,
        ST,
        out,
        SpatialAIMConfig(mapping="nearest_centroid", k_min=1, annotations=["cellType"]),
    )

    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == [
        "annotation_cellType",
        "leiden",
    ]
    assert (out / "reference_scaffold.h5ad").is_file()
    assert not list(out.glob("*/reference_scaffold.h5ad"))

    leiden_dir = out / "leiden" / "nearest_centroid"
    _assert_run_layout(leiden_dir)
    config = yaml.safe_load((leiden_dir / "config.yaml").read_text())
    assert config["start_from_annotation"] is None

    mapping_dir = out / "annotation_cellType" / "nearest_centroid"
    k_dirs = _assert_run_layout(mapping_dir)
    config = yaml.safe_load((mapping_dir / "config.yaml").read_text())
    assert config["start_from_annotation"] == "cellType"

    start = ad.read_h5ad(mapping_dir.parent / "start_clustering.h5ad")
    assert sorted(set(start.obs["start_cluster_name"].astype(str))) == cell_types
    assert set(start.obs["start_cluster"]) == set(range(len(cell_types)))
    # K sweeps from the number of annotated types down to 1.
    assert {int(p.name.split("_")[1]) for p in k_dirs} == set(
        range(1, len(cell_types) + 1)
    )


@_needs_sample
def test_sweep_runs_only_the_given_starts(tmp_path):
    """``leiden_start=False`` sweeps only the annotations (the GUI fills in
    what a mapper lacks), here one annotation without the Leiden start."""
    from spatialaim.config import SpatialAIMConfig
    from spatialaim.cli import run_one_pair

    out = tmp_path / "run"
    run_one_pair(
        SC,
        ST,
        out,
        SpatialAIMConfig(
            mapping="nearest_centroid", leiden_start=False, annotations=["cellType"]
        ),
    )
    assert [p.name for p in out.iterdir() if p.is_dir()] == ["annotation_cellType"]
    _assert_run_layout(out / "annotation_cellType" / "nearest_centroid")


@_needs_sample
def test_sweep_rejects_invalid_annotations(tmp_path):
    """A missing annotation column, or one that leaves a cell unlabelled, stops the
    sweep before anything is computed."""
    import anndata as ad
    import numpy as np

    from spatialaim.config import SpatialAIMConfig
    from spatialaim.cli import run_one_pair

    out = tmp_path / "run"
    with pytest.raises(KeyError, match="no_such_column"):
        run_one_pair(SC, ST, out, SpatialAIMConfig(annotations=["no_such_column"]))
    assert not (out / "reference_scaffold.h5ad").exists()

    adata_sc = ad.read_h5ad(SC)
    labels = adata_sc.obs["cellType"].astype(object)
    labels.iloc[0] = np.nan
    adata_sc.obs["partial"] = labels
    sc_partial = tmp_path / "sc_partial.h5ad"
    adata_sc.write_h5ad(sc_partial)
    with pytest.raises(ValueError, match="unlabelled"):
        run_one_pair(sc_partial, ST, out, SpatialAIMConfig(annotations=["partial"]))
    assert not (out / "reference_scaffold.h5ad").exists()


@_needs_sample
def test_run_cli_end_to_end(tmp_path):
    """The `spatialaim run` entry point maps a single pair and writes the K output."""
    out = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "spatialaim",
            "run",
            "--scdata",
            str(SC),
            "--stdata",
            str(ST),
            "--output_dir",
            str(out),
            "--mapping",
            "nearest_centroid",
            "--k_min",
            "1",
            "--k_max",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert (
        out
        / "leiden"
        / "nearest_centroid"
        / "k_001"
        / "spot_to_state_mapping_soft.h5ad"
    ).is_file()
