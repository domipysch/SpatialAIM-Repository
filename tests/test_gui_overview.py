"""The GUI's start × mapper bookkeeping: which starts a mapper still lacks, and
the Overview tab's table. Built on hand-written run folders, no sweep needed."""

import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("streamlit", reason="gui extra not installed")

_K_COMPARISON = (
    "k,cossim_hard_raw_spot,cossim_hard_raw_gene,spatial_assort_celltype,"
    "spatial_assort_global,modularity_st_expression,modularity_shared\n"
    "2,0.50,0.50,0.60,0.60,0.30,0.30\n"
    "3,0.60,0.60,0.50,0.50,0.40,0.40\n"
)


def _write_run(pair_dir, annotation, mapper, finished=True):
    from spatialaim.io import start_folder_name

    root = pair_dir / start_folder_name(annotation) / mapper
    root.mkdir(parents=True)
    value = "null" if annotation is None else annotation
    (root / "config.yaml").write_text(f"start_from_annotation: {value}\n")
    if finished:
        (root / "k_comparison.csv").write_text(_K_COMPARISON)


def test_missing_starts(tmp_path):
    from spatialaim.gui.compute import missing_starts

    _write_run(tmp_path, None, "wann")
    _write_run(tmp_path, "celltype", "wann", finished=False)
    assert missing_starts(tmp_path, "wann", ["celltype", "subtype"]) == [
        "celltype",
        "subtype",
    ]
    assert missing_starts(tmp_path, "tangram", []) == [None]


def test_overview_table(tmp_path):
    from spatialaim.gui import data_access
    from spatialaim.gui.views.overview import overview_table

    _write_run(tmp_path, None, "wann")
    _write_run(tmp_path, "celltype", "nearest_centroid")
    table = overview_table(data_access.list_starts(tmp_path))
    assert list(table["start"]) == ["Leiden", "celltype"]
    assert list(table["mapper"]) == ["wann", "nearest_centroid"]
    # K = 3: recon 0.6, spatial 0.5, coherence 0.4 -> harmonic mean ≈ 0.486,
    # beating K = 2 (0.5, 0.6, 0.3 -> ≈ 0.429).
    assert list(table["best_k"]) == [3, 3]
    assert table["all"].iloc[0] == pytest.approx(3 / (1 / 0.6 + 1 / 0.5 + 1 / 0.4))
