"""
The read-only online demo (``spatialaim gui --demo_data <dir>``).

The launcher passes the demo data root to the Streamlit process in
``SPATIALAIM_DEMO_DATA``; its presence switches the whole app into demo mode
(``ACTIVE``). The demo only browses precomputed results of the public datasets:
no paths are typed, nothing is run, nothing is exported and nothing is written to
disk. The data root holds::

    <data root>/
    ├── datasets/        # the paper's relational layout: scRNA/, ST/, pairs.csv
    └── results/
        └── <linkage>/   # one batch run per linkage (average/, ward/)
            └── <PairID:03d>_<scName>__<stName>/   # one pair output per pair

A pair output directly in ``results/`` is read too, under the linkage its configs
record. Every pair of ``pairs.csv`` with a pair output holding a start is offered,
under its optional ``Name`` column (else PairID and dataset names). The imprint
(Impressum) and the privacy policy are Markdown files in ``SPATIALAIM_DEMO_LEGAL``
(``imprint.md``, ``privacy.md``), read on every render so they can be
edited without a restart.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import NamedTuple

import streamlit as st

from spatialaim import LINKAGE_METHODS
from spatialaim.gui import data_access

DATA_ENV = "SPATIALAIM_DEMO_DATA"
LEGAL_ENV = "SPATIALAIM_DEMO_LEGAL"

ACTIVE = bool(os.environ.get(DATA_ENV))
DATA_ROOT = Path(os.environ.get(DATA_ENV, "."))
DATASETS_DIR = DATA_ROOT / "datasets"
RESULTS_DIR = DATA_ROOT / "results"
LEGAL_DIR = Path(os.environ.get(LEGAL_ENV) or DATA_ROOT / "legal")

REPO_URL = "https://github.com/domipysch/SpatialAIM-Repository"

# Plotly.js for the client-side plot components, served by Streamlit's static
# file serving (``static/`` next to app.py) instead of a CDN, so a visitor's
# browser contacts no third party. The Docker image downloads it there.
PLOTLY_FILE = Path(__file__).parent / "static" / "plotly.min.js"
PLOTLY_URL = "app/static/plotly.min.js"

_LEGAL_PAGES = {
    "imprint": ("Imprint", "imprint.md"),
    "privacy": ("Privacy", "privacy.md"),
}


def plotly_url() -> str | None:
    """Where the plot components load Plotly.js from in the demo; ``None`` (the
    CDN) on a local GUI or when the file is missing."""
    return PLOTLY_URL if ACTIVE and PLOTLY_FILE.exists() else None


def cache_entries(n: int) -> int | None:
    """``max_entries`` for a cache shared by every session: bounded in the demo,
    where many visitors browse many pairs, unbounded on a local GUI."""
    return n if ACTIVE else None


class DemoPair(NamedTuple):
    """One selectable pair with precomputed results."""

    pair_id: int
    sc_name: str
    st_name: str
    sc_path: Path
    st_path: Path
    segmentation: Path | None
    outputs: dict[str, Path]  # linkage -> pair output, LINKAGE_METHODS order
    tissue: str
    author: str
    st_author: str
    sc_paper: str
    sc_data: str
    st_paper: str
    st_data: str
    name: str

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        if self.sc_name == self.st_name:
            return f"{self.pair_id:03d} · {self.sc_name}"
        return f"{self.pair_id:03d} · {self.sc_name} × {self.st_name}"


def _pair_outputs(folder_name: str) -> dict[str, Path]:
    """A pair's outputs by linkage: ``results/<linkage>/<folder>`` and a
    ``results/<folder>`` under the linkage its configs record (the first
    found wins); only outputs holding a start count."""
    found: dict[str, Path] = {}
    candidates = [(m, RESULTS_DIR / m / folder_name) for m in LINKAGE_METHODS]
    candidates.append((LINKAGE_METHODS[0], RESULTS_DIR / folder_name))
    for default, pair_dir in candidates:
        starts = data_access.list_starts(pair_dir)
        if not starts:
            continue
        recorded = (data_access.linkage_method_from_config(s.folder) for s in starts)
        found.setdefault(next((m for m in recorded if m), default), pair_dir)
    order = {m: i for i, m in enumerate(LINKAGE_METHODS)}
    return dict(sorted(found.items(), key=lambda kv: order.get(kv[0], len(order))))


def _index(path: Path) -> dict[str, dict[str, str]]:
    """The rows of an ``index.csv`` by ``Name``; empty without one."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["Name"]: row for row in csv.DictReader(fh)}


def _first_url(cell: str | None) -> str:
    """The first http(s) URL of an index cell (``Links`` may hold several,
    space-separated); empty without one."""
    return next((t for t in (cell or "").split() if t.startswith("http")), "")


@st.cache_data(ttl=600, show_spinner=False)
def list_pairs() -> list[DemoPair]:
    """The pairs of ``datasets/pairs.csv`` whose inputs and results exist."""
    pairs_csv = DATASETS_DIR / "pairs.csv"
    if not pairs_csv.exists():
        return []
    sc_index = _index(DATASETS_DIR / "scRNA" / "index.csv")
    st_index = _index(DATASETS_DIR / "ST" / "index.csv")
    with open(pairs_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    pairs = []
    for row in rows:
        pair_id, sc_name, st_name = int(row["PairID"]), row["scName"], row["stName"]
        sc_path = DATASETS_DIR / "scRNA" / f"{sc_name}.h5ad"
        st_path = DATASETS_DIR / "ST" / f"{st_name}.h5ad"
        if not (sc_path.is_file() and st_path.is_file()):
            continue
        outputs = _pair_outputs(f"{pair_id:03d}_{sc_name}__{st_name}")
        if not outputs:
            continue
        mask = (st_index.get(st_name) or {}).get("Segmentation", "").strip()
        seg = DATASETS_DIR / "ST" / "Segmentation" / mask if mask else None
        sc_row = sc_index.get(sc_name) or {}
        st_row = st_index.get(st_name) or {}
        pairs.append(
            DemoPair(
                pair_id,
                sc_name,
                st_name,
                sc_path,
                st_path,
                seg if seg is not None and seg.is_file() else None,
                outputs,
                sc_row.get("Tissue", ""),
                sc_row.get("Autor", ""),
                st_row.get("Autor", ""),
                _first_url(sc_row.get("Quelle")),
                _first_url(sc_row.get("Links")),
                _first_url(st_row.get("Quelle")),
                _first_url(st_row.get("Links")),
                (row.get("Name") or "").strip(),
            )
        )
    return pairs


def banner() -> None:
    """The notice at the top of the sidebar: what the demo is and where the tool
    lives."""
    st.sidebar.info(
        "This is an online demo — browse precomputed results on public datasets. "
        f"To run SpatialAIM on your own data, install it from [GitHub]({REPO_URL}).",
        icon=":material/science:",
    )


def no_data_view() -> None:
    """The landing page when the data root holds no browsable pair."""
    st.title("SpatialAIM online demo")
    st.warning("No precomputed results are available at the moment.")


def _legal_view(slug: str) -> None:
    title, filename = _LEGAL_PAGES[slug]
    path = LEGAL_DIR / filename
    if not path.exists():
        st.title(title)
        st.error(f"Not configured: {filename} is missing.")
        return
    st.markdown(path.read_text(encoding="utf-8"))


def legal_pages() -> list:
    """Imprint and privacy policy as hidden pages, reached from the sidebar
    footer on every page."""
    return [
        st.Page(
            lambda s=slug: _legal_view(s),
            title=title,
            url_path=slug,
            visibility="hidden",
        )
        for slug, (title, _) in _LEGAL_PAGES.items()
    ]
