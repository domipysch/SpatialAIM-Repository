"""
The read-only online demo (``spatialaim gui --demo_data <dir>``).

The launcher passes the demo data root to the Streamlit process in
``SPATIALAIM_DEMO_DATA``; its presence switches the whole app into demo mode
(``ACTIVE``). The demo only browses precomputed results of the public datasets:
no paths are typed, nothing is run, nothing is exported and nothing is written to
disk. The data root holds::

    <data root>/
    ├── datasets/        # the paper's relational layout: scRNA/, ST/, pairs.csv
    └── results/         # one pair output per pair, <PairID:03d>_<scName>__<stName>/

Every pair of ``pairs.csv`` whose output holds a start is offered. The imprint
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
    pair_dir: Path
    tissue: str
    author: str
    source: str

    @property
    def label(self) -> str:
        if self.sc_name == self.st_name:
            return f"{self.pair_id:03d} · {self.sc_name}"
        return f"{self.pair_id:03d} · {self.sc_name} × {self.st_name}"


def _index(path: Path) -> dict[str, dict[str, str]]:
    """The rows of an ``index.csv`` by ``Name``; empty without one."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["Name"]: row for row in csv.DictReader(fh)}


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
        pair_dir = RESULTS_DIR / f"{pair_id:03d}_{sc_name}__{st_name}"
        if not (sc_path.is_file() and st_path.is_file()):
            continue
        if not data_access.list_starts(pair_dir):
            continue
        mask = (st_index.get(st_name) or {}).get("Segmentation", "").strip()
        seg = DATASETS_DIR / "ST" / "Segmentation" / mask if mask else None
        sc_row = sc_index.get(sc_name) or {}
        pairs.append(
            DemoPair(
                pair_id,
                sc_name,
                st_name,
                sc_path,
                st_path,
                seg if seg is not None and seg.is_file() else None,
                pair_dir,
                sc_row.get("Tissue", ""),
                sc_row.get("Autor", ""),
                sc_row.get("Quelle", ""),
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
