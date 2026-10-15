"""
User-assigned names for the computed cell types, one set per K.

A name belongs to the ``(output_dir, K)`` pair, not to a mapper: every mapper
under one output dir cuts the same reference tree, so "cell type 3 at K = 7" is
the same population whichever mapper placed it there. The names live in
``<output_dir>/cell_type_names.json``::

    {"7": {"0": "Astrocytes", "3": "L2/3 IT"}, "12": {...}}

Only the names actually set are stored; ``label`` falls back to ``Cell type <n>``
for every other state, so a partly-named K is the normal case. A run started
from an annotation (``--start_from_annotation``) is the exception at K = T (T =
the number of annotated types): there the tree cut is the identity, so every
state *is* one annotated type and defaults to that type's name
(``annotation_names``); a name set by hand still wins.

This is the GUI's only write path onto disk. Nothing in the sweep or the analysis
reads the file -- losing it costs the labels and nothing else. Because the key is
K alone, names do go stale if the same output dir is later rebuilt with different
start clusters or linkage (which the tool asks you to keep in separate output
dirs anyway).
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

from spatialaim.io import START_CLUSTER_TO_STATE_FILENAME, START_CLUSTERING_FILENAME

from . import data_access

logger = logging.getLogger(__name__)

FILENAME = "cell_type_names.json"


def path_for(output_dir: Path) -> Path:
    """Where this output dir's names file lives (whether or not it exists)."""
    return Path(output_dir) / FILENAME


def load_all(output_dir: Path) -> dict[str, dict[str, str]]:
    """The whole file, or ``{}`` if it is absent, unreadable or not a JSON object.

    Never raises: these are cosmetic labels, so a corrupt file degrades to the
    default ``Cell type <n>`` rather than taking the results view down.
    """
    path = path_for(output_dir)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        logger.warning("Could not read %s; falling back to default labels.", path)
        return {}
    return data if isinstance(data, dict) else {}


def annotation_names(output_dir: Path, k: int) -> dict[int, str]:
    """``{state: annotated type}`` when this output dir's runs start from an
    annotation and ``k`` is T, the number of annotated types -- the level where
    the tree cut is the identity, so each state is exactly one annotated type.
    Empty for every other K, for Leiden starts, and when no run root has the
    files yet. All mappers cut the same tree, so any run root serves."""
    if data_access.start_from_annotation_from_config(output_dir) is None:
        return {}
    for root in sorted(Path(output_dir).glob("*/")):
        clustering = root / START_CLUSTERING_FILENAME
        cut = data_access.k_dir(root, k) / START_CLUSTER_TO_STATE_FILENAME
        if clustering.exists() and cut.exists():
            # mtimes in the key: a rerun into the same dir invalidates the cache.
            return dict(
                _annotation_names(
                    str(clustering),
                    str(cut),
                    clustering.stat().st_mtime,
                    cut.stat().st_mtime,
                )
            )
    return {}


@lru_cache(maxsize=32)
def _annotation_names(
    clustering: str, cut: str, _m1: float, _m2: float
) -> tuple[tuple[int, str], ...]:
    """``annotation_names`` for one run root's files (cached per file version)."""
    names = data_access.load_start_cluster_names(Path(clustering).parent)
    states = pd.read_csv(cut)["state"].to_numpy(dtype=int)
    if (
        names is None
        or len(states) != len(names)
        or int(states.max()) + 1 != len(names)
    ):
        return ()  # not the identity level: K != T
    return tuple((int(s), n) for s, n in zip(states, names) if n)


def _stored(output_dir: Path, k: int) -> dict[int, str]:
    """``{state: name}`` for the names set by hand at this K.

    Tolerates hand-edits: entries with a non-numeric state or a blank name are
    skipped rather than surfacing as broken labels.
    """
    per_k = load_all(output_dir).get(str(int(k)))
    if not isinstance(per_k, dict):
        return {}
    out: dict[int, str] = {}
    for state, name in per_k.items():
        try:
            key = int(state)
        except (TypeError, ValueError):
            continue
        if isinstance(name, str) and name.strip():
            out[key] = name.strip()
    return out


def load(output_dir: Path, k: int) -> dict[int, str]:
    """``{state: name}`` for one K, holding only the states that have a name:
    the annotated type names at K = T of an annotation start
    (``annotation_names``), overridden by any name set by hand."""
    return annotation_names(output_dir, k) | _stored(output_dir, k)


def save(output_dir: Path, k: int, names: dict[int, str]) -> None:
    """Replace this K's names, leaving every other K in the file untouched.

    A blank name -- or one equal to the state's annotated type (see
    ``annotation_names``) -- drops that state back to its default label instead
    of being stored, and a K with nothing left set drops out of the file entirely.
    The write goes through a temp file and an atomic replace, so an interrupted
    save cannot leave a half-written file that would read back as no names at all.
    """
    data = load_all(output_dir)
    defaults = annotation_names(output_dir, k)
    cleaned = {
        str(int(s)): n.strip()
        for s, n in names.items()
        if isinstance(n, str) and n.strip() and n.strip() != defaults.get(int(s))
    }
    if cleaned:
        data[str(int(k))] = cleaned
    else:
        data.pop(str(int(k)), None)

    path = path_for(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
    tmp.replace(path)


def set_name(output_dir: Path, k: int, state: int, name: str) -> None:
    """Set (or, when blank, clear) one state's name at this K, keeping the
    others -- the cell-type modal's write, through the same ``save`` as the
    all-states naming dialog."""
    names = _stored(output_dir, k)
    names[int(state)] = name
    save(output_dir, k, names)


def label(state: int, names: dict[int, str] | None) -> str:
    """The display label for one computed cell type: its name, else ``Cell type n``."""
    name = (names or {}).get(int(state))
    return name if name else f"Cell type {int(state)}"
