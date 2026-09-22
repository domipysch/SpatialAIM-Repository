"""User-assigned names for the computed cell types, one set per K.

A name belongs to the ``(output_dir, K)`` pair, not to a mapper: every mapper
under one output dir cuts the same reference tree, so "cell type 3 at K = 7" is
the same population whichever mapper placed it there. The names live in
``<output_dir>/cell_type_names.json``::

    {"7": {"0": "Astrocytes", "3": "L2/3 IT"}, "12": {...}}

Only the names actually set are stored; ``label`` falls back to ``Cell type <n>``
for every other state, so a partly-named K is the normal case.

This is the GUI's only write path onto disk. Nothing in the sweep or the analysis
reads the file -- losing it costs the labels and nothing else. Because the key is
K alone, names do go stale if the same output dir is later rebuilt with different
start clusters or linkage (which the tool asks you to keep in separate output
dirs anyway).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

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


def load(output_dir: Path, k: int) -> dict[int, str]:
    """``{state: name}`` for one K, holding only the states that have a name.

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


def save(output_dir: Path, k: int, names: dict[int, str]) -> None:
    """Replace this K's names, leaving every other K in the file untouched.

    A blank name drops that state back to its default label instead of storing an
    empty string, and a K with nothing left set drops out of the file entirely.
    The write goes through a temp file and an atomic replace, so an interrupted
    save cannot leave a half-written file that would read back as no names at all.
    """
    data = load_all(output_dir)
    cleaned = {
        str(int(s)): n.strip()
        for s, n in names.items()
        if isinstance(n, str) and n.strip()
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


def label(state: int, names: dict[int, str] | None) -> str:
    """The display label for one computed cell type: its name, else ``Cell type n``."""
    name = (names or {}).get(int(state))
    return name if name else f"Cell type {int(state)}"
