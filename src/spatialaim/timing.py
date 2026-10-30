"""
Wall-clock timing of a few sweep phases (the scaffold, and per K the mapping and
the analysis), written as ``timing.csv``: one row per phase, with its start as a
Unix timestamp so an outside monitor can attribute its own samples (e.g. memory)
to a phase.
"""

from __future__ import annotations
import csv
import time
from contextlib import contextmanager
from pathlib import Path

TIMING_CSV = "timing.csv"
_FIELDS = ("phase", "k", "start_epoch", "seconds")


class PhaseTimer:
    """Collects ``(phase, k, start, seconds)`` rows; ``k`` is empty for the
    phases that are not per K."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    @contextmanager
    def phase(self, name: str, k: int | None = None):
        start = time.time()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.rows.append(
                {
                    "phase": name,
                    "k": "" if k is None else k,
                    "start_epoch": f"{start:.3f}",
                    "seconds": f"{time.perf_counter() - t0:.3f}",
                }
            )

    def write(self, folder: Path) -> None:
        with open(Path(folder) / TIMING_CSV, "w", newline="") as fh:
            writer = csv.DictWriter(fh, _FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
