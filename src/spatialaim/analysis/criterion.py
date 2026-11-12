"""
The interface every K-selection criterion implements.

A criterion measures one K's mapping (:meth:`Criterion.analyse`), which is
written as JSON into that K's ``analysis/`` folder, and reads it back as its
``k_comparison.csv`` columns (:meth:`Criterion.collect`). Its score per K is the
harmonic mean of its ``columns``.

To add a criterion: subclass :class:`Criterion` in its own module of
``spatialaim.analysis`` and add an instance to
:data:`spatialaim.analysis.registry.REGISTRY`.
"""

from __future__ import annotations
import json
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from anndata import AnnData


class Criterion(ABC):
    """One K-selection criterion: its metrics per K, and how they are scored."""

    key: str  # name in the registry, the score table and k_selection.json
    label: str  # axis / card title
    short: str  # short label for the scatter-panel titles
    unit: str  # y-axis title of the line card
    filename: str  # the JSON written into each K's analysis folder
    columns: tuple[str, ...]  # the k_comparison.csv columns combined into the score
    curve_labels: tuple[str, ...]  # legend labels for those columns
    axis_from_zero: bool = False  # start this criterion's scatter axis at 0

    @abstractmethod
    def analyse(
        self,
        adata_sc: AnnData,
        adata_st: AnnData,
        start_cluster_to_state: np.ndarray,
    ) -> dict:
        """This K's metrics as a JSON-serialisable dict."""

    @abstractmethod
    def collect(self, data_dir: Path) -> dict[str, float]:
        """This K's ``k_comparison.csv`` columns, read from its analysis folder."""

    def run(
        self,
        adata_sc: AnnData,
        adata_st: AnnData,
        start_cluster_to_state: np.ndarray,
        data_dir: Path,
    ) -> None:
        """Analyse this K and write the metrics to ``data_dir / filename``."""
        metrics = self.analyse(adata_sc, adata_st, start_cluster_to_state)
        with open(Path(data_dir) / self.filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

    def load(self, data_dir: Path) -> dict:
        """The metrics this criterion wrote to ``data_dir``, or ``{}``."""
        return load_json(Path(data_dir) / self.filename)


def load_json(path: Path) -> dict:
    """A JSON dict, or ``{}`` when the file is absent or unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def finite(x) -> float:
    """``x`` as a float, with inf / NaN / None / non-numbers as NaN (a gap in
    the GUI's plots instead of a blown-up autoscaled axis)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")
