"""
Drive an SpatialAIM sweep for one mapper from the GUI.

Runs ``spatialaim.run`` in-process (the same call ``spatialaim run`` makes). The sweep only
writes machine-readable metrics per K -- no figures -- and the GUI renders every
plot on demand from them. One run covers the starts its mapper still lacks in
the pair output (``missing_starts``): of the Leiden start and each annotation,
one after another, each mapper into its own run root
``<pair output>/<start>/<mapper>/`` so several mappers coexist. The sweep runs on
a background thread; the UI polls the number of finished ``k_<kkk>/`` folders
per start for its progress bars.
"""

from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path
from typing import NamedTuple

from spatialaim import SpatialAIMConfig, LINKAGE_METHODS
from spatialaim.cli import run_one_pair
from spatialaim.io import start_folder_name

from . import data_access

logger = logging.getLogger(__name__)

# The GUI does not expose the Leiden resolution; use the SpatialAIMConfig default so the
# sweep and the reference scaffold agree.
DEFAULT_LEIDEN_RESOLUTION = SpatialAIMConfig().leiden_resolution


def start_complete(pair_dir: Path, mapper: str, annotation: str | None) -> bool:
    """Whether ``mapper`` has finished the start ``annotation`` (``None`` =
    Leiden) in the pair output: its K comparison is written last."""
    root = data_access.run_root(Path(pair_dir) / start_folder_name(annotation), mapper)
    return (root / "k_comparison.csv").exists()


def missing_starts(
    pair_dir: Path, mapper: str, annotations: list[str]
) -> list[str | None]:
    """The starts out of Leiden and ``annotations`` that ``mapper`` has not
    finished in the pair output, in sweep order."""
    return [a for a in [None, *annotations] if not start_complete(pair_dir, mapper, a)]


class StartProgress(NamedTuple):
    """How far a run has got in one of its starts."""

    label: str
    done: int
    # ``None`` until the start's clusters are known (its start_clustering.h5ad).
    expected: int | None


class MapperRun:
    """Handle for one background mapper sweep over the given starts
    (``None`` = Leiden); poll it for progress and status."""

    def __init__(
        self,
        mapper: str,
        sc_path: Path,
        st_path: Path,
        pair_dir: Path,
        k_min: int | None,
        k_max: int | None,
        k_step: int,
        linkage_method: str = LINKAGE_METHODS[0],
        starts: list[str | None] | None = None,
        segmentation_path: Path | None = None,
    ) -> None:
        self.mapper = mapper
        self.segmentation_path = (
            Path(segmentation_path) if segmentation_path is not None else None
        )
        self.sc_path = Path(sc_path)
        self.st_path = Path(st_path)
        self.pair_dir = Path(pair_dir)
        self.k_min = k_min
        self.k_max = k_max
        self.k_step = k_step
        self.linkage_method = linkage_method
        self._starts = list(starts) if starts is not None else [None]
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def starts(self) -> list[str | None]:
        """The starts this run sweeps, in order (``None`` = Leiden)."""
        return list(self._starts)

    def root(self, start_from_annotation: str | None) -> Path:
        """This mapper's run root in one start."""
        return data_access.run_root(
            self.pair_dir / start_folder_name(start_from_annotation), self.mapper
        )

    def start(self) -> MapperRun:
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            cfg = SpatialAIMConfig(
                mapping=self.mapper,
                leiden_resolution=DEFAULT_LEIDEN_RESOLUTION,
                linkage_method=self.linkage_method,
                k_min=self.k_min,
                k_max=self.k_max,
                k_step=self.k_step,
                leiden_start=None in self._starts,
                annotations=[a for a in self._starts if a is not None],
            )
            # Same entry point as the CLI. The reference scaffold is cached at the
            # pair output root and shared across all starts and mappers of this
            # pair: build it once, then reuse it.
            run_one_pair(
                self.sc_path,
                self.st_path,
                self.pair_dir,
                cfg,
                self.segmentation_path,
            )
        except Exception:  # noqa: BLE001 - surface any failure to the UI
            self.error = traceback.format_exc()
            logger.exception("Mapper run failed for %s", self.mapper)

    # -- progress ---------------------------------------------------------
    def is_running(self) -> bool:
        return self._thread.is_alive()

    def progress(self) -> list[StartProgress]:
        """Per start, in sweep order: the K folders finished so far and the total
        the sweep will produce."""
        return [
            StartProgress(
                data_access.start_label(a),
                len(data_access.list_ks(self.root(a))),
                self._n_expected(self.root(a)),
            )
            for a in self.starts
        ]

    def finished(self, start_from_annotation: str | None) -> bool:
        """Whether this run is done with one start: every K folder written and
        the K comparison saved. Starts sweep one after another, so a running
        run may already be finished in its first starts; a start it does not
        sweep (finished by an earlier run) counts as done."""
        if not self.is_running():
            return True
        if start_from_annotation not in self.starts:
            return True
        root = self.root(start_from_annotation)
        expected = self._n_expected(root)
        return (
            expected is not None
            and len(data_access.list_ks(root)) >= expected
            and (root / "k_comparison.csv").exists()
        )

    def _n_expected(self, root: Path) -> int | None:
        """Total K folders the sweep produces in ``root``, once the start clusters
        are known; ``None`` until its ``start_clustering.h5ad`` exists (the level
        count depends on the start-cluster count, which is only known then)."""
        labels = data_access.load_start_cluster_labels(root)
        if labels is None:
            return None
        n_start_clusters = int(labels.max()) + 1
        k_hi = min(n_start_clusters, self.k_max) if self.k_max else n_start_clusters
        k_lo = max(1, self.k_min) if self.k_min else 1
        return len(range(k_hi, k_lo - 1, -self.k_step))
