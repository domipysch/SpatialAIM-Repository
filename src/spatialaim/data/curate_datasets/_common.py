"""
Shared helpers for the ``curate_datasets`` scripts.

Every curation script takes a single argument, ``--out-dir``, downloads its
sources into a temporary folder below it, builds the h5ad files and deletes the
downloads again — so only the final datasets survive a run.
"""

import argparse
import contextlib
import logging
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

DATA_ROOT = Path(r"C:\Users\zi69hebi\Dev\10_Alignment\Data")
# The paper's dataset database: the curated 10 references and their ST slices.
# (Built as 02_Datasets_Reproducable; moved here once it replaced the old set.)
OUT_ROOT = DATA_ROOT / "01_Datasets_Paper"

_USER_AGENT = "spatialaim-curate/1.0 (academic dataset reproduction)"
_CHUNK = 1 << 20  # 1 MiB
_REPORT_EVERY = 256 << 20  # log progress every 256 MiB


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_ROOT,
        help=f"Target root; scRNA/ and ST/ are created below it (default: {OUT_ROOT})",
    )
    return parser


def setup_logging() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def prepare_dirs(out_dir: Path) -> tuple[Path, Path]:
    """Create and return (sc_dir, st_dir) below ``out_dir``."""
    sc_dir, st_dir = out_dir / "scRNA", out_dir / "ST"
    for directory in (sc_dir, st_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return sc_dir, st_dir


KEEP_ENV_VAR = "SPATIALAIM_KEEP_DOWNLOADS"


@contextlib.contextmanager
def staging(out_dir: Path, name: str) -> Iterator[Path]:
    """
    Yield the folder the sources are downloaded into, and clean it up after.

    Setting ``SPATIALAIM_KEEP_DOWNLOADS=1`` keeps them in
    ``<out_dir>/_downloads/<name>`` instead, where ``download`` finds and reuses
    them — for debugging a curation script without refetching gigabytes.
    """
    if os.environ.get(KEEP_ENV_VAR):
        kept = out_dir / "_downloads" / name
        kept.mkdir(parents=True, exist_ok=True)
        logger.info("%s set — keeping downloads in %s", KEEP_ENV_VAR, kept)
        yield kept
        logger.info("downloads kept in %s (delete when done debugging)", kept)
    else:
        with tempfile.TemporaryDirectory(dir=out_dir, prefix=f"_tmp_{name}_") as tmp:
            yield Path(tmp)
        logger.info("intermediates removed")


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _remote_size(url: str) -> int | None:
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": _USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length is not None else None
    except (urllib.error.URLError, ValueError, TimeoutError):
        return None


def download(url: str, dest: Path, *, expected_bytes: int | None = None) -> Path:
    """
    Fetch ``url`` to ``dest``.

    ``expected_bytes`` is the size recorded when the source was verified and is
    used as the size check whenever the server reports no Content-Length.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    target = _remote_size(url)
    if target is None:
        target = expected_bytes

    have = dest.stat().st_size if dest.exists() else 0
    if have and target is not None:
        if have == target:
            logger.info("  cached  %s (%s)", dest.name, _human(target))
            return dest
        if have > target:  # not the same file any more
            dest.unlink()
            have = 0

    headers = {"User-Agent": _USER_AGENT}
    mode = "wb"
    if have:
        # only reachable in keep-downloads mode, where a partial file survives
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
        logger.info("  resume  %s at %s / %s", dest.name, _human(have), _human(target))
    else:
        logger.info("  get     %s (%s)", dest.name, _human(target))

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        if have and response.status != 206:  # server ignored the Range header
            mode, have = "wb", 0
        with open(dest, mode) as handle:
            done = have
            next_report = done + _REPORT_EVERY
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    logger.info("          %s / %s", _human(done), _human(target))
                    next_report += _REPORT_EVERY

    size = dest.stat().st_size
    if target is not None and size != target:
        raise IOError(f"{dest.name}: got {size} bytes, expected {target}")
    logger.info("  done    %s (%s)", dest.name, _human(size))
    return dest


def _human(n: int | None) -> str:
    if n is None:
        return "unknown size"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def uppercase(genes: Iterable[str]) -> list[str]:
    """The project-wide gene-name convention (see prepare/normalize_gene_names.py)."""
    return [str(g).upper() for g in genes]


def write(adata, path: Path) -> None:
    """Write via a .part file so an interrupted run leaves no half dataset."""
    tmp = path.with_suffix(".h5ad.part")
    adata.write_h5ad(tmp)
    shutil.move(str(tmp), str(path))
    logger.info(
        "  wrote   %s  (%d x %d, %s)",
        path.name,
        adata.n_obs,
        adata.n_vars,
        _human(path.stat().st_size),
    )
