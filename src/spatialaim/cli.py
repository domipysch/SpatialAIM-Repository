"""
The ``spatialaim`` command line: ``run``, ``gui`` and ``validate``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import warnings
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
import yaml

from spatialaim.config import LINKAGE_METHODS, MAPPING_CHOICES, SpatialAIMConfig
from spatialaim.data.validate import validate_pairs_csv, validate_single_pair
from spatialaim.data.validate.validate_st import SEGMENTATION_DIR_NAME
from spatialaim.gui.__main__ import launch

# Harmless warnings of the sweep stack (imported in run_one_pair): conda ships
# two OpenMP runtimes (threadpoolctl warns on every call), and docrep (via
# squidpy) complains about unknown docstring keys.
warnings.filterwarnings(
    "ignore",
    message=r"(?s).*Found Intel OpenMP.*LLVM OpenMP",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*is not a valid key!",
    category=SyntaxWarning,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args) or 0


# --------------------------------------------------------------------------- #
# spatialaim run
# --------------------------------------------------------------------------- #
def run_one_pair(
    sc_path: Path,
    st_path: Path,
    root_output_folder: Path,
    config: SpatialAIMConfig,
    segmentation_path: Path | None = None,
) -> pd.DataFrame:
    """Run the sweep for one sc/ST pair with ``config``, saving the config as
    ``<root_output_folder>/<mapper>/config.yaml``. ``segmentation_path`` is the
    ST slice's optional segmentation mask."""
    # Imported here: the sweep pulls in scanpy / squidpy, which the light
    # commands (--help, gui, validate) do not need.
    from spatialaim.sweep import run

    root_output_folder = Path(root_output_folder)
    mapping_output_folder = root_output_folder / config.mapping
    mapping_output_folder.mkdir(parents=True, exist_ok=True)
    with open(mapping_output_folder / "config.yaml", "w") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)

    return run(
        sc_path=sc_path,
        st_path=st_path,
        root_output_folder=root_output_folder,
        mapper=config.build_mapper(),
        linkage_method=config.linkage_method,
        leiden_resolution=config.leiden_resolution,
        k_min=config.k_min,
        k_max=config.k_max,
        k_step=config.k_step,
        start_from_annotation=config.start_from_annotation,
        segmentation_path=segmentation_path,
    )


def run_batch(
    pairs_csv: Path,
    sc_dir: Path,
    st_dir: Path,
    output_dir: Path,
    config: SpatialAIMConfig,
) -> list[str]:
    """Run the sweep for every pair in ``pairs_csv``, each into
    ``<output_dir>/<PairID>_<scName>__<stName>``. A failing pair is logged and
    skipped. Returns the error messages (empty if all pairs succeeded).

    Segmentation masks are taken from the ``Segmentation`` column of
    ``<st_dir>/index.csv``, if there is one.
    """
    with open(pairs_csv, newline="") as fh:
        all_pairs = list(csv.DictReader(fh))
    logger.info("Loaded %d pairs from %s", len(all_pairs), pairs_csv)
    st_index = _read_st_index(st_dir)

    errors: list[str] = []
    for pair in all_pairs:
        pair_id = int(pair["PairID"])
        sc_name = pair["scName"]
        st_name = pair["stName"]
        sc_path = sc_dir / f"{sc_name}.h5ad"
        st_path = st_dir / f"{st_name}.h5ad"
        mask_name = (st_index.get(st_name) or {}).get("Segmentation", "").strip()
        segmentation_path = (
            st_dir / SEGMENTATION_DIR_NAME / mask_name if mask_name else None
        )

        required = [sc_path, st_path] + (
            [segmentation_path] if segmentation_path is not None else []
        )
        missing = [p for p in required if not p.exists()]
        if missing:
            msg = f"[Pair {pair_id:>3}] Missing files: {[str(p) for p in missing]}"
            logger.error(msg)
            errors.append(msg)
            continue

        pair_output = output_dir / f"{pair_id:03d}_{sc_name}__{st_name}"
        logger.info("[Pair %3d] %s x %s -> %s", pair_id, sc_name, st_name, pair_output)
        try:
            run_one_pair(sc_path, st_path, pair_output, config, segmentation_path)
        except Exception as exc:
            msg = f"[Pair {pair_id:>3}] FAILED: {exc}"
            logger.error(msg)
            errors.append(msg)

    if errors:
        logger.warning("%d pair(s) failed:", len(errors))
        for e in errors:
            logger.warning("  %s", e)
    else:
        logger.info("All pairs completed successfully.")
    return errors


def _read_st_index(st_dir: Path) -> dict[str, dict[str, str]]:
    """The rows of ``<st_dir>/index.csv`` by slice name; empty without one."""
    path = st_dir / "index.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {row["Name"]: row for row in csv.DictReader(fh)}


def _cmd_run(args: argparse.Namespace) -> int:
    _setup_logging(args.logging == "verbose")
    config = SpatialAIMConfig(
        mapping=args.mapping,
        leiden_resolution=args.leiden_resolution,
        linkage_method=args.linkage_method,
        k_min=args.k_min,
        k_max=args.k_max,
        k_step=args.k_step,
        start_from_annotation=args.start_from_annotation,
    )

    if args.pairs_csv is not None:
        if args.sc_dir is None or args.st_dir is None or args.output_dir is None:
            _fail("--pairs_csv requires --sc_dir, --st_dir, and --output_dir")
        if args.segmentation is not None:
            _fail(
                "--segmentation is single-pair only; in batch mode the masks come "
                "from the Segmentation column of the ST index.csv"
            )
        run_batch(args.pairs_csv, args.sc_dir, args.st_dir, args.output_dir, config)
        return 0

    if args.scdata is None or args.stdata is None or args.output_dir is None:
        _fail(
            "Provide either --scdata/--stdata (single pair) "
            "or --pairs_csv/--sc_dir/--st_dir (batch)"
        )
    if args.segmentation is not None and not args.segmentation.exists():
        _fail(f"--segmentation: file not found: {args.segmentation}")
    run_one_pair(args.scdata, args.stdata, args.output_dir, config, args.segmentation)
    return 0


# --------------------------------------------------------------------------- #
# spatialaim gui / spatialaim validate
# --------------------------------------------------------------------------- #
def _cmd_gui(args: argparse.Namespace) -> int:
    return launch(server_port=args.server_port)


def _cmd_validate(args: argparse.Namespace) -> int:
    if args.pairs_csv is not None:
        if (
            args.scdata is not None
            or args.stdata is not None
            or args.segmentation is not None
        ):
            _fail(
                "--pairs_csv cannot be combined with --scdata/--stdata/--segmentation"
            )
        return validate_pairs_csv(args.pairs_csv, args.sc_dir, args.st_dir)

    if args.scdata is None or args.stdata is None:
        _fail(
            "Provide either --scdata and --stdata (single pair) "
            "or --pairs_csv (every pair of a pairs.csv)"
        )
    if args.sc_dir is not None or args.st_dir is not None:
        _fail("--sc_dir/--st_dir only apply to --pairs_csv")
    return validate_single_pair(args.scdata, args.stdata, args.segmentation)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fail(message: str) -> None:
    """Print ``message`` as an error and exit with code 2, like argparse."""
    print(f"spatialaim: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _pkg_version() -> str:
    try:
        return version("spatial-aim")
    except PackageNotFoundError:  # source tree without install
        return "0+unknown"


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatialaim",
        description="SpatialAIM - annotation-independent mapping of scRNA references onto "
        "high-resolution spatial transcriptomics.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_pkg_version()}"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(sub)
    _add_gui_parser(sub)
    _add_validate_parser(sub)
    return parser


def _add_run_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "run",
        help="Run the SpatialAIM sweep (single pair or batch).",
        description="Run the SpatialAIM agglomerative-K sweep with a modular spot-to-state "
        "mapper, for one sc/st pair or every row of a pairs.csv.",
    )
    # Single pair
    p.add_argument("--scdata", type=Path, default=None, help="Single-pair: sc .h5ad")
    p.add_argument("--stdata", type=Path, default=None, help="Single-pair: ST .h5ad")
    p.add_argument(
        "--segmentation",
        type=Path,
        default=None,
        help="Single-pair: optional cell segmentation mask of the ST slice (.tif, "
        "uint32, linked via obs['segmentation_id']); enables the shape-consistency "
        "metric. In batch mode the masks come from the Segmentation column of the "
        "ST index.csv.",
    )
    # Batch
    p.add_argument("--pairs_csv", type=Path, default=None, help="Batch: pairs.csv")
    p.add_argument(
        "--sc_dir", type=Path, default=None, help="Batch: folder of scRNA .h5ad"
    )
    p.add_argument(
        "--st_dir", type=Path, default=None, help="Batch: folder of ST .h5ad"
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output folder (single) or root for all pair outputs (batch)",
    )
    # Method
    p.add_argument(
        "--mapping",
        choices=list(MAPPING_CHOICES),
        default="nearest_centroid",
        help="Spot-to-state mapping: 'nearest_centroid' (zero-parameter "
        "nearest-centroid by cosine, default), 'wann' (reliability-weighted "
        "adaptive-kNN label transfer, parameter-free), or an external reference "
        "aligner 'tangram' / 'tacco' / 'dot' (each runs out-of-process in its own "
        "conda env, once per K).",
    )
    p.add_argument(
        "--start_from_annotation",
        type=str,
        default=None,
        metavar="OBS_COLUMN",
        help="Start the agglomeration from a pre-existing annotation instead of a "
        "Leiden over-clustering: the named scRNA obs column's cell types become the "
        "start clusters, no over-clustering is computed, and K sweeps from the "
        "number of annotated types down (cells with no label are dropped). "
        "--leiden_resolution then only governs the shared-gene reference Leiden. "
        "Give such runs their own --output_dir: a run root is named after the "
        "mapper alone, so the two modes would otherwise overwrite each other.",
    )
    p.add_argument("--leiden_resolution", type=float, default=3.0)
    p.add_argument(
        "--linkage_method",
        choices=list(LINKAGE_METHODS),
        default=LINKAGE_METHODS[0],
        help="Linkage for the agglomeration tree over the start clusters: "
        "'average' (default, UPGMA) peels small tight groups off a dominant "
        "state; 'ward' carries a size term and tends to produce balanced states.",
    )
    p.add_argument("--k_min", type=int, default=None)
    p.add_argument("--k_max", type=int, default=None)
    p.add_argument("--k_step", type=int, default=1)
    p.add_argument("--logging", choices=["normal", "verbose"], default="normal")
    p.set_defaults(func=_cmd_run)


def _add_gui_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "gui",
        help="Launch the interactive Streamlit results GUI.",
        description="Launch the SpatialAIM results GUI (configure everything in the sidebar).",
    )
    p.add_argument(
        "--server_port",
        type=int,
        default=8501,
        help="Port for the Streamlit server (default 8501).",
    )
    p.set_defaults(func=_cmd_gui)


def _add_validate_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "validate",
        help="Validate a single sc/st pair or every pair of a pairs.csv.",
        description="Check the scRNA and ST h5ads against the SpatialAIM contract (raw "
        "counts, unique uppercase gene names, ST obsm['spatial'], the optional "
        "cell segmentation mask) and the pair against its shared-gene intersection. "
        "With --pairs_csv, an index.csv next to the h5ads additionally cross-checks "
        "the recorded counts and names each ST slice's segmentation mask. Exits "
        "non-zero if any errors are found.",
    )
    # Single pair
    p.add_argument("--scdata", type=Path, default=None, help="Single-pair: sc .h5ad")
    p.add_argument("--stdata", type=Path, default=None, help="Single-pair: ST .h5ad")
    p.add_argument(
        "--segmentation",
        type=Path,
        default=None,
        help="Single-pair: optional cell segmentation mask of the ST slice (.tif, uint32)",
    )
    # Batch
    p.add_argument(
        "--pairs_csv",
        type=Path,
        default=None,
        help="Batch: pairs.csv - validates every pair it lists and each dataset "
        "it references.",
    )
    p.add_argument(
        "--sc_dir",
        type=Path,
        default=None,
        help="Batch: folder of scRNA .h5ad (default: scRNA/ next to the pairs.csv)",
    )
    p.add_argument(
        "--st_dir",
        type=Path,
        default=None,
        help="Batch: folder of ST .h5ad (default: ST/ next to the pairs.csv)",
    )
    p.set_defaults(func=_cmd_validate)


if __name__ == "__main__":
    raise SystemExit(main())
