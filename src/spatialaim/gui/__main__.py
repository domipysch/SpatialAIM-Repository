"""
Launcher for the SpatialAIM GUI.

Starts a Streamlit server running ``spatialaim/gui/app.py``. Everything - the
scRNA-seq/ST paths, output directory, K step and the agglomeration linkage - is set in
the app's sidebar. ``--demo_data`` instead serves the read-only online demo over
the precomputed results in that folder (see ``demo.py``).

    spatialaim gui                      # then set everything in the sidebar
    spatialaim gui --server_port 8600
    spatialaim gui --demo_data /data    # the online demo
    python -m spatialaim.gui            # equivalent
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from importlib.resources import as_file, files
from pathlib import Path

# Streamlit's default primary colour is red, which reads like an error on
# selections, tags and focused inputs; blue (Streamlit's own info blue) instead.
PRIMARY_COLOR = "#1c83e1"

# The demo is a public server: no file watcher or rerun-on-save, no developer
# menu, no uploads, no detailed error pages, and no cookie (the XSRF token only
# guards uploads, which the demo has none of).
_DEMO_FLAGS = [
    "--server.fileWatcherType", "none",
    "--server.runOnSave", "false",
    "--server.maxUploadSize", "1",
    "--server.enableXsrfProtection", "false",
    "--server.enableStaticServing", "true",
    "--client.toolbarMode", "viewer",
    "--client.showErrorDetails", "none",
]  # fmt: skip


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatialaim gui",
        description="Launch the interactive SpatialAIM results GUI (configure everything "
        "in the sidebar).",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=8501,
        help="Port for the Streamlit server (default 8501).",
    )
    add_demo_arguments(parser)
    return parser


def add_demo_arguments(parser: argparse.ArgumentParser) -> None:
    """The online-demo flags, shared with ``spatialaim gui`` in ``cli.py``."""
    parser.add_argument(
        "--demo_data",
        type=Path,
        default=None,
        help="Serve the read-only online demo over the precomputed results in this "
        "folder (datasets/ + results/) instead of the full GUI.",
    )
    parser.add_argument(
        "--legal_dir",
        type=Path,
        default=None,
        help="Demo only: folder with imprint.md and privacy.md "
        "(default <demo_data>/legal).",
    )


def launch(
    server_port: int = 8501,
    demo_data: Path | None = None,
    legal_dir: Path | None = None,
) -> int:
    """Run ``streamlit run spatialaim/gui/app.py`` on ``server_port``; return its exit code.

    The app imports the installed ``spatialaim`` package directly, so PYTHONPATH does
    not need to be set. With ``demo_data`` it serves the online demo headless,
    configured through the environment of the Streamlit process.

    The GUI dependencies are optional; if they are missing, print an install
    hint instead of letting the streamlit subprocess fail obscurely.
    """
    if importlib.util.find_spec("streamlit") is None:
        print(
            "The SpatialAIM GUI needs the optional 'gui' dependencies "
            "(streamlit, plotly, kaleido). Install them with:\n"
            "    pip install 'spatial-aim[gui]'",
            file=sys.stderr,
        )
        return 1

    env = dict(os.environ)
    if demo_data is not None:
        env["SPATIALAIM_DEMO_DATA"] = str(Path(demo_data).resolve())
        if legal_dir is not None:
            env["SPATIALAIM_DEMO_LEGAL"] = str(Path(legal_dir).resolve())

    app_resource = files("spatialaim.gui") / "app.py"
    with as_file(app_resource) as app_path:
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(server_port),
            "--server.headless",
            "true" if demo_data is not None else "false",
            "--browser.gatherUsageStats",
            "false",
            "--theme.primaryColor",
            PRIMARY_COLOR,
            *(_DEMO_FLAGS if demo_data is not None else []),
        ]
        print("Launching SpatialAIM GUI:", " ".join(cmd), flush=True)
        return subprocess.run(cmd, env=env).returncode


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return launch(args.server_port, args.demo_data, args.legal_dir)


if __name__ == "__main__":
    raise SystemExit(main())
