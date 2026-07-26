from __future__ import annotations

import argparse
from pathlib import Path
import sys

from . import __version__
from .tui import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 -m mai",
        description="Open the MAI research workbench.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"MAI workbench {__version__}",
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(
            "The MAI workbench needs an interactive terminal. "
            "Run `python3 -m mai` from a terminal.",
            file=sys.stderr,
        )
        return 2
    repository = Path(__file__).resolve().parents[1]
    return run(repository)


if __name__ == "__main__":
    raise SystemExit(main())
