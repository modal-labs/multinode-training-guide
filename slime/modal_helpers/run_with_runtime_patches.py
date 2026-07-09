"""Run a Python script after applying Modal runtime patches."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

from modal_helpers.runtime_patches import apply_modal_runtime_patches


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m modal_helpers.run_with_runtime_patches SCRIPT [ARGS...]")

    script = sys.argv[1]
    sys.argv = [script, *sys.argv[2:]]
    apply_modal_runtime_patches()
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
