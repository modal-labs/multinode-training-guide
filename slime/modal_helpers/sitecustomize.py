"""Autoload Modal runtime patches in Ray worker Python processes."""

from __future__ import annotations

import os
import sys


def _enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off", ""}


if _enabled("MODAL_RUNTIME_PATCHES_AUTOAPPLY"):
    try:
        if "/root" not in sys.path:
            sys.path.insert(0, "/root")
        from modal_helpers.runtime_patches import apply_modal_runtime_patches

        apply_modal_runtime_patches()
    except Exception as exc:
        if _enabled("MODAL_RUNTIME_PATCHES_STRICT"):
            raise
        print(f"Modal runtime patch autoload skipped: {exc!r}")
