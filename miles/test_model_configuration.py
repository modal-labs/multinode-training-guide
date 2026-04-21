"""Smoke test for the ``ModelConfiguration`` additive introduction.

Verifies:
  1. ``ModelConfiguration`` imports.
  2. ``KimiK25ModelConfiguration`` constructs and exposes ``model_name`` and
     ``model_path``.
  3. ``download_model`` is callable (it is not actually invoked here — that
     would shell out to ``modal run``).
  4. A ``MilesConfig`` subclass holding a ``model_configuration`` attribute
     does NOT render ``--model-configuration`` as a CLI arg. This guards the
     ``_MILES_SKIP`` extension in ``configs/base.py``.

Run from ``miles/``:

    uv run python test_model_configuration.py
"""

from __future__ import annotations

import sys
from pathlib import Path


_MILES_DIR = Path(__file__).resolve().parent
if str(_MILES_DIR) not in sys.path:
    sys.path.insert(0, str(_MILES_DIR))


def test_model_configuration_imports_and_constructs() -> None:
    from configs.model_configuration import (
        KimiK25ModelConfiguration,
        ModelConfiguration,
    )

    assert ModelConfiguration.__doc__ and "Known model families" in ModelConfiguration.__doc__

    kimi = KimiK25ModelConfiguration()
    assert kimi.model_name == "moonshotai/Kimi-K2.5"
    assert isinstance(kimi.model_path, str) and kimi.model_path.endswith("/Kimi-K2.5-bf16")
    assert callable(kimi.download_model)


def test_base_download_model_is_not_implemented() -> None:
    from configs.model_configuration import ModelConfiguration

    try:
        ModelConfiguration().download_model()
    except NotImplementedError:
        pass
    else:
        raise AssertionError("ModelConfiguration.download_model() should raise NotImplementedError")


def test_model_configuration_does_not_leak_as_cli_arg() -> None:
    from configs.base import MilesConfig
    from configs.model_configuration import KimiK25ModelConfiguration

    class _Probe(MilesConfig):
        model_configuration = KimiK25ModelConfiguration()
        actor_num_nodes = 1
        actor_num_gpus_per_node = 8

    probe = _Probe()
    args = probe.cli_args()
    for token in args:
        assert "model-configuration" not in token, (
            f"--model-configuration leaked into cli_args: {args}"
        )


def test_miles_skip_contains_model_configuration() -> None:
    from configs.base import _MILES_SKIP

    assert "model_configuration" in _MILES_SKIP, (
        f"_MILES_SKIP must contain 'model_configuration'; got {_MILES_SKIP}"
    )


def main() -> int:
    tests = [
        test_model_configuration_imports_and_constructs,
        test_base_download_model_is_not_implemented,
        test_model_configuration_does_not_leak_as_cli_arg,
        test_miles_skip_contains_model_configuration,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
    if failures:
        print(f"\n{len(failures)}/{len(tests)} test(s) failed.")
        return 1
    print(f"\nAll {len(tests)} test(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
