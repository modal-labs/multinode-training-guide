"""Model family registry for Miles experiments.

Keeps per-model identity (HF source, prepared artifact path, how to download)
separate from ``MilesConfig`` (training hyperparameters). A ``MilesConfig``
subclass that holds a ``ModelConfiguration`` instance must list
``"model_configuration"`` in ``configs.base._MILES_SKIP`` so the object does
not leak into the auto-generated Miles CLI args via ``cli_args()``.

The filename ``model_configuration.py`` is intentionally excluded from the
experiment-config scanners in ``configs/__init__.py`` and
``miles/modal_train.py::list_configs`` — it is not an experiment.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from configs.base import CHECKPOINTS_PATH


class ModelConfiguration:
    """Known model families. Extend as new models are added."""

    model_name: str
    model_path: str | None = None

    def download_model(self) -> None:
        raise NotImplementedError


_MILES_ROOT = Path(__file__).resolve().parents[1]
_MODAL_TRAIN_PATH = _MILES_ROOT / "modal_train.py"


class KimiK25ModelConfiguration(ModelConfiguration):
    model_name = "moonshotai/Kimi-K2.5"
    model_path = str(CHECKPOINTS_PATH / "Kimi-K2.5-bf16")

    def download_model(self) -> None:
        # Delegate to the Modal app functions defined in modal_train.py via the
        # Modal CLI. Shelling out (rather than importing modal_train directly)
        # avoids module-level side effects such as Modal volume creation at
        # import time and matches the invocation path in
        # docs/agent-modal-training.md.
        env = {**os.environ, "EXPERIMENT_CONFIG": "kimi_k25"}
        subprocess.run(
            ["modal", "run", f"{_MODAL_TRAIN_PATH}::download_model"],
            env=env,
            check=True,
        )
        subprocess.run(
            ["modal", "run", f"{_MODAL_TRAIN_PATH}::convert_kimi_int4_to_bf16"],
            env=env,
            check=True,
        )
