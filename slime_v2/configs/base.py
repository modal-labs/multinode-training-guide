"""Base configuration classes and volume mount paths for slime_v2.

Two separate concerns:

  ModalConfig  — Modal infrastructure (gpu model, async mode, dev overlay)
  SlimeConfig  — SLIME training arguments

Each experiment defines one instance of each. All non-private, non-callable
attributes on a SlimeConfig subclass become SLIME CLI args automatically via
cli_args(). The 'environment' field is the only exception — it is injected
into the Ray job runtime env, not passed to SLIME directly.
"""

import logging
import math
from pathlib import Path
from typing import Any

# ── Volume mount paths ────────────────────────────────────────────────────────

HF_CACHE_PATH = Path("/root/.cache/huggingface")
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")

logger = logging.getLogger(__name__)

# Fields on SlimeConfig that are NOT SLIME CLI args.
_SLIME_SKIP = {"environment", "async_mode", "slime_model_script"}


class ModalConfig:
    """Modal infrastructure configuration — GPU provisioning and image setup only."""

    gpu: str = "H200"  # GPU model only (e.g. "H100", "H200", "B200")
    local_slime: str = ""  # path to local slime repo for dev overlay
    image_run_commands: list[str] = []  # extra image setup commands for this experiment

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class SlimeConfig:
    """Base SLIME training configuration.

    Subclass and set class attributes to configure an experiment.
    All attributes (except 'environment') are forwarded to SLIME as CLI args.
    Each experiment must be fully self-contained — no inherited defaults.

    Fields in _SLIME_SKIP are launcher instructions, not SLIME CLI args:
      environment       — injected into the Ray job runtime env
      async_mode        — selects train_async.py vs train.py
      slime_model_script — path relative to /root/slime to a shell script that
                           defines MODEL_ARGS for model architecture; sourced
                           before running the train command

    Example:

        class MyExperiment(SlimeConfig):
            async_mode = False
            slime_model_script = ""
            hf_checkpoint = "Qwen/Qwen3-8B"
            actor_num_nodes = 1
            actor_num_gpus_per_node = 8
            megatron_to_hf_mode = "bridge"
            ...

        slime = MyExperiment()
    """

    # Launcher instructions — not passed to SLIME CLI (see _SLIME_SKIP).
    environment: dict = {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
    }
    async_mode: bool = False  # True → use train_async.py
    slime_model_script: str = ""  # shell script path relative to /root/slime

    def __init__(self, **kwargs: Any) -> None:
        # Fresh environment dict per instance — never mutate the class-level default.
        self.environment = dict(type(self).environment)
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fields(self) -> dict[str, Any]:
        """Merged field dict from the class hierarchy; instance attrs win."""
        fields: dict[str, Any] = {}
        for cls in reversed(type(self).__mro__):
            if cls is object:
                continue
            fields.update(
                {
                    k: v
                    for k, v in vars(cls).items()
                    if not k.startswith("_") and not callable(v)
                }
            )
        fields.update(vars(self))
        return {k: v for k, v in fields.items() if k not in _SLIME_SKIP}

    # ── Public API ────────────────────────────────────────────────────────────

    def cli_args(self) -> list[str]:
        """SLIME CLI arguments derived from this config.

        Conversion rules:
          field_name → --field-name  (underscore to hyphen)
          True       → --flag        (no value)
          False/None → omitted
          list       → --flag v1 v2 ...
          other      → --flag value
        """
        out: list[str] = []
        for key, val in self._fields().items():
            if val is None or val is False:
                continue
            flag = f"--{key.replace('_', '-')}"
            if val is True:
                out.append(flag)
            elif isinstance(val, list):
                out += [flag] + [str(v) for v in val]
            else:
                out += [flag, str(val)]
        return out

    def total_nodes(self) -> int:
        """Total Modal cluster nodes required by this config.

        Derived from actor/critic/rollout GPU counts. Modal provisions whole
        nodes, so we ceil-divide. If total GPUs aren't a clean multiple of
        gpus_per_node, a warning is logged (unused GPUs on the last node).
        """
        f = self._fields()
        gpus_per_node = f.get("actor_num_gpus_per_node", 8)
        actor_nodes = f.get("actor_num_nodes", 1)
        colocate = f.get("colocate", False)
        use_critic = f.get("use_critic", False)
        critic_nodes = f.get("critic_num_nodes") or actor_nodes
        critic_gpus = f.get("critic_num_gpus_per_node") or gpus_per_node
        rollout_gpus = f.get("rollout_num_gpus")

        training_gpus = actor_nodes * gpus_per_node
        if use_critic:
            training_gpus += critic_nodes * critic_gpus

        if colocate:
            total_gpus = training_gpus
        else:
            rollout_gpus = rollout_gpus or (actor_nodes * gpus_per_node)
            total_gpus = training_gpus + rollout_gpus

        if total_gpus % gpus_per_node != 0:
            logger.warning(
                "total_gpus=%d is not a multiple of gpus_per_node=%d — "
                "%d GPU(s) on the last node will be unused.",
                total_gpus,
                gpus_per_node,
                gpus_per_node - (total_gpus % gpus_per_node),
            )

        return math.ceil(total_gpus / gpus_per_node)
