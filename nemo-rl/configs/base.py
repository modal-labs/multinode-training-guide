"""Base configuration classes and volume mount paths for NeMo-RL.

Two separate concerns:

  ModalConfig  — Modal infrastructure (gpu model, image, dev overlay)
  NemoRLConfig — NeMo-RL recipe: which run script, which base YAML config,
                 cluster shape, and Hydra overrides

NeMo-RL is driven by a YAML config plus Hydra-style ``key=value`` overrides
(see ``examples/run_grpo.py``). Unlike slime — where every config attribute
becomes a CLI flag — NeMo-RL configs carry a single ``overrides`` dict of
dotted Hydra keys, because the keys contain dots that are not valid Python
attribute names.

Each experiment defines one ``ModalConfig`` and one ``NemoRLConfig`` instance.
"""

from pathlib import Path
from typing import Any, Literal

# ── Volume mount paths ────────────────────────────────────────────────────────

# The NeMo-RL container caches Hugging Face artifacts under the default HF home.
HF_CACHE_PATH = Path("/root/.cache/huggingface")
CHECKPOINTS_PATH = Path("/checkpoints")

# Where the NeMo-RL repo lives inside the official image (see docker/Dockerfile).
NEMO_RL_ROOT = "/opt/nemo-rl"

# ── Types ─────────────────────────────────────────────────────────────────────

GPUType = Literal["H100", "H200", "B200", "B300", "A100"]

class ModalConfig:
    """Modal infrastructure configuration — GPU provisioning and image setup only."""

    # Official NeMo-RL release image (https://registry.ngc.nvidia.com/orgs/nvidia/containers/nemo-rl).
    docker_image: str = "nvcr.io/nvidia/nemo-rl:v0.5.0"
    gpu: GPUType = "H100"
    memory: tuple[int, int] | None = (
        None  # per-container memory in MiB; see https://modal.com/docs/guide/resources#memory-limits
    )
    cloud: str | None = None  # e.g. "aws", "gcp"
    region: str | None = None  # e.g. "us-east-2"
    local_nemo_rl: str | None = None  # path to local NeMo-RL repo for dev overlay
    image_run_commands: list[str] = []  # extra commands to run during image build
    image_env: dict[str, str] = {}  # env vars baked into the image (Modal .env())

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

def _hydra_value(val: Any) -> str:
    """Serialize a Python value into a Hydra override RHS string."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (list, tuple)):
        return "[" + ",".join(_hydra_value(v) for v in val) + "]"
    return str(val)


class NemoRLConfig:
    """Base NeMo-RL recipe configuration.

    Subclass and set class attributes to configure an experiment. The launcher
    runs::

        uv run python <entrypoint> --config <base_config> <hydra overrides>

    on the Ray head node, after the Modal cluster's Ray cluster is up.

    Launcher fields:
      entrypoint    — run script relative to /opt/nemo-rl (e.g. examples/run_grpo.py)
      base_config   — YAML config path passed to --config
      num_nodes     — total Modal/Ray nodes (also set as cluster.num_nodes)
      gpus_per_node — GPUs per node (also set as cluster.gpus_per_node)
      hf_model      — model id to prefetch into the HF cache; defaults to the
                      policy.model_name override, else the base config's value
      hf_datasets   — optional dataset repo ids to prefetch in download_data()
      environment   — extra environment variables for the driver process

    Recipe knobs:
      overrides     — dict of dotted Hydra keys → values, applied on top of the
                      base YAML config (e.g. {"policy.model_name": "Qwen/Qwen2.5-1.5B"})

    Example::

        class _Recipe(NemoRLConfig):
            entrypoint = "examples/run_grpo.py"
            base_config = "examples/configs/grpo_math_1B.yaml"
            num_nodes = 1
            gpus_per_node = 8
            overrides = {
                "policy.model_name": "Qwen/Qwen2.5-1.5B",
                "logger.wandb_enabled": True,
            }

        nemo_rl = _Recipe()
    """

    # ── Launcher instructions ───────────────────────────────────────────────────
    entrypoint: str = "examples/run_grpo.py"
    base_config: str = "examples/configs/grpo_math_1B.yaml"
    num_nodes: int = 1
    gpus_per_node: int = 8
    hf_model: str | None = None
    hf_datasets: list[str] = []
    environment: dict[str, str] = {}

    # ── Recipe knobs ────────────────────────────────────────────────────────────
    overrides: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        # Fresh mutable copies per instance — never mutate the class-level defaults.
        self.overrides = dict(type(self).overrides)
        self.environment = dict(type(self).environment)
        self.hf_datasets = list(type(self).hf_datasets)
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ── Public API ──────────────────────────────────────────────────────────────

    def total_nodes(self) -> int:
        """Total Modal cluster nodes required by this recipe."""
        return self.num_nodes

    def resolved_overrides(self, experiment: str | None = None) -> dict[str, Any]:
        """Hydra overrides with launcher-managed defaults merged in.

        The cluster shape and checkpoint directory are forced to match the Modal
        cluster and mounted checkpoints volume. Explicit ``overrides`` win over
        the default checkpoint dir, never over cluster shape. ``experiment`` (the
        config file stem) keys the default checkpoint dir so recipes don't
        collide on the shared volume.
        """
        run_name = experiment or type(self).__name__
        merged: dict[str, Any] = {
            "checkpointing.checkpoint_dir": f"{CHECKPOINTS_PATH}/{run_name}",
        }
        merged.update(self.overrides)
        # Cluster shape always tracks the actual Modal allocation.
        merged["cluster.num_nodes"] = self.num_nodes
        merged["cluster.gpus_per_node"] = self.gpus_per_node
        return merged

    def cli_args(self, experiment: str | None = None) -> list[str]:
        """Argument list for the NeMo-RL run script.

        Produces ``["--config", <base_config>, "k=v", ...]`` with Hydra-encoded
        values (True→true, None→null, lists→[a,b]).
        """
        out = ["--config", self.base_config]
        for key, val in self.resolved_overrides(experiment).items():
            out.append(f"{key}={_hydra_value(val)}")
        return out

    def model_id(self) -> str | None:
        """Hugging Face model id this recipe trains, for prefetching."""
        return self.hf_model or self.overrides.get("policy.model_name")
