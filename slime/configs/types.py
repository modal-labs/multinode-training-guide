"""Types for thin SLIME experiment wrappers."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SlimExperimentConfig:
    """Resolved experiment wrapper config.

    This intentionally keeps only wrapper-level concerns and opaque SLIME args.
    """

    name: str
    source_path: Path
    model_id: str
    model_args_script: str
    model_args_env: dict[str, str] = field(default_factory=dict)
    args_files: tuple[Path, ...] = ()
    args: tuple[str, ...] = ()
    app_name: str = "slime-grpo"
    n_nodes: int = 4
    gpu: str = "H100:8"
    wandb_project: str = "slime-grpo"
    wandb_run_name_prefix: str = ""
    train_script: str = "slime/train_async.py"
    sync: bool = False

