"""Base configuration for SLIME training on Modal.

This module provides a simple pass-through config system that is decoupled from
slime's internal argument structure. Configs just pass raw CLI args to slime.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RLConfig:
    """Minimal config - just metadata and raw args passthrough.

    This config is intentionally simple and decoupled from slime's internal
    argument structure. It passes raw CLI args directly to slime without
    parsing or validation, making it robust to slime version changes.
    """

    # Required metadata
    model_name: str
    model_id: str  # HuggingFace model ID (e.g., "Qwen/Qwen3-4B")

    # Modal settings
    app_name: str = "slime-grpo"
    n_nodes: int = 4
    gpu: str = "H100:8"

    # Training mode
    sync: bool = True

    # Wandb
    wandb_project: str = "slime-grpo"
    wandb_run_name_prefix: str = ""

    # Raw CLI args - passed directly to slime, no parsing/validation
    # Use triple-quoted strings for readability with comments
    slime_args: str = ""

    # Extra args that get appended (for easy overrides)
    extra_args: list[str] = field(default_factory=list)

    @property
    def train_script(self) -> str:
        return "slime/train.py" if self.sync else "slime/train_async.py"

    def _clean_args(self, args: str) -> str:
        """Clean up arg string: remove comments, normalize whitespace."""
        lines = []
        for line in args.split("\n"):
            # Remove comments
            if "#" in line:
                line = line[:line.index("#")]
            line = line.strip()
            if line:
                lines.append(line)
        return " ".join(lines)

    def generate_train_args(self, hf_model_path: str, checkpoints_path: Path, data_path: Path, is_infinite_run: bool) -> str:
        """Generate full command line args for slime."""
        # HF model path resolved from cache (snapshot_download with local_files_only=True)
        base_args = f"--hf-checkpoint {hf_model_path} --ref-load {hf_model_path}"

        cleaned_slime_args = self._clean_args(self.slime_args)
        cleaned_slime_args = cleaned_slime_args.replace("{data_path}", str(data_path))
        cleaned_slime_args = cleaned_slime_args.replace("{checkpoints_path}", str(checkpoints_path))

        extra = " ".join(self.extra_args) if self.extra_args else ""

        return f"{base_args} {cleaned_slime_args} {extra}".strip()


# Common arg templates for convenience
QWEN3_4B_MODEL_ARGS = """
    --num-layers 36 --hidden-size 2560 --ffn-hidden-size 9728
    --num-attention-heads 32 --group-query-attention --num-query-groups 8
    --kv-channels 128 --vocab-size 151936
    --normalization RMSNorm --norm-epsilon 1e-6 --swiglu
    --disable-bias-linear --qk-layernorm
    --use-rotary-position-embeddings --rotary-base 1000000
"""

DEFAULT_TRAINING_ARGS = """
    --tensor-model-parallel-size 2 --sequence-parallel
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --use-dynamic-batch-size --max-tokens-per-gpu 9216
    --megatron-to-hf-mode bridge
    --attention-dropout 0.0 --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
"""

DEFAULT_OPTIMIZER_ARGS = """
    --optimizer adam
    --lr 1e-6 --lr-decay-style constant
    --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
"""

DEFAULT_GRPO_ARGS = """
    --advantage-estimator grpo
    --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2 --eps-clip-high 0.28
"""

DEFAULT_DATA_ARGS = """
    --input-key messages --label-key label
    --apply-chat-template --rollout-shuffle --rm-type math
"""

