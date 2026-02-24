from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RLConfig:
    model_name: str
    model_id: str

    app_name: str = "slime-code-golf"
    n_nodes: int = 4
    gpu: str = "H100:8"
    sync: bool = True

    wandb_project: str = "slime-code-golf"
    wandb_run_name_prefix: str = "qwen8b-mbpp"

    slime_args: str = ""
    extra_args: list[str] = field(default_factory=list)

    # Custom RM runtime settings
    harbor_task_root: str = "/data/mbpp_harbor/tasks"
    harbor_data_volume_name: str = "slime-code-golf-harbor-data"
    harbor_rm_modal_app: str = "slime-harbor-rm"
    harbor_rm_max_concurrency: int = 64
    harbor_rm_timeout_sec: int = 120
    harbor_length_bonus_weight: float = 0.2

    @property
    def train_script(self) -> str:
        return "slime/train.py" if self.sync else "slime/train_async.py"

    def _clean_args(self, args: str) -> str:
        lines: list[str] = []
        for raw_line in args.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if line:
                lines.append(line)
        return " ".join(lines)

    def generate_train_args(
        self,
        hf_model_path: str,
        checkpoints_path: Path,
        data_path: Path,
    ) -> str:
        base_args = f"--hf-checkpoint {hf_model_path} --ref-load {hf_model_path}"

        cleaned = self._clean_args(self.slime_args)
        cleaned = cleaned.replace("{data_path}", str(data_path))
        cleaned = cleaned.replace("{checkpoints_path}", str(checkpoints_path))

        extra = " ".join(self.extra_args) if self.extra_args else ""
        return f"{base_args} {cleaned} {extra}".strip()


QWEN3_8B_MODEL_ARGS = """
    --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288
    --num-attention-heads 32 --group-query-attention --num-query-groups 8
    --kv-channels 128 --vocab-size 151936
    --normalization RMSNorm --norm-epsilon 1e-6 --swiglu
    --disable-bias-linear --qk-layernorm
    --untie-embeddings-and-output-weights
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
