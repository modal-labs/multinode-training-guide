"""Configuration for Qwen3-4B GRPO training on Haiku dataset."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RLConfig:
    """Training config that passes raw CLI args directly to slime."""

    model_name: str
    model_id: str

    # Modal settings
    app_name: str = "slime-grpo"
    n_nodes: int = 4
    gpu: str = "H100:8"

    # Training mode
    sync: bool = True

    # Wandb
    wandb_project: str = "slime-grpo"
    wandb_run_name_prefix: str = ""

    # Raw CLI args passed directly to slime
    slime_args: str = ""

    # Extra args that get appended (for easy overrides)
    extra_args: list[str] = field(default_factory=list)

    @property
    def train_script(self) -> str:
        return "slime/train.py" if self.sync else "slime/train_async.py"

    def _clean_args(self, args: str) -> str:
        """Remove comments and normalize whitespace."""
        lines = []
        for line in args.split("\n"):
            if "#" in line:
                line = line[: line.index("#")]
            line = line.strip()
            if line:
                lines.append(line)
        return " ".join(lines)

    def generate_train_args(self, models_path: Path, data_path: Path, is_infinite_run: bool) -> str:
        base_args = f"--hf-checkpoint {models_path}/{self.model_name} --ref-load {models_path}/{self.model_name}"

        cleaned_slime_args = self._clean_args(self.slime_args)
        cleaned_slime_args = cleaned_slime_args.replace("{data_path}", str(data_path))
        cleaned_slime_args = cleaned_slime_args.replace("{models_path}", str(models_path))

        extra = " ".join(self.extra_args) if self.extra_args else ""

        return f"{base_args} {cleaned_slime_args} {extra}".strip()


# ── Model architecture constants ──

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


# ── Config factory ──

def get_config(run_name: str = "qwen3-4b-haiku") -> RLConfig:
    return RLConfig(
        model_name="Qwen3-4B",
        model_id="Qwen/Qwen3-4B",
        n_nodes=1,
        gpu="H200:8",
        app_name="slime-qwen3-4b-haiku",
        sync=True,
        wandb_project="slime-grpo-haiku",
        wandb_run_name_prefix=run_name,
        slime_args=f"""
            # Model architecture
            {QWEN3_4B_MODEL_ARGS}

            # Training parallelism and optimization
            {DEFAULT_TRAINING_ARGS}

            # Optimizer
            {DEFAULT_OPTIMIZER_ARGS}

            # GRPO algorithm
            {DEFAULT_GRPO_ARGS}

            # Data
            --input-key messages --label-key label
            --apply-chat-template --rollout-shuffle
            --prompt-data {{data_path}}/haiku/train.parquet

            # Custom reward model
            --rm-type remote_rm
            --rm-url https://modal-labs-joy-dev--llm-judge-reward-model-llmjudgeflash.us-east.modal.direct/score

            --num-rollout 50
            --rollout-batch-size 128
            --n-samples-per-prompt 8
            --global-batch-size 64

            # SGLang
            --rollout-num-gpus-per-engine 2
            --sglang-mem-fraction-static 0.7

            --rollout-max-response-len 300

            --rollout-temperature 1
            --rollout-skip-special-tokens

            # Orchestration
            --actor-num-nodes 1
            --actor-num-gpus-per-node 8
            --colocate

            # Eval
            --eval-prompt-data haiku {{data_path}}/haiku/test.parquet
            --eval-interval 20
            --n-samples-per-eval-prompt 8
            --eval-max-response-len 300
            --eval-top-p 1
        """,
    )
