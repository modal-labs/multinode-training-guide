"""Configuration for Qwen3-4B GRPO training on Haiku dataset."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path




_MODEL_INFO = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": ("qwen3-30b-a3b-instruct", "30b"),
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8": ("qwen3-235b-a22b-instruct-fp8", "235b"),
}


class JudgeType(str, Enum):
    STRICT = "strict"
    STRICT_LEVELED = "strict_leveled"
    NO_LLM = "no_llm"  # only use the structure score

class JudgeModelSize(str, Enum):
    QWEN3_30B = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    QWEN3_235B = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

    @property
    def model_name(self) -> str:
        return _MODEL_INFO[self.value][0]

    @property
    def shorthand(self) -> str:
        return _MODEL_INFO[self.value][1]



ACTIVE_JUDGE_TYPE = JudgeType.NO_LLM
ACTIVE_JUDGE_MODEL_SIZE = JudgeModelSize.QWEN3_30B



@dataclass
class RLConfig:
    """Training config that passes raw CLI args directly to slime."""

    model_name: str
    model_id: str

    # Modal settings
    n_nodes: int = 4
    gpu: str = "H100:8"


    # Wandb
    wandb_project: str = "example-haiku"
    wandb_run_name_prefix: str = ""

    # Raw CLI args passed directly to slime
    slime_args: str = ""

    # Extra args that get appended (for easy overrides)
    extra_args: list[str] = field(default_factory=list)

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

    def generate_train_args(self, data_path: Path, is_infinite_run: bool) -> str:
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(self.model_id)
        base_args = f"--hf-checkpoint {model_path} --ref-load {model_path}"

        cleaned_slime_args = self._clean_args(self.slime_args)
        cleaned_slime_args = cleaned_slime_args.replace("{data_path}", str(data_path))

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

def _get_judge_url(judge_type: JudgeType, judge_model_size: JudgeModelSize) -> str:
    return f"https://modal-labs-joy-dev--llm-judge-{judge_model_size.shorthand}-{judge_type.value}-llmjudge.us-east.modal.direct"


def _get_reward_model_args_from_judge_type(judge_type: JudgeType, judge_model_size: JudgeModelSize) -> str:
    if judge_type == JudgeType.STRICT or judge_type == JudgeType.STRICT_LEVELED:
        return f"""--rm-type remote_rm
            --rm-url {_get_judge_url(judge_type, judge_model_size)}/score"""
    elif judge_type == JudgeType.NO_LLM:
        return """--rm-type async_rm
            --custom-rm-path llm_judges.nlp.haiku_rm"""

def get_config(run_name: str = "qwen3-4b-haiku", judge_type = ACTIVE_JUDGE_TYPE, judge_model_size = ACTIVE_JUDGE_MODEL_SIZE) -> RLConfig:
    return RLConfig(
        model_name="Qwen3-4B",
        model_id="Qwen/Qwen3-4B",
        n_nodes=1,
        gpu="H200:8",
        wandb_project="example-haiku",
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
            --apply-chat-template-kwargs '{{"enable_thinking": false}}'
            --prompt-data {{data_path}}/haiku/train.parquet

            # Custom reward model
            {_get_reward_model_args_from_judge_type(judge_type, judge_model_size)}
            
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
