"""Minimal config set copied from slime/configs for the standalone ray example."""

from dataclasses import dataclass, field


@dataclass
class RLConfig:
    model_name: str
    model_id: str
    sync: bool = True
    slime_args: str = ""
    extra_args: list[str] = field(default_factory=list)

    def _clean_args(self, args: str) -> str:
        lines = []
        for line in args.split("\n"):
            if "#" in line:
                line = line[: line.index("#")]
            line = line.strip()
            if line:
                lines.append(line)
        return " ".join(lines)

    def generate_train_args(
        self, hf_model_path: str, checkpoints_path: str, data_path: str
    ) -> str:
        base_args = f"--hf-checkpoint {hf_model_path} --ref-load {hf_model_path}"
        cleaned = self._clean_args(self.slime_args)
        cleaned = cleaned.replace("{data_path}", data_path)
        cleaned = cleaned.replace("{checkpoints_path}", checkpoints_path)
        extra = " ".join(self.extra_args) if self.extra_args else ""
        return f"{base_args} {cleaned} {extra}".strip()


QWEN3_4B_MODEL_ARGS = """
    --num-layers 36 --hidden-size 2560 --ffn-hidden-size 9728
    --num-attention-heads 32 --group-query-attention --num-query-groups 8
    --kv-channels 128 --vocab-size 151936
    --normalization RMSNorm --norm-epsilon 1e-6 --swiglu
    --disable-bias-linear --qk-layernorm
    --use-rotary-position-embeddings --rotary-base 1000000
"""

QWEN3_8B_MODEL_ARGS = """
    --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288
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


def _qwen_4b() -> RLConfig:
    return RLConfig(
        model_name="Qwen3-4B",
        model_id="Qwen/Qwen3-4B",
        sync=True,
        slime_args=f"""
            {QWEN3_4B_MODEL_ARGS}
            {DEFAULT_TRAINING_ARGS}
            {DEFAULT_OPTIMIZER_ARGS}
            {DEFAULT_GRPO_ARGS}
            {DEFAULT_DATA_ARGS}
            --prompt-data {{data_path}}/gsm8k/train.parquet
            --num-rollout 3000
            --rollout-batch-size 32
            --n-samples-per-prompt 8
            --global-batch-size 256
            --eval-prompt-data gsm8k {{data_path}}/gsm8k/test.parquet
            --rollout-num-gpus-per-engine 2
            --sglang-mem-fraction-static 0.7
            --rollout-max-response-len 8192
            --rollout-temperature 1
            --actor-num-nodes 1
            --actor-num-gpus-per-node 8
            --colocate
            --eval-interval 20
            --n-samples-per-eval-prompt 16
            --eval-max-response-len 16384
            --eval-top-p 1
        """,
    )


def _qwen_8b_multi() -> RLConfig:
    return RLConfig(
        model_name="Qwen3-8B",
        model_id="Qwen/Qwen3-8B",
        sync=True,
        slime_args=f"""
            {QWEN3_8B_MODEL_ARGS}
            {DEFAULT_TRAINING_ARGS}
            {DEFAULT_OPTIMIZER_ARGS}
            {DEFAULT_GRPO_ARGS}
            {DEFAULT_DATA_ARGS}
            --prompt-data {{data_path}}/gsm8k/train.parquet
            --num-rollout 3000
            --rollout-batch-size 128
            --n-samples-per-prompt 8
            --global-batch-size 1024
            --eval-prompt-data gsm8k {{data_path}}/gsm8k/test.parquet
            --rollout-num-gpus-per-engine 2
            --sglang-mem-fraction-static 0.7
            --rollout-max-response-len 8192
            --rollout-temperature 1
            --actor-num-nodes 4
            --actor-num-gpus-per-node 8
            --colocate
            --eval-interval 20
            --n-samples-per-eval-prompt 16
            --eval-max-response-len 16384
            --eval-top-p 1
        """,
    )


_CONFIGS = {
    "qwen-4b": _qwen_4b,
    "qwen-8b-multi": _qwen_8b_multi,
}


def get_config(name: str) -> RLConfig:
    if name not in _CONFIGS:
        available = ", ".join(sorted(_CONFIGS.keys()))
        raise ValueError(f"Unknown config: {name}. Available configs: {available}")
    return _CONFIGS[name]()


def list_configs() -> list[str]:
    return sorted(_CONFIGS.keys())
