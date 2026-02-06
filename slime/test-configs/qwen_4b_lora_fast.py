"""Fast iteration config for Qwen3-4B GRPO training with LoRA.

Optimized for quick development cycles:
- Small number of rollout samples
- Minimal eval (high interval)
- Small batch sizes
"""

from configs.base import (
    RLConfig,
    QWEN3_4B_MODEL_ARGS,
    DEFAULT_TRAINING_ARGS,
    DEFAULT_OPTIMIZER_ARGS,
    DEFAULT_GRPO_ARGS,
    DEFAULT_DATA_ARGS,
)


def get_config() -> RLConfig:
    return RLConfig(
        model_name="Qwen3-4B",
        model_id="Qwen/Qwen3-4B",

        # Modal settings
        n_nodes=1,
        gpu="H100:8",
        app_name="slime-qwen3-4b-lora-fast",
        sync=True,

        # Wandb
        wandb_project="slime-grpo-lora",
        wandb_run_name_prefix="qwen3-4b-lora-fast",

        # All slime args as raw CLI string
        slime_args=f"""
            # Model architecture
            {QWEN3_4B_MODEL_ARGS}

            # PEFT/LoRA configuration
            --peft-type lora
            --lora-rank 32
            --lora-alpha 32
            --lora-dropout 0.0
            --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2

            # Training parallelism and optimization
            {DEFAULT_TRAINING_ARGS}

            # Optimizer
            {DEFAULT_OPTIMIZER_ARGS}

            # GRPO algorithm
            {DEFAULT_GRPO_ARGS}

            # Data - minimal for fast iteration
            {DEFAULT_DATA_ARGS}
            --prompt-data {{data_path}}/gsm8k/train.parquet
            --num-rollout 64
            --rollout-batch-size 8
            --n-samples-per-prompt 4
            --global-batch-size 32

            # Eval - skip during fast iteration (very high interval)
            --eval-prompt-data gsm8k {{data_path}}/gsm8k/test.parquet
            --eval-interval 9999
            --n-samples-per-eval-prompt 4
            --eval-max-response-len 4096
            --eval-top-p 1

            # SGLang
            --rollout-num-gpus-per-engine 2
            --sglang-mem-fraction-static 0.7
            --rollout-max-response-len 2048
            --rollout-temperature 1

            # Orchestration
            --actor-num-nodes 1
            --actor-num-gpus-per-node 8
            --colocate
        """,
    )
