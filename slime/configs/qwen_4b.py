"""Configuration for Qwen3-4B GRPO training (pass-through style)."""

from .base import (
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
        app_name="slime-qwen3-4b",
        sync=True,

        # Wandb
        wandb_project="slime-grpo",
        wandb_run_name_prefix="qwen3-4b-gsm8k",

        # All slime args as raw CLI string
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
            {DEFAULT_DATA_ARGS}
            --prompt-data {{data_path}}/gsm8k/train.parquet
            --num-rollout 3000
            --rollout-batch-size 32
            --n-samples-per-prompt 8
            --global-batch-size 256

            # Eval data
            --eval-prompt-data gsm8k {{data_path}}/gsm8k/test.parquet

            # SGLang
            --rollout-num-gpus-per-engine 2
            --sglang-mem-fraction-static 0.7
            --rollout-max-response-len 8192
            --rollout-temperature 1

            # Orchestration
            --actor-num-nodes 1
            --actor-num-gpus-per-node 8
            --colocate

            # Eval
            --eval-interval 20
            --n-samples-per-eval-prompt 16
            --eval-max-response-len 16384
            --eval-top-p 1
        """,
    )
