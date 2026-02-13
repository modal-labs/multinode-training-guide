"""Configuration for Qwen3-8B GRPO training on 4 nodes."""

from .base import (
    RLConfig,
    QWEN3_8B_MODEL_ARGS,
    DEFAULT_TRAINING_ARGS,
    DEFAULT_OPTIMIZER_ARGS,
    DEFAULT_GRPO_ARGS,
    DEFAULT_DATA_ARGS,
)


def get_config() -> RLConfig:
    return RLConfig(
        model_name="Qwen3-8B",
        model_id="Qwen/Qwen3-8B",

        # Modal settings
        n_nodes=4,
        gpu="H100:8",
        app_name="slime-qwen3-8b-multi",
        sync=True,

        # Wandb
        wandb_project="slime-grpo",
        wandb_run_name_prefix="qwen3-8b-gsm8k-4node",

        # All slime args as raw CLI string
        slime_args=f"""
            # Model architecture
            {QWEN3_8B_MODEL_ARGS}

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
            --rollout-batch-size 128
            --n-samples-per-prompt 8
            --global-batch-size 1024

            # Eval data
            --eval-prompt-data gsm8k {{data_path}}/gsm8k/test.parquet

            # SGLang
            --rollout-num-gpus-per-engine 2
            --sglang-mem-fraction-static 0.7
            --rollout-max-response-len 8192
            --rollout-temperature 1

            # Orchestration
            --actor-num-nodes 4
            --actor-num-gpus-per-node 8
            --colocate

            # Eval
            --eval-interval 20
            --n-samples-per-eval-prompt 16
            --eval-max-response-len 16384
            --eval-top-p 1
        """,
    )
