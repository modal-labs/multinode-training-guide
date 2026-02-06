"""Configuration for Qwen3-4B GRPO training on Haiku dataset with LLM judge."""

from .base import (
    RLConfig,
    QWEN3_4B_MODEL_ARGS,
    DEFAULT_TRAINING_ARGS,
    DEFAULT_OPTIMIZER_ARGS,
    DEFAULT_GRPO_ARGS,
)


def get_config() -> RLConfig:
    return RLConfig(
        model_name="Qwen3-4B",
        model_id="Qwen/Qwen3-4B",

        # Modal settings
        n_nodes=1,
        gpu="H200:8",
        app_name="slime-qwen3-4b-haiku",
        sync=True,

        # Wandb
        wandb_project="slime-grpo-haiku",
        wandb_run_name_prefix="qwen3-4b-haiku",

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

            # Data - using custom reward model path instead of --rm-type
            --input-key messages --label-key label
            --apply-chat-template --rollout-shuffle
            --prompt-data {{data_path}}/haiku/train.parquet

            # Custom reward model - SLIME may support this pattern
            # Check SLIME docs for exact flag name
            --rm-type remote_rm
            --rm-url https://modal-labs-joy-dev--llm-judge-reward-model-llmjudge-score.modal.run

            --num-rollout 2
            --rollout-batch-size 16
            --n-samples-per-prompt 4
            --global-batch-size 64

            # SGLang
            --rollout-num-gpus-per-engine 2
            --sglang-mem-fraction-static 0.7
            --rollout-max-response-len 512
            --rollout-temperature 1

            # Orchestration
            --actor-num-nodes 1
            --actor-num-gpus-per-node 8
            --colocate

            # Eval
            --eval-prompt-data haiku {{data_path}}/haiku/test.parquet
            --eval-interval 20
            --n-samples-per-eval-prompt 8
            --eval-max-response-len 512
            --eval-top-p 1
        """,
    )
