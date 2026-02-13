"""Configuration for Qwen3-0.6B GRPO training on Haiku dataset with reasoning."""

from .base import (
    QWEN3_4B_MODEL_ARGS,
    RLConfig,
    DEFAULT_TRAINING_ARGS,
    DEFAULT_OPTIMIZER_ARGS,
    DEFAULT_GRPO_ARGS,
)

from llm_judges.deploy import get_judge_url, ACTIVE_JUDGE_TYPE

NUM_ROLLOUT = 200

def get_config(run_name: str = "qwen3-4b-haiku") -> RLConfig:
    return RLConfig(
        model_name="Qwen3-4B",
        model_id="Qwen/Qwen3-4B",

        # Modal settings
        n_nodes=1,
        gpu="H200:8",
        app_name=f"slime-qwen3-4b-haiku-{ACTIVE_JUDGE_TYPE.value}",
        sync=True,

        # Wandb
        wandb_project="slime-haiku",
        wandb_run_name_prefix=run_name,

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
            --apply-chat-template-kwargs '{{"enable_thinking": false}}'
            --prompt-data {{data_path}}/haiku/train.parquet

            # Custom reward model - MUST extract final answer (see below)
            --rm-type remote_rm
            --rm-url {get_judge_url(ACTIVE_JUDGE_TYPE)}/score

            --num-rollout {NUM_ROLLOUT}
            --rollout-batch-size 128
            --n-samples-per-prompt 8
            --global-batch-size 64

            # SGLang
            --rollout-num-gpus-per-engine 2
            --sglang-mem-fraction-static 0.7
            
            # IMPORTANT: Allow reasoning + answer, but keep it reasonable
            --rollout-max-response-len 100
            
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