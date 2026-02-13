"""Configuration for GLM-4.7 (358B MoE) GRPO training (pass-through style).

Based on: https://huggingface.co/zai-org/GLM-4.7/blob/main/config.json

NOTE: You must download GLM-4.7 to /models/ volume before running.
"""

from .base import (
    RLConfig,
    GLM_4_7_MODEL_ARGS,
    GLM_4_7_TRAINING_ARGS,
    DEFAULT_OPTIMIZER_ARGS,
    DEFAULT_GRPO_ARGS,
    DEFAULT_DATA_ARGS,
)


def get_config() -> RLConfig:
    return RLConfig(
        model_name="GLM-4.7",
        model_id="zai-org/GLM-4.7",

        # Modal settings - 358B model needs heavy distribution
        # TODO: training updates are faster than rollouts
        # 8 training + 4 rollout nodes
        n_nodes=12,
        gpu="B200:8",
        app_name="slime-grpo-glm-4.7",
        sync=True,

        # Wandb
        wandb_project="slime-grpo",
        wandb_run_name_prefix="glm-4.7-grpo",

        # All slime args as raw CLI string
        slime_args=f"""
            # Model architecture (358B MoE)
            {GLM_4_7_MODEL_ARGS}

            # Training parallelism and optimization
            {GLM_4_7_TRAINING_ARGS}

            # Optimizer
            {DEFAULT_OPTIMIZER_ARGS}

            # GRPO algorithm
            {DEFAULT_GRPO_ARGS}

            # Data
            {DEFAULT_DATA_ARGS}
            --prompt-data {{data_path}}/gsm8k/train.parquet
            --num-rollout 10
            --rollout-batch-size 64
            --n-samples-per-prompt 8
            --global-batch-size 512

            # Eval data
            --eval-prompt-data gsm8k {{data_path}}/gsm8k/test.parquet

            # SGLang - 358B model needs TP=8 for inference
            --rollout-num-gpus-per-engine 32 # TODO: SUSSSSSS
            --sglang-dp-size 4
            --sglang-mem-fraction-static 0.7
            --rollout-max-response-len 8192
            --rollout-temperature 1
            --sglang-enable-dp-attention
            --sglang-enable-dp-lm-head
            --sglang-moe-dense-tp-size 1
            --sglang-cuda-graph-max-bs 32
            --sglang-max-running-requests 256
            --sglang-speculative-algorithm EAGLE
            --sglang-speculative-num-steps 2
            --sglang-speculative-eagle-topk 1
            --sglang-speculative-num-draft-tokens 3

            # Orchestration
            --actor-num-nodes 8
            --actor-num-gpus-per-node 8
            --rollout-num-gpus 32

            # Eval
            --eval-interval 20
            --n-samples-per-eval-prompt 2
            --eval-max-response-len 16384
            --eval-temperature 0.6
            --eval-top-p 0.95
        """,
    )
