"""Configuration for GLM-4.7-Flash (30B MoE with MLA) GRPO training (pass-through style).

Based on: scripts/models/glm4.7-30B-A3B.sh and run.sh from THUDM internal

NOTE: You must download GLM-4.7-Flash to /models/ volume before running.

modal run slime/modal_train.py::train_multi_node --config "glm-4-7-flash"
modal run -d slime/modal_train.py::train_multi_node --config "glm-4-7-flash"
"""

from .base import (
    RLConfig,
    GLM_4_7_FLASH_MODEL_ARGS,
    GLM_4_7_FLASH_TRAINING_ARGS,
    DEFAULT_OPTIMIZER_ARGS,
    DEFAULT_GRPO_ARGS,
    DEFAULT_DATA_ARGS,
)


def get_config() -> RLConfig:
    return RLConfig(
        model_name="GLM-4.7-Flash",
        model_id="zai-org/GLM-4.7-Flash",

        # Modal settings
        n_nodes=4,
        gpu="H100:8",
        app_name="slime-grpo-glm-4.7-flash",
        sync=False,

        # Wandb
        wandb_project="slime-grpo",
        wandb_run_name_prefix="glm-4.7-flash-grpo",

        # All slime args as raw CLI string
        slime_args=f"""
            # Model architecture (30B MoE with MLA)
            {GLM_4_7_FLASH_MODEL_ARGS}

            # Training parallelism and optimization
            {GLM_4_7_FLASH_TRAINING_ARGS}

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
            --rollout-num-gpus-per-engine 8
            --sglang-dp-size 8
            --sglang-mem-fraction-static 0.8
            --rollout-max-response-len 32768
            --rollout-temperature 1
            --sglang-enable-dp-attention
            --sglang-enable-dp-lm-head
            --sglang-moe-dense-tp-size 1
            --sglang-cuda-graph-max-bs 64
            --sglang-max-running-requests 512
            --sglang-speculative-algorithm EAGLE
            --sglang-speculative-num-steps 2
            --sglang-speculative-eagle-topk 1
            --sglang-speculative-num-draft-tokens 3

            # Orchestration
            --actor-num-nodes 2
            --actor-num-gpus-per-node 8
            --rollout-num-gpus 16

            # Eval
            --eval-interval 20
            --n-samples-per-eval-prompt 2
            --eval-max-response-len 16384
            --eval-temperature 0.6
            --eval-top-p 0.95
        """,
    )
