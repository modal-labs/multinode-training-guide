from __future__ import annotations

from .base import (
    DEFAULT_GRPO_ARGS,
    DEFAULT_OPTIMIZER_ARGS,
    DEFAULT_TRAINING_ARGS,
    QWEN3_8B_MODEL_ARGS,
    RLConfig,
)


def get_config() -> RLConfig:
    return RLConfig(
        model_name="Qwen3-8B",
        model_id="Qwen/Qwen3-8B",
        app_name="slime-qwen8b-code-golf",
        n_nodes=4,
        gpu="H100:8",
        sync=True,
        wandb_project="slime-code-golf",
        wandb_run_name_prefix="qwen8b-mbpp-harbor-thinking-4k-512",
        harbor_rm_profile=True,
        harbor_rm_log_samples=True,
        slime_args=f"""
            # Model
            {QWEN3_8B_MODEL_ARGS}

            # Training + optimizer + GRPO
            {DEFAULT_TRAINING_ARGS}
            {DEFAULT_OPTIMIZER_ARGS}
            {DEFAULT_GRPO_ARGS}

            # Dataset format
            --input-key messages
            --label-key label
            --apply-chat-template
            --apply-chat-template-kwargs '{{"enable_thinking": true}}'
            --prompt-data {{data_path}}/mbpp_harbor/slime/train.parquet
            --eval-prompt-data mbpp {{data_path}}/mbpp_harbor/slime/test.parquet

            # Rollout / batching (64 * 8 = 512 samples)
            --num-rollout 2000
            --rollout-batch-size 64
            --n-samples-per-prompt 8
            --global-batch-size 512
            --rollout-max-response-len 4096
            --rollout-temperature 0.9
            --eval-max-response-len 4096
            --n-samples-per-eval-prompt 8

            # Custom reward model (Harbor + Modal sandbox scoring)
            --rm-type math
            --custom-rm-path custom_rm.custom_rm

            # SGLang rollout engines
            --rollout-num-gpus-per-engine 2
            --sglang-mem-fraction-static 0.7

            # Distributed orchestration
            --actor-num-nodes 4
            --actor-num-gpus-per-node 8
            --colocate

            # Eval cadence
            --eval-interval 20
            --eval-top-p 1

            # Save checkpoints to volume
            --save {{checkpoints_path}}/qwen8b_code_golf_thinking_4k_512
            --save-interval 20
        """,
    )
