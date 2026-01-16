"""Configuration for Qwen2.5-0.5B-Instruct GRPO training."""

from .base import (
    TrainingConfig,
    ModelArchitectureConfig,
    PerformanceConfig,
    ClusterConfig,
)

_ARCHITECTURE = ModelArchitectureConfig(
    num_layers=24,
    hidden_size=896,
    ffn_hidden_size=4864,
    num_attention_heads=14,
    num_query_groups=2,
    add_qkv_bias=True,  # Qwen2.5 uses QKV bias
)

_PERFORMANCE = PerformanceConfig(
    tensor_model_parallel_size=1,
    max_tokens_per_gpu=9216,
)


def get_config_sync() -> TrainingConfig:
    return TrainingConfig(
        model_name="Qwen2.5-0.5B-Instruct",
        architecture=_ARCHITECTURE,
        performance=_PERFORMANCE,
        cluster=ClusterConfig(
            actor_num_nodes=2,
            actor_num_gpus_per_node=8,
            rollout_num_gpus=16,
            colocate=True,
        ),
        app_name="slime-grpo-qwen-0.5b-sync",
        wandb_run_name_prefix="sync-qwen-0.5b-gsm8k",
    )


def get_config_async() -> TrainingConfig:
    return TrainingConfig(
        model_name="Qwen2.5-0.5B-Instruct",
        architecture=_ARCHITECTURE,
        performance=_PERFORMANCE,
        cluster=ClusterConfig(
            actor_num_nodes=2,
            actor_num_gpus_per_node=8,
            rollout_num_gpus=16,
        ),
        app_name="slime-grpo-qwen-0.5b-async",
        wandb_run_name_prefix="async-qwen-0.5b-gsm8k",
        use_async=True,
    )
