"""Configuration for Qwen3-4B-Instruct-2507 GRPO training."""

from .base import (
    TrainingConfig,
    ModelArchitectureConfig,
    PerformanceConfig,
    ClusterConfig,
)

_ARCHITECTURE = ModelArchitectureConfig(
    num_layers=36,
    hidden_size=2560,
    ffn_hidden_size=9728,
    num_attention_heads=32,
    num_query_groups=8,
    rotary_base=5000000,
    kv_channels=128,
    qk_layernorm=True,
)

_PERFORMANCE = PerformanceConfig(
    tensor_model_parallel_size=2,
    max_tokens_per_gpu=4096,
)


def get_config_sync() -> TrainingConfig:
    return TrainingConfig(
        model_name="Qwen3-4B-Instruct-2507",
        architecture=_ARCHITECTURE,
        performance=_PERFORMANCE,
        app_name="slime-grpo-qwen-4b-sync",
        wandb_run_name_prefix="sync-qwen-3-4b-instruct-gsm8k",
    )


def get_config_async() -> TrainingConfig:
    return TrainingConfig(
        model_name="Qwen3-4B-Instruct-2507",
        architecture=_ARCHITECTURE,
        performance=_PERFORMANCE,
        app_name="slime-grpo-qwen-4b-async",
        wandb_run_name_prefix="async-qwen-3-4b-instruct-gsm8k",
        use_async=True,
    )
