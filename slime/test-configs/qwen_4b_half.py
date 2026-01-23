"""Configuration for Qwen3-4B-Instruct-2507 GRPO training."""

from .base import (
    RLConfig,
    ModelArchitectureConfig,
    MegatronConfig,
    SGLangConfig,
    OrchestrationConfig,
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

_TRAINING = MegatronConfig(
    tensor_model_parallel_size=2,
)

_SGLANG = SGLangConfig(
    rollout_num_gpus_per_engine=8,  # TP size - use all GPUs per engine
    rollout_batch_size=64,
)

_ORCHESTRATION = OrchestrationConfig(
    actor_num_gpus_per_node=8,
    actor_num_nodes=4,
    colocate=True,
)


def get_config() -> RLConfig:
    return RLConfig(
        model_name="Qwen3-4B-Instruct-2507",
        architecture=_ARCHITECTURE,
        training=_TRAINING,
        sglang=_SGLANG,
        orchestration=_ORCHESTRATION,
        app_name="slime-grpo-qwen-4b-async-half-v2",
        wandb_run_name_prefix="async-qwen-3-4b-instruct-gsm8k-half-v2",
        sync=False,
    )
