"""Configuration for Qwen2.5-0.5B-Instruct async GRPO training."""

from .base import (
    RLConfig,
    ModelArchitectureConfig,
    MegatronConfig,
    OrchestrationConfig,
)

_ARCHITECTURE = ModelArchitectureConfig(
    num_layers=24,
    hidden_size=896,
    ffn_hidden_size=4864,
    num_attention_heads=14,
    num_query_groups=2,
    add_qkv_bias=True,  # Qwen2.5 uses QKV bias
)

_TRAINING = MegatronConfig(
    tensor_model_parallel_size=1,
    max_tokens_per_gpu=9216,
)


def get_config() -> RLConfig:
    return RLConfig(
        model_name="Qwen2.5-0.5B-Instruct",
        architecture=_ARCHITECTURE,
        training=_TRAINING,
        orchestration=OrchestrationConfig(
            actor_num_nodes=2,
            actor_num_gpus_per_node=8,
            rollout_num_gpus=16,
        ),
        app_name="slime-grpo-qwen-0.5b-async",
        wandb_run_name_prefix="async-qwen-0.5b-gsm8k",
        sync=False,
    )
