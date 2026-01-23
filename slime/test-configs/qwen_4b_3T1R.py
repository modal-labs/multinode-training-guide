"""Configuration for Qwen3-4B-Instruct-2507 GRPO training."""

from .base import (
    RLConfig,
    ModelArchitectureConfig,
    MegatronConfig,
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
    max_tokens_per_gpu=4096,
)


def get_config() -> RLConfig:
    return RLConfig(
        model_name="Qwen3-4B-Instruct-2507",
        architecture=_ARCHITECTURE,
        training=_TRAINING,
        orchestration=OrchestrationConfig(actor_num_nodes=3, rollout_num_gpus=1*8),  # BROKEN, see Notion
        app_name="slime-grpo-qwen-4b-async-3T1R",
        wandb_run_name_prefix="async-qwen-3-4b-instruct-gsm8k-3T1R",
        sync=False,
    )
