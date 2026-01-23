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
    sglang_mem_fraction_static=0.5,
    rollout_batch_size=256,
)

_ORCHESTRATION = OrchestrationConfig(
    actor_num_nodes=1,
    rollout_num_gpus=3*8,
    over_sampling_batch_size=512,
)


def get_config() -> RLConfig:
    return RLConfig(
        model_name="Qwen3-4B-Instruct-2507",
        architecture=_ARCHITECTURE,
        training=_TRAINING,
        sglang=_SGLANG,
        orchestration=_ORCHESTRATION,
        app_name="slime-grpo-qwen-4b-async-1T3R-v9-rbs256",
        wandb_run_name_prefix="async-qwen-3-4b-instruct-gsm8k-1T3R-v9-rbs256",
        sync=False,
    )
