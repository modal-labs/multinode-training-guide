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
    global_batch_size=8192,
)

_SGLANG = SGLangConfig(
    rollout_num_gpus_per_engine=2,
    # sglang_pipeline_parallel_size=2,
    sglang_data_parallel_size=1,
    sglang_mem_fraction_static=0.9,
    rollout_batch_size=8192,
    rollout_max_response_len=2048,
)

_ORCHESTRATION = OrchestrationConfig(
    actor_num_nodes=1,
    actor_num_gpus_per_node=2,
    rollout_num_gpus=1*2,
    colocate=False,
    num_gpus_per_node=2,
)


def get_config() -> RLConfig:
    return RLConfig(
        model_name="Qwen3-4B-Instruct-2507",
        architecture=_ARCHITECTURE,
        training=_TRAINING,
        sglang=_SGLANG,
        orchestration=_ORCHESTRATION,
        # Modal settings (must match modal_train.py decorator)
        n_nodes=2,
        gpu="H100:2",
        app_name="slime-grpo-qwen-4b-async-1replica-train-staticmem0.9-bs8192-4gpus-maxresponse2048",
        wandb_run_name_prefix="async-qwen-3-4b-instruct-gsm8k--train-1replica-staticmem0.9-bs8192-4gpus-maxresponse2048",
        sync=False,
    )
