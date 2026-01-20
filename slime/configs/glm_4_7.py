"""Configuration for GLM-4.7-30B-A3B (MoE with MLA) GRPO training.

Based on: https://github.com/THUDM/slime/blob/main/scripts/run-glm4.7-30B-A3B.sh

NOTE: You must download GLM-4.7-Flash to /models/ volume before running.
"""

from .base import (
    RLConfig,
    ModelArchitectureConfig,
    MegatronConfig,
    SGLangConfig,
    OrchestrationConfig,
    GRPOConfig,
    EvalConfig,
)

# GLM-4.7-30B-A3B architecture from scripts/models/glm4.7-30B-A3B.sh
# MoE: 64 routed experts, 4 active, 1 shared
# MLA: Multi-Latent Attention
_ARCHITECTURE = ModelArchitectureConfig(
    num_layers=47,  # N_DENSE_LAYERS(1) + N_MOE_LAYERS(46)
    hidden_size=2048,  # NHIDDEN
    ffn_hidden_size=10240,  # FFN_HIDDEN (dense layers)
    num_attention_heads=20,  # NHEADS
    num_query_groups=20,  # GQA not used with MLA
    vocab_size=154880,
    rotary_base=1000000,
    normalization="RMSNorm",
    swiglu=True,
    disable_bias_linear=True,
    add_qkv_bias=True,
    qk_layernorm=True,
)

_TRAINING = MegatronConfig(
    # Parallelism
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=2,
    context_parallel_size=2,
    expert_model_parallel_size=8,
    expert_tensor_parallel_size=1,
    sequence_parallel=True,
    # Batching
    max_tokens_per_gpu=32768,
    use_dynamic_batch_size=True,
    global_batch_size=1024,
    # Optimizer
    optimizer="adam",
    lr=1e-6,
    lr_decay_style="constant",
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.98,
    # Misc
    attention_dropout=0.0,
    hidden_dropout=0.0,
    attention_backend="flash",
)

_SGLANG = SGLangConfig(
    rollout_num_gpus_per_engine=8,
    sglang_data_parallel_size=8,
    sglang_mem_fraction_static=0.8,
    rollout_batch_size=128,
    n_samples_per_prompt=8,
    rollout_max_response_len=32768,
    rollout_temperature=1.0,
)

_ORCHESTRATION = OrchestrationConfig(
    actor_num_nodes=2,
    actor_num_gpus_per_node=8,
    num_gpus_per_node=8,
    colocate=True,
    num_rollout_infinite=3000,
)

_GRPO = GRPOConfig(
    advantage_estimator="grpo",
    use_kl_loss=True,
    kl_loss_coef=0.00,
    kl_loss_type="low_var_kl",
    entropy_coef=0.00,
)

_EVAL = EvalConfig(
    eval_interval=20,
    n_samples_per_eval_prompt=2,
    eval_max_response_len=16384,
    eval_top_k=1,
)

# MoE architecture args (from MODEL_ARGS)
_MOE_ARGS = [
    "--moe-layer-freq '[0]*1+[1]*46'",  # 1 dense + 46 MoE layers
    "--num-experts 64",
    "--moe-shared-expert-intermediate-size 1536",  # MOE_FFN_HIDDEN * MOE_SHARED_EXPERTS
    "--moe-router-topk 4",  # MOE_ACTIVE_ROUTED_EXPERTS
    "--moe-grouped-gemm",
    "--moe-permute-fusion",
    "--moe-ffn-hidden-size 1536",  # MOE_FFN_HIDDEN
    "--moe-router-score-function sigmoid",
    "--moe-router-pre-softmax",
    "--moe-router-enable-expert-bias",
    "--moe-router-bias-update-rate 0",
    "--moe-router-load-balancing-type seq_aux_loss",
    "--moe-router-topk-scaling-factor 1.8",
    "--moe-aux-loss-coeff 0",
    "--moe-router-dtype fp32",
    "--moe-token-dispatcher-type flex",
    "--moe-enable-deepep",
]

# MLA (Multi-Latent Attention) args
_MLA_ARGS = [
    "--multi-latent-attention",
    "--q-lora-rank 768",
    "--kv-lora-rank 512",
    "--qk-head-dim 192",
    "--v-head-dim 256",
    "--kv-channels 192",
    "--qk-pos-emb-head-dim 64",
]

# Other model args
_MODEL_ARGS = [
    "--make-vocab-size-divisible-by 64",
    "--untie-embeddings-and-output-weights",
    "--position-embedding-type rope",
    "--no-position-embedding",
    "--no-rope-fusion",
]

# Recompute (memory optimization)
_RECOMPUTE_ARGS = [
    "--recompute-granularity full",
    "--recompute-method uniform",
    "--recompute-num-layers 1",
    "--decoder-last-pipeline-num-layers 23",
]

# Optimizer offload
_OPTIMIZER_ARGS = [
    "--optimizer-cpu-offload",
    "--overlap-cpu-optimizer-d2h-h2d",
    "--use-precision-aware-optimizer",
]

# SGLang advanced features
_SGLANG_ADVANCED_ARGS = [
    "--sglang-enable-dp-attention",
    "--sglang-enable-dp-lm-head",
    "--sglang-moe-dense-tp-size 1",
    "--sglang-speculative-algorithm EAGLE",
    "--sglang-speculative-num-steps 2",
    "--sglang-speculative-eagle-topk 1",
    "--sglang-speculative-num-draft-tokens 3",
    "--sglang-cuda-graph-max-bs 64",
    "--sglang-max-running-requests 512",
]

# Eval args
_EVAL_ARGS = [
    "--eval-temperature 0.6",
    "--eval-top-p 0.95",
]

_EXTRA_ARGS = _MOE_ARGS + _MLA_ARGS + _MODEL_ARGS + _RECOMPUTE_ARGS + _OPTIMIZER_ARGS + _SGLANG_ADVANCED_ARGS + _EVAL_ARGS


def get_config() -> RLConfig:
    return RLConfig(
        model_name="GLM-4.7",  # Must exist in /models/ volume
        model_org="zai-org",
        architecture=_ARCHITECTURE,
        training=_TRAINING,
        sglang=_SGLANG,
        orchestration=_ORCHESTRATION,
        grpo=_GRPO,
        eval=_EVAL,
        n_nodes=2,
        gpu="H100:8",
        app_name="slime-grpo-glm-4.7-30b-a3b",
        wandb_run_name_prefix="glm-4.7-30b-a3b-grpo",
        wandb_project="slime-grpo",
        sync=False,
        extra_args=_EXTRA_ARGS,
    )
