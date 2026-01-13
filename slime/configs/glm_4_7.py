"""Configuration for GLM-4.7 (358B MoE) GRPO training.

Based on: https://huggingface.co/zai-org/GLM-4.7/blob/main/config.json

NOTE: You must download GLM-4.7 to /models/ volume before running.
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

# GLM-4.7 architecture from HuggingFace config.json
# MoE: 160 routed experts, 8 active per token, 1 shared expert
# First 3 layers are dense, remaining 89 are MoE
_ARCHITECTURE = ModelArchitectureConfig(
    num_layers=92,
    hidden_size=5120,
    ffn_hidden_size=12288,  # intermediate_size (dense layers)
    num_attention_heads=96,
    num_query_groups=8,  # num_key_value_heads (GQA)
    kv_channels=128,  # head_dim
    vocab_size=151552,
    rotary_base=1000000,  # rope_theta
    norm_epsilon=1e-5,  # rms_norm_eps
    normalization="RMSNorm",
    swiglu=True,  # hidden_act: "silu"
    disable_bias_linear=False,
    add_qkv_bias=True,  # attention_bias: true
    qk_layernorm=True,  # use_qk_norm: true
    untie_embeddings_and_output_weights=True,  # tie_word_embeddings: false in HF
    # MoE (Mixture of Experts)
    moe_layer_freq="[0]*3+[1]*89",  # first_k_dense_replace=3, remaining 89 MoE
    num_experts=160,  # n_routed_experts
    moe_shared_expert_intermediate_size=1536,
    moe_router_topk=8,  # num_experts_per_tok
    moe_grouped_gemm=True,
    moe_permute_fusion=True,
    moe_ffn_hidden_size=1536,
    moe_router_score_function="sigmoid",
    moe_router_pre_softmax=True,
    moe_router_enable_expert_bias=True,
    moe_router_bias_update_rate=0,
    moe_router_load_balancing_type="seq_aux_loss",
    moe_router_topk_scaling_factor=2.5,  # routed_scaling_factor
    moe_aux_loss_coeff=0,
    moe_router_dtype="fp32",
    moe_token_dispatcher_type="flex",
    moe_enable_deepep=True,
)

_TRAINING = MegatronConfig(
    # Parallelism - 358B model needs heavy distribution
    # With 64 GPUs: TP=8 (within node) * PP=4 (across nodes) * DP=2 = 64
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=4,
    context_parallel_size=2,
    expert_model_parallel_size=16,  # 160 experts / 8 = 20 experts per GPU
    expert_tensor_parallel_size=1,
    sequence_parallel=True,
    decoder_last_pipeline_num_layers=23,  # 92 layers / 4 pipeline stages = 23
    # Batching
    max_tokens_per_gpu=16384,  # Reduced for large model
    use_dynamic_batch_size=True,
    global_batch_size=512,  # Reduced for memory
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
    # Recompute (memory optimization) - ESSENTIAL for 358B
    recompute_granularity="full",
    recompute_method="uniform",
    recompute_num_layers=1,
    # Optimizer offload - ESSENTIAL for 358B
    optimizer_cpu_offload=True,
    overlap_cpu_optimizer_d2h_h2d=True,
    use_precision_aware_optimizer=True,
)

_SGLANG = SGLangConfig(
    # 358B model needs TP=8 for inference too
    rollout_num_gpus_per_engine=8,
    sglang_dp_size=4,  # 32 rollout GPUs / 8 TP = 4 DP engines
    sglang_mem_fraction_static=0.85,
    rollout_batch_size=64,  # Reduced for large model
    n_samples_per_prompt=8,
    rollout_max_response_len=16384,  # Reduced for memory
    rollout_temperature=1.0,
    sglang_enable_dp_attention=True,
    sglang_enable_dp_lm_head=True,
    sglang_moe_dense_tp_size=1,
    sglang_cuda_graph_max_bs=32,  # Reduced
    sglang_max_running_requests=256,  # Reduced

    sglang_speculative_algorithm="EAGLE",
    sglang_speculative_num_steps=2,
    sglang_speculative_eagle_topk=1,
    sglang_speculative_num_draft_tokens=3,
)

_ORCHESTRATION = OrchestrationConfig(
    actor_num_nodes=8,
    actor_num_gpus_per_node=8,
    rollout_num_gpus=4*8,  # 4 nodes for rollout (32 GPUs)
    colocate=False,  # Separate training and rollout
    num_gpus_per_node=8,
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

# Eval args
_EVAL_ARGS = [
    "--eval-temperature 0.6",
    "--eval-top-p 0.95",
]

_EXTRA_ARGS = _EVAL_ARGS


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
        n_nodes=12,  # 8 training + 4 rollout nodes
        gpu="B200:8",
        app_name="slime-grpo-glm-4.7-1-26-cp2-ep-16",
        wandb_run_name_prefix="glm-4.7-grpo",
        wandb_project="slime-grpo",
        sync=False,
        extra_args=_EXTRA_ARGS,
    )
