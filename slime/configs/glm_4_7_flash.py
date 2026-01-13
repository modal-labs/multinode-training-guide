"""Configuration for GLM-4.7-Flash (30B MoE with MLA) GRPO training.

Based on: scripts/models/glm4.7-30B-A3B.sh and run.sh from THUDM internal

NOTE: You must download GLM-4.7-Flash to /models/ volume before running.

modal run slime/modal_train.py::train_multi_node --config "glm-4-7-flash"
modal run -d slime/modal_train.py::train_multi_node --config "glm-4-7-flash"
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

# GLM-4.7-Flash architecture - from glm4.7-30B-A3B.sh MODEL_ARGS
_ARCHITECTURE = ModelArchitectureConfig(
    # Core architecture
    num_layers=47,  # N_DENSE_LAYERS(1) + N_MOE_LAYERS(46)
    hidden_size=2048,  # NHIDDEN
    ffn_hidden_size=10240,  # FFN_HIDDEN
    num_attention_heads=20,  # NHEADS
    vocab_size=154880,
    norm_epsilon=1e-5,  # rms_norm_eps
    make_vocab_size_divisible_by=64,
    # Position embeddings
    position_embedding_type="rope",
    no_position_embedding=True,
    rotary_base=1000000,
    no_rope_fusion=True,
    # Normalization & activation
    normalization="RMSNorm",
    swiglu=True,
    disable_bias_linear=True,
    add_qkv_bias=True,
    qk_layernorm=True,
    untie_embeddings_and_output_weights=True,
    # MLA (Multi-Latent Attention)
    group_query_attention=False,
    multi_latent_attention=True,
    q_lora_rank=768,
    kv_lora_rank=512,
    qk_head_dim=192,
    v_head_dim=256,
    kv_channels=192,
    qk_pos_emb_head_dim=64,
    # MoE (Mixture of Experts)
    moe_layer_freq="[0]*1+[1]*46",
    num_experts=64,  # MOE_ROUTED_EXPERTS
    moe_shared_expert_intermediate_size=1536,  # MOE_FFN_HIDDEN * MOE_SHARED_EXPERTS
    moe_router_topk=4,  # MOE_ACTIVE_ROUTED_EXPERTS
    moe_grouped_gemm=True,
    moe_permute_fusion=True,
    moe_ffn_hidden_size=1536,  # MOE_FFN_HIDDEN
    moe_router_score_function="sigmoid",
    moe_router_pre_softmax=True,
    moe_router_enable_expert_bias=True,
    moe_router_bias_update_rate=0,
    moe_router_load_balancing_type="aux_loss",
    moe_router_topk_scaling_factor=1.8,
    moe_aux_loss_coeff=0,
    moe_router_dtype="fp32",
    # From MISC_ARGS
    moe_token_dispatcher_type="flex",
    moe_enable_deepep=True,
)

# From PERF_ARGS, OPTIMIZER_ARGS, MISC_ARGS
_TRAINING = MegatronConfig(
    # PERF_ARGS
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=2,
    context_parallel_size=2,
    expert_model_parallel_size=8,
    expert_tensor_parallel_size=1,
    sequence_parallel=True,
    decoder_last_pipeline_num_layers=23,
    recompute_granularity="full",
    recompute_method="uniform",
    recompute_num_layers=1,
    use_dynamic_batch_size=True,
    max_tokens_per_gpu=32768,
    # ROLLOUT_ARGS
    global_batch_size=1024,
    # OPTIMIZER_ARGS
    optimizer="adam",
    lr=1e-6,
    lr_decay_style="constant",
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.98,
    optimizer_cpu_offload=True,
    overlap_cpu_optimizer_d2h_h2d=True,
    use_precision_aware_optimizer=True,
    # MISC_ARGS
    attention_dropout=0.0,
    hidden_dropout=0.0,
    attention_backend="flash",
)

# From SGLANG_ARGS
_SGLANG = SGLangConfig(
    rollout_num_gpus_per_engine=8,
    sglang_dp_size=8,
    sglang_mem_fraction_static=0.8,
    sglang_enable_dp_attention=True,
    sglang_enable_dp_lm_head=True,
    sglang_moe_dense_tp_size=1,
    # Speculative decoding (MTP)
    sglang_speculative_algorithm="EAGLE",
    sglang_speculative_num_steps=2,
    sglang_speculative_eagle_topk=1,
    sglang_speculative_num_draft_tokens=3,
    # Performance
    sglang_cuda_graph_max_bs=64,
    sglang_max_running_requests=512,
    # ROLLOUT_ARGS
    rollout_batch_size=128,
    n_samples_per_prompt=8,
    rollout_max_response_len=32768,
    rollout_temperature=1.0,
)

# From ray job submit command
_ORCHESTRATION = OrchestrationConfig(
    actor_num_nodes=2,
    actor_num_gpus_per_node=8,
    colocate=False,
    num_gpus_per_node=8,
    rollout_num_gpus=2*8,
)

# From GRPO_ARGS
_GRPO = GRPOConfig(
    advantage_estimator="grpo",
    use_kl_loss=True,
    kl_loss_coef=0.00,
    kl_loss_type="low_var_kl",
    kl_coef=0.00,
    entropy_coef=0.00,
)

# From EVAL_ARGS
_EVAL = EvalConfig(
    eval_interval=20,
    n_samples_per_eval_prompt=2,
    eval_max_response_len=16384,
    eval_temperature=0.6,
    eval_top_p=0.95,
)


def get_config() -> RLConfig:
    return RLConfig(
        model_name="GLM-4.7-Flash",
        model_org="zai-org",
        architecture=_ARCHITECTURE,
        training=_TRAINING,
        sglang=_SGLANG,
        orchestration=_ORCHESTRATION,
        grpo=_GRPO,
        eval=_EVAL,
        n_nodes=4,
        gpu="H100:8",
        app_name="slime-grpo-glm-4.7-flash-grpo",
        wandb_run_name_prefix="glm-4.7-flash-grpo",
        wandb_project="slime-grpo",
        sync=False,
    )
