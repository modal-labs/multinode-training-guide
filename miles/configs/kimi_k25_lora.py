"""Kimi-K2.5 LoRA — 16 H200 nodes (8 GPUs each), colocated.

Run:
    EXPERIMENT_CONFIG=kimi_k25_lora modal run -d miles/modal_train.py::train

Knobs to tune:
  - ``lora_rank`` / ``lora_alpha``: 32/32 default.
  - ``target_modules``: comma-separated HF leaf names; miles normalizes to
    Megatron names. ``q_b_proj`` / ``kv_b_proj`` are dropped by SGLang's
    default ``get_hidden_dim`` (see ``target_modules_hf_for_sglang_rollout``).
  - ``lr``: 1e-5 default; PEFT tolerates ~10x full-param.
  - Throughput knobs: see ``kimi_k25.py``.
"""

from configs.kimi_k25 import _Miles as _FullParamMiles, modal  # noqa: F401


class _Miles(_FullParamMiles):
    actor_num_nodes = 16
    tensor_model_parallel_size = 8
    sequence_parallel = True
    pipeline_model_parallel_size = 2
    context_parallel_size = 8
    expert_model_parallel_size = 64
    expert_tensor_parallel_size = 1
    decoder_last_pipeline_num_layers = 30

    wandb_group = "kimi-k25-lora"

    # LoRA config
    lr = 1e-5
    lora_rank = 32
    lora_alpha = 32
    lora_dropout = 0.0
    target_modules = "q_a_proj,kv_a_proj_with_mqa,o_proj,gate_proj,up_proj,down_proj"
    experts_shared_outer_loras = True
    lora_base_cpu_backup = True
    no_gradient_accumulation_fusion = True
    sglang_lora_backend = "triton"
    sglang_lora_use_virtual_experts = True

    # Off-policy IS correction: PPO operates on within-train ratio; TIS clamps
    # the cross-engine (sglang Marlin int4 vs Megatron fake-QAT bf16) ratio
    # with a wider bound than PPO's eps_clip, keeping kernel-rounding bias out
    # of PPO clipping. See analysis/README.md §8.4.
    use_tis = True


miles = _Miles()
