"""Kimi-K2.5 LoRA — 8 H200 nodes (8 GPUs each), colocated.

Run:
    EXPERIMENT_CONFIG=kimi_k25_lora modal run -d miles/modal_train.py::train

What to tune:

- ``lora_rank`` / ``lora_alpha``: rank and scaling. 32/32 is a reasonable default.

- ``target_modules``: comma-separated HF-style tokens (Megatron names work too;
  miles normalizes via ``lora_utils.convert_target_modules_to_megatron``).
  Each token attaches LoRA to *every* module whose HF leaf name matches,
  across all layers.

  Attention (K2.5 uses MLA):
    q_a_proj              hidden  -> q_lora_rank            (MLA-specific)
    q_b_proj              q_lora  -> num_heads * qk_head    (MLA-specific)
    kv_a_proj_with_mqa    hidden  -> kv_lora + qk_rope      (MLA-specific)
    kv_b_proj             kv_lora -> num_heads * (qk+v)     (MLA-specific)
    o_proj                attn_out -> hidden                (any attention)

  FFN — all three FFN kinds, NOT just MoE (one token hits all three):
    gate_proj / up_proj   -> fused linear_fc1 on:
                               - dense MLP (first ``first_k_dense_replace`` layers)
                               - routed experts (``mlp.experts.*``)
                               - shared experts (``mlp.shared_experts``)
    down_proj             -> linear_fc2 on the same three places

  Not targetable by default: router (``mlp.gate.weight``), vision tower,
  mm_projector. Inspect checkpoint keys via ``modal run modal_hf_inspect.py``.

  SGLang caveat: rollout drops ``q_b_proj`` / ``kv_b_proj`` if its default
  ``get_hidden_dim`` lacks them (see ``target_modules_hf_for_sglang_rollout``).

- ``lr``: 1e-5 default; PEFT usually tolerates 10x higher than full-param.

- SGLang rollout flags (don't change unless you know why):
    sglang_lora_backend = "triton"
    sglang_lora_use_virtual_experts = True
    sglang_experts_shared_outer_loras = True
        Megatron-Bridge attaches ONE shared adapter per MoE module (across all
        experts). This flag tells SGLang to allocate expert_dim=1 for all four
        MoE LoRA matrices and broadcast at forward time — matches training
        and saves ~26+ GiB of otherwise wasted per-expert buffers.

Throughput knobs (see ``kimi_k25.py`` for cross-cutting notes):
  - ``max_tokens_per_gpu`` (+ ``log_probs_max_tokens_per_gpu``)
  - ``rollout_max_response_len`` (increase if you see high truncated_ratio)
  - ``rollout_batch_size``
  - ``n_samples_per_prompt``
  - ``sglang_mem_fraction_static``
"""

from configs.kimi_k25 import _Miles as _FullParamMiles, modal  # noqa: F401


class _Miles(_FullParamMiles):
    # LoRA controls freezing; drop the full-param freeze regex.
    only_train_params_name_list = None

    lora_rank = 32
    lora_alpha = 32
    lora_dropout = 0.0

    _MOE_LORA_INTERVAL = 2   # 1=every layer, 2=every 2nd, 3=every 3rd, ...
    _MOE_LORA_LAYERS = list(range(_MOE_LORA_INTERVAL, 61, _MOE_LORA_INTERVAL))
    target_modules = ",".join(
        # MLA attention — every layer:
        ["q_a_proj", "kv_a_proj_with_mqa", "o_proj"]
        # Routed MoE FFN — every Nth layer:
        + [
            f"*.layers.{i}.mlp.experts.{mod}"
            for i in _MOE_LORA_LAYERS
            for mod in ("linear_fc1", "linear_fc2")
        ]
    )
    experts_shared_outer_loras = True

    sglang_lora_backend = "triton"
    disable_parameter_transpose_cache = True
    sglang_lora_use_virtual_experts = True
    sglang_experts_shared_outer_loras = True

    lr = 1e-5
    wandb_group = "kimi-k25-lora"

miles = _Miles()
