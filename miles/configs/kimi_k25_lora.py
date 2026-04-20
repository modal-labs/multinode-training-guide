"""Kimi-K2.5 LoRA (standard ``LoRA``, not CanonicalLoRA) — 8x H200, colocated.

``target_modules`` is a comma-separated list. Each token may be **HF-style** (what
you see in ``weight_map`` suffixes) or **Megatron** (``linear_*``); Miles
``lora_utils.convert_target_modules_to_megatron`` normalizes before PEFT. See
``nan_wonderland/miles/miles/backends/megatron_utils/lora_utils.py``.

**Decoder — MLA attention (pick any subset):**

  HF token                 Megatron equivalent
  -----------------------  ---------------------
  ``q_a_proj``             ``linear_q_down_proj``
  ``q_b_proj``             ``linear_q_up_proj``
  ``kv_a_proj_with_mqa``   ``linear_kv_down_proj``
  ``kv_b_proj``            ``linear_kv_up_proj``
  ``o_proj``               ``linear_proj``

**Decoder — MoE / MLP FFN:**

  HF tokens ``gate_proj``, ``up_proj``, ``down_proj`` map to Megatron
  ``linear_fc1`` (fused SwiGLU gate+up) and ``linear_fc2`` (down). Listing
  ``gate_proj`` and ``up_proj`` still yields one ``linear_fc1`` adapter each
  place that leaf exists (routed experts under ``mlp.experts.*`` and shared
  under ``mlp.shared_experts.*``). The **router** is ``mlp.gate`` in HF
  (``…mlp.gate.weight``), not ``gate_proj``; there is no default token for it.

**Not covered here:** vision tower / ``mm_projector`` weights (separate subtree
in the checkpoint). Use ``modal run modal_hf_inspect.py`` from ``miles/`` for
exact key templates.

**SGLang colocation:** rollout config may drop ``q_b_proj`` / ``kv_b_proj`` if
default ``get_hidden_dim`` lacks them (``target_modules_hf_for_sglang_rollout``).

Run: ``EXPERIMENT_CONFIG=kimi_k25_lora modal run -d miles/modal_train.py::train``
"""

from configs.kimi_k25 import _Miles as _FullParamMiles, modal  # noqa: F401


class _Miles(_FullParamMiles):
    # LoRA controls freezing; drop the full-param freeze regex.
    only_train_params_name_list = None

    lora_rank = 32
    lora_alpha = 32
    lora_dropout = 0.0
    
    # see module docstring for full list
    target_modules = "q_a_proj,kv_a_proj_with_mqa,o_proj,gate_proj,up_proj,down_proj"

    lr = 1e-5
    wandb_group = "kimi-k25-lora"

miles = _Miles()
