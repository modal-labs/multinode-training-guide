"""Kimi-K2.5 LoRA — 8 H200 nodes, MLA attention only.

Run:
    EXPERIMENT_CONFIG=kimi_k25_lora_8nodes_mla modal run -d miles/modal_train.py::train

Inherits the 8-node parallelism from ``kimi_k25.py`` (TP=8, PP=2, CP=4,
EP=32). Adapters target only the MLA attention projections (``q_a_proj``,
``kv_a_proj_with_mqa``, ``o_proj``) — no MoE FFN. The heaviest LoRA path
(grouped MoE experts) is excluded, so memory fits comfortably at 8 nodes.
"""

from configs.kimi_k25 import _Miles as _FullParamMiles, modal  # noqa: F401


class _Miles(_FullParamMiles):
    lora_rank = 32
    lora_alpha = 32
    lora_dropout = 0.0

    actor_num_nodes = 8
    tensor_model_parallel_size = 8
    sequence_parallel = True
    pipeline_model_parallel_size = 2
    context_parallel_size = 4
    expert_model_parallel_size = 32
    expert_tensor_parallel_size = 1
    decoder_last_pipeline_num_layers = 30

    target_modules = "q_a_proj,kv_a_proj_with_mqa,o_proj"

    sglang_lora_backend = "triton"

    no_gradient_accumulation_fusion = True

    lr = 1e-5
    wandb_group = "kimi-k25-lora-8nodes-mla"


miles = _Miles()
