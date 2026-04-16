"""Kimi-K2.5 LoRA training — 8x H200, colocated.

Phase 1 scope: LoRA on MLA projections only (5 modules × 60 layers = 300 adapters).
MoE expert LoRA (linear_fc1/linear_fc2) is intentionally excluded here — sparse
expert + compressed-tensors + LoRA is the shakiest code path and should be
added in a follow-up once MLA LoRA is validated end-to-end.

Run: EXPERIMENT_CONFIG=kimi_k25_lora modal run -d miles/modal_train.py::train
"""

from configs.kimi_k25_fullparam_smoke import _Miles as _FullParamMiles, modal  # noqa: F401


class _Miles(_FullParamMiles):
    # LoRA controls freezing; drop the full-param freeze regex.
    only_train_params_name_list = None

    lora_rank = 32
    lora_alpha = 32
    lora_dropout = 0.0
    target_modules = (
        "linear_q_down_proj,linear_q_up_proj,"
        "linear_kv_down_proj,linear_kv_up_proj,"
        "linear_proj"
    )

    lr = 1e-5
    wandb_group = "kimi-k25-lora"

    def __init__(self):
        super().__init__()
        # Keep base expert weights in fake INT4 during forward so training
        # numerics match SGLang's INT4 inference. LoRA adapters train in bf16;
        # base is frozen so optimizer state stays tiny and the dequant scratch
        # fits in H200 memory.
        self.environment.update({
            "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
            "OPEN_TRAINING_INT4_GROUP_SIZE": "32",
        })


miles = _Miles()
