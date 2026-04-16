"""Kimi-K2.5 LoRA training — 8x H200, colocated.

Run: EXPERIMENT_CONFIG=kimi_k25_lora modal run -d miles/modal_train.py::train
"""

from configs.kimi_k25_fullparam_smoke import _Miles as _FullParamMiles, modal  # noqa: F401


class _Miles(_FullParamMiles):
    only_train_params_name_list = None

    lora_rank = 32
    lora_alpha = 32
    lora_dropout = 0.0
    target_modules = (
        "linear_q_proj,linear_q_down_proj,linear_q_up_proj,"
        "linear_kv_down_proj,linear_kv_up_proj,"
        "linear_proj,linear_fc1,linear_fc2"
    )

    save = "/checkpoints/Kimi-K2.5-lora-ckpt"
    lr = 1e-5
    wandb_group = "kimi-k25-lora"


miles = _Miles()
