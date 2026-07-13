"""GLM-5.2 DAPO 32k padded — FP8 rollout A/B test, 8 nodes x 8 H200.

Identical to glm5_2_744b_a40b_lora_dapo_tilelang_32k_padstress except SGLang
serves the frozen base FP8-quantized (--quantization fp8, dynamic per-token
activation quant). Training stays bf16. Goals:

  1. rollout/prefill+decode speedup vs the bf16 baseline (~9 min/rollout)
  2. train_rollout_logprob_abs_diff — bf16 baseline is ~0.01; this measures
     the train/rollout policy mismatch introduced by FP8 serving

Short run (3 rollouts = 6 train steps), no checkpoints.

    EXPERIMENT_CONFIG=glm5_2_744b_a40b_lora_dapo_tilelang_32k_fp8roll \
        uv run modal run --detach miles/modal_train.py::train
"""

from configs import glm5_2_744b_a40b_lora_dapo_tilelang_32k_padstress as _base

from configs.base import ModalConfig

modal = ModalConfig(
    docker_image=_base.modal.docker_image,
    gpu=_base.modal.gpu,
    memory=_base.modal.memory,
    cloud=_base.modal.cloud,
    region=_base.modal.region,
    patch_files=[*_base.modal.patch_files, "miles/sglang_fp8_lora_fix.py"],
    image_run_commands=[
        *_base.modal.image_run_commands,
        # LoRA-B buffer sizing fix for quantized column-parallel layers
        # (validated on the 5-layer FP8 diag; see sglang_fp8_lora_fix.py).
        "python /tmp/sglang_fp8_lora_fix.py",
    ],
    image_env=dict(_base.modal.image_env),
)


class _Miles(_base._Miles):
    sglang_quantization = "fp8"

    num_rollout = 3
    save = None
    save_interval = None

    wandb_group = "glm5.2-744B-8node-tilelang-dapo-32k-fp8roll"


miles = _Miles()
