"""GLM-5.2 LoRA DAPO training with FP8 rollout — 8 nodes x 8 H200, TileLang/THD, 32k context.

The finalized FP8-rollout setup: the rollout engines serve the official
block-quantized ``zai-org/GLM-5.2-FP8`` checkpoint while the trainer keeps
bf16 ``zai-org/GLM-5.2``. LoRA adapters (bf16) sync to the engines every
rollout. Measured vs the bf16/bf16 baseline: generation ~25-40% faster
(351-450s vs ~540s per rollout), train_rollout_logprob_abs_diff ~0.040
(bf16 ~0.010, online-quant fp8 ~0.057), no sampling NaNs, rewards learn.

Fixes this configuration depends on (see each patch's docstring):
  - glm5_tilelang_safe_indices.patch: TileLang sparse-MLA backward NaNs
    (unsafe padded-index access + aggressive shared-memory merge miscompile).
  - sglang_fp8_lora_fix.py: LoRA-B buffer mis-sizing on quantized
    column-parallel layers crashed engine init under --quantization fp8 /
    quantized checkpoints.
  - sglang_tp1_shared_expert_fix.py: SGLANG_SHARED_EXPERT_TP1-replicated
    shared expert was double-added (once per TP rank) whenever the post-MoE
    all-reduce is deferred/replaced (FlashInfer AllReduce Fusion,
    dp-attention reduce-scatterv); folded into the pre-reduction add.
  - SGLANG_SHARED_EXPERT_TP1=1: the 128x128 block-quantized checkpoint cannot
    TP32-shard the shared expert (2048/32 = 64-row shards < one scale block),
    so replicate it instead (~25 MB/rank).

    EXPERIMENT_CONFIG=glm5_2_744b_a40b_lora_dapo_tilelang_32k_fp8 \
        uv run modal run miles/modal_train.py::download_model
    EXPERIMENT_CONFIG=glm5_2_744b_a40b_lora_dapo_tilelang_32k_fp8 \
        uv run modal run miles/modal_train.py::download_data
    EXPERIMENT_CONFIG=glm5_2_744b_a40b_lora_dapo_tilelang_32k_fp8 \
        uv run modal run --detach miles/modal_train.py::train
"""

from configs import glm5_2_744b_a40b_lora_dapo_tilelang_32k as _base
from configs.base import CHECKPOINTS_PATH, ModalConfig

_FP8_CHECKPOINT = "zai-org/GLM-5.2-FP8"

modal = ModalConfig(
    docker_image=_base.modal.docker_image,
    gpu=_base.modal.gpu,
    memory=_base.modal.memory,
    cloud=_base.modal.cloud,
    region=_base.modal.region,
    patch_files=[
        *_base.modal.patch_files,
        "miles/sglang_fp8_lora_fix.py",
        "miles/sglang_tp1_shared_expert_fix.py",
    ],
    image_run_commands=[
        *_base.modal.image_run_commands,
        "python /tmp/sglang_fp8_lora_fix.py",
        "python /tmp/sglang_tp1_shared_expert_fix.py",
    ],
    image_env=dict(_base.modal.image_env),
)


class _Miles(_base._Miles):
    environment = {
        **_base._Miles.environment,
        "SGLANG_SHARED_EXPERT_TP1": "1",
    }

    # The pre-quantized checkpoint carries its own quantization_config;
    # do NOT also force online quantization.
    sglang_quantization = None

    # Rollout base differs from hf_checkpoint, so adapter sync must be
    # opted into explicitly.
    sglang_config = {
        "sglang": [
            {
                "name": "actor",
                "model_path": _FP8_CHECKPOINT,
                "update_weights": True,
                "num_gpus_per_engine": 32,
                "server_groups": [
                    {"worker_type": "regular", "num_gpus": 64},
                ],
            }
        ]
    }

    # 4096-token response budget so completions can finish and reward can
    # improve (1024 truncated heavily on dapo-math).
    rollout_max_response_len = 4096

    # 2 gradient steps per rollout (128 samples / global_batch_size 64).
    num_rollout = 50
    save = f"{CHECKPOINTS_PATH}/GLM-5.2-lora-dapo-32k-fp8-ckpt"
    save_interval = 10

    wandb_group = "glm5.2-744B-8node-tilelang-dapo-32k-fp8"

    def download_model(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(self.hf_checkpoint, max_workers=32)
        snapshot_download(_FP8_CHECKPOINT, max_workers=32)


miles = _Miles()
