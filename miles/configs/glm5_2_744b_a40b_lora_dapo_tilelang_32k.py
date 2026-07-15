"""GLM-5.2 full-model TileLang/THD on dapo-math — 32k context + 1k rollout, 8 nodes x 8 H200.

Uses the validated 64-GPU TileLang setup (safe-indices kernel patch, full
activation recompute, extended SGLang load timeout) with the dapo-math task.
No context parallelism (CP=1); 32k context fits without it (validated 2026-07-12).

    EXPERIMENT_CONFIG=glm5_2_744b_a40b_lora_dapo_tilelang_32k \
        uv run modal run miles/modal_train.py::download_data
    EXPERIMENT_CONFIG=glm5_2_744b_a40b_lora_dapo_tilelang_32k \
        uv run modal run --detach miles/modal_train.py::train
"""

from configs import glm5_2_744b_a40b_lora as _base
from configs.base import CHECKPOINTS_PATH, DATA_PATH, ModalConfig

_PATCH = "miles/glm5_tilelang_safe_indices.patch"

modal = ModalConfig(
    docker_image=_base.modal.docker_image,
    gpu=_base.modal.gpu,
    memory=_base.modal.memory,
    cloud=_base.modal.cloud,
    region=_base.modal.region,
    patch_files=[_PATCH],
    image_run_commands=[
        *_base.modal.image_run_commands,
        (
            "cd /usr/local/lib/python3.12/dist-packages && "
            "git apply --check /tmp/glm5_tilelang_safe_indices.patch && "
            "git apply /tmp/glm5_tilelang_safe_indices.patch"
        ),
        (
            "python -c \"from pathlib import Path; "
            "p = Path('/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py'); "
            "s = p.read_text(); "
            "old = 'UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480'; "
            "assert s.count(old) == 1; "
            "p.write_text(s.replace(old, 'UNBALANCED_MODEL_LOADING_TIMEOUT_S = 1800'))\""
        ),
        # LoRA checkpoint fix: dp_rank_0's mkdir on the shared volume is not
        # visible to containers on other nodes, so every rank must mkdir itself
        # before writing its training_state_rank{N}.pt.
        (
            "python -c \"from pathlib import Path; "
            "p = Path('/root/miles/miles/backends/megatron_utils/lora_utils.py'); "
            "s = p.read_text(); "
            "old = 'if is_dp_rank_0:\\n        save_path.mkdir(parents=True, exist_ok=True)'; "
            "new = 'save_path.mkdir(parents=True, exist_ok=True)'; "
            "assert s.count(old) == 1; "
            "p.write_text(s.replace(old, new))\""
        ),
    ],
    image_env=dict(_base.modal.image_env),
)


class _Miles(_base._Miles):
    dsa_attention_backend = "tilelang"
    qkv_format = "thd"
    data_pad_size_multiplier = None
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1

    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"

    seq_length = 32768
    rollout_max_context_len = 32768
    rollout_max_response_len = 1024

    num_rollout = 50
    save = f"{CHECKPOINTS_PATH}/GLM-5.2-lora-dapo-32k-ckpt"
    save_interval = 10

    wandb_group = "glm5.2-744B-8node-tilelang-dapo-32k"

    def download_data(self) -> None:
        import os

        from huggingface_hub import snapshot_download

        os.makedirs(f"{DATA_PATH}/dapo-math-17k", exist_ok=True)
        snapshot_download(
            repo_id="zhuzilin/dapo-math-17k",
            repo_type="dataset",
            local_dir=f"{DATA_PATH}/dapo-math-17k",
        )


miles = _Miles()
