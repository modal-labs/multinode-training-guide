"""Qwen3-30B-A3B GRPO on DAPO-Math-17k — non-colocated 2-node actor + rollout.

Follows the same inheritance pattern as glm47_flash_dapo_noncolocate while
keeping the underlying Qwen3-30B-A3B training settings aligned with the repo.s
run-qwen3-30B-A3B.sh defaults and the guide docs.
"""

from configs import qwen3_dapo as _base
from configs.base import CHECKPOINTS_PATH, ModalConfig

modal = ModalConfig(
    gpu="H200",
    # local_slime="/home/ec2-user/nan_wonderland/slime",
    # patch_files=["/home/ec2-user/nan_wonderland/delta_sync_history_20260331/patches/sglang_delta_compression_working_non_colocate.patch"],
    # image_run_commands=[
    #     "uv pip install --system zstandard",
    #     "cd /sgl-workspace/sglang && patch -p1 < /tmp/sglang_delta_compression_working_non_colocate.patch && cd /root/slime",
    # ],
)


class _Slime(_base._Slime):
    ref_load = f"{CHECKPOINTS_PATH}/Qwen3-30B-A3B_torch_dist_tp4pp2"
    # ── Infrastructure ────────────────────────────────────────────────────────
    actor_num_nodes = 2
    colocate = False
    rollout_num_gpus = 16

    # ── Rollout ───────────────────────────────────────────────────────────────
    sglang_enable_dp_lm_head = True
    sglang_moe_dense_tp_size = 1
    sglang_max_running_requests = 512

    sglang_mem_fraction_static = 0.8
    sglang_cuda_graph_max_bs = 64
    optimizer_cpu_offload = False
    overlap_cpu_optimizer_d2h_h2d = False
    use_precision_aware_optimizer = False

    use_fault_tolerance = True

    # ── Training ──────────────────────────────────────────────────────────────
    tensor_model_parallel_size = 4
    sequence_parallel = True
    pipeline_model_parallel_size = 2
    context_parallel_size = 2
    skip_eval_before_train = True

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_group = "qwen3-30b-a3b-dapo-math-noncolocate"


slime = _Slime()
