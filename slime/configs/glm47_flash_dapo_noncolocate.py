"""GLM-4.7-Flash (30B-A3B MoE) GRPO on DAPO-Math-17k — non-colocated 2-node (2×8 H100 actor + 1×8 rollout).

TP=2, PP=2, CP=2, EP=8 → TP×EP=16, requires 3 nodes total.
Checkpoint: convert with nproc=4 (TP=2, PP=2, decoder_last=23) → GLM-4.7-Flash_torch_dist_tp2pp2
"""

from configs.base import CHECKPOINTS_PATH
from configs.glm47_flash_dapo import _Slime as _Glm47Slime


class _Slime(_Glm47Slime):
    ref_load = f"{CHECKPOINTS_PATH}/GLM-4.7-Flash_torch_dist_tp2pp2"

    # ── Infrastructure ────────────────────────────────────────────────────────
    actor_num_nodes = 2
    colocate = False
    rollout_num_gpus = 16

    # ── Rollout ───────────────────────────────────────────────────────────────
    sglang_mem_fraction_static = 0.8
    sglang_speculative_num_steps = 3
    sglang_speculative_num_draft_tokens = 4
    sglang_cuda_graph_max_bs = 64

    # ── Training ──────────────────────────────────────────────────────────────
    tensor_model_parallel_size = 2
    sequence_parallel = True
    pipeline_model_parallel_size = 2
    context_parallel_size = 2
    decoder_last_pipeline_num_layers = 23
    skip_eval_before_train = True

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_group = "glm4.7-flash-dapo-math-noncolocate"


slime = _Slime()
