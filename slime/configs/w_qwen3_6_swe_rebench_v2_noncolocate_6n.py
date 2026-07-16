"""Qwen3.6-35B-A3B SWE-rebench-V2 (Python) agentic RL — noncolocate, six nodes.

Scale-out sibling of ``w_qwen3_6_swe_rebench_v2_noncolocate_3n``: same data
(SWE-rebench-V2 Python subset), model, algorithm, and dp-off TP2 engine recipe;
only the **topology** grows BOTH the training and rollout fleets.

Topology: 2 training nodes (16 GPU, Megatron) + 4 rollout nodes (32 GPU, 16× TP2
SGLang engines behind sgl-router) = 6 nodes / 48 GPU. The second train node just
doubles the data-parallel width: the inherited parallelism (TP2 × CP2 × PP1 ×
EP8) holds — model-parallel group stays 4, so DP goes 2 → 4 and expert-DP 1 → 2,
both clean against world_size=16 and global_batch_size=256.

Everything else (data repoint, eval transfer slice, download_data, model,
checkpoint, agent env, algorithm, optimizer) is inherited from the 3n config.

    EXPERIMENT_CONFIG=w_qwen3_6_swe_rebench_v2_noncolocate_6n \
        uv run --no-dev modal run -d slime/modal_train.py::train
"""

from configs import w_qwen3_6_swe_rebench_v2_noncolocate_3n as _base
from configs.base import CHECKPOINTS_PATH, run_tag

# Same Modal image / dev overlay as the 3n config.
modal = _base.modal

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3.6-35b-a3b-swe-rebench-v2-noncolocate-6n")


class _Slime(_base._Slime):
    # ── Topology: 2 train nodes + 4 rollout nodes ──────────────────────────────
    actor_num_nodes = 2            # 2 training nodes (16 GPU, Megatron; DP 2 → 4)
    actor_num_gpus_per_node = 8
    rollout_num_gpus = 32          # 4 rollout nodes → 32 // 2 = 16 TP2 engines

    # ── WandB / debug dumps (retag so they don't collide with the 3n run) ──────
    wandb_group = _RUN_TAG
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )


slime = _Slime()
