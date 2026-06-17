"""Qwen3.6-35B-A3B DAPO-Math GRPO — colocated, single node (1× H100:8).

Same DAPO-Math-17k data / deepscaler reward / AIME-2024 eval as
``w_qwen3_6_dapo_colocate_2n`` (inherited wholesale, including download_data),
but every training / parallelism / rollout / engine knob is swapped for the
args that training-gym's ``Qwen3_6_35b_Recipe`` produces — i.e. the
``tutorials/singlenode/000_qwen35b`` tutorial, ported into this repo's
self-contained slime-config format.

That recipe is sized for a single 8× **H100** (80 GB) box, which is why its
args differ from the H200 SWE base:
  - PP 1 -> 2 and EP 8 -> 4: split the MoE across the pipeline with fewer expert
    shards to fit 80 GB (vs the H200's 141 GB).
  - SGLang DP-attention on (dp_size=4, ep_size=4, dp-lm-head, 512 max running
    requests): two TP-shrunk engines instead of one TP8 engine.
  - Smaller batches / longer responses: rollout_batch_size 16, global 128,
    rollout_max_response_len 16384, max_tokens_per_gpu 8192.
  - PP2 needs its own torch_dist conversion target (``..._tp2pp2``). One-time:
        EXPERIMENT_CONFIG=w_qwen3_6_dapo_colocate_1n_tggym \\
        modal run slime/modal_train.py::convert_hf_to_megatron_checkpoint
    (If your existing ``Qwen3.6-35B-A3B_torch_dist`` reshards across PP, just
    point ``ref_load`` back at it.)

Notes on faithfulness to 000_qwen35b.py:
  - ``num_rollout = 10`` is the tutorial's quick-demo value; the recipe default
    is 3000. Bump it for a real run.
  - Eval is kept from the dapo_2n base (AIME every 5 rollouts). The training-gym
    recipe runs eval as a separate post-train serve, so it sets
    ``eval_interval = None`` — set that here to match the recipe exactly.
  - The H100-sized parallelism also runs (under-utilized) on H200; flip
    ``modal.gpu`` back to "H200" to stay on the rest of the guide's infra.
"""

import os

from configs import w_qwen3_6_dapo_colocate_2n as _base
from configs.base import CHECKPOINTS_PATH, HF_CACHE_PATH, ModalConfig, run_tag

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3.6-35b-a3b-dapo-math-colocate-1n-tggym")

_WANDB_IMAGE_ENV = {
    k: v for k in ("WANDB_PROJECT", "WANDB_GROUP") if (v := os.environ.get(k)) is not None
}

# H100 box (the recipe's gpu_type); otherwise identical to the SWE base image.
modal = ModalConfig(
    gpu="H100",
    local_slime="/Users/junlin/Documents/Research/async-rl/slime",
    image_env=_WANDB_IMAGE_ENV,
    image_run_commands=["pip install modal", f"rm -rf {HF_CACHE_PATH}"],
)


class _Slime(_base._Slime):
    # Data / reward / eval / download_data all inherited from
    # w_qwen3_6_dapo_colocate_2n. Below: only the deltas that make the training
    # knobs match training-gym's Qwen3_6_35b_Recipe (H100-sized).

    # ── Parallelism (H100-sized: TP2 / PP2 / CP2 / EP4) ───────────────────────
    pipeline_model_parallel_size = 2  # base: 1
    expert_model_parallel_size = 4  # base: 8
    # TP2, CP2, sequence_parallel, ETP1 already match the SWE base.

    # PP2 needs its own torch_dist conversion target (see docstring).
    # ref_load = f"{CHECKPOINTS_PATH}/Qwen3.6-35B-A3B_torch_dist_tp2pp2"
    ref_load = f"{CHECKPOINTS_PATH}/Qwen3.6-35B-A3B_torch_dist"

    # ── Rollout sizing ────────────────────────────────────────────────────────
    num_rollout = 2  # tutorial's quick-demo value; recipe default is 3000
    rollout_batch_size = 16  # base: 32
    rollout_max_response_len = 16384  # base: 8192
    global_batch_size = 128  # rollout_batch_size * n_samples_per_prompt // steps
    rollout_num_gpus_per_engine = 4  # dapo base: 8 → two TP-shrunk engines / 8 GPUs

    # ── SGLang engine: DP-attention, H100-sized ───────────────────────────────
    sglang_mem_fraction_static = 0.75  # dapo base: 0.65
    sglang_enable_dp_attention = False  # base: False
    sglang_dp_size = 1
    # sglang_ep_size = 4  # base: 8
    # sglang_enable_dp_lm_head = True
    sglang_max_running_requests = 512
    sglang_cuda_graph_bs = [1, 2, 4, 8] + list(range(16, 257, 8))
    # The recipe leaves custom all-reduce on; the SWE base disables it.
    sglang_disable_custom_all_reduce = True
    sglang_speculative_algorithm = "Eagle"

    # ── Training ──────────────────────────────────────────────────────────────
    max_tokens_per_gpu = 8192  # base: 16384


    # ── WandB / dumps: fresh tag so we don't log under the dapo_2n run ─────────
    wandb_group = _RUN_TAG
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )


slime = _Slime()
