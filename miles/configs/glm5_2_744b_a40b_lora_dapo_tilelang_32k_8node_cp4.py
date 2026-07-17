"""GLM-5.2 LoRA DAPO 32k padded — 8 nodes, CP4, NO activation recompute.

Same TP8 x CP4 geometry (and hence identical per-rank memory picture) as the
16-node config, on the readily-schedulable 8-node block: replica = 32 GPUs,
DP = 2. See glm5_2_744b_a40b_lora_dapo_tilelang_32k_16node for the CP design
notes and the megatron_dsa_cp_assert_fix rationale.

Correctness oracle: train_rollout_logprob_abs_diff (~0.01 healthy, >1 means
broken CP attention math — stop the run if seen).

    EXPERIMENT_CONFIG=glm5_2_744b_a40b_lora_dapo_tilelang_32k_8node_cp4 \
        uv run modal run --detach miles/modal_train.py::train
"""

from configs import glm5_2_744b_a40b_lora_dapo_tilelang_32k_16node as _base
from configs.base import CHECKPOINTS_PATH

modal = _base.modal


class _Miles(_base._Miles):
    actor_num_nodes = 8

    save = f"{CHECKPOINTS_PATH}/GLM-5.2-lora-dapo-32k-8node-cp4-ckpt"

    wandb_group = "glm5.2-744B-8node-cp4-tilelang-dapo-32k-norecompute"


miles = _Miles()
