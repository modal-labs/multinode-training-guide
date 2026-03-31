"""Qwen3-VL-30B-A3B-Instruct GRPO on Geo3K — non-colocated 2-node (8+8 GPUs)."""

from configs.base import ModalConfig
from configs.qwen3vl_geo3k_vlm import _Slime as _BaseSlime

modal = ModalConfig(gpu="H200")


class _Slime(_BaseSlime):
    # Non-colocated: 1 node × 8 GPUs for actor, 1 node × 8 GPUs for rollout
    colocate = False
    rollout_num_gpus = 8


slime = _Slime()
