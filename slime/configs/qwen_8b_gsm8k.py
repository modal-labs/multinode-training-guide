"""Qwen3-8B GRPO on GSM8K — 4 nodes, colocated.

Inherits all settings from qwen_4b_gsm8k and overrides what differs:
model checkpoint, architecture, node count, and batch sizes.
"""

from configs import qwen_4b_gsm8k as _base

modal = _base.modal


class _Slime(_base._Slime):
    # ── Model ─────────────────────────────────────────────────────────────────
    hf_checkpoint = "Qwen/Qwen3-8B"

    # ── Rollout ───────────────────────────────────────────────────────────────
    rollout_batch_size = 128
    global_batch_size = 1024

    # ── Model architecture (Qwen3-8B) ─────────────────────────────────────────
    hidden_size = 4096
    ffn_hidden_size = 12288

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_group = "qwen3-8b-gsm8k"


slime = _Slime()
