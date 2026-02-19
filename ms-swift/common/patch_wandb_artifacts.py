"""
Patch Megatron's wandb_utils to skip artifact upload on checkpoint save.

ms-swift redirects args.save during save_checkpoint (to checkpoint-N subdirs),
which causes Megatron's wandb_utils.on_save_checkpoint_success to look for
latest_checkpointed_iteration.txt in the wrong location. The file is written
by rank 0 but the wandb artifact upload runs on is_last_rank() (different container).

This plugin is loaded via --external_plugins and runs early enough to patch
before training starts. Metrics/loss WandB logging is unaffected — only the
checkpoint artifact upload is disabled.

Usage:
    megatron sft ... --external_plugins /root/common/patch_wandb_artifacts.py
"""
try:
    from megatron.training import wandb_utils
    _original = wandb_utils.on_save_checkpoint_success

    def _noop_on_save_checkpoint_success(*args, **kwargs):
        pass

    wandb_utils.on_save_checkpoint_success = _noop_on_save_checkpoint_success
except ImportError:
    pass