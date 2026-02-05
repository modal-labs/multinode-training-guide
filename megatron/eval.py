"""
GLM-4.7 LoRA Evaluation Script

Loads the latest LoRA checkpoint and runs validation to verify loss.
Used to diagnose training/inference loss discrepancy.
"""

import argparse
import os
from functools import wraps

import torch
from megatron.core.transformer.transformer_block import TransformerBlock

from megatron.bridge import AutoBridge
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
)
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    FinetuningDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step


def parse_args():
    p = argparse.ArgumentParser(description="GLM-4.7 LoRA Evaluation")
    p.add_argument(
        "--preprocessed_dir", required=True, help="Path to preprocessed dataset"
    )
    p.add_argument(
        "--checkpoint_dir",
        required=True,
        help=(
            "Path to LoRA checkpoint directory "
            "(e.g., /checkpoints/glm47_lora)"
        ),
    )
    p.add_argument(
        "--base_checkpoint",
        required=True,
        help="Path to base Megatron checkpoint (required for PEFT)",
    )
    p.add_argument(
        "--hf_model", default="zai-org/GLM-4.7", help="HuggingFace model ID"
    )
    p.add_argument(
        "--context",
        choices=["64k", "128k"],
        default="128k",
        help="Context length",
    )
    p.add_argument(
        "--lora_rank",
        type=int,
        default=128,
        help="LoRA rank (must match training)",
    )
    p.add_argument(
        "--eval_iters",
        type=int,
        default=5,
        help="Number of validation iterations (default 5 to match training)",
    )
    p.add_argument(
        "--full_checkpoint",
        action="store_true",
        help="Load full checkpoint (Base + LoRA) from checkpoint_dir",
    )
    return p.parse_args()


def patch_peft_filter_to_use_name_matching():
    """
    Patch Megatron-Bridge to filter adapters by NAME only.

    The original code checks key[1].requires_grad, which is unreliable
    in distributed checkpoints where values are ShardedTensors/Views
    that may have requires_grad=False even for trainable parameters.
    """
    from megatron.bridge.peft import base

    def _robust_adapter_key_filter(self, key) -> bool:
        name = key[0] if isinstance(key, tuple) else key
        return name in self.params_to_save or ".adapter." in name

    base.PEFT.adapter_key_filter = _robust_adapter_key_filter
    print(
        "[PEFT Filter Patch] Replaced adapter_key_filter to use name matching"
    )


def patch_disable_peft_filtering_entirely():
    """
    Disable PEFT filtering for checkpoint loading.
    This allows loading from a full model checkpoint (Base + LoRA).
    """
    from megatron.bridge.training import checkpointing

    def _pass_through_state_dict(state_dict, peft_config):
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print("[PEFT] Disabled adapter filtering for checkpoint load")
        return state_dict

    checkpointing.apply_peft_adapter_filter_to_state_dict = _pass_through_state_dict
    print("[PEFT] Disabled PEFT filtering for checkpoint loading")


def patch_moe_expert_sharded_state_dict():
    """
    Fix MoE Expert sharded_state_dict to include LoRA adapters.
    Applied during loading to ensure expected keys include adapters.
    """
    from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP

    rank = int(os.environ.get("RANK", 0))

    def make_patched_sharded_state_dict(original_fn, class_name):
        @wraps(original_fn)
        def _patched_sharded_state_dict(
            self, prefix="", sharded_offsets=(), metadata=None
        ):
            state_dict = original_fn(
                self, prefix, sharded_offsets, metadata
            )

            adapter_count = 0
            for name, module in self.named_modules():
                if name == "":
                    continue
                if "adapter" in name.lower() or "lora" in name.lower():
                    for param_name, param in module.named_parameters(
                        recurse=False
                    ):
                        full_key = f"{prefix}{name}.{param_name}"
                        if full_key not in state_dict:
                            state_dict[full_key] = param
                            adapter_count += 1

            if adapter_count > 0 and rank == 0:
                print(
                    f"[{class_name} Patch] Added {adapter_count} "
                    "adapter params to sharded_state_dict"
                )

            return state_dict

        return _patched_sharded_state_dict

    if hasattr(GroupedMLP, "sharded_state_dict"):
        original_grouped = GroupedMLP.sharded_state_dict
        GroupedMLP.sharded_state_dict = make_patched_sharded_state_dict(
            original_grouped, "GroupedMLP"
        )
        print("[Patch] Patched GroupedMLP.sharded_state_dict for loading")

    if hasattr(SequentialMLP, "sharded_state_dict"):
        original_sequential = SequentialMLP.sharded_state_dict
        SequentialMLP.sharded_state_dict = make_patched_sharded_state_dict(
            original_sequential, "SequentialMLP"
        )
        print("[Patch] Patched SequentialMLP.sharded_state_dict for loading")


def patch_checkpoint_load_with_logging():
    """Patch checkpoint load to print LoRA weight norms after loading.

    IMPORTANT: Must patch setup.load_checkpoint, not checkpointing.load_checkpoint.
    setup.py imports `load_checkpoint` directly into its namespace, so patching
    the original module does not affect the already-imported reference.

    Signature: load_checkpoint(state, model, optimizer, opt_param_scheduler, ...)
    """
    from megatron.bridge.training import setup

    _original_load = setup.load_checkpoint

    @wraps(_original_load)
    def _load_with_logging(state, model, *args, **kwargs):
        result = _original_load(state, model, *args, **kwargs)

        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print("\n" + "=" * 70)
            print("LORA ADAPTER INSPECTION AFTER CHECKPOINT LOAD")
            print("=" * 70)
            m = model[0] if isinstance(model, list) else model
            total_norm = 0.0
            dense_count = 0
            expert_count = 0
            router_count = 0
            sample_names = []
            for name, param in m.named_parameters():
                if ".adapter." in name and param.numel() > 0:
                    norm = param.data.norm().item()
                    total_norm += norm**2
                    if len(sample_names) < 3:
                        sample_names.append(f"{name}: norm={norm:.4f}")
                    if ".experts." in name or "expert" in name.lower():
                        expert_count += 1
                    elif "router" in name.lower():
                        router_count += 1
                    else:
                        dense_count += 1
            total_count = dense_count + expert_count + router_count
            print(f"  Dense adapters: {dense_count}")
            print(f"  Expert adapters: {expert_count}")
            print(f"  Router adapters: {router_count}")
            print(f"  TOTAL: {total_count} adapters")
            print(f"  Combined L2 norm: {(total_norm ** 0.5):.6f}")
            if expert_count == 0:
                print("  WARNING: No expert adapters loaded")
            print("\n  Sample adapter names:")
            for sample in sample_names:
                print(f"    {sample}")
            print("=" * 70 + "\n")

        return result

    setup.load_checkpoint = _load_with_logging
    print(
        "[Checkpoint Patch] Added LoRA weight norm logging on load "
        "(patching setup.load_checkpoint)"
    )


def main():
    args = parse_args()

    # Validate checkpoint directory exists and has tracker file
    tracker_file = os.path.join(
        args.checkpoint_dir, "latest_checkpointed_iteration.txt"
    )
    if not os.path.exists(args.checkpoint_dir):
        raise RuntimeError(
            f"Checkpoint directory not found: {args.checkpoint_dir}"
        )
    if not os.path.exists(tracker_file):
        raise RuntimeError(f"No tracker file found: {tracker_file}")

    with open(tracker_file, "r") as f:
        iteration = int(f.read().strip())

    if args.full_checkpoint:
        patch_disable_peft_filtering_entirely()

    # patch_peft_filter_to_use_name_matching()
    # patch_moe_expert_sharded_state_dict()
    # patch_checkpoint_load_with_logging()

    print("=" * 60)
    print("GLM-4.7 LoRA EVALUATION (Megatron Native)")
    print("=" * 60)
    print(f"  Base model: {args.base_checkpoint}")
    print(f"  LoRA checkpoint: {args.checkpoint_dir}")
    print(f"  Full checkpoint load: {args.full_checkpoint}")
    print(f"  Loading iteration: {iteration}")
    print(f"  Dataset: {args.preprocessed_dir}")
    print(f"  Eval iters: {args.eval_iters}")
    print("=" * 60)

    # Patch for PP + Recompute + Frozen Base gradient fix
    _original_forward = TransformerBlock.forward

    @wraps(_original_forward)
    def _patched_forward(self, hidden_states, *args_, **kwargs):
        if (
            torch.is_tensor(hidden_states)
            and not hidden_states.requires_grad
            and hidden_states.is_floating_point()
        ):
            hidden_states = hidden_states.detach().requires_grad_(True)
        return _original_forward(self, hidden_states, *args_, **kwargs)

    TransformerBlock.forward = _patched_forward

    lora_config = LoRA(
        dim=args.lora_rank,
        alpha=32,
        dropout=0.05,
    )

    print(f"Loading config from: {args.hf_model}")
    bridge = AutoBridge.from_hf_pretrained(args.hf_model, trust_remote_code=True)
    model_cfg = bridge.to_megatron_provider(load_weights=False)

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=0,
        lr_decay_iters=1,
        max_lr=1e-4,
        min_lr=0.0,
    )

    seq_length = 131_072 if args.context == "128k" else 65_536

    config = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=0,
            eval_interval=1,
            eval_iters=args.eval_iters,
            global_batch_size=1,
            micro_batch_size=1,
            skip_train=True,
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            overlap_param_gather=False,
            overlap_grad_reduce=False,
        ),
        dataset=FinetuningDatasetConfig(
            dataset_root=args.preprocessed_dir,
            seq_length=seq_length,
            seed=5678,
            dataloader_type="batch",
            num_workers=4,
            do_validation=True,
            do_test=False,
        ),
        logger=LoggerConfig(
            log_interval=1,
            tensorboard_dir="/tmp/tensorboard",
            wandb_project=None,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=args.hf_model,
        ),
        checkpoint=CheckpointConfig(
            save_interval=999999,
            save=None,
            load=args.checkpoint_dir,
            pretrained_checkpoint=args.base_checkpoint,
            ckpt_format="torch_dist",
            load_optim=False,
            load_rng=False,
        ),
        rng=RNGConfig(seed=5678),
        peft=lora_config,
        mixed_precision="bf16_mixed",
    )

    # Match training parallelism: TP=2, PP=4, EP=4, CP=4
    config.model.tensor_model_parallel_size = 2
    config.model.pipeline_model_parallel_size = 4
    config.model.expert_model_parallel_size = 4
    config.model.context_parallel_size = 4
    config.model.calculate_per_token_loss = True
    config.model.seq_length = seq_length
    config.model.sequence_parallel = True
    config.model.attention_backend = "flash"

    # MoE optimization
    config.model.moe_grouped_gemm = True

    # Operator fusion
    config.model.masked_softmax_fusion = True
    config.model.bias_activation_fusion = True
    config.model.bias_dropout_fusion = True
    config.model.apply_rope_fusion = True

    # Memory optimization
    config.model.recompute_granularity = "full"
    config.model.recompute_method = "uniform"
    config.model.recompute_num_layers = 1

    print("Starting evaluation...")
    finetune(config=config, forward_step_func=forward_step)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
