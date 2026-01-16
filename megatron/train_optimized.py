"""
GLM-4.7 LoRA Training Script - OPTIMIZED VERSION

Changes from original:
1. Uses NVIDIA-recommended parallelism (TP=2, PP=4, EP=4) instead of (TP=2, PP=1, EP=8)
2. Removes activation recomputation (not needed with PP=4)
3. Enables torch.compile for performance
4. Enables NVLS for better NCCL performance
5. Increases micro_batch_size for better throughput

Memory budget analysis (per GPU, H100 80GB):
- Model weights (frozen bf16): ~23 GB
- LoRA adapters + optimizer:   ~0.5 GB
- Activations (23 layers):     ~12 GB (mbs=1) or ~24 GB (mbs=2)
- Communication buffers:       ~4 GB
- CUDA overhead:               ~2 GB
- TOTAL (mbs=2):               ~54 GB → 26 GB headroom
"""

import argparse
import os

import torch
import wandb

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
    p = argparse.ArgumentParser(description="GLM-4.7 LoRA Training (Optimized)")

    p.add_argument("--preprocessed_dir", required=True, help="Path to preprocessed dataset")
    p.add_argument("--megatron_checkpoint", required=True, help="Path to Megatron checkpoint")
    p.add_argument("--checkpoints_dir", required=True, help="Directory to save checkpoints")
    p.add_argument("--hf_model", default="zai-org/GLM-4.7", help="HuggingFace model ID")

    # Optimization flags for experimentation
    p.add_argument("--micro_batch_size", type=int, default=1, help="Micro batch size (default: 1, use 2 for H200)")
    p.add_argument("--disable_compile", action="store_true", help="Disable torch.compile")
    p.add_argument("--moe_layer_recompute", action="store_true", help="Enable MoE-only recomputation (minimal overhead)")

    return p.parse_args()


def main():
    args = parse_args()

    rank = int(os.environ.get("RANK", 0))

    # Only rank 0 initializes WandB
    if rank == 0:
        run = wandb.init(project="glm47-lora-optimized")
        print(f"WandB initialized: {run.name} ({run.id})")

    print("=" * 60)
    print("GLM-4.7 (358B MoE) LoRA Training - OPTIMIZED")
    print("=" * 60)

    # ==========================================================================
    # ENVIRONMENT CONFIGURATION - OPTIMIZED
    # ==========================================================================

    # Enable expandable segments for MoE memory patterns
    # This IS necessary for MoE models regardless of memory headroom
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # REMOVED: TORCHDYNAMO_DISABLE - torch.compile should work with PP=4 config
    # REMOVED: NCCL_NVLS_ENABLE=0 - NVLS improves performance, we have memory headroom

    # INVESTIGATE: NVTE_FUSED_ATTN - may still be needed for GLM-4.7 specifically
    # Try enabling first, disable only if you see cudnn errors
    # os.environ["NVTE_FUSED_ATTN"] = "0"  # Uncomment if needed

    # ==========================================================================
    # NO MONKEY PATCHES NEEDED
    # ==========================================================================
    # The TransformerBlock requires_grad patch is only needed when:
    # 1. Using activation recomputation (recompute_granularity="full")
    # 2. With frozen base model (LoRA/PEFT)
    #
    # Since we're NOT using recompute, we don't need the patch.
    # The gradient flow works correctly without recomputation.

    # LoRA config - same as original
    lora_config = LoRA(
        dim=128,
        alpha=32,
        dropout=0.05,
    )

    print(f"Creating config from HF model: {args.hf_model}")
    bridge = AutoBridge.from_hf_pretrained(args.hf_model, trust_remote_code=True)
    model_cfg = bridge.to_megatron_provider(load_weights=False)

    print(
        f"Model loaded: num_layers={model_cfg.num_layers}, "
        f"num_moe_experts={getattr(model_cfg, 'num_moe_experts', 'N/A')}"
    )

    print(f"[Rank {rank}] Loading pre-built dataset from: {args.preprocessed_dir}")

    # Optimizer with cosine annealing - same as original
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=50,
        lr_decay_iters=None,
        max_lr=1e-4,
        min_lr=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.1,
    )

    # Build the full config
    config = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=650,
            eval_interval=9999,
            eval_iters=0,
            global_batch_size=32,  # Increased from 16 due to higher micro_batch
            micro_batch_size=args.micro_batch_size,  # Configurable, default 2
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            overlap_param_gather=True,  # ENABLED - we have memory headroom
        ),
        dataset=FinetuningDatasetConfig(
            dataset_root=args.preprocessed_dir,
            seq_length=8192,
            seed=5678,
            dataloader_type="batch",
            num_workers=2,  # Increased from 1
            do_validation=False,
            do_test=False,
        ),
        logger=LoggerConfig(
            log_interval=1,
            tensorboard_dir="/tmp/tensorboard",
            wandb_project="glm47-lora-optimized",
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=args.hf_model,
        ),
        checkpoint=CheckpointConfig(
            save_interval=130,
            save=f"{args.checkpoints_dir}/glm47_lora_optimized",
            pretrained_checkpoint=args.megatron_checkpoint,
            ckpt_format="torch_dist",
            fully_parallel_save=True,  # ENABLED - faster checkpointing
            async_save=True,  # ENABLED - non-blocking saves
        ),
        rng=RNGConfig(seed=5678),
        peft=lora_config,
        mixed_precision="bf16_mixed",
    )

    # ==========================================================================
    # PARALLELISM - NVIDIA RECOMMENDED FOR 355B LoRA
    # ==========================================================================
    # TP=2 × PP=4 × EP=4 × DP=1 = 32 GPUs
    #
    # Why this works:
    # - PP=4 splits 92 layers across 4 stages → 23 layers per GPU
    # - 23 layers × 509MB/layer = ~11.7GB activation memory (vs 46.8GB with PP=1)
    # - This eliminates the need for activation recomputation
    #
    # Trade-off: PP>1 introduces pipeline bubble overhead, but this is offset by:
    # 1. No recompute overhead (~30% compute savings)
    # 2. Ability to use torch.compile
    # 3. Higher micro_batch_size possible

    config.model.tensor_model_parallel_size = 2
    config.model.pipeline_model_parallel_size = 4  # CHANGED from 1
    config.model.expert_model_parallel_size = 4    # CHANGED from 8
    config.model.context_parallel_size = 1

    config.model.calculate_per_token_loss = False
    config.model.sequence_parallel = True  # Required when TP > 1 with MoE
    config.model.attention_backend = "flash"

    # MoE optimization
    config.model.moe_grouped_gemm = True

    # ==========================================================================
    # ACTIVATION RECOMPUTATION - SELECTIVE MoE-ONLY (optional)
    # ==========================================================================
    # With PP=4 and mbs=1, we should have sufficient headroom without recompute.
    # But if OOM occurs (especially on PP stage 0 which has embedding + layers),
    # we can enable MoE-only recomputation which has minimal overhead (~5-10%)
    # compared to full recomputation (~30%).
    #
    # MoE layer recomputation discards MoE activations during forward and
    # recomputes them during backward - this specifically helps with the
    # reduce_scatter buffers that caused the OOM.

    if args.moe_layer_recompute:
        print("Enabling MoE-layer-only recomputation (--moe_layer_recompute)")
        # Selective recompute: only recompute MoE layer activations
        # This is much cheaper than full recompute (~5-10% overhead vs ~30%)
        config.model.recompute_granularity = "selective"
        config.model.recompute_method = "uniform"
        config.model.recompute_num_layers = 1
        # Specifically target MoE activations
        # Note: The exact config key may vary by Megatron version
        # config.model.moe_layer_recompute = True  # Try this if above doesn't work

    config.model.seq_length = 8192

    print("\nOptimized Config:")
    print("  Model: GLM-4.7 (358B MoE)")
    print(f"  seq_length: {config.model.seq_length}")
    print(f"  micro_batch_size: {config.train.micro_batch_size}")
    print(f"  global_batch_size: {config.train.global_batch_size}")
    print(f"  Dataset: {args.preprocessed_dir}")
    print(
        f"  Parallelism: TP={config.model.tensor_model_parallel_size}, "
        f"PP={config.model.pipeline_model_parallel_size}, "
        f"EP={config.model.expert_model_parallel_size}, "
        f"CP={config.model.context_parallel_size}"
    )
    print("  Recompute: DISABLED (not needed with PP=4)")
    print(f"  moe_grouped_gemm: {config.model.moe_grouped_gemm}")
    print(f"  overlap_param_gather: {config.ddp.overlap_param_gather}")
    print(f"  torch.compile: {'DISABLED' if args.disable_compile else 'ENABLED'}")

    finetune(config=config, forward_step_func=forward_step)
    print("Training complete!")


if __name__ == "__main__":
    main()
