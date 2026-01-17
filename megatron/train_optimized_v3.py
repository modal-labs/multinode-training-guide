"""
GLM-4.7 LoRA Training Script - OPTIMIZED V3

V3 Goals: Leverage the ~17GB/GPU unused VRAM headroom observed in V2.

V2 status:
- mbs=2, full recompute: ~63GB peak, 17GB headroom, ~375 TFLOP/s
- Stable but leaving performance on the table

V3 Experiments:
1. mbs=3 with full recompute (PRIMARY)
   - Expected: ~71-75GB peak
   - Benefit: ~50% more compute per forward/backward
   - Risk: Medium - MoE backward spikes might push us close to limit

2. mbs=4 with full recompute (AGGRESSIVE)
   - Expected: ~78-82GB peak
   - Benefit: ~100% more compute per forward/backward
   - Risk: High - likely to OOM on MoE backward pass

3. mbs=2 with recompute_num_layers=2 (CONSERVATIVE)
   - Expected: ~68-72GB peak
   - Benefit: ~15% less recompute overhead
   - Risk: Medium - stores 2x activations between checkpoints

Memory math with full recompute (scaling from V2 observations):
- V2 (mbs=2): ~63GB peak
- Forward activations scale linearly with mbs
- Estimated: mbs=3 → ~63 + (63-23)*0.5 = ~83GB? Too aggressive.

Actually, let me recalculate more carefully:
- Model weights: ~23GB (fixed)
- LoRA + optim: ~0.5GB (fixed)
- Comm buffers: ~4GB (fixed)
- CUDA overhead: ~2GB (fixed)
- Fixed total: ~30GB

- Recompute activations scale with mbs:
  - V2 observed: 63 - 30 = ~33GB for mbs=2
  - mbs=3: ~50GB activations → 80GB total (risky but might work)
  - mbs=4: ~66GB activations → 96GB total (too high)

So mbs=3 is the sweet spot to try first.
"""

import argparse
import os
from functools import wraps

import torch
import wandb
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
    p = argparse.ArgumentParser(description="GLM-4.7 LoRA Training (Optimized V3)")

    p.add_argument("--preprocessed_dir", required=True, help="Path to preprocessed dataset")
    p.add_argument("--megatron_checkpoint", required=True, help="Path to Megatron checkpoint")
    p.add_argument("--checkpoints_dir", required=True, help="Directory to save checkpoints")
    p.add_argument("--hf_model", default="zai-org/GLM-4.7", help="HuggingFace model ID")

    # V3: Default to mbs=3 to leverage unused VRAM
    # With TP=2, PP=4: DP = 32/(2×4) = 4
    # gbs must be divisible by mbs × DP = 3 × 4 = 12
    p.add_argument("--micro_batch_size", type=int, default=3,
                   help="Micro batch size (default: 3 in V3)")
    p.add_argument("--global_batch_size", type=int, default=36,
                   help="Global batch size (must be divisible by mbs × DP=12)")

    # Recompute options
    p.add_argument("--recompute_num_layers", type=int, default=1,
                   help="Checkpoint every N layers (1=full recompute, 2=half overhead)")
    p.add_argument("--no_recompute", action="store_true",
                   help="Disable recomputation entirely (will likely OOM)")

    return p.parse_args()


def main():
    args = parse_args()

    rank = int(os.environ.get("RANK", 0))

    # Only rank 0 initializes WandB
    if rank == 0:
        run = wandb.init(project="glm47-lora-optimized-v3")
        print(f"WandB initialized: {run.name} ({run.id})")

    print("=" * 70)
    print("GLM-4.7 (358B MoE) LoRA Training - OPTIMIZED V3")
    print("=" * 70)

    # ==========================================================================
    # ENVIRONMENT CONFIGURATION
    # ==========================================================================

    # Required for MoE memory patterns - allows fragmentation-friendly allocation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Layer norm SM margin - prevents persistent LN kernels from blocking DP comm
    os.environ["NVTE_FWD_LAYERNORM_SM_MARGIN"] = "16"
    os.environ["NVTE_BWD_LAYERNORM_SM_MARGIN"] = "16"

    # Enable TP overlap on H100 (Hopper)
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    # ==========================================================================
    # CRITICAL PATCH: Fix for Recompute + Frozen Base Model (LoRA)
    # ==========================================================================

    _original_transformer_forward = TransformerBlock.forward

    @wraps(_original_transformer_forward)
    def _patched_transformer_forward(self, hidden_states, *args_, **kwargs):
        if (
            torch.is_tensor(hidden_states)
            and not hidden_states.requires_grad
            and hidden_states.is_floating_point()
        ):
            hidden_states = hidden_states.detach().requires_grad_(True)
        return _original_transformer_forward(self, hidden_states, *args_, **kwargs)

    TransformerBlock.forward = _patched_transformer_forward
    print("[PEFT+Recompute FIX] Patched TransformerBlock.forward for gradient flow")

    # LoRA config
    lora_config = LoRA(
        dim=128,
        alpha=32,
        dropout=0.05,
    )

    print(f"Creating config from HF model: {args.hf_model}")
    bridge = AutoBridge.from_hf_pretrained(args.hf_model, trust_remote_code=True)
    model_cfg = bridge.to_megatron_provider(load_weights=False)

    print(
        f"Model: num_layers={model_cfg.num_layers}, "
        f"num_moe_experts={getattr(model_cfg, 'num_moe_experts', 'N/A')}"
    )

    # Optimizer with cosine annealing
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

    # ==========================================================================
    # TRAINING CONFIG
    # ==========================================================================

    config = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=650,
            eval_interval=9999,
            eval_iters=0,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
            # Manual GC to reduce jitter
            manual_gc_interval=10,
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            # DISABLED for PEFT - overlaps expect all params to have gradients
            overlap_param_gather=False,
            overlap_grad_reduce=False,
            bucket_size=100_000_000,  # 100MB buckets
        ),
        dataset=FinetuningDatasetConfig(
            dataset_root=args.preprocessed_dir,
            seq_length=8192,
            seed=5678,
            dataloader_type="batch",
            num_workers=4,
            do_validation=False,
            do_test=False,
        ),
        logger=LoggerConfig(
            log_interval=1,
            tensorboard_dir="/tmp/tensorboard",
            wandb_project="glm47-lora-optimized-v3",
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=args.hf_model,
        ),
        checkpoint=CheckpointConfig(
            save_interval=130,
            save=f"{args.checkpoints_dir}/glm47_lora_v3",
            pretrained_checkpoint=args.megatron_checkpoint,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            async_save=True,
        ),
        rng=RNGConfig(seed=5678),
        peft=lora_config,
        mixed_precision="bf16_mixed",
    )

    # ==========================================================================
    # PARALLELISM - NVIDIA RECOMMENDED
    # ==========================================================================

    config.model.tensor_model_parallel_size = 2
    config.model.pipeline_model_parallel_size = 4
    config.model.expert_model_parallel_size = 4
    config.model.context_parallel_size = 1

    config.model.calculate_per_token_loss = False
    config.model.sequence_parallel = True
    config.model.attention_backend = "flash"

    # MoE optimization
    config.model.moe_grouped_gemm = True

    # ==========================================================================
    # ACTIVATION RECOMPUTATION CONFIGURATION
    # ==========================================================================

    if not args.no_recompute:
        config.model.recompute_granularity = "full"
        config.model.recompute_method = "uniform"
        config.model.recompute_num_layers = args.recompute_num_layers

        if args.recompute_num_layers == 1:
            print(f"Recompute: FULL (checkpoint every layer)")
        else:
            print(f"Recompute: PARTIAL (checkpoint every {args.recompute_num_layers} layers)")
            print(f"  - This reduces recompute overhead but increases activation memory")
    else:
        print("Recompute: DISABLED - WARNING: Will likely OOM with mbs >= 2")

    config.model.seq_length = 8192

    # ==========================================================================
    # OPERATOR FUSION - ENABLED
    # ==========================================================================
    config.model.masked_softmax_fusion = True
    config.model.bias_activation_fusion = True
    config.model.bias_dropout_fusion = True
    config.model.apply_rope_fusion = True

    # ==========================================================================
    # V3 SUMMARY
    # ==========================================================================

    # Calculate gradient accumulation steps
    accum_steps = args.global_batch_size // args.micro_batch_size

    print("\n" + "=" * 70)
    print("V3 Configuration Summary")
    print("=" * 70)
    print(f"  Model: GLM-4.7 (358B MoE)")
    print(f"  seq_length: {config.model.seq_length}")
    print(f"  micro_batch_size: {config.train.micro_batch_size}")
    print(f"  global_batch_size: {config.train.global_batch_size}")
    print(f"  gradient_accumulation_steps: {accum_steps}")
    print(f"\nParallelism:")
    print(f"  TP={config.model.tensor_model_parallel_size}")
    print(f"  PP={config.model.pipeline_model_parallel_size}")
    print(f"  EP={config.model.expert_model_parallel_size}")
    print(f"  CP={config.model.context_parallel_size}")
    print(f"\nMemory Optimizations:")
    if not args.no_recompute:
        print(f"  recompute_granularity: full")
        print(f"  recompute_num_layers: {args.recompute_num_layers}")
    else:
        print(f"  recompute: DISABLED")
    print(f"\nExpected Memory (per GPU):")
    print(f"  Model weights (frozen): ~23 GB")
    print(f"  LoRA + optimizer: ~0.5 GB")
    print(f"  Comm buffers: ~4 GB")
    print(f"  CUDA overhead: ~2 GB")
    print(f"  Activations (mbs={args.micro_batch_size}): ~{16.5 * args.micro_batch_size:.1f} GB (with full recompute)")
    estimated_total = 30 + 16.5 * args.micro_batch_size + 5  # +5GB for MoE backward spikes
    print(f"  Estimated Total: ~{estimated_total:.0f} GB")
    print(f"  H100 Headroom: ~{80 - estimated_total:.0f} GB")

    if estimated_total > 78:
        print(f"\n  WARNING: Estimated memory is close to H100 limit!")
        print(f"           MoE backward spikes may cause OOM.")

    print("=" * 70 + "\n")

    finetune(config=config, forward_step_func=forward_step)
    print("Training complete!")


if __name__ == "__main__":
    main()
