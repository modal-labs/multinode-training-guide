"""
GLM-4.7 LoRA Training Script

This script is mounted into Modal containers and executed via torchrun.
Configuration is passed via command-line arguments.
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
    p = argparse.ArgumentParser(description="GLM-4.7 LoRA Training")

    p.add_argument("--run_id", required=True, help="WandB run ID")
    p.add_argument("--preprocessed_dir", required=True, help="Path to preprocessed dataset")
    p.add_argument("--megatron_checkpoint", required=True, help="Path to Megatron checkpoint")
    p.add_argument("--checkpoints_dir", required=True, help="Directory to save checkpoints")
    p.add_argument("--hf_model", default="zai-org/GLM-4.7", help="HuggingFace model ID")

    return p.parse_args()


def main():
    args = parse_args()

    rank = int(os.environ.get("RANK", 0))

    # Only rank 0 initializes WandB
    if rank == 0:
        wandb.init(project="glm47-lora", id=args.run_id, resume="allow")
        print(f"WandB initialized with run_id: {args.run_id}")

    print(f"[Rank {rank}] Using run_id: {args.run_id}")

    print("=" * 60)
    print(f"GLM-4.7 (358B MoE) - Run: {args.run_id}")
    print("=" * 60)

    # Monkey-patch for PP=1 + Recompute + Frozen Base gradient fix
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
    print("[PEFT+Recompute FIX] Patched TransformerBlock.forward")

    # LoRA config
    lora_config = LoRA(
        dim=128,
        alpha=32,
        dropout=0.05,
    )

    # GLM-4.7 uses same architecture class (Glm4MoeForCausalLM) as GLM-4.5
    # So we can use from_hf_pretrained directly - it will read the correct architecture
    print(f"Creating config from HF model: {args.hf_model}")

    # Use AutoBridge to get the model provider from HF config
    # This reads the actual architecture (92 layers, 160 experts) from HF
    bridge = AutoBridge.from_hf_pretrained(args.hf_model, trust_remote_code=True)
    model_cfg = bridge.to_megatron_provider(load_weights=False)

    print(
        f"Model loaded: num_layers={model_cfg.num_layers}, "
        f"num_moe_experts={getattr(model_cfg, 'num_moe_experts', 'N/A')}"
    )

    print(f"[Rank {rank}] Loading pre-built dataset from: {args.preprocessed_dir}")

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

    # Build the full config
    config = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=650,
            eval_interval=9999,
            eval_iters=0,
            global_batch_size=16,
            micro_batch_size=1,
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=DistributedDataParallelConfig(check_for_nan_in_grad=True),
        dataset=FinetuningDatasetConfig(
            dataset_root=args.preprocessed_dir,
            seq_length=131072,  # 128k context
            seed=5678,
            dataloader_type="batch",
            num_workers=1,
            do_validation=False,
            do_test=False,
        ),
        logger=LoggerConfig(
            log_interval=1,
            tensorboard_dir="/tmp/tensorboard",
            wandb_project="glm47-lora",
            wandb_exp_name=args.run_id,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=args.hf_model,
        ),
        checkpoint=CheckpointConfig(
            save_interval=130,
            save=f"{args.checkpoints_dir}/glm47_lora_{args.run_id}",
            pretrained_checkpoint=args.megatron_checkpoint,
            ckpt_format="torch_dist",
            fully_parallel_save=False,  # Must be False - Modal multiprocessing is limited
            async_save=False,  # Disable async save - Modal queue issues
        ),
        rng=RNGConfig(seed=5678),
        peft=lora_config,
        mixed_precision="bf16_mixed",
    )

    print("Config created successfully")

    # Parallelism for GLM-4.7 on 32 GPUs (TP=2 x PP=1 x EP=8 x CP=2 = 32)
    config.model.tensor_model_parallel_size = 2
    config.model.pipeline_model_parallel_size = 1
    config.model.expert_model_parallel_size = 8
    config.model.context_parallel_size = 2  # 64k per GPU
    # config.model.cp_comm_type = "a2a+p2p"
    config.model.calculate_per_token_loss = True  # Required for CP>1
    config.model.sequence_parallel = True
    config.model.attention_backend = "flash"

    # MoE optimization
    config.model.moe_grouped_gemm = True

    # DDP optimization
    # config.ddp.overlap_param_gather = True

    # Memory optimization - activation recomputation
    config.model.recompute_granularity = "full"
    config.model.recompute_method = "uniform"
    config.model.recompute_num_layers = 1  # Must be 1 for MTP (Multi-Token Prediction)
    # Sequence length
    config.model.seq_length = 131072  # 128k context

    print("Config:")
    print("  Model: GLM-4.7 (358B MoE)")
    print(f"  seq_length: {config.model.seq_length}")
    print(f"  Dataset: {args.preprocessed_dir} (PRE-BUILT)")
    print(
        f"  TP={config.model.tensor_model_parallel_size}, "
        f"PP={config.model.pipeline_model_parallel_size}, "
        f"EP={config.model.expert_model_parallel_size}, "
        f"CP={config.model.context_parallel_size}"
    )
    print(f"  moe_grouped_gemm: {config.model.moe_grouped_gemm}")
    print(f"  overlap_param_gather: {config.ddp.overlap_param_gather}")

    finetune(config=config, forward_step_func=forward_step)
    print("Training complete!")


if __name__ == "__main__":
    main()
