"""Base configuration for SLIME training on Modal.

This module provides a simple pass-through config system that is decoupled from
slime's internal argument structure. Configs just pass raw CLI args to slime.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import textwrap


@dataclass
class RLConfig:
    """Minimal config - just metadata and raw args passthrough.

    This config is intentionally simple and decoupled from slime's internal
    argument structure. It passes raw CLI args directly to slime without
    parsing or validation, making it robust to slime version changes.
    """

    # Required metadata
    model_name: str
    model_id: str  # HuggingFace model ID (e.g., "Qwen/Qwen3-4B")

    # Modal settings
    app_name: str = "slime-grpo"
    n_nodes: int = 4
    gpu: str = "H100:8"

    # Training mode
    sync: bool = True

    # Wandb
    wandb_project: str = "slime-grpo"
    wandb_run_name_prefix: str = ""

    # Raw CLI args - passed directly to slime, no parsing/validation
    # Use triple-quoted strings for readability with comments
    slime_args: str = ""

    # Extra args that get appended (for easy overrides)
    extra_args: list[str] = field(default_factory=list)

    @property
    def train_script(self) -> str:
        return "slime/train.py" if self.sync else "slime/train_async.py"

    def _clean_args(self, args: str) -> str:
        """Clean up arg string: remove comments, normalize whitespace."""
        lines = []
        for line in args.split("\n"):
            # Remove comments
            if "#" in line:
                line = line[:line.index("#")]
            line = line.strip()
            if line:
                lines.append(line)
        return " ".join(lines)

    def generate_train_args(self, hf_model_path: str, checkpoints_path: Path, data_path: Path, is_infinite_run: bool) -> str:
        """Generate full command line args for slime."""
        # HF model path resolved from cache (snapshot_download with local_files_only=True)
        base_args = f"--hf-checkpoint {hf_model_path} --ref-load {hf_model_path}"

        cleaned_slime_args = self._clean_args(self.slime_args)
        cleaned_slime_args = cleaned_slime_args.replace("{data_path}", str(data_path))
        cleaned_slime_args = cleaned_slime_args.replace("{checkpoints_path}", str(checkpoints_path))

        extra = " ".join(self.extra_args) if self.extra_args else ""

        return f"{base_args} {cleaned_slime_args} {extra}".strip()


# Common arg templates for convenience
QWEN3_4B_MODEL_ARGS = """
    --num-layers 36 --hidden-size 2560 --ffn-hidden-size 9728
    --num-attention-heads 32 --group-query-attention --num-query-groups 8
    --kv-channels 128 --vocab-size 151936
    --normalization RMSNorm --norm-epsilon 1e-6 --swiglu
    --disable-bias-linear --qk-layernorm
    --use-rotary-position-embeddings --rotary-base 1000000
"""

QWEN3_0_5B_MODEL_ARGS = """
    --num-layers 24 --hidden-size 1024 --ffn-hidden-size 3072
    --num-attention-heads 16 --group-query-attention --num-query-groups 8
    --kv-channels 64 --vocab-size 151936
    --normalization RMSNorm --norm-epsilon 1e-6 --swiglu
    --disable-bias-linear --qk-layernorm
    --use-rotary-position-embeddings --rotary-base 1000000
"""

DEFAULT_TRAINING_ARGS = """
    --tensor-model-parallel-size 2 --sequence-parallel
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --use-dynamic-batch-size --max-tokens-per-gpu 9216
    --megatron-to-hf-mode bridge
    --attention-dropout 0.0 --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
"""

DEFAULT_OPTIMIZER_ARGS = """
    --optimizer adam
    --lr 1e-6 --lr-decay-style constant
    --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
"""

DEFAULT_GRPO_ARGS = """
    --advantage-estimator grpo
    --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2 --eps-clip-high 0.28
"""

DEFAULT_DATA_ARGS = """
    --input-key messages --label-key label
    --apply-chat-template --rollout-shuffle --rm-type math
"""

# GLM-4.7 (358B MoE) model architecture args
# Based on: https://huggingface.co/zai-org/GLM-4.7/blob/main/config.json
GLM_4_7_MODEL_ARGS = """
    --num-layers 92 --hidden-size 5120 --ffn-hidden-size 12288
    --num-attention-heads 96 --group-query-attention --num-query-groups 8
    --kv-channels 128 --vocab-size 151552
    --normalization RMSNorm --norm-epsilon 1e-5 --swiglu
    --add-qkv-bias --qk-layernorm
    --untie-embeddings-and-output-weights
    --use-rotary-position-embeddings --rotary-base 1000000
    --num-experts 160
    --moe-layer-freq "[0]*3+[1]*89"
    --moe-shared-expert-intermediate-size 1536
    --moe-router-topk 8
    --moe-grouped-gemm --moe-permute-fusion
    --moe-ffn-hidden-size 1536
    --moe-router-score-function sigmoid
    --moe-router-pre-softmax
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk-scaling-factor 2.5
    --moe-aux-loss-coeff 0
    --moe-router-dtype fp32
    --moe-token-dispatcher-type flex
    --moe-enable-deepep
"""

# GLM-4.7-Flash (30B MoE with MLA) model architecture args
# Based on: scripts/models/glm4.7-30B-A3B.sh
GLM_4_7_FLASH_MODEL_ARGS = """
    --num-layers 47 --hidden-size 2048 --ffn-hidden-size 10240
    --num-attention-heads 20 --vocab-size 154880
    --make-vocab-size-divisible-by 64
    --normalization RMSNorm --norm-epsilon 1e-5 --swiglu
    --disable-bias-linear --add-qkv-bias --qk-layernorm
    --untie-embeddings-and-output-weights
    --position-embedding-type rope --no-position-embedding
    --use-rotary-position-embeddings --rotary-base 1000000 --no-rope-fusion
    --multi-latent-attention
    --q-lora-rank 768 --kv-lora-rank 512
    --qk-head-dim 192 --v-head-dim 256 --kv-channels 192
    --qk-pos-emb-head-dim 64
    --num-experts 64
    --moe-layer-freq "[0]*1+[1]*46"
    --moe-shared-expert-intermediate-size 1536
    --moe-router-topk 4
    --moe-grouped-gemm --moe-permute-fusion
    --moe-ffn-hidden-size 1536
    --moe-router-score-function sigmoid
    --moe-router-pre-softmax
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk-scaling-factor 1.8
    --moe-aux-loss-coeff 0
    --moe-router-dtype fp32
    --moe-token-dispatcher-type flex
    --moe-enable-deepep
"""

# GLM training args with MoE parallelism
GLM_4_7_TRAINING_ARGS = """
    --tensor-model-parallel-size 8 --pipeline-model-parallel-size 4
    --context-parallel-size 2
    --expert-model-parallel-size 16 --expert-tensor-parallel-size 1
    --sequence-parallel
    --decoder-last-pipeline-num-layers 23
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --use-dynamic-batch-size --max-tokens-per-gpu 16384
    --megatron-to-hf-mode bridge
    --attention-dropout 0.0 --hidden-dropout 0.0
    --attention-backend flash
    --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
"""

GLM_4_7_FLASH_TRAINING_ARGS = """
    --tensor-model-parallel-size 4 --pipeline-model-parallel-size 2
    --context-parallel-size 2
    --expert-model-parallel-size 8 --expert-tensor-parallel-size 1
    --sequence-parallel
    --decoder-last-pipeline-num-layers 23
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --use-dynamic-batch-size --max-tokens-per-gpu 32768
    --megatron-to-hf-mode bridge
    --attention-dropout 0.0 --hidden-dropout 0.0
    --attention-backend flash
    --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
"""
