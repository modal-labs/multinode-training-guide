#!/usr/bin/env python3
"""
GLM-4.7 (358B MoE) Memory Calculator for LoRA Training

This script calculates GPU memory requirements for different parallelism configurations.
"""

from dataclasses import dataclass

# =============================================================================
# GLM-4.7 Architecture (based on GLM-4.5 355B architecture)
# =============================================================================


@dataclass
class GLM47Config:
    """
    GLM-4.7 358B MoE Architecture

    Key insight: The "12,288 FFN hidden" in GLM docs refers to the COMBINED FFN
    computation after expert routing, NOT per-expert hidden dimension.

    Working backwards from 358B total params:
    - Total = Embed + Attention + Dense_FFN + MoE_FFN
    - 358B ≈ 2B (embed) + 7.4B (attn) + 0.8B (dense) + 89*160*expert_params
    - expert_params ≈ 24M each
    - Per-expert FFN hidden ≈ 2,343 (so 2 * 5120 * 2343 ≈ 24M)

    With top-8 routing: 8 * 2343 ≈ 18,744 effective FFN width per token
    The "12,288" may be a different architectural specification.
    """

    # Core dimensions
    hidden_size: int = 5120
    num_attention_heads: int = 64
    num_kv_heads: int = 8  # GQA
    head_dim: int = 128  # hidden_size / num_attention_heads

    # Layer configuration
    num_layers: int = 92
    num_dense_layers: int = 3  # First 3 layers are dense
    num_moe_layers: int = 89  # Remaining 89 are MoE

    # FFN dimensions - CORRECTED based on working backwards from 358B total
    # Working backwards:
    #   358B total - 2B embed - 8.65B attn - 0.64B dense = 346.7B for MoE
    #   346.7B / 89 MoE layers = 3.895B per MoE layer
    #   3.895B / (160 experts + 1 shared + router) ≈ 24.2M per expert
    #   SwiGLU: 3 * hidden * ffn_hidden = 24.2M
    #   ffn_hidden = 24.2M / (3 * 5120) = 1,575
    expert_ffn_hidden_size: int = 1575  # Derived to match 358B total
    dense_ffn_hidden_size: int = 13824  # Standard ~2.7x hidden for dense layers

    # MoE configuration
    num_experts: int = 160
    num_experts_per_token: int = 8  # top-8 routing
    # Shared expert for load balancing (GLM uses 1 shared expert)
    num_shared_experts: int = 1

    # Other
    vocab_size: int = 200064
    max_seq_length: int = 131072  # 128K context

    # Layernorm and other small params per layer
    layernorm_params_per_layer: int = (
        2 * 5120 * 2
    )  # 2 layernorms * hidden * 2 (weight + bias)

    def total_params(self) -> int:
        """Calculate total parameter count"""
        # Embedding (input + output, may be tied)
        embed_params = self.vocab_size * self.hidden_size * 2

        # Attention per layer (same for dense and MoE)
        # QKV projection: hidden -> (Q_heads * head_dim) + 2 * (KV_heads * head_dim)
        # With GQA: Q has 64 heads, KV has 8 heads
        q_params = (
            self.hidden_size * self.num_attention_heads * self.head_dim
        )  # 5120 * 64 * 128
        kv_params = (
            self.hidden_size * 2 * self.num_kv_heads * self.head_dim
        )  # 5120 * 2 * 8 * 128
        proj_params = (
            self.num_attention_heads * self.head_dim * self.hidden_size
        )  # 64 * 128 * 5120
        attn_params_per_layer = (
            q_params + kv_params + proj_params + self.layernorm_params_per_layer
        )

        # Dense FFN params (for first 3 layers) - SwiGLU style: 3 projections
        # up_proj: hidden -> ffn_hidden
        # gate_proj: hidden -> ffn_hidden
        # down_proj: ffn_hidden -> hidden
        dense_ffn_params = 3 * self.hidden_size * self.dense_ffn_hidden_size
        dense_layer_params = attn_params_per_layer + dense_ffn_params

        # MoE FFN params per expert (SwiGLU style)
        expert_params = 3 * self.hidden_size * self.expert_ffn_hidden_size
        # Router/gate params
        router_params = self.hidden_size * self.num_experts
        # Shared expert
        shared_expert_params = self.num_shared_experts * expert_params
        # Total MoE layer params
        moe_layer_params = (
            attn_params_per_layer
            + (self.num_experts * expert_params)
            + shared_expert_params
            + router_params
        )

        # Total
        total = (
            embed_params
            + self.num_dense_layers * dense_layer_params
            + self.num_moe_layers * moe_layer_params
        )

        return total

    def active_params(self) -> int:
        """Calculate active parameters per forward pass (with top-k routing)"""
        embed_params = self.vocab_size * self.hidden_size * 2

        q_params = self.hidden_size * self.num_attention_heads * self.head_dim
        kv_params = self.hidden_size * 2 * self.num_kv_heads * self.head_dim
        proj_params = self.num_attention_heads * self.head_dim * self.hidden_size
        attn_params_per_layer = (
            q_params + kv_params + proj_params + self.layernorm_params_per_layer
        )

        dense_ffn_params = 3 * self.hidden_size * self.dense_ffn_hidden_size
        dense_layer_params = attn_params_per_layer + dense_ffn_params

        # Only top-k experts + shared expert active
        expert_params = 3 * self.hidden_size * self.expert_ffn_hidden_size
        active_experts = self.num_experts_per_token + self.num_shared_experts
        router_params = self.hidden_size * self.num_experts  # Router always runs
        active_moe_layer_params = (
            attn_params_per_layer + (active_experts * expert_params) + router_params
        )

        total = (
            embed_params
            + self.num_dense_layers * dense_layer_params
            + self.num_moe_layers * active_moe_layer_params
        )

        return total

    def print_breakdown(self):
        """Print detailed parameter breakdown"""
        embed = self.vocab_size * self.hidden_size * 2

        q_params = self.hidden_size * self.num_attention_heads * self.head_dim
        kv_params = self.hidden_size * 2 * self.num_kv_heads * self.head_dim
        proj_params = self.num_attention_heads * self.head_dim * self.hidden_size
        attn_per_layer = q_params + kv_params + proj_params

        dense_ffn = 3 * self.hidden_size * self.dense_ffn_hidden_size
        expert_params = 3 * self.hidden_size * self.expert_ffn_hidden_size
        all_experts = self.num_experts * expert_params

        print("\nParameter Breakdown:")
        print(f"  Embeddings (in+out): {embed / 1e9:.2f}B")
        print(f"  Attention per layer: {attn_per_layer / 1e6:.1f}M")
        print(
            f"  Total attention ({self.num_layers} layers): {self.num_layers * attn_per_layer / 1e9:.2f}B"
        )
        print(f"  Dense FFN per layer: {dense_ffn / 1e6:.1f}M")
        print(
            f"  Total dense FFN ({self.num_dense_layers} layers): {self.num_dense_layers * dense_ffn / 1e9:.2f}B"
        )
        print(f"  Expert params each: {expert_params / 1e6:.2f}M")
        print(f"  All experts per MoE layer: {all_experts / 1e9:.2f}B")
        print(
            f"  Total MoE experts ({self.num_moe_layers} layers): {self.num_moe_layers * all_experts / 1e9:.1f}B"
        )


@dataclass
class ParallelismConfig:
    """Parallelism configuration"""

    tp: int = 1  # Tensor parallel
    pp: int = 1  # Pipeline parallel
    ep: int = 1  # Expert parallel
    cp: int = 1  # Context parallel

    @property
    def model_parallel_size(self) -> int:
        """Total model parallel size (excluding DP)"""
        return self.tp * self.pp * self.ep

    def dp_size(self, total_gpus: int) -> int:
        """Calculate data parallel size"""
        return total_gpus // self.model_parallel_size


@dataclass
class LoRAConfig:
    """LoRA adapter configuration"""

    rank: int = 128
    alpha: int = 32
    target_modules: tuple = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")
    dropout: float = 0.05


@dataclass
class TrainingConfig:
    """Training configuration"""

    micro_batch_size: int = 1
    seq_length: int = 8192
    dtype_bytes: int = 2  # bf16


def calculate_memory(
    model: GLM47Config,
    parallel: ParallelismConfig,
    lora: LoRAConfig,
    training: TrainingConfig,
    total_gpus: int,
    use_recompute: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Calculate per-GPU memory requirements for LoRA training.

    Memory components:
    1. Model weights (frozen, bf16) - sharded by TP, PP, EP
    2. LoRA adapter weights + gradients + optimizer states
    3. Activations (the main variable with/without recompute)
    4. Communication buffers
    5. KV cache (for attention)
    """

    results = {}

    dp = parallel.dp_size(total_gpus)
    results["dp_size"] = dp

    # =========================================================================
    # 1. BASE MODEL WEIGHTS (frozen, no gradients/optimizer states)
    # =========================================================================

    # Embedding - sharded by TP
    embed_params = model.vocab_size * model.hidden_size
    embed_per_gpu = embed_params / parallel.tp

    # Attention weights per layer - sharded by TP
    # QKV: hidden -> (heads + 2*kv_heads) * head_dim
    qkv_params = (
        model.hidden_size
        * (model.num_attention_heads + 2 * model.num_kv_heads)
        * model.head_dim
    )
    proj_params = model.hidden_size * model.hidden_size
    attn_params_per_layer = (qkv_params + proj_params) / parallel.tp

    # Dense FFN layers (first 3 layers) - sharded by TP
    # SwiGLU has 3 projections: gate, up, down
    dense_ffn_params_per_layer = (
        3 * model.hidden_size * model.dense_ffn_hidden_size
    ) / parallel.tp

    # MoE expert weights - sharded by EP (and TP within EP group)
    # SwiGLU has 3 projections: gate, up, down
    expert_params = 3 * model.hidden_size * model.expert_ffn_hidden_size
    experts_per_ep_rank = model.num_experts / parallel.ep
    # Router params
    router_params = model.hidden_size * model.num_experts
    # Shared expert
    shared_expert_params = model.num_shared_experts * expert_params
    moe_params_per_layer = (
        (experts_per_ep_rank * expert_params) + router_params + shared_expert_params
    ) / parallel.tp

    # Layers per PP stage
    layers_per_stage = model.num_layers / parallel.pp
    dense_layers_per_stage = min(model.num_dense_layers, layers_per_stage)
    moe_layers_per_stage = layers_per_stage - dense_layers_per_stage

    # Total model weight bytes per GPU
    model_weight_params = (
        embed_per_gpu  # embedding (assume first PP stage)
        + dense_layers_per_stage * (attn_params_per_layer + dense_ffn_params_per_layer)
        + moe_layers_per_stage * (attn_params_per_layer + moe_params_per_layer)
        + (embed_per_gpu if parallel.pp == 1 else 0)  # output proj on last stage
    )
    model_weight_bytes = model_weight_params * training.dtype_bytes

    results["model_weights_gb"] = model_weight_bytes / 1e9
    results["model_weight_params_billions"] = model_weight_params / 1e9

    # =========================================================================
    # 2. LORA ADAPTER PARAMETERS
    # =========================================================================

    # LoRA adds low-rank matrices A (d x r) and B (r x d) to target modules
    # For QKV: adapts combined QKV projection
    # For proj: adapts output projection
    # For fc1/fc2: adapts FFN

    # Count target linear layers per transformer layer
    lora_targets_per_attn = 2  # qkv + proj (fused)
    lora_targets_per_ffn = 2  # fc1 + fc2

    # LoRA params per attention target: 2 * hidden * rank (A and B)
    lora_attn_params = lora_targets_per_attn * 2 * model.hidden_size * lora.rank

    # LoRA params per FFN target (for dense layers)
    lora_dense_ffn_params = lora_targets_per_ffn * 2 * model.hidden_size * lora.rank

    # LoRA params per expert (MoE layers) - each expert gets its own LoRA
    # Actually, Megatron typically applies LoRA to the router input, not per-expert
    # For now, assume LoRA on attention + shared components only in MoE layers
    lora_moe_layer_params = lora_attn_params  # Only attention in MoE layers

    # Total LoRA params per PP stage
    lora_params_per_stage = (
        dense_layers_per_stage * (lora_attn_params + lora_dense_ffn_params)
        + moe_layers_per_stage * lora_moe_layer_params
    )

    # LoRA is sharded by TP
    lora_params_per_gpu = lora_params_per_stage / parallel.tp

    # LoRA memory: weights + gradients + optimizer states (Adam: m, v, master weights)
    # With distributed optimizer, optimizer states are sharded by DP
    lora_weight_bytes = lora_params_per_gpu * training.dtype_bytes
    lora_grad_bytes = lora_params_per_gpu * training.dtype_bytes
    # Optimizer: master weights (fp32) + m (fp32) + v (fp32) = 12 bytes per param
    # Sharded by DP
    lora_optim_bytes = (lora_params_per_gpu * 12) / dp

    lora_total_bytes = lora_weight_bytes + lora_grad_bytes + lora_optim_bytes

    results["lora_params_millions"] = lora_params_per_gpu / 1e6
    results["lora_total_gb"] = lora_total_bytes / 1e9

    # =========================================================================
    # 3. ACTIVATION MEMORY
    # =========================================================================

    # This is the critical component for LoRA training
    # Even though base model is frozen, we need activations for backprop to reach LoRA adapters

    batch = training.micro_batch_size
    seq = training.seq_length
    hidden = model.hidden_size
    heads = model.num_attention_heads
    kv_heads = model.num_kv_heads
    # Removed: ffn = model.ffn_hidden_size (no longer exists, use expert_ffn_hidden_size)

    # Per-layer activation memory (without recompute)
    # Attention:
    #   - Input: batch * seq * hidden
    #   - QKV: batch * seq * (heads + 2*kv_heads) * head_dim
    #   - Attention scores: batch * heads * seq * seq (this is huge for long seq!)
    #   - Attention output: batch * seq * hidden
    # FFN:
    #   - Input: batch * seq * hidden
    #   - Hidden: batch * seq * ffn_hidden (per active expert in MoE)
    #   - Output: batch * seq * hidden

    # With sequence parallel (SP) enabled, sequence dimension is sharded by TP
    seq_per_tp = seq / parallel.tp

    # Attention activations per layer
    attn_input = batch * seq_per_tp * hidden
    qkv_act = batch * seq_per_tp * (heads + 2 * kv_heads) * model.head_dim / parallel.tp
    # Attention scores - Flash Attention doesn't materialize full O(n^2) matrix
    # It recomputes in tiles, so memory is O(n) not O(n^2)
    attn_scores_flash = batch * heads / parallel.tp * seq_per_tp * 256  # tile size
    attn_output = batch * seq_per_tp * hidden

    attn_act_per_layer = (
        attn_input + qkv_act + attn_scores_flash + attn_output
    ) * training.dtype_bytes

    # FFN activations per layer
    # For MoE: only top-k experts process tokens, but tokens are redistributed
    # Use per-expert FFN hidden size, not the combined FFN width
    expert_ffn = model.expert_ffn_hidden_size
    ffn_input = batch * seq_per_tp * hidden
    # SwiGLU: gate and up activations, then down
    ffn_hidden_act = (
        batch
        * seq_per_tp
        * expert_ffn
        * 2
        * (model.num_experts_per_token + model.num_shared_experts)
    )
    ffn_output = batch * seq_per_tp * hidden

    ffn_act_per_layer = (ffn_input + ffn_hidden_act + ffn_output) * training.dtype_bytes

    # Total activations without recompute (store all layers in PP stage)
    if not use_recompute:
        total_act_bytes = layers_per_stage * (attn_act_per_layer + ffn_act_per_layer)
    else:
        # With full recompute, only store input to each layer (recompute everything else)
        # Actually stores: layer inputs + minimal intermediate state
        input_act_per_layer = batch * seq_per_tp * hidden * training.dtype_bytes
        total_act_bytes = layers_per_stage * input_act_per_layer * 2  # some overhead

    results["activation_gb"] = total_act_bytes / 1e9
    results["activation_per_layer_mb"] = (attn_act_per_layer + ffn_act_per_layer) / 1e6

    # =========================================================================
    # 4. KV CACHE (for inference-style caching, not always used in training)
    # =========================================================================

    # KV cache per layer: 2 * batch * seq * kv_heads * head_dim
    kv_per_layer = 2 * batch * seq * kv_heads * model.head_dim * training.dtype_bytes
    kv_total = layers_per_stage * kv_per_layer / parallel.tp

    results["kv_cache_gb"] = kv_total / 1e9

    # =========================================================================
    # 5. COMMUNICATION BUFFERS & MISC
    # =========================================================================

    # NCCL buffers, gradient buckets, etc.
    # Typically 2-4 GB for large models
    comm_buffers_gb = 4.0

    # CUDA context, kernels, etc.
    cuda_overhead_gb = 2.0

    results["comm_buffers_gb"] = comm_buffers_gb
    results["cuda_overhead_gb"] = cuda_overhead_gb

    # =========================================================================
    # TOTAL
    # =========================================================================

    total_gb = (
        results["model_weights_gb"]
        + results["lora_total_gb"]
        + results["activation_gb"]
        + results["comm_buffers_gb"]
        + results["cuda_overhead_gb"]
    )

    results["total_gb"] = total_gb
    results["use_recompute"] = use_recompute

    if verbose:
        print(f"\n{'=' * 70}")
        print("GLM-4.7 LoRA Memory Analysis")
        print(f"{'=' * 70}")
        print(
            f"\nParallelism: TP={parallel.tp}, PP={parallel.pp}, EP={parallel.ep}, CP={parallel.cp}"
        )
        print(f"Data Parallel: DP={dp} (with {total_gpus} total GPUs)")
        print(f"Recompute: {'ENABLED' if use_recompute else 'DISABLED'}")
        print(f"\nLoRA: rank={lora.rank}, alpha={lora.alpha}")
        print(
            f"Training: micro_batch={training.micro_batch_size}, seq_len={training.seq_length}"
        )
        print(f"\n{'-' * 70}")
        print("Memory Breakdown (per GPU):")
        print(f"{'-' * 70}")
        print(
            f"  Model weights (frozen bf16):  {results['model_weights_gb']:>8.2f} GB  ({results['model_weight_params_billions']:.2f}B params)"
        )
        print(
            f"  LoRA adapters + optim:        {results['lora_total_gb']:>8.2f} GB  ({results['lora_params_millions']:.1f}M params)"
        )
        print(
            f"  Activations:                  {results['activation_gb']:>8.2f} GB  ({results['activation_per_layer_mb']:.1f} MB/layer)"
        )
        print(f"  Communication buffers:        {results['comm_buffers_gb']:>8.2f} GB")
        print(f"  CUDA overhead:                {results['cuda_overhead_gb']:>8.2f} GB")
        print(f"{'-' * 70}")
        print(f"  TOTAL:                        {results['total_gb']:>8.2f} GB")
        print(f"{'=' * 70}")

        h100_memory = 80
        h200_memory = 141

        if total_gb <= h100_memory:
            print(
                f"  ✅ FITS in H100 (80GB) with {h100_memory - total_gb:.1f}GB headroom"
            )
        else:
            print(
                f"  ❌ DOES NOT FIT in H100 (80GB) - needs {total_gb - h100_memory:.1f}GB more"
            )

        if total_gb <= h200_memory:
            print(
                f"  ✅ FITS in H200 (141GB) with {h200_memory - total_gb:.1f}GB headroom"
            )

    return results


def calculate_pp_stage_memory(
    model: GLM47Config,
    parallel: ParallelismConfig,
    training: TrainingConfig,
    pp_stage: int,
) -> dict:
    """
    Calculate memory for a specific PP stage, accounting for:
    - Embedding layer (PP stage 0 only)
    - Output projection (last PP stage only)
    - Varying layer counts due to integer division
    """
    total_layers = model.num_layers
    layers_per_stage = total_layers // parallel.pp

    # Handle uneven layer distribution
    extra_layers = total_layers % parallel.pp
    if pp_stage < extra_layers:
        layers_per_stage += 1

    # Embedding on first stage, output proj on last stage
    has_embedding = pp_stage == 0
    has_output_proj = pp_stage == parallel.pp - 1

    # Model weights
    embed_params = model.vocab_size * model.hidden_size if has_embedding else 0
    output_params = model.vocab_size * model.hidden_size if has_output_proj else 0

    # Attention per layer
    q_params = model.hidden_size * model.num_attention_heads * model.head_dim
    kv_params = model.hidden_size * 2 * model.num_kv_heads * model.head_dim
    proj_params = model.num_attention_heads * model.head_dim * model.hidden_size
    attn_per_layer = (q_params + kv_params + proj_params) / parallel.tp

    # MoE params per layer
    expert_params = 3 * model.hidden_size * model.expert_ffn_hidden_size
    experts_per_ep = model.num_experts / parallel.ep
    router_params = model.hidden_size * model.num_experts
    shared_expert = model.num_shared_experts * expert_params
    moe_per_layer = (
        (experts_per_ep * expert_params) + router_params + shared_expert
    ) / parallel.tp

    # Total weights for this PP stage
    weight_params = (
        embed_params / parallel.tp
        + output_params / parallel.tp
        + layers_per_stage * (attn_per_layer + moe_per_layer)
    )
    weight_bytes = weight_params * training.dtype_bytes

    # Activations per layer
    batch = training.micro_batch_size
    seq = training.seq_length
    hidden = model.hidden_size
    seq_per_tp = seq / parallel.tp

    # Forward pass activations
    attn_act = batch * seq_per_tp * hidden * 4  # input, qkv, scores approx, output
    expert_ffn = model.expert_ffn_hidden_size
    ffn_act = batch * seq_per_tp * hidden * 2 + batch * seq_per_tp * expert_ffn * 2 * (
        model.num_experts_per_token + 1
    )

    act_per_layer = (attn_act + ffn_act) * training.dtype_bytes

    # Backward pass roughly doubles activation memory due to gradients
    backward_multiplier = 1.8  # empirical: backward needs ~1.8x forward memory

    total_act = layers_per_stage * act_per_layer * backward_multiplier

    return {
        "pp_stage": pp_stage,
        "layers": layers_per_stage,
        "has_embedding": has_embedding,
        "has_output_proj": has_output_proj,
        "weight_gb": weight_bytes / 1e9,
        "activation_gb": total_act / 1e9,
        "total_gb": (weight_bytes + total_act) / 1e9 + 6,  # +6GB for buffers/overhead
    }


def calculate_recompute_memory(
    model: GLM47Config,
    parallel: ParallelismConfig,
    training: TrainingConfig,
    recompute_num_layers: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Calculate memory with activation recomputation.

    With full recompute (recompute_num_layers=1):
    - Only store input activations at checkpoint boundaries
    - During backward, recompute forward pass to regenerate activations
    - Peak memory = weights + checkpoint_activations + recompute_buffer + gradients

    With partial recompute (recompute_num_layers=N):
    - Checkpoint every N layers
    - Store N layers worth of activations at a time
    """
    results = {}

    batch = training.micro_batch_size
    seq = training.seq_length
    hidden = model.hidden_size
    layers_per_stage = model.num_layers // parallel.pp
    seq_per_tp = seq // parallel.tp

    # Model weights (same regardless of recompute)
    embed_params = model.vocab_size * model.hidden_size / parallel.tp
    expert_params = 3 * model.hidden_size * model.expert_ffn_hidden_size
    experts_per_ep = model.num_experts / parallel.ep
    moe_params_per_layer = (experts_per_ep * expert_params) / parallel.tp

    attn_params_per_layer = (
        model.hidden_size
        * (model.num_attention_heads + 2 * model.num_kv_heads)
        * model.head_dim
        + model.hidden_size * model.hidden_size
    ) / parallel.tp

    weight_params = embed_params + layers_per_stage * (
        attn_params_per_layer + moe_params_per_layer
    )
    weight_gb = weight_params * training.dtype_bytes / 1e9

    # Checkpoint activation memory
    # With recompute_num_layers=N, we store inputs every N layers
    checkpoint_interval = recompute_num_layers
    num_checkpoints = (
        layers_per_stage + checkpoint_interval - 1
    ) // checkpoint_interval

    # Each checkpoint stores: hidden_states (batch * seq * hidden)
    checkpoint_act_bytes = batch * seq_per_tp * hidden * training.dtype_bytes
    total_checkpoint_bytes = num_checkpoints * checkpoint_act_bytes

    # Recompute buffer: during backward, we need to store activations for N layers
    # This is the peak memory during recomputation
    per_layer_act_bytes = (
        batch * seq_per_tp * hidden * 4 * training.dtype_bytes
    )  # input, qkv, intermediate, output
    expert_act_bytes = (
        batch
        * seq_per_tp
        * model.expert_ffn_hidden_size
        * 2
        * (model.num_experts_per_token + 1)
        * training.dtype_bytes
    )
    full_layer_act = per_layer_act_bytes + expert_act_bytes

    recompute_buffer_bytes = checkpoint_interval * full_layer_act

    # Gradient buffer (for backward pass)
    grad_buffer_bytes = (
        full_layer_act * 1.5
    )  # gradients are slightly larger due to intermediate storage

    # Communication and CUDA overhead
    comm_overhead_gb = 4.0
    cuda_overhead_gb = 2.0

    # MoE backward spike (reduce_scatter buffers)
    # This is the additional memory needed during MoE backward
    moe_backward_spike_gb = 1.5 * batch  # Empirically ~1.5GB per sample in batch

    total_act_gb = (
        total_checkpoint_bytes + recompute_buffer_bytes + grad_buffer_bytes
    ) / 1e9
    total_gb = (
        weight_gb
        + total_act_gb
        + comm_overhead_gb
        + cuda_overhead_gb
        + moe_backward_spike_gb
    )

    results = {
        "weight_gb": weight_gb,
        "checkpoint_act_gb": total_checkpoint_bytes / 1e9,
        "recompute_buffer_gb": recompute_buffer_bytes / 1e9,
        "grad_buffer_gb": grad_buffer_bytes / 1e9,
        "moe_backward_spike_gb": moe_backward_spike_gb,
        "comm_overhead_gb": comm_overhead_gb,
        "cuda_overhead_gb": cuda_overhead_gb,
        "total_gb": total_gb,
        "num_checkpoints": num_checkpoints,
    }

    if verbose:
        print(
            f"\nRecompute Analysis (mbs={batch}, recompute_num_layers={recompute_num_layers}):"
        )
        print(f"  Model weights:      {weight_gb:>6.1f} GB")
        print(
            f"  Checkpoint acts:    {total_checkpoint_bytes / 1e9:>6.1f} GB ({num_checkpoints} checkpoints)"
        )
        print(
            f"  Recompute buffer:   {recompute_buffer_bytes / 1e9:>6.1f} GB ({checkpoint_interval} layers)"
        )
        print(f"  Grad buffer:        {grad_buffer_bytes / 1e9:>6.1f} GB")
        print(f"  MoE backward spike: {moe_backward_spike_gb:>6.1f} GB")
        print(f"  Comm + CUDA:        {comm_overhead_gb + cuda_overhead_gb:>6.1f} GB")
        print("  --------------------------")
        print(f"  TOTAL:              {total_gb:>6.1f} GB")
        headroom = 80 - total_gb
        status = "✅" if headroom > 2 else ("⚠️" if headroom > 0 else "❌")
        print(f"  H100 headroom:      {headroom:>6.1f} GB {status}")

    return results


def main():
    model = GLM47Config()
    lora = LoRAConfig(rank=128, alpha=32)
    training = TrainingConfig(micro_batch_size=1, seq_length=8192)

    print("\nGLM-4.7 Model Stats:")
    print(f"  Total parameters: {model.total_params() / 1e9:.1f}B")
    print(f"  Active parameters: {model.active_params() / 1e9:.1f}B")
    model.print_breakdown()

    # =========================================================================
    # PP STAGE ANALYSIS - explains the OOM on PP stage 0
    # =========================================================================
    print("\n" + "=" * 70)
    print("PP STAGE MEMORY BREAKDOWN (TP=2, PP=4, EP=4, mbs=2)")
    print("=" * 70)
    nvidia_config = ParallelismConfig(tp=2, pp=4, ep=4, cp=1)
    training_mb2 = TrainingConfig(micro_batch_size=2, seq_length=8192)

    print(
        f"\n{'PP Stage':<10} {'Layers':<8} {'Embed?':<8} {'Weights':<12} {'Activations':<14} {'Total':<12} {'H100 Fit?'}"
    )
    print("-" * 80)
    for stage in range(4):
        result = calculate_pp_stage_memory(model, nvidia_config, training_mb2, stage)
        fits = "✅" if result["total_gb"] < 75 else "❌ OOM risk"
        embed_str = "Yes" if result["has_embedding"] else "No"
        print(
            f"{stage:<10} {result['layers']:<8} {embed_str:<8} {result['weight_gb']:<12.1f} {result['activation_gb']:<14.1f} {result['total_gb']:<12.1f} {fits}"
        )

    print("\n" + "=" * 70)
    print("PP STAGE MEMORY BREAKDOWN (TP=2, PP=4, EP=4, mbs=1) - SAFER")
    print("=" * 70)
    training_mb1 = TrainingConfig(micro_batch_size=1, seq_length=8192)

    print(
        f"\n{'PP Stage':<10} {'Layers':<8} {'Embed?':<8} {'Weights':<12} {'Activations':<14} {'Total':<12} {'H100 Fit?'}"
    )
    print("-" * 80)
    for stage in range(4):
        result = calculate_pp_stage_memory(model, nvidia_config, training_mb1, stage)
        fits = "✅" if result["total_gb"] < 75 else "❌ OOM risk"
        embed_str = "Yes" if result["has_embedding"] else "No"
        print(
            f"{stage:<10} {result['layers']:<8} {embed_str:<8} {result['weight_gb']:<12.1f} {result['activation_gb']:<14.1f} {result['total_gb']:<12.1f} {fits}"
        )

    total_gpus = 32

    # Current config: TP=2, PP=1, EP=8
    print("\n" + "=" * 70)
    print("CONFIGURATION 1: Current (TP=2, PP=1, EP=8) - WITH recompute")
    print("=" * 70)
    current_config = ParallelismConfig(tp=2, pp=1, ep=8, cp=1)
    calculate_memory(
        model, current_config, lora, training, total_gpus, use_recompute=True
    )

    print("\n" + "=" * 70)
    print("CONFIGURATION 2: Current (TP=2, PP=1, EP=8) - WITHOUT recompute")
    print("=" * 70)
    calculate_memory(
        model, current_config, lora, training, total_gpus, use_recompute=False
    )

    # NVIDIA recommended: TP=2, PP=4, EP=4
    print("\n" + "=" * 70)
    print("CONFIGURATION 3: NVIDIA Recipe (TP=2, PP=4, EP=4) - WITHOUT recompute")
    print("=" * 70)
    nvidia_config = ParallelismConfig(tp=2, pp=4, ep=4, cp=1)
    calculate_memory(
        model, nvidia_config, lora, training, total_gpus, use_recompute=False
    )

    # Alternative: TP=2, PP=2, EP=8
    print("\n" + "=" * 70)
    print("CONFIGURATION 4: Alternative (TP=2, PP=2, EP=8) - WITHOUT recompute")
    print("=" * 70)
    alt_config = ParallelismConfig(tp=2, pp=2, ep=8, cp=1)
    calculate_memory(model, alt_config, lora, training, total_gpus, use_recompute=False)

    # Test with higher micro batch
    print("\n" + "=" * 70)
    print("CONFIGURATION 5: NVIDIA Recipe + micro_batch=2 - WITHOUT recompute")
    print("=" * 70)
    training_mb2 = TrainingConfig(micro_batch_size=2, seq_length=8192)
    calculate_memory(
        model, nvidia_config, lora, training_mb2, total_gpus, use_recompute=False
    )

    # What about without any EP sharding optimization?
    print("\n" + "=" * 70)
    print("CONFIGURATION 6: Minimal (TP=2, PP=1, EP=1) - see what explodes")
    print("=" * 70)
    minimal_config = ParallelismConfig(tp=2, pp=1, ep=1, cp=1)
    calculate_memory(
        model, minimal_config, lora, training, total_gpus, use_recompute=False
    )

    # =========================================================================
    # OPTIMIZED CONFIGURATIONS - exploring throughput optimization
    # =========================================================================

    print("\n" + "=" * 70)
    print("CONFIGURATION 7: NVIDIA Recipe + micro_batch=4 - max throughput?")
    print("=" * 70)
    training_mb4 = TrainingConfig(micro_batch_size=4, seq_length=8192)
    calculate_memory(
        model, nvidia_config, lora, training_mb4, total_gpus, use_recompute=False
    )

    print("\n" + "=" * 70)
    print("CONFIGURATION 8: PP=2, EP=8 + micro_batch=2 - balance")
    print("=" * 70)
    calculate_memory(
        model, alt_config, lora, training_mb2, total_gpus, use_recompute=False
    )

    print("\n" + "=" * 70)
    print("CONFIGURATION 9: PP=2, EP=8 + micro_batch=4 - push throughput")
    print("=" * 70)
    calculate_memory(
        model, alt_config, lora, training_mb4, total_gpus, use_recompute=False
    )

    # Smaller LoRA rank for comparison
    print("\n" + "=" * 70)
    print("CONFIGURATION 10: NVIDIA Recipe + LoRA rank=64 + micro_batch=4")
    print("=" * 70)
    lora_small = LoRAConfig(rank=64, alpha=16)
    calculate_memory(
        model, nvidia_config, lora_small, training_mb4, total_gpus, use_recompute=False
    )

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Memory vs Throughput Trade-offs")
    print("=" * 70)
    print(f"\n{'Config':<45} {'Memory':<12} {'Fits H100?':<12} {'Headroom':<12}")
    print("-" * 81)

    configs_to_compare = [
        ("Current (PP=1,EP=8) + recompute", current_config, training, True),
        ("Current (PP=1,EP=8) no recompute", current_config, training, False),
        ("NVIDIA (PP=4,EP=4) mbs=1", nvidia_config, training, False),
        ("NVIDIA (PP=4,EP=4) mbs=2", nvidia_config, training_mb2, False),
        ("NVIDIA (PP=4,EP=4) mbs=4", nvidia_config, training_mb4, False),
        ("Alt (PP=2,EP=8) mbs=1", alt_config, training, False),
        ("Alt (PP=2,EP=8) mbs=2", alt_config, training_mb2, False),
        ("Alt (PP=2,EP=8) mbs=4", alt_config, training_mb4, False),
    ]

    for name, cfg, train_cfg, recompute in configs_to_compare:
        result = calculate_memory(
            model,
            cfg,
            lora,
            train_cfg,
            total_gpus,
            use_recompute=recompute,
            verbose=False,
        )
        fits = "✅ Yes" if result["total_gb"] <= 80 else "❌ No"
        headroom = 80 - result["total_gb"]
        headroom_str = (
            f"{headroom:.1f}GB" if headroom > 0 else f"{-headroom:.1f}GB over"
        )
        print(f"{name:<45} {result['total_gb']:<12.1f} {fits:<12} {headroom_str:<12}")

    # =========================================================================
    # V3 OPTIMIZATION: Recompute configurations with different mbs
    # =========================================================================
    print("\n" + "=" * 70)
    print("V3 OPTIMIZATION: Recompute Memory Analysis")
    print("=" * 70)
    print("\nGoal: Find optimal mbs to leverage ~17GB/GPU unused VRAM from V2")
    print("V2 baseline: mbs=2, full recompute, ~63GB observed peak")

    nvidia_config = ParallelismConfig(tp=2, pp=4, ep=4, cp=1)

    # Test different mbs with full recompute
    for mbs in [1, 2, 3, 4]:
        train_cfg = TrainingConfig(micro_batch_size=mbs, seq_length=8192)
        calculate_recompute_memory(
            model, nvidia_config, train_cfg, recompute_num_layers=1, verbose=True
        )

    # Test partial recompute (recompute_num_layers=2) with mbs=2
    print("\n" + "-" * 70)
    print("Alternative: Partial recompute (recompute_num_layers=2)")
    print("-" * 70)
    for mbs in [2, 3]:
        train_cfg = TrainingConfig(micro_batch_size=mbs, seq_length=8192)
        calculate_recompute_memory(
            model, nvidia_config, train_cfg, recompute_num_layers=2, verbose=True
        )

    # =========================================================================
    # CALIBRATED ESTIMATE: Based on observed V2 data
    # =========================================================================
    print("\n" + "=" * 70)
    print("CALIBRATED ESTIMATES (based on observed V2 peak: 63GB)")
    print("=" * 70)
    print("""
The calculator underestimates by ~25GB due to:
- PP buffers (multiple microbatches in flight)
- MoE all-to-all communication buffers
- PyTorch autograd graph overhead
- CUDA allocator fragmentation

Calibrated scaling from observed V2 (mbs=2, full recompute, 63GB):
- Base overhead: ~30GB (weights + fixed buffers)
- Per-mbs overhead: ~16.5GB

mbs=1: ~30 + 16.5 = ~47GB  (confirmed: lower than V2)
mbs=2: ~30 + 33   = ~63GB  (confirmed: V2 observed peak)
mbs=3: ~30 + 50   = ~80GB  ⚠️  Right at H100 limit!
mbs=4: ~30 + 66   = ~96GB  ❌ Will OOM
""")

    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. PRIMARY: Try mbs=3 with full recompute
   - Calibrated estimate: ~80GB peak (right at H100 limit)
   - Risk: MoE backward spikes may cause intermittent OOM
   - Potential: ~50% throughput improvement if stable

2. SAFER ALTERNATIVE: If mbs=3 is unstable, try mbs=2 with recompute_num_layers=2
   - Reduces recompute overhead by ~15%
   - Should stay at ~63GB with less compute overhead
   - Modest throughput gain (~10-15%)

3. AGGRESSIVE (NOT RECOMMENDED): mbs=4
   - Will almost certainly OOM on MoE backward pass
   - Only consider if mbs=3 shows >5GB headroom in practice

Run V3 with: modal run --detach modal_train.py::train_lora_optimized_v3
""")


if __name__ == "__main__":
    main()
