# pyright: reportUnknownMemberType=false
"""Patch registry for the DeepSeek-V4-Flash SFT example.

Each patch is isolated here so the training file shows the recipe rather than a wall
of source edits.  The `why` and `ablation` fields document the failure that justified
keeping the patch.  Disable one or more patches for experiments with:

    DSV4_DISABLED_PATCHES=patch_name[,patch_name...] modal run ...

## Patch categories

There are three independent categories — a total of 16 patches, but only the first
two categories apply to training:

### 1. Model registration (2 patches) — required at ALL sequence lengths
Transformers 4.57 does not register `deepseek_v4` as a known model type, so
AutoConfig / MODEL_FOR_CAUSAL_LM_MAPPING lookups fail.  These two patches append
the registration to the installed transformers package.

### 2. Long-context memory (7 patches) — five reduce peak memory at 60k
At 60k tokens, five patches reduce peak memory. Three of them sever autograd for
any adapter upstream of attention, so they are only safe with linear_proj-only
LoRA (these three are GRADIENT_UNSAFE_TRAINING_PATCHES):
  - attention concat in-place rewrite + del          (~60 MiB/layer saved)
  - CSA gather/softmax seq-chunking                  (7.5 GiB → 4 MiB peak per chunk)
  - RoPE rotary chunking (megatron_rope_seq_chunking)  (~30 MiB/head saved)

The other two reduce memory without touching adapter gradients (the mask is not
differentiable; the indexer's no_grad path only fires when its aux loss is off),
so they are gradient-safe at any target_modules:
  - padding-mask diagonal extraction          (avoids 900M-element reduction)
  - DSA indexer head-chunking + no_grad       (43 GiB → 22 GiB peak)

Two more are gradient-safe and always stay on:
  - rope config (YaRN field propagation)      (required at ALL lengths)
  - RoPE CP shape fix (megatron_rope_cp_shape_fix)  (required for CP>1)

At the default 4k sequence length, none of the memory patches OOM — only the rope
config patch is required so Megatron can initialize YaRN position embeddings. To
train adapters upstream of attention (e.g. linear_qkv / all-linear), disable the
three gradient-unsafe patches and absorb their memory by raising context parallelism.

The three gradient-unsafe patches are validated only for `linear_proj` LoRA. The
training entrypoint rejects broader target modules while those patches are enabled.

### 3. vLLM BF16 serving (7 patches) — only needed for inference / eval
vLLM 0.22.1 expects FP8/MXFP4 scale tensors and CUTLASS DSL kernels that are
absent when serving the merged BF16 checkpoint.  These patches add fallbacks so
the exported model can be served without quantization.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class PatchSpec:
    name: str
    command: str
    why: str
    required_for: str
    verify_command: str | None = None
    ablation: str = "not yet ablated"


GRADIENT_UNSAFE_TRAINING_PATCHES = frozenset(
    {
        "mcore_bridge_attention_memory",
        "megatron_csa_chunked_unfused_attention",
        "megatron_rope_seq_chunking",
    }
)


def disabled_patch_names() -> set[str]:
    return {
        name.strip()
        for name in os.environ.get("DSV4_DISABLED_PATCHES", "").split(",")
        if name.strip()
    }


def apply_image_patches(image, patches: Iterable[PatchSpec]):
    disabled = disabled_patch_names()
    applied: list[str] = []
    for patch in patches:
        if patch.name in disabled:
            image = image.run_commands(f"echo Skipping DeepSeek patch: {patch.name}")
            continue
        image = image.run_commands(f"echo Applying DeepSeek patch: {patch.name}")
        image = image.run_commands(patch.command)
        applied.append(patch.name)
        if patch.verify_command is not None:
            image = image.run_commands(patch.verify_command)
    return image.env(
        {
            "DSV4_APPLIED_PATCHES": ",".join(applied),
            "DSV4_DISABLED_PATCHES": ",".join(sorted(disabled)),
        }
    )


DEEPSEEK_V4_CONFIG_PATCH = r"""cat >>/usr/local/lib/python3.11/site-packages/transformers/models/auto/configuration_auto.py <<'PY'
try:
    try:
        from ...configuration_utils import PreTrainedConfig as _BaseConfig
    except ImportError:
        from ...configuration_utils import PretrainedConfig as _BaseConfig

    class DeepseekV4Config(_BaseConfig):
        model_type = "deepseek_v4"
        has_no_defaults_at_init = True
        keys_to_ignore_at_inference = ["past_key_values"]
        attribute_map = {
            "intermediate_size": "moe_intermediate_size",
            "num_local_experts": "n_routed_experts",
        }
        default_compress_rates = {
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        }
        default_num_hash_layers = 3
        default_partial_rotary_factor = 64 / 512

        def __init__(self, **kwargs):
            compress_ratios = kwargs.pop("compress_ratios", None)
            compress_rate_csa = kwargs.pop("compress_rate_csa", None)
            compress_rate_hca = kwargs.pop("compress_rate_hca", None)
            num_hash_layers = kwargs.pop("num_hash_layers", None)
            qk_rope_head_dim = kwargs.pop("qk_rope_head_dim", None)
            super().__init__(**kwargs)

            n_layers = getattr(self, "num_hidden_layers", 0)
            if getattr(self, "compress_rates", None) is None:
                self.compress_rates = dict(self.default_compress_rates)
            if compress_rate_csa is not None:
                self.compress_rates["compressed_sparse_attention"] = compress_rate_csa
            if compress_rate_hca is not None:
                self.compress_rates["heavily_compressed_attention"] = compress_rate_hca

            if getattr(self, "layer_types", None) is None and compress_ratios is not None:
                ratio_to_layer_type = {
                    0: "sliding_attention",
                    4: "compressed_sparse_attention",
                    128: "heavily_compressed_attention",
                }
                self.layer_types = [ratio_to_layer_type[r] for r in compress_ratios]
            if getattr(self, "layer_types", None) is None:
                interleave = [
                    "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
                    for i in range(max(n_layers - 2, 0))
                ]
                self.layer_types = ["heavily_compressed_attention"] * min(n_layers, 2) + interleave
            self.layer_types = list(self.layer_types[:n_layers])

            if getattr(self, "mlp_layer_types", None) is None:
                n_hash = num_hash_layers if num_hash_layers is not None else self.default_num_hash_layers
                self.mlp_layer_types = ["hash_moe"] * min(n_layers, n_hash) + ["moe"] * max(
                    0, n_layers - n_hash
                )
            self.mlp_layer_types = list(self.mlp_layer_types[:n_layers])

            if getattr(self, "partial_rotary_factor", None) is None:
                self.partial_rotary_factor = (
                    qk_rope_head_dim / getattr(self, "head_dim", 1)
                    if qk_rope_head_dim is not None
                    else self.default_partial_rotary_factor
                )
            self.qk_rope_head_dim = int(getattr(self, "head_dim", 1) * self.partial_rotary_factor)

    if "deepseek_v4" not in CONFIG_MAPPING:
        CONFIG_MAPPING.register("deepseek_v4", DeepseekV4Config)
except Exception:
    pass
PY"""

DEEPSEEK_V4_MODELING_PATCH = r"""cat >>/usr/local/lib/python3.11/site-packages/transformers/models/auto/modeling_auto.py <<'PY'
try:
    from ...modeling_utils import PreTrainedModel as _PreTrainedModel
    from ...models.auto.configuration_auto import CONFIG_MAPPING

    class DeepseekV4ForCausalLM(_PreTrainedModel):
        config_class = CONFIG_MAPPING["deepseek_v4"]
        base_model_prefix = "model"
        _no_split_modules = []

        def __init__(self, config):
            super().__init__(config)

        def forward(self, *args, **kwargs):
            raise NotImplementedError("DeepSeek-V4 export only uses this as a meta model")

    MODEL_FOR_CAUSAL_LM_MAPPING.register(CONFIG_MAPPING["deepseek_v4"], DeepseekV4ForCausalLM)
except Exception:
    pass
PY"""

DEEPSEEK_V4_CONFIG_VERIFY = r"""python - <<'PY'
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
config_cls = CONFIG_MAPPING['deepseek_v4']
assert config_cls.__name__ == 'DeepseekV4Config'
PY"""

DEEPSEEK_V4_MODELING_VERIFY = r"""python - <<'PY'
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
config_cls = CONFIG_MAPPING['deepseek_v4']
assert MODEL_FOR_CAUSAL_LM_MAPPING[config_cls].__name__ == 'DeepseekV4ForCausalLM'
PY"""

MCORE_BRIDGE_ROPE_CONFIG_PATCH = r"""python - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.11/site-packages/mcore_bridge/config/model_config.py")
text = path.read_text()
rope_fields_old = (
    "    original_max_position_embeddings: Optional[int] = None\n"
    "    partial_rotary_factor: Optional[float] = None\n"
)
rope_fields_new = (
    "    original_max_position_embeddings: int = 4096\n"
    "    rotary_scaling_factor: float = 40\n"
    "    beta_fast: float = 32\n"
    "    beta_slow: float = 1\n"
    "    mscale: float = 1.0\n"
    "    mscale_all_dim: float = 0.0\n"
    "    partial_rotary_factor: Optional[float] = None\n"
)
if rope_fields_old not in text:
    raise RuntimeError("mcore_bridge rope field patch target not found")
text = text.replace(rope_fields_old, rope_fields_new)

rope_scaling_old = (
    "            if 'type' in self.rope_scaling and 'rope_type' not in self.rope_scaling:\n"
    "                self.rope_scaling['rope_type'] = self.rope_scaling['type']\n"
)
rope_scaling_new = (
    "            if 'type' in self.rope_scaling and 'rope_type' not in self.rope_scaling:\n"
    "                self.rope_scaling['rope_type'] = self.rope_scaling['type']\n"
    "            if 'factor' in self.rope_scaling:\n"
    "                self.rotary_scaling_factor = self.rope_scaling['factor']\n"
    "            if 'original_max_position_embeddings' in self.rope_scaling:\n"
    "                self.original_max_position_embeddings = self.rope_scaling['original_max_position_embeddings']\n"
    "            if 'beta_fast' in self.rope_scaling:\n"
    "                self.beta_fast = self.rope_scaling['beta_fast']\n"
    "            if 'beta_slow' in self.rope_scaling:\n"
    "                self.beta_slow = self.rope_scaling['beta_slow']\n"
    "            if 'mscale' in self.rope_scaling:\n"
    "                self.mscale = self.rope_scaling['mscale']\n"
    "            if 'mscale_all_dim' in self.rope_scaling:\n"
    "                self.mscale_all_dim = self.rope_scaling['mscale_all_dim']\n"
    "            if self.llm_model_type == 'deepseek_v4' and 'main' not in self.rope_scaling:\n"
    "                self.rope_scaling = {\n"
    "                    'main': dict(self.rope_scaling),\n"
    "                    'compress': dict(self.rope_scaling),\n"
    "                }\n"
)
if rope_scaling_old not in text:
    raise RuntimeError("mcore_bridge rope scaling patch target not found")
text = text.replace(rope_scaling_old, rope_scaling_new)
path.write_text(text)
PY"""

MCORE_BRIDGE_ROPE_CONFIG_VERIFY = r"""python - <<'PY'
from pathlib import Path

text = Path("/usr/local/lib/python3.11/site-packages/mcore_bridge/config/model_config.py").read_text()
assert "rotary_scaling_factor: float = 40" in text
assert "if self.llm_model_type == 'deepseek_v4' and 'main' not in self.rope_scaling" in text
PY"""

MCORE_BRIDGE_MEMORY_PATCH = r"""python - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.11/site-packages/mcore_bridge/model/gpts/deepseek_v4.py")
text = path.read_text()
old = (
    "        core_attn_out = torch.cat([content_part, rot_part], dim=-1)\n"
    "        core_attn_out = core_attn_out.view(seq_len, core_attn_out.size(1), -1)\n"
)
new = (
    "        core_attn_out = torch.cat([content_part, rot_part], dim=-1)\n"
    "        del content_part, rot_part\n"
    "        core_attn_out = core_attn_out.view(seq_len, core_attn_out.size(1), -1)\n"
)
if old not in text:
    raise RuntimeError("mcore_bridge DeepSeek V4 attention output memory patch target not found")
text = text.replace(old, new)
old = "            query = torch.cat([q_no_pe, q_pos_emb], dim=-1)\n"
new = (
    "            query = q.detach()\n"
    "            query[..., q_no_pe.shape[-1] :] = q_pos_emb\n"
    "            del q_no_pe, q_pos_emb\n"
)
if old not in text:
    raise RuntimeError("mcore_bridge DeepSeek V4 query concat memory patch target not found")
text = text.replace(old, new)
old = "            kv = torch.cat([kv_no_pe, k_pos_emb], dim=-1).unsqueeze(-2)\n"
new = (
    "            kv = kv.detach()\n"
    "            kv[..., kv_no_pe.shape[-1] :] = k_pos_emb\n"
    "            del kv_no_pe, k_pos_emb\n"
    "            kv = kv.unsqueeze(-2)\n"
)
if old not in text:
    raise RuntimeError("mcore_bridge DeepSeek V4 kv concat memory patch target not found")
text = text.replace(old, new)
path.write_text(text)
PY"""

MCORE_BRIDGE_MEMORY_VERIFY = r"""python - <<'PY'
from pathlib import Path

text = Path("/usr/local/lib/python3.11/site-packages/mcore_bridge/model/gpts/deepseek_v4.py").read_text()
assert "del content_part, rot_part" in text
assert "query = q.detach()" in text
assert "kv = kv.detach()" in text
PY"""
MCORE_BRIDGE_LONG_CONTEXT_MASK_PATCH = r"""python - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.11/site-packages/mcore_bridge/model/gpt_model.py")
text = path.read_text()
old = (
    "        if mcore_016 and full_attention_mask is not None:\n"
    "            assert packed_seq_params is None\n"
    "            padding_mask = ~((~full_attention_mask).sum(dim=(1, 2)) > 0)\n"
    "            if self.config.context_parallel_size > 1:\n"
    "                padding_mask = split_cp_inputs(padding_mask, None, 1)\n"
    "            tp_size = self.config.tensor_model_parallel_size\n"
    "            if self.config.sequence_parallel and tp_size > 1:\n"
    "                assert padding_mask.shape[1] % tp_size == 0, f'padding_mask.shape: {padding_mask.shape}'\n"
    "                padding_mask = torch.chunk(padding_mask, tp_size, dim=1)[mpu.get_tensor_model_parallel_rank()]\n"
    "            extra_block_kwargs['padding_mask'] = padding_mask.contiguous()\n"
)
new = (
    "        if mcore_016 and full_attention_mask is not None:\n"
    "            assert packed_seq_params is None\n"
    "            if full_attention_mask.numel() > 2_000_000_000:\n"
    "                padding_mask = full_attention_mask.diagonal(dim1=2, dim2=3).squeeze(1)\n"
    "            else:\n"
    "                padding_mask = ~((~full_attention_mask).sum(dim=(1, 2)) > 0)\n"
    "            if self.config.context_parallel_size > 1:\n"
    "                padding_mask = split_cp_inputs(padding_mask, None, 1)\n"
    "            tp_size = self.config.tensor_model_parallel_size\n"
    "            if self.config.sequence_parallel and tp_size > 1:\n"
    "                assert padding_mask.shape[1] % tp_size == 0, f'padding_mask.shape: {padding_mask.shape}'\n"
    "                padding_mask = torch.chunk(padding_mask, tp_size, dim=1)[mpu.get_tensor_model_parallel_rank()]\n"
    "            extra_block_kwargs['padding_mask'] = padding_mask.contiguous()\n"
)
if old not in text:
    raise RuntimeError("mcore_bridge long-context padding-mask patch target not found")
path.write_text(text.replace(old, new))
PY"""

MCORE_BRIDGE_LONG_CONTEXT_MASK_VERIFY = r"""python - <<'PY'
from pathlib import Path

text = Path("/usr/local/lib/python3.11/site-packages/mcore_bridge/model/gpt_model.py").read_text()
assert "full_attention_mask.diagonal(dim1=2, dim2=3).squeeze(1)" in text
PY"""

MCORE_DSA_ZERO_LOSS_TOPK_PATCH = r"""python - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.11/site-packages/megatron/core/transformer/experimental_attention_variant/dsa.py")
text = path.read_text()
old = '''    # Compute attention scores: q @ k^T
    # [seqlen_q, batch, index_n_heads, index_head_dim] @ [seqlen_k, batch, index_head_dim]^T
    #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
    index_scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())

    # Apply ReLU activation.
    index_scores = torch.relu(index_scores)

    # Weight each head by attention weights.
    # [seqlen_q, batch, index_n_heads, seqlen_k] * [seqlen_q, batch, index_n_heads, 1]
    #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
    index_scores = index_scores * weights.unsqueeze(-1)

    # Sum across attention heads.
    # [seqlen_q, batch, index_n_heads, seqlen_k] -> [seqlen_q, batch, seqlen_k]
    index_scores = index_scores.sum(dim=2)

    # Transpose to [batch, seqlen_q, seqlen_k].
    index_scores = index_scores.transpose(0, 1)

    return index_scores
'''
new = '''    index_scores = None
    head_chunk = 2
    for start in range(0, q.size(2), head_chunk):
        end = min(start + head_chunk, q.size(2))
        partial = torch.einsum('sbhd,tbd->sbht', q[:, :, start:end].float(), k.float())
        partial = torch.relu(partial)
        partial = partial * weights[:, :, start:end].unsqueeze(-1)
        partial = partial.sum(dim=2)
        index_scores = partial if index_scores is None else index_scores + partial

    return index_scores.transpose(0, 1)
'''
if old not in text:
    raise RuntimeError("DeepSeek DSA index-score patch target not found")
text = text.replace(old, new)
needle = '    ' + (chr(34) * 3) + 'Naive implementation of forward pass for indexer loss.' + (chr(34) * 3) + '\n'
replacement = needle + '''    if loss_coeff == 0.0:
        with torch.no_grad():
            _, topk_indices = fused_qk_topk_naive(q, k, weights, topk, mask)
        return topk_indices, q.new_zeros(())

'''
if needle not in text:
    raise RuntimeError("DeepSeek DSA indexer-loss patch target not found")
path.write_text(text.replace(needle, replacement, 1))
PY"""

MCORE_DSA_ZERO_LOSS_TOPK_VERIFY = r"""python - <<'PY'
from pathlib import Path

text = Path("/usr/local/lib/python3.11/site-packages/megatron/core/transformer/experimental_attention_variant/dsa.py").read_text()
assert "head_chunk = 2" in text
assert "with torch.no_grad()" in text
PY"""

MCORE_CSA_CHUNKED_UNFUSED_PATCH = r"""python - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.11/site-packages/megatron/core/transformer/experimental_attention_variant/csa.py")
text = path.read_text()
old = '''    sq, b, np_, hn = query.size()

    # --- Gather KV at topk positions ---
    # kv_full: [n_kv, b, hn] -> [b, n_kv, hn]
    kv_t = kv_full.permute(1, 0, 2)

    safe_indices = topk_indices.clamp(min=0).long()  # [b, sq, topk]
    safe_indices_exp = safe_indices.unsqueeze(-1).expand(-1, -1, -1, hn)  # [b, sq, topk, hn]
    # [b, n_kv, hn] -> [b, 1, n_kv, hn] -> gather -> [b, sq, topk, hn]
    kv_gathered = torch.gather(
        kv_t.unsqueeze(1).expand(-1, sq, -1, -1), dim=2, index=safe_indices_exp
    )

    # --- Attention scores ---
    # query: [sq, b, np, hn] -> [b, np, sq, hn]
    q = query.permute(1, 2, 0, 3).float()
    kv_g = kv_gathered.float()  # [b, sq, topk, hn]

    # [b, np, sq, topk]
    scores = torch.einsum("bnsh,bskh->bnsk", q, kv_g) * softmax_scale

    # Mask invalid
    invalid_mask = (topk_indices < 0).unsqueeze(1)  # [b, 1, sq, topk]
    scores = scores.masked_fill(invalid_mask, float("-inf"))

    # --- Softmax with attention sink ---
    sink = attn_sink.view(1, np_, 1, 1).float()
    scores_max = scores.max(dim=-1, keepdim=True).values  # [b, np, sq, 1]
    scores_max = torch.max(scores_max, sink)

    exp_scores = torch.exp(scores - scores_max)  # [b, np, sq, topk]
    exp_sink = torch.exp(sink - scores_max)  # [1, np, 1, 1]

    sum_exp = exp_scores.sum(dim=-1, keepdim=True) + exp_sink
    attn_weights = exp_scores / sum_exp  # [b, np, sq, topk]

    # --- Weighted sum ---
    output = torch.einsum("bnsk,bskh->bnsh", attn_weights, kv_g)
    output = output.to(query.dtype)

    # [b, np, sq, hn] -> [sq, b, np, hn] -> [sq, b, np * hn]
    output = output.permute(2, 0, 1, 3).contiguous()
    output = output.reshape(sq, b, np_ * hn)
    return output
'''
new = '''    sq, b, np_, hn = query.size()

    kv_t = kv_full.detach().permute(1, 0, 2)
    sink = attn_sink.view(1, np_, 1, 1).float()
    output_full = query.detach()
    seq_chunk = 16

    for start in range(0, sq, seq_chunk):
        end = min(start + seq_chunk, sq)
        topk_chunk = topk_indices[:, start:end]
        safe_indices = topk_chunk.clamp(min=0).long()
        safe_indices_exp = safe_indices.unsqueeze(-1).expand(-1, -1, -1, hn)
        kv_gathered = torch.gather(
            kv_t.unsqueeze(1).expand(-1, end - start, -1, -1), dim=2, index=safe_indices_exp
        )

        q = query[start:end].detach().permute(1, 2, 0, 3)
        kv_g = kv_gathered
        scores = torch.einsum("bnsh,bskh->bnsk", q, kv_g).float() * softmax_scale
        invalid_mask = (topk_chunk < 0).unsqueeze(1)
        scores = scores.masked_fill(invalid_mask, float("-inf"))

        scores_max = scores.max(dim=-1, keepdim=True).values
        scores_max = torch.max(scores_max, sink)
        exp_scores = torch.exp(scores - scores_max)
        exp_sink = torch.exp(sink - scores_max)
        sum_exp = exp_scores.sum(dim=-1, keepdim=True) + exp_sink
        attn_weights = (exp_scores / sum_exp).to(query.dtype)
        del scores, scores_max, exp_scores, exp_sink, invalid_mask, sum_exp

        output = torch.einsum("bnsk,bskh->bnsh", attn_weights, kv_g)
        output = output.to(query.dtype)
        output_full[start:end] = output.permute(2, 0, 1, 3)
        del attn_weights, kv_gathered, kv_g, output, q, safe_indices, safe_indices_exp

    return output_full.reshape(sq, b, np_ * hn)
'''
if old not in text:
    raise RuntimeError("DeepSeek CSA unfused attention patch target not found")
path.write_text(text.replace(old, new))
PY"""

MCORE_CSA_CHUNKED_UNFUSED_VERIFY = r"""python - <<'PY'
from pathlib import Path

text = Path("/usr/local/lib/python3.11/site-packages/megatron/core/transformer/experimental_attention_variant/csa.py").read_text()
assert "seq_chunk = 16" in text
assert "output_full" in text
PY"""

# Shape fix only: CP padding + cos_/sin_ length reconciliation. These are
# gradient-safe (no detach) and required whenever CP>1. The rotary-cache
# repeat path corrupts positions if it ever cycles (repeats > 1), so it warns
# loudly when that happens — under CP padding the mismatch should be absorbed
# by a single tail slice, never a true cyclic repeat.
MCORE_DSV4_ROPE_SHAPE_FIX_PATCH = r"""python - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.11/site-packages/megatron/core/models/common/embeddings/rope_utils.py")
text = path.read_text()
if "padding = (-pos_emb.size(seq_dim)) % (2 * cp_size)" not in text:
    lines = text.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if "    pos_emb = pos_emb.view(" in line:
            lines[idx:idx] = [
                "    padding = (-pos_emb.size(seq_dim)) % (2 * cp_size)\n",
                "    if padding:\n",
                "        pad_shape = list(pos_emb.shape)\n",
                "        pad_shape[seq_dim] = padding\n",
                "        pos_emb = torch.cat(\n",
                "            [\n",
                "                pos_emb,\n",
                "                pos_emb.narrow(seq_dim, pos_emb.size(seq_dim) - 1, 1).expand(*pad_shape),\n",
                "            ],\n",
                "            dim=seq_dim,\n",
                "        )\n",
            ]
            break
    else:
        raise RuntimeError("Megatron RoPE CP view patch target not found")
    text = "".join(lines)
if "if cos_.size(0) != t.size(0):" not in text:
    lines = text.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if "t = (t * cos_)" in line:
            lines[idx:idx] = [
                "    if cos_.size(0) != t.size(0):\n",
                "        repeats = (t.size(0) + cos_.size(0) - 1) // cos_.size(0)\n",
                "        if repeats > 1:\n",
                "            import warnings\n",
                "            warnings.warn(\n",
                "                f'[DSV4] RoPE cos/sin cache ({cos_.size(0)}) shorter than input '\n",
                "                f'({t.size(0)}); cyclic repeat x{repeats} will wrap positions and '\n",
                "                'corrupt positional encodings. Expected only a single-tail CP-padding '\n",
                "                'mismatch (repeats == 1).',\n",
                "                stacklevel=2,\n",
                "            )\n",
                "        repeat_shape = [1] * cos_.dim()\n",
                "        repeat_shape[0] = repeats\n",
                "        cos_ = cos_.repeat(*repeat_shape)[: t.size(0)]\n",
                "        sin_ = sin_.repeat(*repeat_shape)[: t.size(0)]\n",
                "\n",
            ]
            break
    else:
        raise RuntimeError("Megatron RoPE CP patch target not found")
    text = "".join(lines)
path.write_text(text)
PY"""

MCORE_DSV4_ROPE_SHAPE_FIX_VERIFY = r"""python - <<'PY'
from pathlib import Path

text = Path("/usr/local/lib/python3.11/site-packages/megatron/core/models/common/embeddings/rope_utils.py").read_text()
assert "padding = (-pos_emb.size(seq_dim)) % (2 * cp_size)" in text
assert "if cos_.size(0) != t.size(0):" in text
assert "corrupt positional encodings" in text
PY"""

# Memory-only: chunked + detached RoPE application. This severs the autograd
# graph through the rotary multiply (detach), which is only safe when no LoRA
# adapter upstream of attention needs that gradient (i.e. target_modules is
# linear_proj-only). Disable with megatron_rope_seq_chunking to keep gradients.
MCORE_DSV4_ROPE_SEQ_CHUNKING_PATCH = r"""python - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.11/site-packages/megatron/core/models/common/embeddings/rope_utils.py")
text = path.read_text()
if "rope_chunk = 1024" not in text:
    lines = text.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if line == "    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)\n":
            lines[idx : idx + 1] = [
                "    if t.size(0) > 4096:\n",
                "        output = t.detach()\n",
                "        rope_chunk = 1024\n",
                "        for start in range(0, t.size(0), rope_chunk):\n",
                "            end = min(start + rope_chunk, t.size(0))\n",
                "            t_chunk = t[start:end].detach()\n",
                "            output[start:end] = (t_chunk * cos_[start:end]) + (\n",
                "                _rotate_half(t_chunk, rotary_interleaved) * sin_[start:end]\n",
                "            )\n",
                "        if t_pass.numel() == 0:\n",
                "            return output\n",
                "        return torch.cat((output, t_pass), dim=-1)\n",
                "\n",
                "    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)\n",
            ]
            break
    else:
        raise RuntimeError("Megatron RoPE chunk patch target not found")
    text = "".join(lines)
path.write_text(text)
PY"""

MCORE_DSV4_ROPE_SEQ_CHUNKING_VERIFY = r"""python - <<'PY'
from pathlib import Path

text = Path("/usr/local/lib/python3.11/site-packages/megatron/core/models/common/embeddings/rope_utils.py").read_text()
assert "rope_chunk = 1024" in text
assert "output = t.detach()" in text
PY"""

VLLM_SCALE_FMT_PATCH = r"""python - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py')
text = path.read_text()
old = '        self.scale_fmt = config.quantization_config["scale_fmt"]\n'
new = '        self.scale_fmt = getattr(config, "quantization_config", {"scale_fmt": "ue8m0"})["scale_fmt"]\n'
if old not in text:
    raise RuntimeError('DeepSeek V4 vLLM scale_fmt patch target not found')
path.write_text(text.replace(old, new))
PY"""

VLLM_BF16_SCALE_WEIGHTS_PATCH = r"""python - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py')
text = path.read_text()
old = '                param = params_dict[name]\n                weight_loader = param.weight_loader\n'
new = '                param = params_dict.get(name)\n                if param is None:\n                    if name.endswith("weight_scale_inv"):\n                        break\n                    raise KeyError(name)\n                weight_loader = param.weight_loader\n'
if old not in text:
    raise RuntimeError('DeepSeek V4 vLLM missing BF16 scale patch target not found')
text = text.replace(old, new, 1)
old = '                    param = params_dict[name]\n                    weight_loader = getattr(\n'
new = '                    param = params_dict.get(name)\n                    if param is None:\n                        if name.endswith("weight_scale_inv"):\n                            continue\n                        raise KeyError(name)\n                    weight_loader = getattr(\n'
if old not in text:
    raise RuntimeError('DeepSeek V4 vLLM default BF16 scale patch target not found')
text = text.replace(old, new, 1)
old = '                        param = params_dict[name_mapped]\n                        # We should ask the weight loader to return success or not\n'
new = '                        param = params_dict.get(name_mapped)\n                        if param is None:\n                            if "weight_scale" in name_mapped:\n                                continue\n                            raise KeyError(name_mapped)\n                        # We should ask the weight loader to return success or not\n'
if old not in text:
    raise RuntimeError('DeepSeek V4 vLLM expert BF16 scale patch target not found')
path.write_text(text.replace(old, new, 1))
PY"""

VLLM_BF16_ATTENTION_PATCH = r"""python - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/attention.py')
text = path.read_text()
old = '        if current_platform.is_rocm():\n'
new = '        if current_platform.is_rocm() or not hasattr(self.wo_a, "weight_scale_inv"):\n'
if old not in text:
    raise RuntimeError('DeepSeek V4 vLLM BF16 attention patch target not found')
path.write_text(text.replace(old, new, 1))
PY"""

VLLM_DISABLE_CUTEDSL_INDEXER_PATCH = r"""python - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/common/ops/fused_indexer_q.py')
text = path.read_text()
old = 'from vllm.utils.import_utils import has_cutedsl\n'
new = 'def has_cutedsl():\n    return False\n'
if old not in text:
    raise RuntimeError('DeepSeek V4 vLLM CUTLASS DSL fallback patch target not found')
path.write_text(text.replace(old, new, 1))
PY"""

VLLM_DISABLE_CUTEDSL_CACHE_PATCH = r"""python - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/common/ops/cache_utils.py')
text = path.read_text()
old = 'from vllm.utils.import_utils import has_cutedsl\n'
new = 'def has_cutedsl():\n    return False\n'
if old not in text:
    raise RuntimeError('DeepSeek V4 vLLM cache CUTLASS DSL fallback patch target not found')
path.write_text(text.replace(old, new, 1))
PY"""

VLLM_COMPRESSOR_FALLBACK_PATCH = r"""python - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/compressor.py')
text = path.read_text()
old = '            if self.head_dim == 512:\n'
new = '            if False and self.head_dim == 512:\n'
if old not in text:
    raise RuntimeError('DeepSeek V4 vLLM compressor Triton fallback patch target not found')
path.write_text(text.replace(old, new, 1))
PY"""

VLLM_UNINSTALL_CUTLASS_DSL = r"""python -m pip uninstall -y nvidia-cutlass-dsl"""

TRANSFORMERS_DSV4_PATCHES = (
    PatchSpec(
        name="transformers_deepseek_v4_config",
        command=DEEPSEEK_V4_CONFIG_PATCH,
        verify_command=DEEPSEEK_V4_CONFIG_VERIFY,
        why="Transformers 4.57.4 does not register deepseek_v4 in CONFIG_MAPPING/MODEL_FOR_CAUSAL_LM_MAPPING in this image, but ms-swift export and tokenizer/config loading use AutoConfig/AutoModel mappings.",
        required_for="model download smoke tests, ms-swift export, and vLLM-compatible HF views",
        ablation="cheap smoke ablation: disabling this prevents AutoConfig/MODEL_FOR_CAUSAL_LM mapping for deepseek_v4",
    ),
    PatchSpec(
        name="transformers_deepseek_v4_meta_model",
        command=DEEPSEEK_V4_MODELING_PATCH,
        verify_command=DEEPSEEK_V4_MODELING_VERIFY,
        why="ms-swift export needs a registered causal-LM class as a meta model even though inference uses vLLM, not this forward implementation.",
        required_for="Megatron-to-HF export",
        ablation="cheap smoke ablation: disabling this leaves CONFIG_MAPPING present but no MODEL_FOR_CAUSAL_LM_MAPPING entry",
    ),
)

MSSWIFT_MEGATRON_PATCHES = (
    PatchSpec(
        name="mcore_bridge_rope_config",
        command=MCORE_BRIDGE_ROPE_CONFIG_PATCH,
        verify_command=MCORE_BRIDGE_ROPE_CONFIG_VERIFY,
        why="mcore-bridge 1.4.2 does not propagate DSv4 YaRN rope_scaling fields (factor, beta_fast, beta_slow, mscale) from HF config. Without them, RoPE initialization fails.",
        required_for="any DSv4 Megatron SFT (all sequence lengths)",
        ablation="REQUIRED: disabling it crashes during model init with missing ModelConfig.beta_fast.",
    ),
    PatchSpec(
        name="mcore_bridge_attention_memory",
        command=MCORE_BRIDGE_MEMORY_PATCH,
        verify_command=MCORE_BRIDGE_MEMORY_VERIFY,
        why="torch.cat([q_no_pe, q_pos_emb]) and torch.cat([kv_no_pe, k_pos_emb]) each allocate a full-sequence temporary. At S=30k/rank (60k with CP=2), that is ~30k*B*512*2bytes ≈ 60 MiB per concat per layer. In-place write + del frees the source tensors immediately.",
        required_for="60k SFT (not needed at 4k)",
        ablation="REQUIRED at 60k: disabling it OOMs before step 1 in qkv_up_proj_and_rope_apply torch.cat (1.76 GiB allocation with ~1.2 GiB free).",
    ),
    PatchSpec(
        name="mcore_bridge_long_context_padding_mask",
        command=MCORE_BRIDGE_LONG_CONTEXT_MASK_PATCH,
        verify_command=MCORE_BRIDGE_LONG_CONTEXT_MASK_VERIFY,
        why="full_attention_mask is [B,1,S,S]. At S=30k (60k with CP=2), ~(~mask).sum() over 900M elements exceeds GPU memory. Extracting the diagonal gives the same per-token validity for the non-packed SFT case.",
        required_for="60k SFT (not needed at 4k where mask is only 16M elements)",
        ablation="REQUIRED at 60k: disabling it OOMs before step 1 in the transformer forward when the full [B,1,S,S] padding mask path stays enabled.",
    ),
    PatchSpec(
        name="megatron_dsa_chunked_indexer",
        command=MCORE_DSA_ZERO_LOSS_TOPK_PATCH,
        verify_command=MCORE_DSA_ZERO_LOSS_TOPK_VERIFY,
        why="DSA indexer computes [S_q, B, n_heads, S_k] scores. At S=30k, n_heads=4, FP32: 30k*1*4*30k*4 ≈ 43 GiB. Chunking by head pairs reduces peak to ~22 GiB. no_grad top-k when loss_coeff=0 avoids materializing the backward graph.",
        required_for="60k SFT (not needed at 4k where the full tensor is ~75 MiB)",
        ablation="REQUIRED at 60k: disabling it OOMs before step 1 in dsa.py _compute_index_scores, allocating ~50 GiB per rank.",
    ),
    PatchSpec(
        name="megatron_csa_chunked_unfused_attention",
        command=MCORE_CSA_CHUNKED_UNFUSED_PATCH,
        verify_command=MCORE_CSA_CHUNKED_UNFUSED_VERIFY,
        why="CSA gathers [B, S, topk, hn] then computes attention over the full sequence. At S=30k, topk=256, hn=512, BF16: 30k*256*512*2 ≈ 7.5 GiB per batch. Chunking to seq_chunk=16 reduces this to ~4 MiB per chunk.",
        required_for="60k SFT (not needed at 4k where the gather is ~1 GiB)",
        ablation="REQUIRED at 60k: disabling it OOMs before step 1 in csa.py when casting the gathered KV tensor (~19.5 GiB allocation).",
    ),
    PatchSpec(
        name="megatron_rope_cp_shape_fix",
        command=MCORE_DSV4_ROPE_SHAPE_FIX_PATCH,
        verify_command=MCORE_DSV4_ROPE_SHAPE_FIX_VERIFY,
        why="Two gradient-safe shape fixes: (1) CP padding — pos_emb length must be divisible by 2*CP for the context-parallel split/gather, required when CP>1. (2) cos_/sin_ length reconciliation — handles the shape mismatch when the rotary cache is shorter than the (CP-padded) input; warns loudly if it would ever cyclically repeat (repeats>1) and wrap positions. Neither edit detaches, so both are safe at any target_modules.",
        required_for="any CP>1 run (60k validation scales CP to 4-8).",
        ablation="REQUIRED for CP>1: disabling it crashes before step 1 with RoPE tensor length mismatches (e.g. 7216 vs 3608).",
    ),
    PatchSpec(
        name="megatron_rope_seq_chunking",
        command=MCORE_DSV4_ROPE_SEQ_CHUNKING_PATCH,
        verify_command=MCORE_DSV4_ROPE_SEQ_CHUNKING_VERIFY,
        why="RoPE chunking — t*cos_ + rotate_half(t)*sin_ allocates a full-sequence temporary; at S=30k, head_dim=512, BF16: ~30 MiB per head. Chunking to 1024 reduces peak, but the chunked path detach()es, severing autograd through the rotary multiply. Safe only when no adapter upstream of attention needs that gradient (linear_proj-only LoRA). Disable it to train qkv/all-linear adapters; raise CP instead to recover the memory.",
        required_for="60k SFT memory at low CP with linear_proj-only LoRA (not needed at 4k, and droppable at high CP).",
        ablation="Memory-only: gradient-unsafe at target_modules other than linear_proj; the 60k-without-memory-patches plan disables this and absorbs the memory via CP=4-8.",
    ),
    *TRANSFORMERS_DSV4_PATCHES,
)

VLLM_PATCHES = (
    PatchSpec(
        name="vllm_scale_fmt_default",
        command=VLLM_SCALE_FMT_PATCH,
        why="vLLM expects quantization_config.scale_fmt for quantized DeepSeek-V4 checkpoints; the exported BF16 merged checkpoint has no quantization_config, so default to ue8m0 only where vLLM reads the field.",
        required_for="vLLM startup for merged BF16 checkpoint",
        ablation="post-SFT eval required this for server startup",
    ),
    PatchSpec(
        name="vllm_ignore_missing_bf16_scale_weights",
        command=VLLM_BF16_SCALE_WEIGHTS_PATCH,
        why="The merged BF16 safetensors do not contain FP8/MXFP4 scale tensors such as weight_scale_inv; vLLM loader must skip only those absent scale tensors while still erroring on real missing weights.",
        required_for="vLLM weight loading for merged BF16 checkpoint",
        ablation="post-SFT eval required this for loading exported safetensors",
    ),
    PatchSpec(
        name="vllm_bf16_attention_scale_fallback",
        command=VLLM_BF16_ATTENTION_PATCH,
        why="The BF16 attention module does not expose wo_a.weight_scale_inv, so vLLM must take the non-quantized fallback path rather than the quantized scale path.",
        required_for="vLLM forward for merged BF16 checkpoint",
        ablation="post-SFT eval required this for inference",
    ),
    PatchSpec(
        name="vllm_disable_cutedsl_indexer",
        command=VLLM_DISABLE_CUTEDSL_INDEXER_PATCH,
        why="The CUTLASS DSL fused indexer path failed in this image; forcing the non-CUTEDSL path keeps inference on the working implementation.",
        required_for="vLLM DSv4 indexer inference",
        ablation="post-SFT eval required this for stable inference",
    ),
    PatchSpec(
        name="vllm_disable_cutedsl_cache",
        command=VLLM_DISABLE_CUTEDSL_CACHE_PATCH,
        why="The CUTLASS DSL cache utility path failed in this image; forcing the fallback matches the working indexer path.",
        required_for="vLLM DSv4 cache/indexer inference",
        ablation="post-SFT eval required this for stable inference",
    ),
    PatchSpec(
        name="vllm_compressor_triton_fallback",
        command=VLLM_COMPRESSOR_FALLBACK_PATCH,
        why="The specialized head_dim=512 compressor branch failed for this image/checkpoint combination; the generic fallback served 60k eval successfully.",
        required_for="vLLM 60k inference",
        ablation="post-SFT eval required this for stable inference",
    ),
    PatchSpec(
        name="vllm_remove_cutlass_dsl_package",
        command=VLLM_UNINSTALL_CUTLASS_DSL,
        why="Uninstalling nvidia-cutlass-dsl prevents import-time selection of broken CUTEDSL kernels after the source-level fallbacks are applied.",
        required_for="vLLM startup/inference stability",
        ablation="post-SFT eval kept this enabled",
    ),
)
