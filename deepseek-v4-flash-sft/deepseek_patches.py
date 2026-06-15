# pyright: reportUnknownMemberType=false
"""Patch registry for the DeepSeek-V4-Flash SFT example.

Each patch is isolated here so the training file shows the recipe rather than a wall
of source edits.  The `why` field documents the failure that justified keeping the
patch.

## Patch categories

There are two independent categories — a total of 7 patches. `deepseek_v4` is
registered natively in transformers 5.8.0+ (the images pin 5.10.2), so no model
registration patches are needed; only the first category applies to training:

### 1. Megatron Bridge integration (2 patches) — gradient-safe, no memory hacks
These edit mcore-bridge source to make DSv4 init and THD CP work; they do not
detach, so they are safe at any `target_modules` (linear_proj, qkv, all-linear):
  - rope config (YaRN field propagation)            (required at ALL lengths)
  - DSv4 THD CP boundary tensor plumbing            (required at CP>1 packed)

This example does NOT carry the detach-based memory patches that an earlier
revision used to fit 60k on 2 nodes. Those patches severed autograd through
attention, so they were only correct with `linear_proj`-only LoRA. They were
removed in favor of raising context parallelism and model partitioning (the
long-context attempt is now CP=8 / EP=16 / 16 nodes), which targets the same
memory pressure while keeping gradients correct for any target module. See the
"Experimental Megatron/ms-swift 60k path" section of README.md.

### 2. vLLM BF16 serving (5 patches) — only needed for inference / eval
vLLM 0.22.1 expects FP8/MXFP4 scale tensors that are absent when serving the
merged BF16 checkpoint, and ships CUTLASS DSL kernels that fail in this image.
These patches add scale fallbacks and uninstall the CUTLASS DSL package so the
exported model can be served without quantization.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class PatchSpec:
    name: str
    command: str
    why: str
    verify_command: str | None = None


def apply_image_patches(image, patches: Iterable[PatchSpec]):
    for patch in patches:
        image = image.run_commands(patch.command)
        if patch.verify_command is not None:
            image = image.run_commands(patch.verify_command)
    return image


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

MCORE_BRIDGE_DSV4_THD_CP_BOUNDARY_PATCH = r"""python - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.11/site-packages/mcore_bridge/model/gpts/deepseek_v4.py")
text = path.read_text()
marker = "MODAL_DSV4_THD_CP_BRIDGE_PATCH"
if marker in text:
    raise RuntimeError("mcore_bridge DSv4 THD CP boundary patch already applied")

gpt_path = Path("/usr/local/lib/python3.11/site-packages/mcore_bridge/model/gpt_model.py")
gpt_text = gpt_path.read_text()
gpt_marker = "MODAL_DSV4_SKIP_PACKED_ROPE_PREINDEX_CP"
if gpt_marker in gpt_text:
    raise RuntimeError("mcore_bridge DSv4 packed CP RoPE preindex patch already applied")

gpt_needle = (
    "        if self.position_embedding_type == 'rope' and packed_seq and "
    "not self.config.apply_rope_fusion:\n"
)
gpt_replacement = (
    "        skip_dsv4_cp_rope_preindex = (\n"
    "            getattr(self.config, 'mcore_model_type', None) == 'deepseek_v4'\n"
    "            and self.config.context_parallel_size > 1\n"
    "        )\n"
    "        # MODAL_DSV4_SKIP_PACKED_ROPE_PREINDEX_CP\n"
    "        if (\n"
    "            self.position_embedding_type == 'rope'\n"
    "            and packed_seq\n"
    "            and not self.config.apply_rope_fusion\n"
    "            and not skip_dsv4_cp_rope_preindex\n"
    "        ):\n"
)
if gpt_needle not in gpt_text:
    raise RuntimeError("Could not find Bridge GPTModel packed RoPE preindex hook")
gpt_path.write_text(gpt_text.replace(gpt_needle, gpt_replacement, 1))

patch = r'''

# MODAL_DSV4_THD_CP_BRIDGE_PATCH
# Adapt mcore-bridge's DSv4 override to Megatron-LM#5087. The upstream PR added
# a THD context-parallel boundary exchange before KV projection; Bridge 1.5.2
# still overrides the upstream methods and otherwise drops those tensors.
from megatron.core.transformer.experimental_attention_variant import csa_cp_utils as _modal_cp_utils

try:
    from megatron.core.fusions.fused_mla_yarn_rope_apply import (
        fused_mla_rope_inplace as _modal_fused_mla_rope_inplace,
    )
except Exception:
    _modal_fused_mla_rope_inplace = None


def _modal_rope_tensor(rotary_pos_emb):
    return rotary_pos_emb[0] if isinstance(rotary_pos_emb, tuple) else rotary_pos_emb


def _modal_dsv4_get_query_key_value_tensors(
    self,
    hidden_states,
    key_value_states=None,
    position_ids=None,
    packed_seq_params=None,
    inference_context=None,
    rotary_pos_emb=None,
    *,
    inference_params=None,
    boundary_hidden=None,
):
    assert hidden_states.ndim == 3, (
        f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"
    )
    if packed_seq_params is not None:
        assert packed_seq_params.local_cp_size is None, (
            "dynamic_context_parallel is not supported with MLA yet and is planned for future. "
            "Please disable dynamic_context_parallel."
        )
    assert inference_context is None and inference_params is None, (
        "Inference is not supported for DSv4HybridSelfAttention."
    )

    packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
    if packed_seq:
        cu_seqlens_q = (
            packed_seq_params.cu_seqlens_q_padded
            if packed_seq_params.cu_seqlens_q_padded is not None
            else packed_seq_params.cu_seqlens_q
        )
        cu_seqlens_kv = (
            packed_seq_params.cu_seqlens_kv_padded
            if packed_seq_params.cu_seqlens_kv_padded is not None
            else packed_seq_params.cu_seqlens_kv
        )
        rope_max_seqlen_q = packed_seq_params.max_seqlen_q
        rope_max_seqlen_kv = packed_seq_params.max_seqlen_kv
    else:
        cu_seqlens_q = cu_seqlens_kv = None
        rope_max_seqlen_q = rope_max_seqlen_kv = None

    q_compressed, _ = self.linear_q_down_proj(hidden_states)
    kv_compressed = hidden_states
    boundary_kv_compressed = boundary_hidden

    if packed_seq_params is not None:
        q_compressed = q_compressed.squeeze(1)
        kv_compressed = kv_compressed.squeeze(1)
        if boundary_kv_compressed is not None:
            boundary_kv_compressed = boundary_kv_compressed.squeeze(1)

    if self.config.q_lora_rank is not None:
        q_compressed = apply_module(self.q_layernorm)(q_compressed)

    rotary_pos_emb = _modal_rope_tensor(rotary_pos_emb)
    mscale = 1.0

    def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, boundary_kv_compressed=None):
        q, _ = self.linear_q_up_proj(q_compressed)
        q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)
        q = _q_rms_norm(q, self.config.layernorm_epsilon)

        boundary_rows = 0
        if boundary_kv_compressed is not None:
            boundary_rows = boundary_kv_compressed.shape[0]
            kv_projection_input = torch.cat([boundary_kv_compressed, kv_compressed], dim=0)
        else:
            kv_projection_input = kv_compressed

        kv, _ = self.linear_kv_proj(kv_projection_input)
        kv = self.kv_layernorm(kv)
        boundary_kv = None

        cp_group = self.pg_collection.cp
        cp_size = cp_group.size() if cp_group is not None else 1
        pos_dim = self.config.qk_pos_emb_head_dim
        q_nope_dim = q.shape[-1] - pos_dim
        kv_nope_dim = kv.shape[-1] - pos_dim
        use_thd_cp = packed_seq and cp_size > 1

        if self.config.apply_rope_fusion:
            if _modal_fused_mla_rope_inplace is None:
                raise RuntimeError("Fused MLA RoPE apply is not imported successfully")
            rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                rope_max_seqlen_q if packed_seq else q.shape[0],
                dtype=hidden_states.dtype,
                packed_seq=packed_seq,
                mscale=mscale,
            )
            if use_thd_cp:
                global_start = cp_group.rank() * q.shape[0]
                query = _modal_cp_utils.apply_thd_cp_local_rope_fused(
                    q,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    q_nope_dim,
                    pos_dim,
                    cu_seqlens_q,
                    global_start,
                )
                kv = _modal_cp_utils.apply_thd_cp_local_rope_fused(
                    kv.unsqueeze(-2),
                    rotary_pos_cos,
                    rotary_pos_sin,
                    kv_nope_dim,
                    pos_dim,
                    cu_seqlens_kv,
                    global_start - boundary_rows,
                )
                if boundary_kv_compressed is not None:
                    boundary_kv = kv[:boundary_rows]
                    kv = kv[boundary_rows:]
                key = value = kv
            else:
                query = _modal_fused_mla_rope_inplace(
                    q,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    q_nope_dim,
                    pos_dim,
                    cu_seqlens_q,
                    cp_group.rank(),
                    cp_size,
                    remove_interleaving=True,
                )
                kv = _modal_fused_mla_rope_inplace(
                    kv.unsqueeze(-2),
                    rotary_pos_cos,
                    rotary_pos_sin,
                    kv_nope_dim,
                    pos_dim,
                    cu_seqlens_kv,
                    cp_group.rank(),
                    cp_size,
                    remove_interleaving=True,
                )
                key = value = kv
        elif use_thd_cp:
            global_start = cp_group.rank() * q.shape[0]
            query = _modal_cp_utils.apply_thd_cp_local_rope_unfused(
                q,
                rotary_pos_emb,
                q_nope_dim,
                pos_dim,
                cu_seqlens_q,
                global_start,
                self.config,
            )
            kv = _modal_cp_utils.apply_thd_cp_local_rope_unfused(
                kv.unsqueeze(-2),
                rotary_pos_emb,
                kv_nope_dim,
                pos_dim,
                cu_seqlens_kv,
                global_start - boundary_rows,
                self.config,
            )
            if boundary_kv_compressed is not None:
                boundary_kv = kv[:boundary_rows]
                kv = kv[boundary_rows:]
            key = value = kv
        else:
            rotary_local = rotary_pos_emb[: q.size(0)]
            q_no_pe, q_pos_emb = torch.split(q, [q.shape[-1] - pos_dim, pos_dim], dim=-1)
            q_pos_emb = apply_rotary_pos_emb(
                q_pos_emb,
                rotary_local,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
                mscale=mscale,
                cp_group=cp_group,
                mla_rotary_interleaved=True,
                mla_output_remove_interleaving=True,
                max_seqlen=rope_max_seqlen_q,
            )
            query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

            kv_no_pe, k_pos_emb = torch.split(kv, [kv.size(-1) - pos_dim, pos_dim], dim=-1)
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb,
                rotary_local,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
                mscale=mscale,
                cp_group=cp_group,
                mla_rotary_interleaved=True,
                mla_output_remove_interleaving=True,
                max_seqlen=rope_max_seqlen_kv,
            )
            kv = torch.cat([kv_no_pe, k_pos_emb], dim=-1).unsqueeze(-2)
            key = value = kv

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        if boundary_kv is not None:
            boundary_kv = boundary_kv.contiguous()
            return query, key, value, boundary_kv
        return query, key, value

    if self.recompute_up_proj:
        quantization = self.config.fp8 or self.config.fp4
        self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=quantization)
        if boundary_kv_compressed is None:
            query, key, value = self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply,
                q_compressed,
                kv_compressed,
            )
            boundary_kv = None
        else:
            query, key, value, boundary_kv = self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply,
                q_compressed,
                kv_compressed,
                boundary_kv_compressed,
            )
    elif boundary_kv_compressed is None:
        query, key, value = qkv_up_proj_and_rope_apply(q_compressed, kv_compressed)
        boundary_kv = None
    else:
        query, key, value, boundary_kv = qkv_up_proj_and_rope_apply(
            q_compressed,
            kv_compressed,
            boundary_kv_compressed,
        )

    result = (query, key, value, q_compressed, kv_compressed)
    if boundary_kv is not None:
        return result + (boundary_kv,)
    return result


def _modal_dsv4_forward(
    self,
    hidden_states,
    attention_mask,
    key_value_states=None,
    inference_context=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    rotary_pos_cos_sin=None,
    attention_bias=None,
    packed_seq_params=None,
    position_ids=None,
    sequence_len_offset=None,
    *,
    inference_params=None,
):
    rotary_pos_emb = rotary_pos_emb[self.rope_layer_type]
    assert attention_bias is None, "Attention bias should not be passed into DSv4HybridAttention."
    assert rotary_pos_cos is None and rotary_pos_sin is None, (
        "DSv4HybridAttention does not support Flash Decoding"
    )
    assert not rotary_pos_cos_sin, "Flash-infer rope has not been tested with DSv4HybridAttention."
    assert inference_context is None and inference_params is None, (
        "Inference is not supported for DSv4HybridAttention."
    )

    cp_group = self.pg_collection.cp
    cp_size = cp_group.size() if cp_group is not None else 1
    qkv_format = packed_seq_params.qkv_format if packed_seq_params is not None else None
    use_thd_cp = cp_size > 1 and qkv_format == "thd"

    boundary_hidden = None
    if use_thd_cp:
        boundary_hidden = _modal_cp_utils.exchange_cp_boundary_hidden(
            hidden_states,
            self._dsv4_compress_ratio,
            self.config.csa_window_size,
            cp_group,
        )

    qkv = self.get_query_key_value_tensors(
        hidden_states,
        key_value_states,
        position_ids,
        packed_seq_params,
        rotary_pos_emb=rotary_pos_emb,
        inference_context=inference_context,
        boundary_hidden=boundary_hidden,
    )
    if use_thd_cp:
        query, key, value, q_compressed, kv_compressed, boundary_kv = qkv
    else:
        query, key, value, q_compressed, kv_compressed = qkv
        boundary_kv = None

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    core_attn_manager = off_interface(self.offload_core_attention and self.training, query, "core_attn")
    with core_attn_manager as query:
        core_attn_out = self.core_attention(
            query,
            key,
            value,
            attention_mask,
            packed_seq_params=packed_seq_params,
            x=hidden_states,
            qr=q_compressed,
            boundary_hidden=boundary_hidden,
            boundary_kv=boundary_kv,
        )
    forced_released_tensors = [query, key, value]
    if boundary_kv is not None:
        forced_released_tensors.append(boundary_kv)
    core_attn_out = core_attn_manager.group_offload(
        core_attn_out,
        forced_released_tensors=forced_released_tensors,
    )

    if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
        core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

    if self.recompute_up_proj:
        assert self.qkv_up_checkpoint is not None
        self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
        self.qkv_up_checkpoint = None

    seq_len = core_attn_out.size(0)
    n_heads = self.num_attention_heads_per_partition
    pos_dim = self.config.qk_pos_emb_head_dim
    nope_dim = self.config.v_head_dim - pos_dim
    core_attn_out = core_attn_out.view(seq_len, core_attn_out.size(1), n_heads, -1)

    packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
    if packed_seq:
        cu_seqlens_kv = (
            packed_seq_params.cu_seqlens_kv_padded
            if packed_seq_params.cu_seqlens_kv_padded is not None
            else packed_seq_params.cu_seqlens_kv
        )
        rope_max_seqlen_kv = packed_seq_params.max_seqlen_kv
    else:
        cu_seqlens_kv = None
        rope_max_seqlen_kv = None

    rotary_pos_emb_tensor = _modal_rope_tensor(rotary_pos_emb)
    if use_thd_cp:
        global_start = cp_group.rank() * core_attn_out.shape[0]
        core_attn_out = _modal_cp_utils.apply_thd_cp_local_rope_unfused(
            core_attn_out,
            rotary_pos_emb_tensor,
            nope_dim,
            pos_dim,
            cu_seqlens_kv,
            global_start,
            self.config,
            inverse=True,
        )
    else:
        content_part, rot_part = torch.split(
            core_attn_out,
            [core_attn_out.size(-1) - pos_dim, pos_dim],
            dim=-1,
        )
        rot_part_in = rot_part.squeeze(1) if packed_seq else rot_part
        rot_part_out = apply_rotary_pos_emb(
            rot_part_in,
            rotary_pos_emb_tensor,
            self.config,
            cu_seqlens=cu_seqlens_kv,
            mscale=1.0,
            cp_group=cp_group,
            mla_rotary_interleaved=True,
            inverse=True,
            mla_output_remove_interleaving=True,
            max_seqlen=rope_max_seqlen_kv,
        )
        rot_part = rot_part_out.unsqueeze(1) if packed_seq else rot_part_out
        core_attn_out = torch.cat([content_part, rot_part], dim=-1)

    core_attn_out = core_attn_out.view(seq_len, core_attn_out.size(1), -1)

    if self._o_group_proj_is_grouped_linear:
        s, b = core_attn_out.size(0), core_attn_out.size(1)
        core_attn_out = core_attn_out.view(s, b, self.o_local_groups, -1)
        core_attn_out = core_attn_out.permute(2, 0, 1, 3).contiguous()
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.size(-1))
        m_splits = [s * b] * self.o_local_groups
        core_attn_out = self.linear_o_group_proj(core_attn_out, m_splits)
        core_attn_out = core_attn_out.view(self.o_local_groups, s, b, -1)
        core_attn_out = core_attn_out.permute(1, 2, 0, 3).contiguous()
        core_attn_out = core_attn_out.reshape(s, b, -1)
    else:
        core_attn_out = core_attn_out.view(
            core_attn_out.size(0),
            core_attn_out.size(1),
            self.o_local_groups,
            -1,
        )
        wo_a_weight = self.linear_o_group_proj.view(self.o_local_groups, self.config.o_lora_rank, -1)
        core_attn_out = torch.einsum("...gd,grd->...gr", core_attn_out, wo_a_weight)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)

    attn_proj_manager = off_interface(self.offload_attn_proj, core_attn_out, "attn_proj")
    with attn_proj_manager as core_attn_out:
        output, bias = self.linear_proj(core_attn_out)
    output = attn_proj_manager.group_offload(output, forced_released_tensors=[core_attn_out])

    return output, bias


DSv4HybridSelfAttention.get_query_key_value_tensors = _modal_dsv4_get_query_key_value_tensors
DSv4HybridSelfAttention.forward = _modal_dsv4_forward
'''

path.write_text(text + patch)
PY"""

MCORE_BRIDGE_DSV4_THD_CP_BOUNDARY_VERIFY = r"""python - <<'PY'
import inspect

from mcore_bridge.model.gpts.deepseek_v4 import DSv4HybridSelfAttention
from mcore_bridge.model.gpt_model import GPTModel

sig = inspect.signature(DSv4HybridSelfAttention.get_query_key_value_tensors)
assert "boundary_hidden" in sig.parameters
source = inspect.getsource(DSv4HybridSelfAttention.forward)
assert "boundary_kv=boundary_kv" in source
assert "exchange_cp_boundary_hidden" in source
gpt_source = inspect.getsource(GPTModel.forward)
assert "MODAL_DSV4_SKIP_PACKED_ROPE_PREINDEX_CP" in gpt_source
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

MSSWIFT_MEGATRON_PATCHES = (
    PatchSpec(
        name="mcore_bridge_rope_config",
        command=MCORE_BRIDGE_ROPE_CONFIG_PATCH,
        verify_command=MCORE_BRIDGE_ROPE_CONFIG_VERIFY,
        why="mcore-bridge does not propagate DSv4 YaRN rope_scaling fields (factor, beta_fast, beta_slow, mscale) from HF config. Without them, RoPE initialization fails.",
    ),
    PatchSpec(
        name="mcore_bridge_dsv4_thd_cp_boundary",
        command=MCORE_BRIDGE_DSV4_THD_CP_BOUNDARY_PATCH,
        verify_command=MCORE_BRIDGE_DSV4_THD_CP_BOUNDARY_VERIFY,
        why="mcore-bridge 1.5.2 overrides Megatron's DSv4 attention and drops the THD context-parallel boundary_hidden/boundary_kv tensors added by NVIDIA/Megatron-LM#5087.",
    ),
)

VLLM_PATCHES = (
    PatchSpec(
        name="vllm_scale_fmt_default",
        command=VLLM_SCALE_FMT_PATCH,
        why="vLLM expects quantization_config.scale_fmt for quantized DeepSeek-V4 checkpoints; the exported BF16 merged checkpoint has no quantization_config, so default to ue8m0 only where vLLM reads the field.",
    ),
    PatchSpec(
        name="vllm_ignore_missing_bf16_scale_weights",
        command=VLLM_BF16_SCALE_WEIGHTS_PATCH,
        why="The merged BF16 safetensors do not contain FP8/MXFP4 scale tensors such as weight_scale_inv; vLLM loader must skip only those absent scale tensors while still erroring on real missing weights.",
    ),
    PatchSpec(
        name="vllm_bf16_attention_scale_fallback",
        command=VLLM_BF16_ATTENTION_PATCH,
        why="The BF16 attention module does not expose wo_a.weight_scale_inv, so vLLM must take the non-quantized fallback path rather than the quantized scale path.",
    ),
    PatchSpec(
        name="vllm_compressor_triton_fallback",
        command=VLLM_COMPRESSOR_FALLBACK_PATCH,
        why="The specialized head_dim=512 compressor branch failed for this image/checkpoint combination; the generic fallback served 60k eval successfully.",
    ),
    PatchSpec(
        name="vllm_remove_cutlass_dsl_package",
        command=VLLM_UNINSTALL_CUTLASS_DSL,
        why="vLLM 0.22.1 gates the CUTLASS DSL indexer/cache kernels (which fail in this image) on has_cutedsl() == _has_module('cutlass'). Uninstalling nvidia-cutlass-dsl makes has_cutedsl() return False everywhere, which is the primary mechanism that forces the working non-CUTEDSL fallback paths.",
    ),
)
