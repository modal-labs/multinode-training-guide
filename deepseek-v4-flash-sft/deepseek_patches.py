# pyright: reportUnknownMemberType=false
"""Patch registry for the DeepSeek-V4-Flash SFT example.

Each patch is isolated here so the training file shows the recipe rather than a wall
of source edits.  The `why` field documents the failure that justified keeping the
patch.

## Patch categories

There are two independent categories — a total of 7 patches. `deepseek_v4` is
registered natively in transformers 5.8.0+ (the images pin 5.10.2), so no model
registration patches are needed; only the first category applies to training:

### 1. Megatron correctness (2 patches) — gradient-safe, no memory hacks
Both edit mcore-bridge / Megatron source to make DSv4 init and CP work; neither
detaches, so both are safe at any `target_modules` (linear_proj, qkv, all-linear):
  - rope config (YaRN field propagation)            (required at ALL lengths)
  - RoPE CP shape fix (megatron_rope_cp_shape_fix)  (required for CP>1)

This example does NOT carry the detach-based memory patches that an earlier
revision used to fit 60k on 2 nodes. Those patches severed autograd through
attention, so they were only correct with `linear_proj`-only LoRA. They were
removed in favor of raising context parallelism (the default is now CP=4 / 4
nodes), which absorbs the same memory while keeping gradients correct for any
target module. See the "Long-context (60k) SFT" section of README.md.

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
        why="mcore-bridge 1.4.2 does not propagate DSv4 YaRN rope_scaling fields (factor, beta_fast, beta_slow, mscale) from HF config. Without them, RoPE initialization fails.",
    ),
    PatchSpec(
        name="megatron_rope_cp_shape_fix",
        command=MCORE_DSV4_ROPE_SHAPE_FIX_PATCH,
        verify_command=MCORE_DSV4_ROPE_SHAPE_FIX_VERIFY,
        why="Two gradient-safe shape fixes: (1) CP padding — pos_emb length must be divisible by 2*CP for the context-parallel split/gather, required when CP>1. (2) cos_/sin_ length reconciliation — handles the shape mismatch when the rotary cache is shorter than the (CP-padded) input; warns loudly if it would ever cyclically repeat (repeats>1) and wrap positions. Neither edit detaches, so both are safe at any target_modules.",
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
