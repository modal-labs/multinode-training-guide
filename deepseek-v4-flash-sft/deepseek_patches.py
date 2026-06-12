# pyright: reportUnknownMemberType=false
"""Patch registry for the DeepSeek-V4-Flash SFT example.

Each patch is isolated here so the training file shows the recipe rather than a wall
of source edits.  The `why` and `ablation` fields document the failure that justified
keeping the patch.  Disable one or more patches for experiments with:

    DSV4_DISABLED_PATCHES=patch_name[,patch_name...] modal run ...

## Patch categories

There are three independent categories — a total of 11 patches, but only the first
two categories apply to training:

### 1. Model registration (2 patches) — required at ALL sequence lengths
Transformers 4.57 does not register `deepseek_v4` as a known model type, so
AutoConfig / MODEL_FOR_CAUSAL_LM_MAPPING lookups fail.  These two patches append
the registration to the installed transformers package.

### 2. Megatron correctness (2 patches) — gradient-safe, no memory hacks
Both edit mcore-bridge / Megatron source to make DSv4 init and CP work; neither
detaches, so both are safe at any `target_modules` (linear_proj, qkv, all-linear):
  - rope config (YaRN field propagation)            (required at ALL lengths)
  - RoPE CP shape fix (megatron_rope_cp_shape_fix)  (required for CP>1)

This example does NOT carry the detach-based memory patches that an earlier
revision used to fit 60k on 2 nodes. Those patches severed autograd through
attention, so they were only correct with `linear_proj`-only LoRA. They were
removed in favor of raising context parallelism (the default is now CP=4 / 4
nodes), which absorbs the same memory while keeping gradients correct for any
target module. See the README's scaling section for the long-context recipe.

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
        name="megatron_rope_cp_shape_fix",
        command=MCORE_DSV4_ROPE_SHAPE_FIX_PATCH,
        verify_command=MCORE_DSV4_ROPE_SHAPE_FIX_VERIFY,
        why="Two gradient-safe shape fixes: (1) CP padding — pos_emb length must be divisible by 2*CP for the context-parallel split/gather, required when CP>1. (2) cos_/sin_ length reconciliation — handles the shape mismatch when the rotary cache is shorter than the (CP-padded) input; warns loudly if it would ever cyclically repeat (repeats>1) and wrap positions. Neither edit detaches, so both are safe at any target_modules.",
        required_for="any CP>1 run (60k validation scales CP to 4-8).",
        ablation="REQUIRED for CP>1: disabling it crashes before step 1 with RoPE tensor length mismatches (e.g. 7216 vs 3608).",
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
