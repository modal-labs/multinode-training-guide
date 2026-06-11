# PR #87 — DSv4-Flash 60k: drop the memory patches, commit to CP=4

PR: https://github.com/modal-labs/multinode-training-guide/pull/87
Branch: `devin/1781209254-dsv4-rope-patch-split` (stacked on #84)

## Goal

Make 60k-context LoRA SFT of DeepSeek-V4-Flash work **without the
gradient-unsafe memory patches**, then — per your call that "4 nodes / CP=4 is
totally fine" — **clean the PR up to that committed path**: delete every patch
that isn't necessary and make CP=4 the long-context default.

## The core finding

At 60k context, **CP=4 (4 nodes, 32×B200) peaks at ~81 GiB/GPU with no memory
patches at all** — less than half of a B200 (180 GB). The patches that rewrote
attention with `detach()` to save memory were therefore unnecessary, and they
were actively harmful: their `detach()`/`no_grad` paths silently sever autograd
through attention, so LoRA adapters on any module upstream of the output
projection (`linear_q_up_proj`, `linear_kv_proj`, `all-linear`, …) received
**exactly zero gradient** with no error raised. Removing them makes gradients
correct for any `target_modules`.

## What changed in the code

### `deepseek_patches.py` — removed 5 training-side memory patches
| Removed patch | Type | Why safe to drop at CP=4 |
|---|---|---|
| `mcore_bridge_attention_memory` | gradient-unsafe | concat temporaries absorbed by 15k/rank seq |
| `megatron_dsa_chunked_indexer` | gradient-unsafe | `no_grad` top-k severed qkv grads; DSA scores fit at 15k/rank |
| `megatron_csa_chunked_unfused_attention` | gradient-unsafe | CSA gathered-KV fits at 15k/rank |
| `megatron_rope_seq_chunking` | gradient-unsafe | `detach()` rotary chunking; the split-out memory half of the RoPE patch |
| `mcore_bridge_long_context_padding_mask` | memory-only | pure O(S²) mask reduction; not needed with ~99 GiB free |

Also deleted the `GRADIENT_UNSAFE_TRAINING_PATCHES` frozenset and the now-dead
patch/verify string constants, and updated the module docstring. **11 patches
remain** (2 transformers registration + 2 gradient-safe Megatron correctness +
7 vLLM inference).

**Kept** (gradient-safe / required): `transformers_deepseek_v4_config`,
`transformers_deepseek_v4_meta_model`, `mcore_bridge_rope_config`,
`megatron_rope_cp_shape_fix` (needed for any `CP>1`), and the 7 vLLM patches.

### `modal_train.py`
- `CP_SIZE` stays `1` for the standard 4k single-node path
  (`train_model`/`export`/`smoke`) so `TP*EP*PP*CP = 8` divides one 8-GPU node.
- New `LONG_CONTEXT_CP_SIZE = 4` is the default only for the `long_context_loop`
  60k recipe (launch with `N_NODES=4`).
- Removed the `target_modules`-vs-patches guard (`_validate_...`) and its helper
  — with the unsafe patches gone there is nothing to guard against.
- (Earlier in the PR) fixed `--target_modules` being passed as a single argv
  token; ms-swift's flag is `nargs='+'`, so multi-module values are now split
  into separate tokens.

### `PLAN_60K_WITHOUT_MEMORY_PATCHES.md`
Rewritten as a concise, generic decision record: the committed CP=4 outcome,
why the patches were droppable, how to run 60k, and a short validation summary.
Removed the machine-specific workspace name and moved the verbose per-run
numbers into the PR description (per AGENTS.md "keep repo docs generic").

## Validation (all on Modal, B200)

| Step | Config | Result | Peak/GPU |
|---|---|---|---|
| 1 — smoke | 4k, 1×8, patches off, `linear_proj` | PASS: 5/5, loss 2.32→2.18 | 85.8 GiB |
| 2 canary | 4k, 1×8, **off**, broad MLA | PASS: `lora_B` nonzero **86/86** q_up+kv, grad_norm 1.1–1.2 | 86.5 GiB |
| 2 control | 4k, 1×8, **on**, broad MLA | PASS (proves trap): **0/86** q_up+kv, grad_norm 0.12–0.17 | 81.7 GiB |
| 3 — 60k | 61440, **CP=4, 4×8 (32)**, 4 unsafe off, broad | PASS: 5/5, grad_norm 0.5–0.77, **86/86** q_up+kv | **81.0 GiB** |
| 4 — e2e | export → vLLM serve @60k → eval | PASS: **100% action-item recall** | — |
| 5 — strip recheck | 61440, **CP=4, 4×8**, **all 5 removed** incl. padding-mask, broad | PASS: 5/5, grad_norm 0.53–0.78; **86/86** q_up+kv, 43/43 proj | **80.98 GiB** |

Gradient flow is proven by `inspect_lora_adapter`, which reads the saved DCP
checkpoint and reports per-module `lora_B` norms (`lora_B` is zero-initialized,
so nonzero ⇒ that adapter received gradient).

## Review + CI

- Devin Review's 🔴 finding was a **real regression I introduced**: a global
  `CP_SIZE=4` broke the default 4k entrypoint (`8 % 32 != 0`). Fixed by scoping
  CP=4 to `long_context_loop` only — committed in `716fa59`.
- Replied to all review comments (workspace name, one-off data, CP regression,
  target_modules split).
- `ruff check` + `pyright` clean; Devin Review CI passing.

## Module-name note

DeepSeek-V4 uses MLA, not fused QKV — there is **no `linear_qkv` module**. The
attention-input projections are `linear_q_up_proj` and `linear_kv_proj`; the
output projection is `linear_proj`. Targeting a nonexistent module silently
matches 0 modules (peft raises nothing), which is its own trap.
