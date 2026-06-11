# Plan: 60k-context LoRA SFT without the gradient-unsafe memory patches

## Background

`deepseek_patches.py` contains 15 patches. Four of the six training-side patches
rewrite attention internals with a `detach()` + in-place-write pattern to reduce
peak memory at 60k tokens:

- `mcore_bridge_attention_memory`
- `mcore_bridge_long_context_padding_mask` (semantics change, lower risk)
- `megatron_dsa_chunked_indexer` (mathematically equivalent chunking; the
  `no_grad` top-k is only valid when the indexer loss coefficient is 0)
- `megatron_csa_chunked_unfused_attention`
- the RoPE-chunking portion of `megatron_rope_cp_and_chunking`

The `detach()`-based rewrites sever the autograd graph through attention. This
is survivable **only** because the current recipe targets LoRA at
`linear_proj` (the attention output projection), whose adapter gradients do not
require backprop through the attention math. Targeting any module upstream of
attention (`linear_qkv`, embeddings, `all-linear`, MLP adapters feeding into
attention of later layers via residuals is fine, but qkv adapters are not)
yields **silently zero or biased gradients** — no error is raised.

Goal of this plan: validate 60k-context training with those patches **disabled**,
keeping only the support-gap patches that are safe at any `target_modules`:

| Keep (safe, required) | Why |
|---|---|
| `transformers_deepseek_v4_config` | `deepseek_v4` not registered in transformers 4.57 |
| `transformers_deepseek_v4_meta_model` | needed for Megatron→HF export |
| `mcore_bridge_rope_config` | YaRN rope fields missing from mcore-bridge 1.4.2; model init fails without it |
| `megatron_rope_cp_and_chunking` — **CP padding + cos/sin length sub-patches only** | CP>1 is structurally broken without the shape fix (see step 0) |
| All 7 vLLM patches | inference-only; no training gradients involved |

| Drop (gradient-unsafe at other target modules) | Memory it was saving |
|---|---|
| `mcore_bridge_attention_memory` | ~60 MiB/layer concat temporaries |
| `mcore_bridge_long_context_padding_mask` | O(S²) mask reduction (900M elements at 30k/rank) |
| `megatron_dsa_chunked_indexer` | DSA indexer scores: ~43 GiB/rank at 30k/rank |
| `megatron_csa_chunked_unfused_attention` | CSA gathered-KV: ~19.5 GiB/rank at 30k/rank |
| RoPE-chunking sub-patch of `megatron_rope_cp_and_chunking` | ~30 MiB/head temporaries |

## Strategy: replace code patches with context parallelism

All of the dropped patches address allocations that scale with per-rank sequence
length S_rank — the dominant ones quadratically (DSA indexer is
`S_rank × heads × S_rank` FP32). Instead of patching the kernels, shrink S_rank
by raising CP:

| CP | S_rank @60k | DSA indexer peak (≈43 GiB @30k, ∝ S_rank²) | CSA gather peak (≈19.5 GiB @30k, ∝ S_rank) |
|---:|---:|---:|---:|
| 2 (current) | 30k | ~43 GiB | ~19.5 GiB |
| 4 | 15k | ~11 GiB | ~10 GiB |
| 8 | 7.5k | ~2.7 GiB | ~5 GiB |

On B200 (180 GB), CP=8 should comfortably absorb every allocation the dropped
patches were avoiding. CP=4 is worth attempting first as it is half the cost.

## Step 0 — split `megatron_rope_cp_and_chunking` (required prerequisite)

The patch currently bundles three sub-patches; we need two of them (CP padding,
cos/sin length fix) and must drop the third (detach-based RoPE chunking).

1. In `deepseek_patches.py`, split `MCORE_DSV4_CP_ROPE_PATCH` into:
   - `megatron_rope_cp_shape_fix` — the `padding = (-pos_emb.size(seq_dim)) % (2 * cp_size)`
     block and the `if cos_.size(0) != t.size(0)` block.
   - `megatron_rope_seq_chunking` — the `rope_chunk = 1024` block (detach-based).
2. Caveat to verify while testing: the cos/sin length fix **repeats the rotary
   cache cyclically** when it is shorter than the input. If it ever triggers
   with a repeat factor > 1, positions wrap (`pos % cache_len`) and the model
   silently sees wrong positional encodings. Add an assert that
   `repeats == 1`-equivalent padding only (i.e. the mismatch is from CP padding,
   not a short cache), or log loudly when it fires.

## Step 1 — smoke test at 4k, patches disabled (cheap, 1×8 B200)

Confirm the baseline still works with everything risky off:

```bash
DSV4_DISABLED_PATCHES=mcore_bridge_attention_memory,mcore_bridge_long_context_padding_mask,megatron_dsa_chunked_indexer,megatron_csa_chunked_unfused_attention,megatron_rope_seq_chunking \
modal run -d modal_train.py::train -- --max-length 4096 --train-iters 5
```

Expected: trains 5 steps (none of the dropped patches are needed at 4k per the
ablation table in the PR description).

## Step 2 — gradient-correctness canary (the actual point)

Before any 60k run, prove gradients flow where the old patches broke them:

1. Run a short job (4k, 5 steps) with
   `--target-modules linear_qkv,linear_proj` (or ms-swift's `all-linear`)
   and patches disabled as in Step 1.
2. Assert every LoRA adapter receives a nonzero gradient on step 1. Simplest
   check: after step 1, log
   `[(n, p.grad.abs().sum().item()) for n, p in model.named_parameters() if "lora" in n]`
   inside the trainer (a temporary hook or ms-swift callback is fine), or
   compare adapter weights before/after 5 steps — all must change.
3. As a negative control, run the same job with the old patches **enabled**
   and confirm the qkv adapters do NOT change (validates the check itself).

## Step 3 — 60k with CP scaled up, patches disabled

Same disabled set as Step 1. Try in order:

1. **CP=4, 4 nodes** (32×B200): `--cp-size 4 --max-length 61440`
   plus matching `n_nodes=4` cluster size. EP/TP per current defaults
   (`EP=8, TP=1, PP=1`); total GPUs must be divisible by TP×EP×PP×CP.
2. If OOM in the DSA indexer or CSA gather → **CP=8, 8 nodes** (64×B200).
3. Watch for the specific allocation sites the ablation table identified:
   `dsa.py _compute_index_scores`, `csa.py` gathered-KV cast,
   `qkv_up_proj_and_rope_apply` concat, and the `[B,1,S,S]` padding-mask
   reduction in `gpt_model.py`. Any OOM should name one of these.
4. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is already set in the
   image; keep it.

Success criteria: 5 SFT steps at 60k complete; loss decreases; no NaN.

## Step 4 — end-to-end validation

Re-run the existing pipeline on the CP-scaled config:

1. Export Megatron LoRA → merged HF safetensors (unchanged path).
2. Serve with the vLLM image (all 7 vLLM patches stay enabled).
3. Run the 60k summarization eval; expect parity with the patched run
   (100% action-item recall in the original validation).
4. Repeat Step 3+4 once with `--target-modules linear_qkv,linear_proj` to
   demonstrate the configuration the old patches could not support.

## Step 5 — cleanup decisions

- If CP=4 works: update the example defaults to CP=4 and delete the four
  dropped patches (and the RoPE-chunking sub-patch) from `deepseek_patches.py`.
- If only CP=8 works: keep the patches available behind
  `DSV4_DISABLED_PATCHES` inverted logic (opt-in rather than default), and keep
  the current `target_modules == "linear_proj"` guard whenever they are enabled.
- Either way, document the cost tradeoff: 2×8 B200 with unsafe patches
  (linear_proj-only LoRA) vs 4–8×8 B200 with correct gradients on any module.

## Known limits / alternatives

- This does not fix throughput: the unfused DSA/CSA reference kernels are slow
  regardless of memory. For production-grade long-context DSv4-Flash training,
  the better long-term path is Miles' fused TileLang kernels
  (`radixark/miles`, `miles_plugins/models/deepseek_v4/ops/kernel/`), which have
  real backward passes — but DSv4 attention LoRA is not wired up there today
  (custom module names `wq_a/wq_b/wkv/wo_a/wo_b` are not in its LoRA target
  mapping, and `wo_a` is consumed via direct `.weight` access).
- Upstream NVIDIA Megatron-LM has no fused DSA kernels and no CSA
  implementation at all as of 2026-06; the pinned commit comes from a fork.

## Validation results (executed 2026-06-11, Modal `peyton-agents`)

All runs use the disabled set **`mcore_bridge_attention_memory,
megatron_dsa_chunked_indexer, megatron_csa_chunked_unfused_attention,
megatron_rope_seq_chunking`** (the 4 gradient-unsafe patches). The
`mcore_bridge_long_context_padding_mask` patch was reclassified as gradient-safe
on the base branch (commit `fbaae62`) and is kept enabled.

**Module-name correction:** the plan's `linear_qkv` does not exist on
DeepSeek-V4 (it uses MLA, not fused QKV). The real attention-input projections
are `linear_q_up_proj` (RoPE query up-proj) and `linear_kv_proj` (KV latent).
All "broad target" runs below use
`--target-modules linear_q_up_proj,linear_kv_proj,linear_proj`. Targeting the
nonexistent `linear_qkv` silently trains nothing there (peft matches 0 modules),
which is itself a trap worth noting.

A prerequisite bug was also fixed: `modal_train.py` passed `--target_modules` as
a single argv token, so any multi-module value was never split and ms-swift
looked for a module literally named `"a,b"`. Fixed by splitting on comma into
separate tokens.

Gradient flow is verified with the new `inspect_lora_adapter` Modal function,
which reads the saved DCP checkpoint and reports per-module `lora_B` norms
(`lora_B` is zero-initialized, so nonzero ⇒ that adapter received gradient).

| Step | Config | Result | Peak/GPU |
|---|---|---|---|
| 1 — smoke | 4k, 1×8 B200, patches off, `linear_proj` | PASS: 5/5, loss 2.32→2.18, grad_norm 0.3–0.4 | 85.8 GiB |
| 2 canary | 4k, 1×8 B200, patches **off**, broad MLA targets | PASS: 5/5, grad_norm 1.1–1.2; `lora_B` nonzero **86/86** q_up+kv, 43/43 proj | 86.5 GiB |
| 2 control | 4k, 1×8 B200, patches **on**, broad MLA targets | PASS (proves the trap): grad_norm 0.12–0.17; `lora_B` nonzero **0/86** q_up+kv, 43/43 proj | 81.7 GiB |
| 3 — 60k | **61440**, **CP=4, 4×8 B200 (32)**, patches off, broad MLA targets | PASS: 5/5, grad_norm 0.5–0.77, no NaN; `lora_B` nonzero **86/86** q_up+kv, 43/43 proj | **81.0 GiB** |
| 4 — e2e | export CP=4 ckpt → merged HF → vLLM serve @ 60k → summarization eval | PASS: export OK, server ready ~520s, **100% action-item recall** (5/5 both examples) | — |

### Key findings

1. **The gradient bug is real and silent.** Step 2's A/B is decisive: with the 4
   patches enabled, the attention-input adapters' `lora_B` stays *exactly* 0
   (0/86) — they receive no gradient — while `linear_proj` trains normally. The
   total grad_norm collapses from ~1.1 (correct) to ~0.15 (proj-only). No error
   is raised. Disabling the patches restores 86/86 nonzero adapters.

2. **CP=4 is more than enough; CP=8 is unnecessary.** At 60k with the patches
   off, CP=4 peaks at **81 GiB/GPU** — less than half of B200's 180 GB, and
   essentially the same peak as the 4k runs. The freed allocations the patches
   used to avoid (DSA indexer, CSA gather, attention concat) are fully absorbed
   by the per-rank sequence shrinking to 15k. There was never an OOM at any of
   the named allocation sites.

3. **End-to-end parity holds.** A checkpoint trained under the new regime
   (patches off, CP=4, broad MLA LoRA) exports and serves cleanly through the
   unchanged vLLM path and scores 100% action-item recall at 60k — matching the
   original patched-run validation.

### Recommendation for Step 5

CP=4 works, so per the plan this enables deleting the four gradient-unsafe
patches and the RoPE-chunking sub-patch, and flipping the example default to
CP=4. The real tradeoff to weigh before doing so:

- **Keep patches (current default), `linear_proj`-only LoRA:** 60k fits on
  **2×8 B200 (CP=2)**, cheaper, but gradients are silently severed for any
  target module other than `linear_proj`.
- **Drop patches, CP=4:** needs **4×8 B200**, but gradients are correct for any
  target module (`linear_q_up_proj`, `linear_kv_proj`, `all-linear`, …).

Because dropping the patches removes the cheap 2-node path some users may rely
on, this PR keeps both options available via the existing opt-in
`DSV4_DISABLED_PATCHES` mechanism + the `target_modules`-vs-patches guard, rather
than hard-deleting them. The validation above is the evidence for whichever
default the maintainers choose.
