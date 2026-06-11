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
