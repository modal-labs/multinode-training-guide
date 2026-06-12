# DeepSeek-V4-Flash SFT example — cleanup plan + testing plan

Handoff plan for finishing the cleanup of this example (PR #84 branch
`devin/1780686358-deepseek-v4-flash-sft`). Each task below is independent unless
noted; do them as separate commits in the order listed so failures are easy to
bisect. **Delete this file in the final commit once all tasks are done.**

Already done on this branch (do not redo): the 5 gradient-unsafe memory patches
were removed in favor of CP=4 (#87), `PR87_cleanup_summary.md` was deleted, and
the stale README/`inspect_lora_adapter` references to the removed patches were
fixed.

Current state: `deepseek_patches.py` carries **11 patches** — 2 transformers
registration, 2 Megatron correctness, 7 vLLM BF16 serving.

---

## Task 1 — Drop the two redundant CUTLASS-DSL source patches

In vLLM 0.22.1, `has_cutedsl()` is `_has_module("cutlass")`
(`vllm/utils/import_utils.py:475`). The image already runs
`vllm_remove_cutlass_dsl_package` (`pip uninstall -y nvidia-cutlass-dsl`), which
makes `has_cutedsl()` return `False` everywhere. The two source patches that
inline `def has_cutedsl(): return False` are therefore no-ops.

- Delete `vllm_disable_cutedsl_indexer` and `vllm_disable_cutedsl_cache`
  (`VLLM_DISABLE_CUTEDSL_INDEXER_PATCH`, `VLLM_DISABLE_CUTEDSL_CACHE_PATCH` and
  their `PatchSpec` entries in `VLLM_PATCHES`).
- Keep `vllm_remove_cutlass_dsl_package`. Update its `why` — the current text
  has the causality backwards (the uninstall is the primary mechanism, not a
  backstop for the source patches).
- Update the patch counts in the `deepseek_patches.py` module docstring
  (11 → 9; vLLM 7 → 5).

**Test:** Testing Plan step T3 (serve + eval). To be thorough, first run
`python -c "from vllm.utils.import_utils import has_cutedsl; assert not has_cutedsl()"`
inside the built vLLM image.

## Task 2 — Replace the transformers registration patches with an upgrade

`transformers_deepseek_v4_config` and `transformers_deepseek_v4_meta_model`
exist only because the images pin `transformers==4.57.4`. Native `deepseek_v4`
support landed in transformers **v5.8.0**. All constraints allow an upgrade:
ms-swift (pinned commit and latest) requires `>=4.33,<5.11.0`; mcore-bridge
1.4.2/1.4.3 requires `>=4.33,<5.11.0`; vLLM 0.22.1 requires `>=4.56.0`
(excludes only 5.0–5.5.0). The pinned ms-swift commit already registers
`deepseek-ai/DeepSeek-V4-Flash` natively.

- Bump `transformers==4.57.4` → `transformers==5.10.2` in both `download_image`
  and `msswift_image` in `modal_train.py`.
- Delete both registration patches and their verify commands from
  `deepseek_patches.py`; remove `TRANSFORMERS_DSV4_PATCHES` and unwrap
  `download_image` (it no longer needs `apply_image_patches` at all).
- Update the module docstring counts (9 → 7) and the
  `PLAN_60K_WITHOUT_MEMORY_PATCHES.md` kept-patch table if Task 5 hasn't
  removed that file yet.

**Risk:** transformers 5.x is a major version running against the torch-2.8
modelscope base image. If the smoke test (T1) fails on import/API changes,
fall back to keeping the 4.57.4 pin + patches and note the blocker in the PR.
Do not partially upgrade (one image on 5.x, the other on 4.x) — the exported
config must stay consistent between the export and serving paths.

**Test:** full Testing Plan (T1–T4). This task touches model load, export, and
serving, so nothing short of the full loop validates it.

## Task 3 — Deduplicate the vLLM serve/eval boilerplate

`deploy_and_eval_merged` and `eval_summarization` in `modal_train.py` duplicate
~120 lines: the `vllm.entrypoints.openai.api_server` Popen invocation, the
720-iteration health poll, the `_chat` HTTP helper, the log-tail-on-error
handler, and the terminate/kill teardown. Only real differences:
`--gpu-memory-utilization` (0.9 vs 0.92) and the `_chat` timeout (120s vs 300s).

- Extract a module-level context manager, e.g.
  `_vllm_server(model_dir, max_model_len, gpu_memory_utilization)` that yields a
  `chat(prompt, max_tokens, timeout)` callable and owns startup wait, error
  log dumping, and teardown.
- Both functions keep their current parameter values.

**Test:** `uv run ruff check` + `uv run pyright` locally, then T3 (both eval
entrypoints exercise the helper).

## Task 4 — Small config fixes

- Remove `VLLM_USE_V1` from `vllm_image` env: the variable no longer exists in
  vLLM 0.22.1 (`vllm/envs.py`); it's dead config.
- `download_image`: drop the `torch==2.9.1` pin if possible — the functions on
  that image (snapshot download, JSONL generation, tokenizer token counting)
  should not need torch. If `AutoTokenizer` for this model pulls in torch at
  import time, instead align the pin with the training image (torch 2.8.x) so
  the two images don't disagree.
- Strip the leftover ablation scaffolding from `deepseek_patches.py`: the
  `ablation` and `required_for` `PatchSpec` fields carry no actionable
  information anymore (`why` is enough), and the `DSV4_DISABLED_PATCHES` env
  knob + Skipping/Applying echo layers were the one-off ablation harness.
  Keep `name`, `command`, `why`, `verify_command`.

**Test:** T1 + T3. The torch change only needs T2's dataset-prep step plus
`download_model` to re-run cleanly.

## Task 5 — Fold the PLAN doc into the README and delete it

`PLAN_60K_WITHOUT_MEMORY_PATCHES.md` is a decision record about patches that no
longer exist; per `AGENTS.md`, one-off notes belong in the PR, not the repo.
The durable content to move into `README.md` (Scaling section):

- The CP-vs-memory table (CP 2/4/8 → per-rank sequence length and peak
  allocations) and the rationale that CP absorbs long-context memory.
- The "no `linear_qkv` module" warning: DSv4 uses MLA; attention-input
  projections are `linear_q_up_proj`/`linear_kv_proj`, output is `linear_proj`;
  peft silently matches 0 modules for nonexistent names.
- The known-limits note (unfused DSA/CSA reference kernels are slow; upstream
  Megatron-LM has no fused DSA / no CSA as of 2026-06).

Then delete the file. The validation history is preserved in the PR #84/#87
descriptions.

**Test:** docs-only; `uv run ruff check` for the repo lint CI.

## Task 6 — Update the PR #84 description

The PR body still says "15 total patches", shows an ablation table for the 5
since-deleted memory patches, and describes the removed gradient-safety guard.
Rewrite it to the post-cleanup state: patch count and categories matching
`deepseek_patches.py`, CP=4 as the long-context mechanism, no guard.

## Deferred (do not do now; leave a PR comment instead)

- **Next vLLM release:** vLLM `main` has reworked `vllm/models/deepseek_v4/`
  since 0.22.1 — `scale_fmt` is hardcoded to `"ue8m0"` (upstreams
  `vllm_scale_fmt_default`) and native BF16 KV-cache paths replace the branch
  `vllm_bf16_attention_scale_fallback` edits. When the first release after
  0.22.1 ships, bump the image tag and retest T3/T4 with those patches removed;
  `vllm_ignore_missing_bf16_scale_weights` and `vllm_compressor_triton_fallback`
  need an empirical retest (the loader still indexes `params_dict[name]` on
  main; the `head_dim == 512` compressor branch still exists).
- **Upstreaming:** the two irreducible patches should become upstream PRs —
  mcore-bridge (YaRN rope fields in `config/model_config.py`; still missing in
  1.4.3) and Megatron-LM (RoPE CP padding + cos/sin length reconciliation in
  `rope_utils.py`; still missing on main).

---

# Testing plan

All commands run from `deepseek-v4-flash-sft/` with Modal auth configured and
the `huggingface-secret` Modal secret present. Total GPU cost is dominated by
T3/T4; T1–T2 are cheap. Run lint/typecheck (`uv run ruff check`,
`uv run pyright`) before every push.

### T1 — Image build + smoke test (cheap, run after every task)

```bash
modal run modal_train.py::smoke_test
```

Validates: both patched images build (all `verify_command`s run at build time
and hard-fail on patch drift), transformers loads the DSv4 config/tokenizer
(`model_type=deepseek_v4`), chat template applies, `megatron sft --help` works.
This is the primary gate for Task 2 (transformers 5.x).

### T2 — Dataset prep (cheap)

```bash
modal run modal_train.py::prepare_dataset --hf-dataset openai/gsm8k \
  --data-folder gsm8k --split train --max-examples 256
modal run modal_train.py::prepare_summary_dataset --num-examples 8
```

Validates the download image (Task 4 torch change) end to end.

### T3 — Short 4k train → export → serve → eval (1×8 B200, ~hours)

```bash
modal run --detach modal_train.py::train_model \
  --data-folder gsm8k --train-iters 5 --save-interval 5 --run-id cleanup-smoke
modal run --detach modal_train.py::export_and_eval \
  --run-id cleanup-smoke --eval-limit 20
```

Pass criteria: 5/5 training steps with decreasing loss; export produces
`merged-hf` safetensors; the vLLM server starts (this is the gate for Tasks 1,
3, 4 — any removed-patch regression shows up as a startup crash or a
`KeyError` during weight loading); smoke inference returns coherent answers;
GSM8K eval completes (accuracy itself is not the gate at 5 steps).

### T4 — Gradient-correctness + long-context check (4×8 B200, expensive; only required for Task 2)

```bash
N_NODES=4 modal run --detach modal_train.py::long_context_loop \
  --run-id cleanup-60k \
  --target-modules linear_q_up_proj,linear_kv_proj,linear_proj
modal run modal_train.py::inspect_lora_adapter --run-id cleanup-60k
```

Pass criteria: training completes 5 steps at 61440 tokens / CP=4 without OOM
(reference peak: ~81 GiB/GPU); `inspect_lora_adapter` reports **nonzero
`lora_B` on all attention-input adapters** (the regression this example
specifically guards against is silently-zero gradients); post-training
summarization recall ≥ baseline (reference: 0% baseline → 100% post-SFT).

### Suggested task → test matrix

| Task | T1 | T2 | T3 | T4 |
|---|---|---|---|---|
| 1 cutedsl removal | x | | x | |
| 2 transformers 5.x | x | x | x | x |
| 3 dedupe serve code | x | | x | |
| 4 config fixes | x | x | x | |
| 5 docs fold | lint only | | | |

To economize, batch Tasks 1, 3, 4 into one T3 run, and run T4 once after
Task 2 lands.
