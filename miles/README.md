# Miles example

Multi-node Miles RL training on Modal using the same Ray bootstrap pattern as
[`ray/modal_train.py`](../ray/modal_train.py), but with Miles recipes stored as
native CLI flag files instead of Python config classes.

## Prerequisites

- A Modal account with multi-node access.
- A `huggingface-secret` Modal secret containing `HF_TOKEN` for gated model
  downloads.
- Optionally export `WANDB_API_KEY` in your local shell before `modal run` to
  auto-enable Weights & Biases logging for training runs.

## Recipes

List the built-in recipes:

```bash
modal run miles/modal_train.py --list-recipes
```

Current recipes:

- `qwen25-0p5b-lora`: single-node smoke test adapted from the upstream Miles
  LoRA example.
- `qwen3-30b-a3b-lora`: first-pass Qwen3-30B-A3B bridge-mode LoRA validation
  recipe, restricted to attention targets (`linear_qkv`, `linear_proj`).
- `qwen3-30b-a3b-lora-fewstep`: trimmed attention-only recipe that is intended
  to prove a few full RL updates on Modal.
- `qwen3-30b-a3b-experts-lora`: second-pass Qwen3-30B-A3B recipe widened to
  expert `linear_fc1` and `linear_fc2` targets after the baseline path works.
- `qwen3-30b-a3b-experts-fewstep`: trimmed expert-target recipe built from the
  working few-step shape.

Testing/debug recipe files live under [`recipes/tests/`](./recipes/tests).

## Prepare assets

Prepare a small GSM8K dataset in the shared volume:

```bash
modal run miles/modal_train.py::prepare_dataset
```

Download a recipe's base model into the shared Hugging Face cache:

```bash
modal run miles/modal_train.py::download_model --recipe qwen3-30b-a3b-lora
```

## Train

The cluster size is chosen at import time by `MILES_N_NODES`, so set it in the
same shell invocation as `modal run`.

Single-node smoke test:

```bash
MILES_N_NODES=1 modal run miles/modal_train.py --recipe qwen25-0p5b-lora
```

Qwen3-30B-A3B baseline LoRA validation:

```bash
MILES_N_NODES=1 modal run miles/modal_train.py --recipe qwen3-30b-a3b-lora
```

Qwen3-30B-A3B few-step attention-only validation:

```bash
MILES_N_NODES=1 modal run miles/modal_train.py --recipe qwen3-30b-a3b-lora-fewstep
```

Qwen3-30B-A3B expert-target LoRA follow-up:

```bash
MILES_N_NODES=1 modal run miles/modal_train.py --recipe qwen3-30b-a3b-experts-lora
```

Qwen3-30B-A3B expert-target few-step validation:

```bash
MILES_N_NODES=1 modal run miles/modal_train.py --recipe qwen3-30b-a3b-experts-fewstep
```

## Qwen3 Notes

- Start with standard LoRA, not DoRA. Miles' current rollout sync and adapter
  filtering are LoRA-specific and keyed off `lora_A` / `lora_B` names, so DoRA
  is not the first validation target.
- The baked Qwen3 recipes are single-node `H100:8` shapes. They are intended to
  validate end-to-end bridge-mode LoRA with colocated rollout first, not to
  exhaustively cover every parallelism combination.
- Source inspection suggests the training path should handle TP / PP / EP / CP
  because the bridge setup forwards all of those settings into Megatron-Bridge,
  and Megatron-Bridge's PEFT tests cover pipeline-style model chunk lists. That
  is still weaker than an actual Miles e2e validation for each shape.
- Miles currently supports LoRA weight sync only for colocated rollout engines.
  Distributed non-colocated rollout sync is not yet implemented for LoRA.
- The baseline Qwen3 recipe stays close to the upstream Miles single-node
  Qwen3-30B-A3B shape. The expert-target recipe is a follow-on experiment, not
  the initial correctness target.

## Observed On Modal

The current wrapper includes runtime patches in
[`modal_patches/sitecustomize.py`](./modal_patches/sitecustomize.py) that:

- register Megatron-Bridge's `LinearCrossEntropyModule` as column-parallel
  before Hugging Face weights are loaded, which fixes bridge-mode Qwen3 load on
  `output_layer.weight`;
- serialize colocated LoRA weight buckets in a builtins-only format and
  rehydrate them inside SGLang, which fixes the Modal colocated LoRA sync path;
- sanitize non-finite SGLang logprob values before JSON serialization;
- sanitize invalid SGLang sampling probability rows before `torch.multinomial`.

What the Modal runs have validated so far on `modal-labs`:

- `qwen3-30b-a3b-lora` gets through bridge-mode LoRA creation and attention
  module injection (`linear_qkv`, `linear_proj`), and it can start loading the
  Hugging Face checkpoint into Megatron.
- `qwen3-30b-a3b-lora-fewstep` now gets through full RL training on Modal. In
  recent runs it passed rollout, weight sync, and actor training repeatedly and
  reached at least `train/step` 6 on a single-node `H100:8` shape.
- `qwen3-30b-a3b-experts-lora` goes further: it creates LoRA with
  `linear_qkv`, `linear_proj`, `linear_fc1`, and `linear_fc2`, injects those
  expert targets under `decoder.layers.*.mlp.experts.*`, loads weights, pushes
  the adapter into SGLang, and starts `Eval gsm8k`.
- `qwen3-30b-a3b-experts-fewstep` has validated the widened target surface on
  Modal: Miles creates LoRA with `linear_fc1` / `linear_fc2`, injects those
  expert modules, completes weight sync, and reaches rollout collection plus
  actor training. A detached confirmation of a full expert-target train step is
  still in progress.
- The remaining instability has been in the colocated SGLang rollout path, not
  in LoRA target discovery. The main concrete runtime failures we hit were:
  non-finite logprobs breaking HTTP JSON serialization, and invalid sampling
  probability tensors breaking `torch.multinomial`.

Current interpretation:

- Qwen3-30B-A3B MoE LoRA support in Miles is real enough to instantiate,
  target, load, and export adapters for both attention and expert MLP layers.
- Attention-only Qwen3-30B-A3B LoRA is now runtime-validated for repeated RL
  updates on `modal-labs`.
- The remaining risk is concentrated in the colocated SGLang rollout lifecycle,
  which is coupled to `offload_rollout` / `enable_memory_saver=True` in the
  current Miles SGLang engine setup, especially once expert-target LoRA is
  enabled.

Useful options:

- `--dry-run`: print the assembled Miles command with a `$MODEL_PATH`
  placeholder without launching the cluster.
- `--extra-args "...flags..."`: append ad hoc Miles CLI overrides.
- `--extra-args-file path/to/file.args`: append overrides from a local text
  file.
- `--custom-config path/to/overrides.yaml`: pass a flat YAML override map to
  Miles via `--custom-config-path`.
- `--allow-cluster-mismatch`: bypass recipe/node-count validation if you are
  intentionally adapting a canned recipe.
- `USE_LOCAL_MILES=/path/to/miles`: overlay a local Miles checkout on top of
  the pinned container image.
- `MILES_IMAGE=radixark/miles:...`: override the pinned image tag. The current
  default is `radixark/miles:dev-202603231227`.

The wrapper intentionally owns only Modal/Ray plumbing plus a small set of
cluster-critical flags. All model and training settings live in
[`miles/recipes/`](./recipes/).
