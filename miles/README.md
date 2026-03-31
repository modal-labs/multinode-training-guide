# Miles on Modal

Run Miles RL training on Modal with recipe files stored under [`recipes/`](./recipes/).
The wrapper handles Modal and Ray orchestration; model and training flags stay in
recipe arg files.

## Prerequisites

- A Modal account with multi-node access.
- A `huggingface-secret` Modal secret containing `HF_TOKEN`.
- Optional: `WANDB_API_KEY` in your local shell for Weights & Biases logging.
- Optional: `modal deploy miles/modal_train.py`. The local entrypoint will try
  the deployed `MilesCluster` first and fall back to an ephemeral app if it is
  not deployed.

## Prepare Shared Assets

Prepare the default GSM8K dataset:

```bash
modal run miles/modal_train.py::prepare_dataset
```

Download a model for a built-in recipe:

```bash
modal run miles/modal_train.py::download_model --recipe qwen3-30b-a3b-lora
```

Or download any model directly:

```bash
modal run miles/modal_train.py::download_model --model-id Qwen/Qwen3-30B-A3B
```

## Recipes

List the available recipes:

```bash
modal run miles/modal_train.py --list-recipes
```

Recommended starting points:

- `qwen3-30b-a3b-lora`: default Qwen3 recipe.
- `qwen3-30b-a3b-lora-fewstep`: smallest end-to-end Qwen3 validation recipe.
- `qwen3-30b-a3b-experts-lora`: explicit expert-target variant.
- `qwen3-30b-a3b-experts-fewstep`: trimmed expert-target validation recipe.
- `qwen25-0p5b-lora`: small smoke test.

Testing and debug recipes live under [`recipes/tests/`](./recipes/tests).

## Train

Set `MILES_N_NODES` in the same shell invocation as `modal run`.

Single-node Qwen3 few-step validation:

```bash
MILES_N_NODES=1 modal run miles/modal_train.py --recipe qwen3-30b-a3b-lora-fewstep
```

Single-node Qwen3 default recipe:

```bash
MILES_N_NODES=1 modal run miles/modal_train.py --recipe qwen3-30b-a3b-lora
```

Single-node expert-target follow-up:

```bash
MILES_N_NODES=1 modal run miles/modal_train.py --recipe qwen3-30b-a3b-experts-fewstep
```

Non-colocated Qwen3 validation:

```bash
MILES_N_NODES=2 modal run miles/modal_train.py \
  --recipe qwen3-30b-a3b-lora-fewstep \
  --no-colocate \
  --actor-nodes 1 \
  --allow-cluster-mismatch
```

Small smoke test:

```bash
MILES_N_NODES=1 modal run miles/modal_train.py --recipe qwen25-0p5b-lora
```

## Ad Hoc Runs

You can launch without a predefined recipe by passing args directly:

```bash
MILES_N_NODES=1 modal run miles/modal_train.py \
  --model-id Qwen/Qwen3-30B-A3B \
  --args-file miles/recipes/qwen3-30b-a3b-lora.args \
  --extra-args "--train-samples 8 --eval-interval 1" \
  --run-name qwen3-adhoc
```

## Useful Options

- `--dry-run`: print the assembled Miles command without launching a job.
- `--args` / `--args-file`: provide the base Miles CLI args.
- `--extra-args` / `--extra-args-file`: append overrides to a recipe or ad hoc run.
- `--custom-config`: pass a YAML override file through to Miles.
- `--run-name`: override the checkpoint subdirectory name.
- `--allow-cluster-mismatch`: bypass recipe node-count checks.
- `USE_LOCAL_MILES=/path/to/miles`: overlay a local Miles checkout.
- `MILES_IMAGE=radixark/miles:...`: override the pinned container image.

## Notes

- The default Qwen3 recipes use standard all-layer LoRA over
  `linear_qkv`, `linear_proj`, `linear_fc1`, and `linear_fc2`.
- Start with the few-step recipes when validating a new environment.
- Modal-specific runtime compatibility patches live in
  [`modal_patches/sitecustomize.py`](./modal_patches/sitecustomize.py).
