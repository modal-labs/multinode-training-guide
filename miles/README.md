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
- `glm4-7-flash-lora`: first real GLM MoE validation recipe.
- `glm5-744b-a40b-4layer-lora`: GLM-5 testing recipe using the 4-layer script
  shape from upstream Miles.
- `glm5-744b-a40b-20layer-lora`: larger GLM-5 testing recipe using the 20-layer
  script shape from upstream Miles.
- `glm5-744b-a40b-lora`: full GLM-5 starter recipe.

## Prepare assets

Prepare a small GSM8K dataset in the shared volume:

```bash
modal run miles/modal_train.py::prepare_dataset
```

Download a recipe's base model into the shared Hugging Face cache:

```bash
modal run miles/modal_train.py::download_model --recipe glm4-7-flash-lora
```

## Train

The cluster size is chosen at import time by `MILES_N_NODES`, so set it in the
same shell invocation as `modal run`.

Single-node smoke test:

```bash
MILES_N_NODES=1 modal run miles/modal_train.py --recipe qwen25-0p5b-lora
```

GLM-4.7-Flash multi-node validation:

```bash
MILES_N_NODES=4 modal run miles/modal_train.py --recipe glm4-7-flash-lora
```

GLM-5 4-layer testing recipe:

```bash
MILES_N_NODES=1 modal run miles/modal_train.py --recipe glm5-744b-a40b-4layer-lora --gpu H200:8
```

GLM-5 20-layer testing recipe:

```bash
MILES_N_NODES=2 modal run miles/modal_train.py --recipe glm5-744b-a40b-20layer-lora --gpu H200:8
```

Full GLM-5 starter recipe:

```bash
MILES_N_NODES=8 modal run miles/modal_train.py --recipe glm5-744b-a40b-lora --gpu H200:8
```

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
