# miles — Modal launcher for Miles training

Thin Modal launcher for running [Miles](https://github.com/radixark/miles) RL
training jobs on Modal GPU clusters.

## Prerequisites

- Modal CLI installed and authenticated
- Modal environment selected: `export MODAL_ENVIRONMENT=<your-env>`
- Modal secrets:
  - `huggingface-secret` for model/data download and post-processing
  - `wandb-secret` for configs with `use_wandb = True`

Run all commands from the repo root.

## Common Workflow

Set the config name once:

```bash
export EXPERIMENT_CONFIG=qwen3_4b_lora_smoke
```

List available configs:

```bash
modal run miles/modal_train.py::list_configs
```

Download required model artifacts:

```bash
modal run miles/modal_train.py::download_model
```

Download required data artifacts:

```bash
modal run miles/modal_train.py::download_data
```

Run optional post-processing only when the config needs it:

```bash
modal run miles/modal_train.py::post_process_model
modal run miles/modal_train.py::post_process_data
```

Convert to Megatron `torch_dist` only for raw-mode configs:

```bash
modal run miles/modal_train.py::convert_hf_to_megatron_checkpoint
```

Bridge-mode configs do not need this conversion step.

Launch training:

```bash
modal run -d miles/modal_train.py::train
```

Use `-d` to keep the Modal job running after the terminal disconnects.

## Run Kimi-K2.5

`kimi_k25` uses bridge mode, so do not run
`convert_hf_to_megatron_checkpoint`.

```bash
export EXPERIMENT_CONFIG=kimi_k25

modal run miles/modal_train.py::download_model
modal run miles/modal_train.py::post_process_model
modal run miles/modal_train.py::download_data
modal run -d miles/modal_train.py::train
```

For this config:

- `download_model` downloads `source_hf_checkpoint` and applies the Kimi source patch.
- `post_process_model` creates `/checkpoints/Kimi-K2.5-int4` for rollout and `/checkpoints/Kimi-K2.5-bf16` for training(`--ref_load`).
- `download_data` downloads the training dataset to `/data`.
- `train` launches the bridge-mode training job directly.

## Launcher Model

Each experiment lives in `miles/configs/<name>.py` and exposes:

- `modal`: `ModalConfig` for image, GPU type, patches, and Modal resources
- `miles`: `MilesConfig` for Miles CLI arguments and preparation hooks

`MilesConfig` attributes become CLI flags automatically:

- `lora_rank = 64` becomes `--lora-rank 64`
- `colocate = True` becomes `--colocate`
- `colocate = False` is omitted

Launcher-only fields are not passed to Miles:

| Field | Purpose |
| --- | --- |
| `environment` | Ray job environment variables |
| `async_mode` | Selects `train_async.py` instead of `train.py` |
| `miles_model_script` | Shell script sourced for `MODEL_ARGS` |
| `source_hf_checkpoint` | Upstream repo/path used for download or config-specific conversion |
| `megatron_conversion_hf_checkpoint` | Optional raw Megatron conversion input; defaults to `hf_checkpoint` |

## Hooks

Every config should make `download_model` and `download_data` usable.
`download_model` has a default implementation; `download_data` must be
implemented by the config.

| Hook | Resource | Expected use |
| --- | --- | --- |
| `download_model()` | CPU | Download source model files into the HF cache |
| `download_data()` | CPU | Download or prepare training data under `/data` |
| `post_process_model()` | GPU | Optional model conversion or derived model artifacts |
| `post_process_data()` | GPU | Optional data processing that needs GPU |

The default `post_process_model()` and `post_process_data()` are no-ops. Run
their Modal entrypoints only when the config documents a reason.

## Checkpoint Fields

Use `hf_checkpoint` for the checkpoint Miles should train or serve from.

Use `source_hf_checkpoint` when the upstream model is different from
`hf_checkpoint`. For example, Kimi downloads `moonshotai/Kimi-K2.5`, then
post-processing writes the training checkpoint to
`/checkpoints/Kimi-K2.5-int4`.

Use `megatron_conversion_hf_checkpoint` only when raw Megatron conversion should
read a different HF-format checkpoint than `hf_checkpoint`.

All three fields may be Hugging Face repo IDs or absolute mounted paths.

## Volumes

| Volume | Mount path | Purpose |
| --- | --- | --- |
| `huggingface-cache` | `/root/.cache/huggingface` | HF snapshots |
| `miles-data` | `/data` | Training and eval data |
| `miles-checkpoints` | `/checkpoints` | Derived checkpoints |

## Add A Config

Create `miles/configs/<name>.py`:

```python
from configs.base import DATA_PATH, ModalConfig, MilesConfig

modal = ModalConfig(gpu="H200")


class _Miles(MilesConfig):
    miles_model_script = "scripts/models/qwen3-4B.sh"
    async_mode = False

    hf_checkpoint = "Qwen/Qwen3-4B"
    megatron_to_hf_mode = "bridge"

    actor_num_nodes = 1
    actor_num_gpus_per_node = 4
    colocate = True

    prompt_data = f"{DATA_PATH}/my_dataset/train.jsonl"
    input_key = "prompt"
    label_key = "label"
    rm_type = "deepscaler"

    def download_data(self) -> None:
        import os

        from huggingface_hub import snapshot_download

        os.makedirs(f"{DATA_PATH}/my_dataset", exist_ok=True)
        snapshot_download(
            repo_id="org/my-dataset",
            repo_type="dataset",
            local_dir=f"{DATA_PATH}/my_dataset",
        )


miles = _Miles()
```

No registration step is needed. The launcher discovers config files
automatically.

## Inline Config Values

`eval_config`, `custom_config_path`, and `sglang_config` may be Python dicts.
The launcher writes them to temporary YAML files before training.

`train_env_vars`, `apply_chat_template_kwargs`, and `multimodal_keys` may be
Python dicts. The launcher serializes them as JSON CLI values.

## Dev Overlay

To test local Miles changes without rebuilding the base image:

```python
modal = ModalConfig(
    gpu="H200",
    local_miles="/path/to/your/miles",
)
```

## Image Patches

To inject local patch files into the image:

```python
modal = ModalConfig(
    gpu="H200",
    patch_files=["patches/sglang_fix.patch"],
    image_run_commands=["cd /sgl-workspace/sglang && git apply /tmp/sglang_fix.patch"],
)
```

Each patch file is copied to `/tmp/<filename>` inside the image.
