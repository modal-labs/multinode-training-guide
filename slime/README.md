# slime — Modal launcher for SLIME training

Thin Modal launcher that runs [SLIME](https://github.com/THUDM/slime) RL training on GPU clusters.

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
export EXPERIMENT_CONFIG=glm47_flash_dapo
```

List available configs:

```bash
modal run slime/modal_train.py::list_configs
```

Download required model artifacts:

```bash
modal run slime/modal_train.py::download_model
```

Download required data artifacts:

```bash
modal run slime/modal_train.py::download_data
```

Run optional post-processing only when the config needs it:

```bash
modal run slime/modal_train.py::post_process_model
modal run slime/modal_train.py::post_process_data
```

Convert to Megatron `torch_dist` only for raw-mode configs:

```bash
modal run slime/modal_train.py::convert_hf_to_megatron_checkpoint
```

Bridge-mode configs do not need this conversion step.

Launch training:

```bash
modal run -d slime/modal_train.py::train
```

Use `-d` (detached) to keep training running after you close your terminal.

## Launcher Model

Each experiment lives in `slime/configs/<name>.py` and exposes:

- `modal`: `ModalConfig` for image, GPU type, patches, and Modal resources
- `slime`: `SlimeConfig` for SLIME CLI arguments and preparation hooks

`ModalConfig.docker_image` is the Docker image reference passed to
`modal.Image.from_registry(...)`. Set it in the config when an experiment needs
a specific SLIME image tag.

`SlimeConfig` attributes become CLI flags automatically:

- `colocate = True` becomes `--colocate`
- `rollout_batch_size = 64` becomes `--rollout-batch-size 64`
- `colocate = False` is omitted

Launcher-only fields are not passed to SLIME:

| Field | Purpose |
| --- | --- |
| `environment` | Ray job environment variables |
| `async_mode` | Selects `train_async.py` instead of `train.py` |
| `slime_model_script` | Shell script sourced for `MODEL_ARGS` |
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

Use `hf_checkpoint` for the checkpoint SLIME should train or serve from.

Use `source_hf_checkpoint` when the upstream model is different from
`hf_checkpoint`.

Use `megatron_conversion_hf_checkpoint` only when raw Megatron conversion should
read a different HF-format checkpoint than `hf_checkpoint`.

All three fields may be Hugging Face repo IDs or absolute mounted paths.

## Volumes

| Volume | Mount path | Purpose |
| --- | --- | --- |
| `huggingface-cache` | `/root/.cache/huggingface` | HF snapshots |
| `slime-data` | `/data` | Training and eval data |
| `slime-checkpoints` | `/checkpoints` | Derived checkpoints |

## Add A Config

Create `slime/configs/<name>.py`:

```python
from configs.base import ModalConfig, SlimeConfig, DATA_PATH

modal = ModalConfig(gpu="H200")


class _Slime(SlimeConfig):
    # Launcher instructions (not passed to SLIME CLI)
    slime_model_script = "scripts/models/qwen3-8B.sh"  # sources MODEL_ARGS
    async_mode = False

    # Model
    hf_checkpoint = "Qwen/Qwen3-8B"
    load = "Qwen/Qwen3-8B"
    megatron_to_hf_mode = "bridge"  # or "raw" (requires conversion)

    # Infrastructure
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8
    colocate = True

    # Data
    prompt_data = f"{DATA_PATH}/my_dataset/train.parquet"
    input_key = "problem"
    label_key = "answer"
    rm_type = "math"

    # ... all other SLIME args as snake_case attributes

    def download_data(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="org/my-dataset",
            repo_type="dataset",
            local_dir=f"{DATA_PATH}/my_dataset",
        )


slime = _Slime()
```

No registration step is needed. The launcher discovers config files
automatically.

## YAML config fields

`eval_config`, `custom_config_path`, and `sglang_config` normally take file paths in SLIME.
In Python configs you can write them as inline dicts — the launcher materializes them to temp YAML files automatically:

```python
class _Slime(SlimeConfig):
    eval_config = {
        "eval": {
            "defaults": {"max_response_len": 16384},
            "datasets": [
                {"name": "aime", "path": "/data/aime.jsonl", "rm_type": "deepscaler"},
            ],
        }
    }
    custom_config_path = {
        "max_turns": 3,
        "rollout_interaction_env_path": "examples.my_env.rollout",
    }
```

`train_env_vars`, `apply_chat_template_kwargs`, and `multimodal_keys` may be
Python dicts. The launcher serializes them as JSON CLI values.

## Dev overlay

To test local SLIME changes without rebuilding the image, set `local_slime` in your `ModalConfig`:

```python
modal = ModalConfig(
    gpu="H200",
    local_slime="/path/to/your/slime",
)
```

## Applying patches to the image

To inject local patch files into the image (e.g. to patch SGLang), use `patch_files` and `image_run_commands`:

```python
modal = ModalConfig(
    gpu="H200",
    patch_files=["patches/sglang_fix.patch"],
    image_run_commands=["cd /sgl-workspace/sglang && git apply /tmp/sglang_fix.patch"],
)
```

Each file in `patch_files` is added to the image at `/tmp/<filename>`.
