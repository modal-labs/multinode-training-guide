# miles — Modal launcher for Miles training

Thin Modal launcher that runs [Miles](https://github.com/radixark/miles) RL training on GPU clusters.

## Prerequisites

- Modal CLI installed and authenticated
- Set your Modal environment: `export MODAL_ENVIRONMENT=<your-env>`
- Modal secrets:
  - `huggingface-secret` — required for `prepare_model` and `prepare_data`
  - `wandb-secret` — required only for experiments with `use_wandb = True`

## Running an experiment

All commands take the experiment name via `EXPERIMENT_CONFIG`. Run from the repo root.

### 1. List available experiments

```bash
modal run miles/modal_train.py::list_configs
```

### 2. Prepare model (one-time)

Downloads the experiment's HF checkpoint to the `huggingface-cache` volume and applies any experiment-specific model fixes.

```bash
EXPERIMENT_CONFIG=qwen3_4b_lora_smoke modal run miles/modal_train.py::prepare_model
```

### 3. Prepare data (one-time)

Downloads and preprocesses the training dataset to the `miles-data` volume.
Only required if the experiment defines a `prepare_data()` function (see [Adding an experiment](#adding-an-experiment)).

```bash
EXPERIMENT_CONFIG=qwen3_4b_lora_smoke modal run miles/modal_train.py::prepare_data
```

### 4. Convert checkpoint (one-time, raw mode only)

Converts the HF checkpoint to `torch_dist` format. Only required when `megatron_to_hf_mode = "raw"`.
Skip this step if using bridge mode.

```bash
EXPERIMENT_CONFIG=qwen3_4b_lora_smoke modal run modal_train.py::convert_checkpoint
```

### 5. Run training

```bash
EXPERIMENT_CONFIG=qwen3_4b_lora_smoke modal run -d modal_train.py::train
```

Use `-d` (detached) to keep training running after you close your terminal.

## Docker Image

Uses `radixark/miles:dev-202604101227` as the base image.

## Architecture

Each experiment defines two module-level objects in `configs/<name>.py`:

- `modal` — `ModalConfig` instance (GPU type, dev overlays, image patches)
- `miles` — `MilesConfig` subclass instance (all Miles CLI arguments)

`MilesConfig` attributes are automatically converted to CLI flags:
- `lora_rank = 64` → `--lora-rank 64`
- `colocate = True` → `--colocate`
- `colocate = False` → omitted

### Special Fields

These `MilesConfig` fields are **not** passed as CLI args:

| Field | Purpose |
|-------|---------|
| `environment` | Injected into the Ray job runtime env |
| `async_mode` | Selects `train_async.py` vs `train.py` |
| `miles_model_script` | Shell script sourced for `MODEL_ARGS` (e.g., `scripts/models/qwen3-4B.sh`) |

### Volumes

| Volume | Mount Path | Purpose |
|--------|-----------|---------|
| `huggingface-cache` | `/root/.cache/huggingface` | Model checkpoints |
| `miles-data` | `/data` | Training datasets |
| `miles-checkpoints` | `/checkpoints` | Converted checkpoints |

## Adding an experiment

### 1. Create the config file

Create `configs/<your_experiment>.py`. Each config file must expose two module-level instances:
- `modal` — a `ModalConfig` instance (GPU type, image patches)
- `miles` — a `MilesConfig` subclass instance (all Miles training arguments)

```python
from configs.base import ModalConfig, MilesConfig, DATA_PATH

modal = ModalConfig(gpu="H200")


class _Miles(MilesConfig):
    # Launcher instructions (not passed to Miles CLI)
    miles_model_script = "scripts/models/qwen3-4B.sh"  # sources MODEL_ARGS
    async_mode = False

    # Model
    hf_checkpoint = "Qwen/Qwen3-4B"
    megatron_to_hf_mode = "bridge"  # or "raw" (requires convert_checkpoint)

    # Infrastructure
    actor_num_nodes = 1
    actor_num_gpus_per_node = 4
    colocate = True

    # Data
    prompt_data = f"{DATA_PATH}/my_dataset/train.jsonl"
    input_key = "prompt"
    label_key = "label"
    rm_type = "deepscaler"

    # ... all other Miles args as snake_case attributes


miles = _Miles()
```

Every attribute on `_Miles` (except `environment`, `async_mode`, `miles_model_script`) is forwarded to
Miles as a CLI argument: `field_name` → `--field-name`. See `configs/base.py` for full rules.

### 2. Add `prepare_model()` / `prepare_data()` methods (if needed)

Override `prepare_model()` if your experiment needs model-specific preparation beyond a plain HF download.
The base implementation already calls `snapshot_download(self.hf_checkpoint)`.

```python
class _Miles(MilesConfig):
    ...
    def prepare_model(self) -> None:
        super().prepare_model()
        # apply model-specific local patches if needed
```

If your experiment needs to download or preprocess a dataset, override `prepare_data()` on `_Miles`.
It runs inside the Modal container with the `miles-data` volume mounted at `DATA_PATH`.

```python
class _Miles(MilesConfig):
    ...
    def prepare_data(self) -> None:
        import os
        from huggingface_hub import snapshot_download

        os.makedirs(f"{DATA_PATH}/my_dataset", exist_ok=True)
        snapshot_download(
            repo_id="org/my-dataset",
            repo_type="dataset",
            local_dir=f"{DATA_PATH}/my_dataset",
        )
```

If `prepare_data()` is not overridden, `prepare_data` will raise `NotImplementedError` — simply skip that step.

### 3. Run the workflow

`EXPERIMENT_CONFIG` is the config filename without `.py`:

```bash
EXPERIMENT_CONFIG=my_experiment modal run miles/modal_train.py::prepare_model
EXPERIMENT_CONFIG=my_experiment modal run miles/modal_train.py::prepare_data  # if prepare_data() defined
EXPERIMENT_CONFIG=my_experiment modal run miles/modal_train.py::convert_checkpoint  # if megatron_to_hf_mode = "raw"
EXPERIMENT_CONFIG=my_experiment modal run -d miles/modal_train.py::train
```

No registration step needed — the launcher discovers configs automatically from the `configs/` directory.

## YAML config fields

`eval_config`, `custom_config_path`, and `sglang_config` normally take file paths in Miles.
In Python configs you can write them as inline dicts — the launcher materializes them to temp YAML files automatically:

```python
class _Miles(MilesConfig):
    eval_config = {
        "eval": {
            "defaults": {"max_response_len": 16384},
            "datasets": [
                {"name": "aime", "path": "/data/aime.jsonl", "rm_type": "deepscaler"},
            ],
        }
    }
```

## JSON config fields

`train_env_vars`, `apply_chat_template_kwargs`, and `multimodal_keys` are parsed by Miles with `json.loads()`.
If set as dicts in Python configs, the launcher serializes them with `json.dumps()` automatically.

## Dev overlay

To test local Miles changes without rebuilding the image, set `local_miles` in your `ModalConfig`:

```python
modal = ModalConfig(
    gpu="H200",
    local_miles="/path/to/your/miles",
)
```

## Applying patches to the image

To inject local patch files into the image (e.g. to patch SGLang), use `patch_files` and `image_run_commands`:

```python
modal = ModalConfig(
    gpu="H200",
    patch_files=["miles/patches/sglang_fix.patch"],
    image_run_commands=["cd /sgl-workspace/sglang && git apply /tmp/sglang_fix.patch"],
)
```

Each file in `patch_files` is added to the image at `/tmp/<filename>`.
