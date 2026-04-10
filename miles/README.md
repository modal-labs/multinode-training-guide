# Miles Modal Launcher

Modal-based launcher for the [Miles](https://github.com/radixark/miles) training framework. Follows the same contract as the `slime/` launcher.

## Quick Start

```bash
# List available experiments
modal run miles/modal_train.py::list_configs

# Run an experiment (3 steps)
export EXPERIMENT_CONFIG=qwen3_4b_lora_smoke
modal run miles/modal_train.py::download_model
modal run miles/modal_train.py::prepare_dataset
modal run -d miles/modal_train.py::train
```

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

## Writing a Config

See `configs/qwen3_4b_lora_smoke.py` for a complete example. Key patterns:

1. Subclass `MilesConfig` and set attributes as class fields
2. Use `miles_model_script` to source model architecture args
3. Override `prepare_data()` to download datasets
4. Set per-experiment env vars in `__init__` via `self.environment.update({...})`
5. Export `modal = ModalConfig(...)` and `miles = YourConfig()` at module level

## Dev Overlay

To test local Miles changes without rebuilding the Docker image:

```python
modal = ModalConfig(
    gpu="H200",
    local_miles="/path/to/local/miles",
)
```

This mounts your local Miles repo at `/root/miles` inside the container.
