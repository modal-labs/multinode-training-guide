# slime_v2 — Modal launcher for SLIME training

Thin Modal launcher that runs [SLIME](https://github.com/SLIT-AI/SLIME) RL training on GPU clusters.

## Prerequisites

- Modal CLI installed and authenticated
- Set your Modal environment: `export MODAL_ENVIRONMENT=<your-env>`

## Running an experiment

All commands take the experiment name via `EXPERIMENT_CONFIG`. Run from the repo root.

### 1. List available experiments

```bash
modal run slime_v2/modal_train.py::list_configs
```

### 2. Download model (one-time)

Downloads the experiment's HF checkpoint to the `huggingface-cache` volume.

```bash
EXPERIMENT_CONFIG=glm4.7-flash-dapo modal run slime_v2/modal_train.py::download_model
```

### 3. Prepare dataset (one-time)

Downloads and preprocesses the training dataset to the `slime-data` volume.

```bash
EXPERIMENT_CONFIG=glm4.7-flash-dapo modal run slime_v2/modal_train.py::prepare_dataset
```

### 4. Convert checkpoint (one-time, raw mode only)

Converts the HF checkpoint to `torch_dist` format. Only required when `megatron_to_hf_mode = "raw"`.
Skip this step if using bridge mode.

```bash
EXPERIMENT_CONFIG=glm4.7-flash-dapo modal run slime_v2/modal_train.py::convert_checkpoint
```

### 5. Run training

```bash
EXPERIMENT_CONFIG=glm4.7-flash-dapo modal run -d slime_v2/modal_train.py::train
```

Use `-d` (detached) to keep training running after you close your terminal.

## Adding an experiment

1. Create `configs/<your_experiment>.py` with `modal` and `slime` instances (see existing configs for examples).
2. Register it in `configs/__init__.py`.

## Dev overlay

To test local SLIME changes without rebuilding the image, set `local_slime` in your `ModalConfig`:

```python
modal = ModalConfig(
    gpu="H200",
    local_slime="/path/to/your/slime",
)
```
