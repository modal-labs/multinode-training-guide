# nemo-rl — Modal launcher for NeMo-RL training

Thin Modal launcher that runs [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) RL
training (GRPO, SFT, DPO, …) on multi-node Modal GPU clusters.

It mirrors the [`slime/`](../slime) launcher in this repo: each experiment is a
Python config, a clustered Modal function brings up a Ray cluster across the
allocation, and the driver runs on the head node. The difference is that NeMo-RL
is driven by a YAML config plus Hydra `key=value` overrides rather than raw CLI
flags.

## Prerequisites

- Modal CLI installed and authenticated
- Modal environment selected: `export MODAL_ENVIRONMENT=<your-env>`
- Modal secrets:
  - `huggingface-secret` — `HF_TOKEN` for model/data download (gated models)
  - `wandb-secret` — `WANDB_API_KEY` for configs with `logger.wandb_enabled=true`

Run all commands from this directory (`nemo-rl/`).

## Common Workflow

Set the config name once:

```bash
export EXPERIMENT_CONFIG=qwen2_5_1_5b_math
```

List available configs:

```bash
modal run modal_train.py::list_configs
```

Download the model into the HF cache volume:

```bash
modal run modal_train.py::download_model
```

Download dataset: 

```bash
modal run modal_train.py::download_data
```

Launch training:

```bash
modal run -d modal_train.py::train
```

Use `-d` (detached) to keep training running after you close your terminal. The
Ray dashboard URL is printed at the start of the run.

## Launcher Model

Each experiment lives in `configs/<name>.py` and exposes:

- `modal`: `ModalConfig` for image, GPU type, and Modal resources
- `nemo_rl`: `NemoRLConfig` for the recipe (run script, base YAML, cluster
  shape, and Hydra overrides)

The launcher runs this on the Ray head node:

```bash
cd /opt/nemo-rl && uv run python <entrypoint> --config <base_config> <overrides...>
```

### How multi-node works

1. `train` is wrapped in `modal.experimental.clustered(num_nodes, rdma=True)`,
   so Modal provisions `num_nodes` RDMA-connected GPU nodes.
2. Each node reads its rank from `modal.experimental.get_cluster_info()`.
3. Rank 0 starts the Ray head and waits until all nodes and GPUs have joined;
   ranks 1…N start Ray workers pointed at the head and idle.
4. Rank 0 runs the NeMo-RL driver, which attaches to the existing Ray cluster
   (`RAY_ADDRESS`) and schedules its actors across the whole allocation.

This is the equivalent of NemoRL's Slurm script [`ray.sub`](https://github.com/NVIDIA-NeMo/RL/blob/main/ray.sub).

## Volumes

| Volume | Mount path | Purpose |
| --- | --- | --- |
| `huggingface-cache` | `/root/.cache/huggingface` | HF model + dataset cache |
| `nemo-rl-checkpoints` | `/checkpoints` | Training checkpoints |

## Add A Config

Create `configs/<name>.py`:

```python
from configs.base import ModalConfig, NemoRLConfig

modal = ModalConfig(gpu="H100")


class _Recipe(NemoRLConfig):
    entrypoint = "examples/run_grpo.py"
    base_config = "examples/configs/grpo_math_8B.yaml"

    num_nodes = 2
    gpus_per_node = 8

    hf_model = "meta-llama/Llama-3.1-8B-Instruct"
    hf_datasets = ["nvidia/OpenMathInstruct-2"]

    overrides = {
        "policy.model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "policy.dtensor_cfg.tensor_parallel_size": 8,
        "logger.wandb_enabled": True,
        "logger.wandb.name": "my-run",
    }


nemo_rl = _Recipe()
```

The base_config points to a path in the NemoRL repo under examples/configs 
## Dev overlay

To run local NeMo-RL changes without rebuilding the image, point `local_nemo_rl`
at your checkout. It is copied over `/opt/nemo-rl` in the image:

```python
modal = ModalConfig(
    gpu="H100",
    local_nemo_rl="/path/to/your/RL",
)
```

The container still uses its baked `/opt/nemo_rl_venv`, so this only overlays
source code, not dependencies. If you change dependencies, rebuild/extend the
image (e.g. via `image_run_commands`).
