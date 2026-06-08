# SkyRL-TX Qwen SFT and RL on Modal

This example runs [SkyRL-TX](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-tx)
as a Tinker-compatible training server on a Modal multi-node GPU cluster.
It uses `Qwen/Qwen3-8B` by default and exercises both:

- supervised fine-tuning with Tinker's `cross_entropy` loss
- policy-gradient RL with Tinker's `ppo` loss over sampled arithmetic rollouts

The default topology is 2 nodes × 4 H100 GPUs. Within each node SkyRL-TX uses
tensor parallelism; across nodes it uses JAX FSDP.

## Prerequisites

- Modal CLI installed and authenticated
- Modal environment selected: `export MODAL_ENVIRONMENT=<your-env>`
- Modal secret `huggingface-secret` with `HF_TOKEN`
- Access to multi-node GPU clusters

Run all commands from the repo root.

## Quickstart

Download the Qwen checkpoint into the persistent Hugging Face cache volume:

```bash
modal run skyrl-tx/modal_train.py::download_model
```

Run the supervised fine-tuning smoke:

```bash
modal run --detach skyrl-tx/modal_train.py::run_sft
```

Run the RL smoke:

```bash
modal run --detach skyrl-tx/modal_train.py::run_rl
```

Use detached mode for the training jobs; the image build, model load, JAX
initialization, and first compile can take several minutes.

## Cluster sizing

The launcher reads these environment variables at import time:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SKYRL_TX_N_NODES` | `2` | Number of Modal containers in the JAX cluster |
| `SKYRL_TX_GPUS_PER_NODE` | `4` | GPUs visible to each JAX process |
| `SKYRL_TX_GPU_TYPE` | `H100` | Modal GPU type |

For the default `Qwen/Qwen3-8B` run:

```text
total GPUs = SKYRL_TX_N_NODES × SKYRL_TX_GPUS_PER_NODE = 8
tensor_parallel_size = SKYRL_TX_GPUS_PER_NODE = 4
fully_sharded_data_parallel_size = SKYRL_TX_N_NODES = 2
```

To run on 2 nodes × 8 H100s:

```bash
SKYRL_TX_GPUS_PER_NODE=8 modal run --detach skyrl-tx/modal_train.py::run_sft
```

## How it works

`run_sft` and `run_rl` are `@modal.experimental.clustered` functions.
Rank 0 starts the SkyRL-TX Tinker API server:

```bash
uv run --extra gpu --extra tinker --extra jax -m skyrl.tinker.api \
  --base-model Qwen/Qwen3-8B \
  --backend jax \
  --backend-config '{"tensor_parallel_size": 4, "fully_sharded_data_parallel_size": 2, ...}'
```

Ranks 1..N start SkyRL-TX JAX workers:

```bash
uv run --extra gpu --extra tinker --extra jax -m skyrl.backends.jax \
  --coordinator-address <rank-0-ip>:7777 \
  --num-processes <N> \
  --process-id <rank>
```

After the API server reports healthy, rank 0 runs either `sft_client.py` or
`rl_client.py` against `http://localhost:8000`.

## Volumes

| Volume | Mount path | Purpose |
| --- | --- | --- |
| `skyrl-tx-hf-cache` | `/root/.cache/huggingface` | Qwen model cache |
| `skyrl-tx-checkpoints` | `/checkpoints` | SkyRL-TX LoRA checkpoints |

## Adjusting the smoke

Both entrypoints expose small training-loop knobs:

```bash
modal run --detach skyrl-tx/modal_train.py::run_sft --steps 16 --lora-rank 8
modal run --detach skyrl-tx/modal_train.py::run_rl --steps 8 --samples-per-prompt 4
```

The clients intentionally use tiny arithmetic datasets so the example validates
the end-to-end SkyRL-TX path without requiring a full benchmark-scale run.
