# checkpoint-bench

Async checkpoint I/O benchmark on Modal (PyTorch 2.9 DCP + `DefaultStager`).

Compares four modes on a 2-node H100 cluster:

| Mode | What blocks the training thread |
|---|---|
| **blocking** | everything (save + commit) |
| **async** | staging (GPU→CPU copy) |
| **fully_async** | ~nothing (enqueue only), commit on main |
| **pipeline** | ~nothing — save *and* commit in background |

## Usage

```bash
# Default: 60 GiB checkpoint
modal run modal_async_checkpoint.py

# Custom size
modal run modal_async_checkpoint.py --ckpt-gib 120

# Smaller test run
modal run modal_async_checkpoint.py --ckpt-gib 10
```

## Config

Edit the constants at the top of `modal_async_checkpoint.py`:

```python
N_NODES = 2   # number of Modal containers
N_GPUS  = 8   # GPUs per node (H100s)
```
