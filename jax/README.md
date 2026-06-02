# jax

Small JAX + Equinox training examples, run on [Modal](https://modal.com) across a 2-node × 8×H100 cluster.

## About Jax
[Jax](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html) is a functional, pythonic library for machine learning and high-performance computing. Similar to PyTorch, it has all of the helpful primitives for constructing tensors, performing matrix multiplications, and efficiently parallelizing computation across hardware topologies.

## About Equinox
[Equinox](https://docs.kidger.site/equinox/) is a third-party library to Jax that allows you to store state, specifically model parameters, as a compatible Jax [PyTree](https://docs.jax.dev/en/latest/key-concepts.html#pytrees). The psuedo-stateful frontend of Equinox makes switching from PyTorch to Jax for neural network training natural. Best of all, you get to JIT compile your forward/backward passes.

## Layout

| file | what it is |
| --- | --- |
| `models.py` | Reusable modules: `Linear`, `BatchNorm`, `GroupNorm`, `MLP`, and `RNN`. |
| `train_mlp.py` | Pure training primitives for the toy MLP regression (`y = x**2`) task: loss, update, checkpointing, `train_loop(mesh, nproc, ...)`. No Modal dependency. |
| `train_rnn.py` | Pure training primitives for the character-level RNN on `moby_dick_cetology.txt`: dataset helpers, loss, update, `generate`, checkpointing, `train_loop(mesh, nproc, ...)`. No Modal dependency. |
| `modal_train.py` | All Modal orchestration (image, app, volumes, distributed init, `@app.function` wrappers, local entrypoints) for both models. |
| `moby_dick_cetology.txt` | [Chapter 32](https://www.reddit.com/r/mobydick/comments/opfisa/when_do_the_infamously_boring_chapters_come_in/) "Cetology" of *Moby-Dick* (~29k chars, 65-char vocab). RNN training data. |

## Prerequisites

- Modal account with access to H100 GPUs
- Workspace with multi-node + RDMA access enabled.

## Running

```bash
# MLP
modal run jax/modal_train.py::mlp_train     # train on 2 × 8×H100
modal run jax/modal_train.py::mlp_sample    # load latest checkpoint and run one forward pass

# RNN
modal run jax/modal_train.py::rnn_train     # train on 2 × 8×H100
modal run jax/modal_train.py::rnn_sample    # load latest checkpoint and sample text
```

Each `*_train` entrypoint:
1. Spins up the image (CUDA, [JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html), [Equinox](https://docs.kidger.site/equinox/), optax, matplotlib).
2. Initializes `jax.distributed` across the 2 containers and builds an `(8, nproc)` device mesh.
3. Calls into `train_mlp.train_loop` / `train_rnn.train_loop`, which checkpoint every epoch to the mounted volume.
4. `volume.commit()`s at the end so checkpoints are visible to the matching `*_sample` job.

## Distributed Training in Jax
To run your jobs *multi-node* with Jax, create a device mesh with `jax.make_mesh(axis_shapes, axis_names)` where shape is `(8, 2)` and names are `("i", "j")` on a 2x8:H100 cluster. In `train_rnn.py` and `train_mlp.py`, we specify a `NamedSharding` with a PartitionSpec of `P(("i", "j"), None)`. The input data's first axis, which in our case is the batch dimension, will be sharded along both the `"i"` and `"j"` axes. With [`explicit sharding`](https://docs.jax.dev/en/latest/parallel.html), we tell Jax how to *globally* divvy up the input data without specifying which specific devices should receive these inputs. This sharding strategy is equivalent to [*data parallelism*](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html).