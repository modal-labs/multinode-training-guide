import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress GRPC errors

import re
import time

import jax
from jax.sharding import NamedSharding, PartitionSpec as P
import optax
import equinox as eqx
import matplotlib.pyplot as plt

from models import MLP, BatchNormState


def loss_fn(model, x, y_true, state):
    out, state = model(x, state)
    return (jax.numpy.mean((y_true - out) ** 2), state)


@eqx.filter_jit
def update(model, opt_state, x, y_true, optimizer, state):
    (loss, state), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, x, y_true, state
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, state


def save_checkpoint(model, state, epoch, path):
    eqx.tree_serialise_leaves(
        os.path.join(path, f"model_epoch_{epoch}.eqx"), (model, state)
    )


def load_checkpoint(model, state, filepath):
    return eqx.tree_deserialise_leaves(filepath, (model, state))


def find_latest_checkpoint(path: str):
    """Return (filepath, epoch) of the highest-epoch checkpoint in `path`,
    or (None, -1) if nothing is there yet."""
    if not os.path.isdir(path):
        return None, -1
    pattern = re.compile(r"model_epoch_(\d+)\.eqx$")
    best_file, best_epoch = None, -1
    for fname in os.listdir(path):
        m = pattern.match(fname)
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_file = os.path.join(path, fname)
    return best_file, best_epoch


def plot_losses(losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def train_loop(
    checkpoint_dir: str,
    mesh,
    nproc: int,
    learning_rate: float = 3e-4,
    batch_size: int = 32,
    epochs: int = 25,
    steps_per_epoch: int = 100,
    hidden_size: int = 64,
    seed: int = 5678,
):
    """Run the full MLP training loop. Assumes `jax.distributed` is already
    initialized and `mesh` is a ready-to-use `jax.Mesh`.

    Writes per-epoch `(model, state)` checkpoints to `checkpoint_dir`, plus a
    final `model_epoch_{epochs}.eqx` with finalized (inference-time) BatchNorm
    running stats. Returns `(model, state)`.
    """
    optimizer = optax.adam(learning_rate)

    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    model = MLP(in_shape=4, hidden_dim=hidden_size, out_shape=4, key=model_key)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    state = BatchNormState(
        training_time=True,
        mean=jax.numpy.zeros(hidden_size),
        var=jax.numpy.zeros(hidden_size),
    )

    data_sharding = NamedSharding(mesh, P(("i", "j"), None))
    weight_sharding = NamedSharding(mesh, P())

    all_losses = []

    for epoch in range(epochs):
        average_t = 0
        for _ in range(steps_per_epoch):
            key, batch_key = jax.random.split(key)
            x = jax.random.normal(batch_key, (batch_size, 4))

            global_x = jax.device_put(x, data_sharding)
            global_y_true = jax.device_put(global_x**2, data_sharding)

            t0 = time.perf_counter()

            model = jax.tree.map(
                lambda x: (
                    jax.device_put(x, weight_sharding)
                    if isinstance(x, jax.Array)
                    else x
                ),
                model,
            )
            opt_state = jax.tree.map(
                lambda x: (
                    jax.device_put(x, weight_sharding)
                    if isinstance(x, jax.Array)
                    else x
                ),
                opt_state,
            )
            state = jax.tree.map(
                lambda x: (
                    jax.device_put(x, weight_sharding)
                    if isinstance(x, jax.Array)
                    else x
                ),
                state,
            )

            model, opt_state, loss, state = update(
                model, opt_state, global_x, global_y_true, optimizer, state
            )
            t1 = time.perf_counter()

            average_t = (t1 - t0) / nproc

            all_losses.append(float(loss))

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss {all_losses[-1]:.6f}, "
            f"Average time per step {average_t:.9f}s"
        )
        if jax.process_index() == 0:
            save_checkpoint(model, state, epoch, checkpoint_dir)

    state = BatchNormState(
        training_time=False,
        mean=state.mean / (steps_per_epoch * epochs),
        var=state.var / (steps_per_epoch * epochs),
    )

    if jax.process_index() == 0:
        # Final checkpoint carries the inference-ready BatchNorm stats.
        save_checkpoint(model, state, epochs, checkpoint_dir)
        plot_losses(all_losses, os.path.join(checkpoint_dir, "loss_curve_mlp.png"))
    return model, state
