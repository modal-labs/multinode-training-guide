import os
import time
import jax
from jax.sharding import NamedSharding, PartitionSpec as P
import optax
import equinox as eqx
import matplotlib.pyplot as plt
import modal
import modal.experimental
import sys
from model import MLP, BatchNormState

CHECKPOINT_DIR = "/checkpoints"
LOCAL_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .pip_install("jax[cuda12]", "equinox", "matplotlib", "optax")
    .add_local_dir(LOCAL_CODE_DIR, remote_path="/root")
)
app = modal.App("jax-mlp-training", image=image)
volume = modal.Volume.from_name("jax-mlp-weights", create_if_missing=True)


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


def save_checkpoint(model, epoch, path):
    eqx.tree_serialise_leaves(os.path.join(path, f"model_epoch_{epoch}.eqx"), model)


def load_checkpoint(model, filepath):
    return eqx.tree_deserialise_leaves(filepath, model)


def plot_losses(losses, save_path="loss_curve.png"):
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


# TODO(atoniolo76): setup Jax mesh for distributed training
@app.function(
    gpu="H100:8",
    volumes={
        CHECKPOINT_DIR: volume,
    },
)
@modal.experimental.clustered(size=2, rdma=True)
def train(
    learning_rate=3e-4,
    batch_size=32,
    epochs=25,
    steps_per_epoch=100,
    hidden_size=64,
    seed=5678,
):
    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    master_addr = cluster_info.container_ipv4_ips[0]
    nproc = len(cluster_info.container_ipv4_ips)

    jax.distributed.initialize(f"{master_addr}:12345", nproc, node_rank)

    print(f"Number of devices: {len(jax.devices())}")

    mesh = jax.make_mesh((8, nproc), ("i", "j"))
    print(f"Mesh: {mesh}")
    optimizer = optax.adam(learning_rate)

    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    model = MLP(in_shape=4, hidden_dim=hidden_size, out_shape=4, key=model_key)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    all_losses = []

    state = BatchNormState(
        training_time=True,
        mean=jax.numpy.zeros(hidden_size),
        var=jax.numpy.zeros(hidden_size),
    )

    for epoch in range(epochs):
        average_t = 0
        for _ in range(steps_per_epoch):
            key, batch_key = jax.random.split(key)
            x = jax.random.normal(batch_key, (batch_size, 4))

            data_sharding = NamedSharding(mesh, P(("i", "j"), None))
            weight_sharding = NamedSharding(mesh, P())
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
            f"Epoch {epoch + 1}/{epochs}, Loss {all_losses[-1]:.6f}, Average time per step {average_t:.9f}s"
        )
        save_checkpoint(model, epoch, CHECKPOINT_DIR)

    state = BatchNormState(
        training_time=False,
        mean=state.mean / (steps_per_epoch * epochs),
        var=state.var / (steps_per_epoch * epochs),
    )

    # Commit volume
    volume.commit()

    plot_losses(all_losses)
    return model, state


@app.function(
    gpu="H100:1",
)
def predict(x: jax.Array):
    print("Starting training on two nodes...")
    call = train.spawn()
    model, state = call.get()
    return model(x, state)


@app.local_entrypoint()
def main():
    x = jax.random.normal(jax.random.PRNGKey(5678), (1, 4))
    y, state = predict.remote(x)
    print("Sample input: ", x)
    print("Sample output: ", y)
