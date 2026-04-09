import os
import time
import jax
import optax
import equinox as eqx
import matplotlib.pyplot as plt
import modal
import modal.experimental
from model import MLP

CHECKPOINT_DIR = "/checkpoints"
LOCAL_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim()
    .pip_install("jax[cuda12]", "equinox", "matplotlib", "optax")
    .add_local_dir(LOCAL_CODE_DIR, remote_path="/root")
)
app = modal.App("jax-mlp-training", image=image)
volume = modal.Volume.from_name("jax-mlp-weights", create_if_missing=True)


def loss_fn(model, x, y_true):
    return jax.numpy.mean((y_true - model(x)) ** 2)


@eqx.filter_jit
def update(model, opt_state, x, y_true, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y_true)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


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
    optimizer = optax.adam(learning_rate)

    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    model = MLP(in_shape=4, hidden_dim=hidden_size, out_shape=1, key=model_key)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    all_losses = []

    for epoch in range(epochs):
        average_t = 0
        for step in range(steps_per_epoch):
            key, batch_key = jax.random.split(key)
            x = jax.random.normal(batch_key, (batch_size, 4))
            y_true = x**2

            t0 = time.perf_counter()
            model, opt_state, loss = update(model, opt_state, x, y_true, optimizer)
            t1 = time.perf_counter()

            average_t = t1 - t0

            all_losses.append(float(loss))

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss {all_losses[-1]:.6f}, Average time per step {average_t:.9f}s"
        )
        save_checkpoint(model, epoch, CHECKPOINT_DIR)

    plot_losses(all_losses)
    return model


if __name__ == "__main__":
    train()
