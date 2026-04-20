import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress GRPC errors

import re
import time

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from models import RNN


LOCAL_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(LOCAL_CODE_DIR, "moby_dick_cetology.txt")


def load_dataset(path: str = DATASET_PATH) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return stoi, itos


def encode(text: str, stoi: dict) -> jax.Array:
    return jnp.array([stoi[c] for c in text], dtype=jnp.int32)


def decode(tokens, itos: dict) -> str:
    return "".join(itos[int(t)] for t in tokens)


def sample_batch(key, data: jax.Array, batch_size: int, seq_len: int):
    """Sample `batch_size` random windows and split into (inputs, targets),
    where targets are inputs shifted by one position."""
    max_start = data.shape[0] - seq_len - 1
    starts = jax.random.randint(key, (batch_size,), 0, max_start)
    idx = starts[:, None] + jnp.arange(seq_len)[None, :]
    x = data[idx]
    y = data[idx + 1]
    return x, y


def run_sequence(model: RNN, xs: jax.Array, hidden=None):
    """Scan the RNN cell over a sequence.

    xs: (T, in_shape). Returns (outs, final_hidden) with outs: (T, out_shape).
    """
    if hidden is None:
        hidden = model.init_hidden()

    def step(h, x):
        y, h = model(x, h)
        return h, y

    final_hidden, outs = jax.lax.scan(step, hidden, xs)
    return outs, final_hidden


def loss_fn(model: RNN, x: jax.Array, y: jax.Array, vocab_size: int) -> jax.Array:
    """Next-character cross-entropy loss.

    x, y: (B, T) int tokens, where y[b, t] is the character following x[b, t].
    """
    x_onehot = jax.nn.one_hot(x, vocab_size)

    def forward(seq):
        logits, _ = run_sequence(model, seq)
        return logits

    logits = jax.vmap(forward)(x_onehot)  # (B, T, V)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    return -target_log_probs.mean()


@eqx.filter_jit
def update(model, opt_state, x, y, optimizer, vocab_size):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, vocab_size)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def generate(model: RNN, prompt: str, stoi: dict, itos: dict, length: int, key):
    vocab_size = len(stoi)
    tokens = [stoi[c] for c in prompt]

    prompt_onehot = jax.nn.one_hot(jnp.array(tokens), vocab_size)
    _, hidden = run_sequence(model, prompt_onehot)

    out = list(tokens)
    last_token = tokens[-1]
    for _ in range(length):
        x = jax.nn.one_hot(jnp.array(last_token), vocab_size)
        logits, hidden = model(x, hidden)
        key, sub = jax.random.split(key)
        last_token = int(jax.random.categorical(sub, logits))
        out.append(last_token)

    return decode(out, itos)


def save_checkpoint(model, epoch, path):
    eqx.tree_serialise_leaves(os.path.join(path, f"model_epoch_{epoch}.eqx"), model)


def load_checkpoint(model, filepath):
    return eqx.tree_deserialise_leaves(filepath, model)


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
    plt.ylabel("Loss (cross-entropy)")
    plt.title("RNN Training Loss")
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
    dataset_path: str = DATASET_PATH,
    learning_rate: float = 3e-3,
    batch_size: int = 64,
    seq_len: int = 128,
    epochs: int = 25,
    steps_per_epoch: int = 200,
    hidden_dim: int = 512,
    seed: int = 5678,
):
    """Run the full training loop. Assumes `jax.distributed` is already
    initialized and that `mesh` is a ready-to-use `jax.Mesh`.

    Writes per-epoch checkpoints to `checkpoint_dir`. Returns
    `(model, stoi, itos)`.
    """
    text = load_dataset(dataset_path)
    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)
    data = encode(text, stoi)
    print(f"vocab_size={vocab_size}, dataset_len={data.shape[0]}")

    optimizer = optax.adam(learning_rate)

    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    model = RNN(
        in_shape=vocab_size,
        hidden_dim=hidden_dim,
        out_shape=vocab_size,
        key=model_key,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Shard batch across all devices; replicate weights.
    data_sharding = NamedSharding(mesh, P(("i", "j"), None))
    weight_sharding = NamedSharding(mesh, P())

    all_losses = []

    # Explicit-mode meshes need to be entered so NamedSharding pspecs can
    # resolve against them inside jit'd + scan'd computations.
    with jax.sharding.set_mesh(mesh):
        for epoch in range(epochs):
            average_t = 0
            for _ in range(steps_per_epoch):
                key, batch_key = jax.random.split(key)
                x, y = sample_batch(batch_key, data, batch_size, seq_len)

                global_x = jax.device_put(x, data_sharding)
                global_y = jax.device_put(y, data_sharding)

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

                model, opt_state, loss = update(
                    model, opt_state, global_x, global_y, optimizer, vocab_size
                )
                t1 = time.perf_counter()

                average_t = (t1 - t0) / nproc

                all_losses.append(float(loss))

            print(
                f"Epoch {epoch + 1}/{epochs}, Loss {all_losses[-1]:.6f}, "
                f"Average time per step {average_t:.9f}s"
            )
            if jax.process_index() == 0:
                save_checkpoint(model, epoch, checkpoint_dir)

    if jax.process_index() == 0:
        plot_losses(all_losses, os.path.join(checkpoint_dir, "loss_curve_rnn.png"))
    return model, stoi, itos
