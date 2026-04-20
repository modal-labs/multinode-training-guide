from __future__ import annotations

import os

import modal
import modal.experimental


CHECKPOINT_DIR = "/checkpoints"
LOCAL_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .pip_install(
        "jax[cuda12]==0.8.2",
        "equinox==0.13.6",
        "optax==0.2.6",
        "matplotlib~=3.10.8",
    )
    .add_local_dir(LOCAL_CODE_DIR, remote_path="/root")
)

with image.imports():
    import jax

    import train_mlp as mlp_lib
    import train_rnn as rnn_lib
    from models import MLP, BatchNormState, RNN

app = modal.App("jax-training", image=image)

mlp_volume = modal.Volume.from_name("jax-mlp-weights", create_if_missing=True)
rnn_volume = modal.Volume.from_name("jax-rnn-weights", create_if_missing=True)


def _init_distributed_mesh():
    """Initialize `jax.distributed` using Modal cluster info and return
    `(mesh, nproc)` for an `(8, nproc)` device mesh named `("i", "j")`."""
    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    master_addr = cluster_info.container_ipv4_ips[0]
    nproc = len(cluster_info.container_ipv4_ips)

    jax.distributed.initialize(f"{master_addr}:12345", nproc, node_rank)
    print(f"Number of devices: {len(jax.devices())}")

    mesh = jax.make_mesh((8, nproc), ("i", "j"))
    print(f"Mesh: {mesh}")
    return mesh, nproc


@app.function(
    gpu="H100:8",
    volumes={CHECKPOINT_DIR: mlp_volume},
    timeout=60 * 60,
)
@modal.experimental.clustered(size=2, rdma=True)
def train_mlp(
    learning_rate: float = 3e-4,
    batch_size: int = 32,
    epochs: int = 25,
    steps_per_epoch: int = 100,
    hidden_size: int = 64,
    seed: int = 5678,
):
    mesh, nproc = _init_distributed_mesh()
    model, state = mlp_lib.train_loop(
        checkpoint_dir=CHECKPOINT_DIR,
        mesh=mesh,
        nproc=nproc,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        hidden_size=hidden_size,
        seed=seed,
    )
    mlp_volume.commit()
    return model, state


@app.function(
    gpu="H100:1",
    volumes={CHECKPOINT_DIR: mlp_volume},
)
def predict_mlp(seed: int = 5678, hidden_size: int = 64):
    """Load the latest MLP checkpoint from the volume and run a forward pass
    on a random Gaussian input seeded with `seed`.

    `hidden_size` must match what the checkpoint was trained with.
    """
    mlp_volume.reload()
    ckpt_path, epoch = mlp_lib.find_latest_checkpoint(CHECKPOINT_DIR)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {CHECKPOINT_DIR}. "
            f"Run `modal run jax/modal_train.py::mlp_train` first."
        )
    print(f"Loading checkpoint: {ckpt_path} (epoch {epoch + 1})")

    template_model = MLP(
        in_shape=4, hidden_dim=hidden_size, out_shape=4, key=jax.random.PRNGKey(0)
    )
    template_state = BatchNormState(
        training_time=False,
        mean=jax.numpy.zeros(hidden_size),
        var=jax.numpy.zeros(hidden_size),
    )
    model, state = mlp_lib.load_checkpoint(template_model, template_state, ckpt_path)

    x = jax.random.normal(jax.random.PRNGKey(seed), (1, 4))
    y, state = model(x, state)
    return x, y


@app.local_entrypoint()
def mlp_train():
    """Launch MLP training on the cluster."""
    print("Starting MLP training on two nodes...")
    train_mlp.remote()
    print("Training finished. Run `mlp_sample` to sample.")


@app.local_entrypoint()
def mlp_sample():
    """Load latest MLP checkpoint and run a single-sample forward pass."""
    x, y = predict_mlp.remote()
    print("Sample input:", x)
    print("Sample output:", y)


@app.function(
    gpu="H100:8",
    volumes={CHECKPOINT_DIR: rnn_volume},
    timeout=60 * 60,
)
@modal.experimental.clustered(size=2, rdma=True)
def train_rnn(
    dataset_path: str | None = None,
    learning_rate: float = 3e-3,
    batch_size: int = 64,
    seq_len: int = 128,
    epochs: int = 25,
    steps_per_epoch: int = 200,
    hidden_dim: int = 512,
    seed: int = 5678,
):
    if dataset_path is None:
        dataset_path = rnn_lib.DATASET_PATH

    mesh, nproc = _init_distributed_mesh()
    model, stoi, itos = rnn_lib.train_loop(
        checkpoint_dir=CHECKPOINT_DIR,
        mesh=mesh,
        nproc=nproc,
        dataset_path=dataset_path,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seq_len=seq_len,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        hidden_dim=hidden_dim,
        seed=seed,
    )
    rnn_volume.commit()
    return model, stoi, itos


@app.function(
    gpu="H100:1",
    volumes={CHECKPOINT_DIR: rnn_volume},
    timeout=60 * 60,
)
def predict_rnn(
    prompt: str = "The whale ",
    length: int = 500,
    seed: int = 0,
    hidden_dim: int = 512,
    dataset_path: str | None = None,
):
    """Load the latest RNN checkpoint from the volume and sample from it.

    `hidden_dim` must match what the checkpoint was trained with.
    """
    if dataset_path is None:
        dataset_path = rnn_lib.DATASET_PATH

    rnn_volume.reload()
    ckpt_path, epoch = rnn_lib.find_latest_checkpoint(CHECKPOINT_DIR)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {CHECKPOINT_DIR}. "
            f"Run `modal run jax/modal_train.py::rnn_train` first."
        )
    print(f"Loading checkpoint: {ckpt_path} (epoch {epoch + 1})")

    text = rnn_lib.load_dataset(dataset_path)
    stoi, itos = rnn_lib.build_vocab(text)
    vocab_size = len(stoi)

    template = RNN(
        in_shape=vocab_size,
        hidden_dim=hidden_dim,
        out_shape=vocab_size,
        key=jax.random.PRNGKey(0),
    )
    model = rnn_lib.load_checkpoint(template, ckpt_path)

    key = jax.random.PRNGKey(seed)
    return rnn_lib.generate(
        model, prompt=prompt, stoi=stoi, itos=itos, length=length, key=key
    )


@app.local_entrypoint()
def rnn_train():
    """Launch RNN training on the cluster."""
    print("Starting RNN training on two nodes...")
    train_rnn.remote()
    print("Training finished. Run `rnn_sample` to sample.")


@app.local_entrypoint()
def rnn_sample():
    """Load latest RNN checkpoint and print a generated sample."""
    sample = predict_rnn.remote(prompt="The whale ", length=500)
    print("Sample output:")
    print(sample)
