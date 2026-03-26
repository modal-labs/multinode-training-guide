import asyncio
import os
import shlex
import subprocess
import time

import modal
import modal.experimental

from configs import get_module
from configs.base import HF_CACHE_PATH, DATA_PATH, CHECKPOINTS_PATH

# ── Experiment ────────────────────────────────────────────────────────────────

experiment = os.environ.get("EXPERIMENT_CONFIG", "")
exp_mod = get_module(experiment) if experiment else None
modal_cfg = exp_mod.modal if exp_mod else None
slime_cfg = exp_mod.slime if exp_mod else None

# ── Image ─────────────────────────────────────────────────────────────────────

SLIME_ROOT = "/root/slime"

image = (
    modal.Image.from_registry("slimerl/slime:nightly-dev-20260326a")
    .entrypoint([])
    .add_local_python_source("configs", copy=True)
)
if modal_cfg:
    if modal_cfg.image_run_commands:
        image = image.run_commands(*modal_cfg.image_run_commands)
    if modal_cfg.local_slime:
        image = image.add_local_dir(
            modal_cfg.local_slime,
            remote_path=SLIME_ROOT,
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
        )

with image.imports():
    import ray
    from ray.job_submission import JobSubmissionClient

# ── Volumes ───────────────────────────────────────────────────────────────────

hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("slime-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("slime-checkpoints", create_if_missing=True)

modal_volumes = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(DATA_PATH): data_volume,
    str(CHECKPOINTS_PATH): checkpoints_volume,
}

# ── Modal App ─────────────────────────────────────────────────────────────────

app = modal.App(experiment)

# ── Utilities ─────────────────────────────────────────────────────────────────


@app.local_entrypoint()
def list_configs():
    """Print all available experiments."""
    from configs import CONFIGS

    print("Available experiments:")
    for name in sorted(CONFIGS):
        mod = CONFIGS[name]
        nodes = mod.slime.total_nodes()
        gpu = f"{mod.modal.gpu}:{mod.slime.actor_num_gpus_per_node}"
        print(f"  {name:<24} {nodes} node(s) × {gpu}")


@app.function(
    image=image,
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
    timeout=2 * 60 * 60,
)
def download_model(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Download the experiment model to the HF cache volume."""
    from huggingface_hub import snapshot_download

    slime_cfg = get_module(experiment).slime
    path = snapshot_download(repo_id=slime_cfg.hf_checkpoint)
    print(f"Downloaded to {path}")
    hf_cache_volume.commit()


@app.function(
    image=image,
    volumes={str(DATA_PATH): data_volume},
    timeout=2 * 60 * 60,
)
def prepare_dataset(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Run the experiment's prepare_data() to populate the data volume."""
    exp_mod = get_module(experiment)
    if not hasattr(exp_mod, "prepare_data"):
        raise RuntimeError(f"Experiment {experiment!r} has no prepare_data() function.")
    data_volume.reload()
    exp_mod.prepare_data()
    data_volume.commit()


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}:{slime_cfg.actor_num_gpus_per_node}" if modal_cfg else None,
    volumes=modal_volumes,
    timeout=4 * 60 * 60,
)
def convert_checkpoint(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Convert HF checkpoint to torch_dist format when megatron_to_hf_mode is raw."""
    from huggingface_hub import snapshot_download

    slime_cfg = get_module(experiment).slime

    if getattr(slime_cfg, "megatron_to_hf_mode", None) == "bridge":
        print(
            f"Experiment {experiment!r} is in bridge mode — no torch_dist conversion needed."
        )
        return

    if not slime_cfg.slime_model_script:
        raise RuntimeError(
            f"Experiment {experiment!r} has no slime_model_script set in SlimeConfig."
        )

    hf_cache_volume.reload()
    checkpoints_volume.reload()

    hf_path = snapshot_download(slime_cfg.hf_checkpoint, local_files_only=True)
    save_path = str(slime_cfg.ref_load)
    tp = getattr(slime_cfg, "tensor_model_parallel_size", 1)
    pp = getattr(slime_cfg, "pipeline_model_parallel_size", 1)
    decoder_first = getattr(slime_cfg, "decoder_first_pipeline_num_layers", None)
    decoder_last = getattr(slime_cfg, "decoder_last_pipeline_num_layers", None)
    mtp_num_layers = getattr(slime_cfg, "mtp_num_layers", None)

    if tp == 1:
        nproc = slime_cfg.actor_num_gpus_per_node
        parallel_args = ""
    else:
        nproc = tp * pp
        parallel_args = (
            f"--tensor-model-parallel-size {tp} "
            f"--pipeline-model-parallel-size {pp} "
            + (
                f"--decoder-first-pipeline-num-layers {decoder_first} "
                if decoder_first
                else ""
            )
            + (
                f"--decoder-last-pipeline-num-layers {decoder_last} "
                if decoder_last
                else ""
            )
        )

    mtp_arg = f"--mtp-num-layers {mtp_num_layers} " if mtp_num_layers else ""

    cmd = (
        f"source {SLIME_ROOT}/{slime_cfg.slime_model_script} && "
        f"torchrun --nproc-per-node={nproc} {SLIME_ROOT}/tools/convert_hf_to_torch_dist.py "
        f"${{MODEL_ARGS[@]}} {parallel_args}{mtp_arg}"
        f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
    )
    print(f"Running: bash -c {cmd!r}")
    subprocess.run(
        ["bash", "-c", cmd],
        check=True,
        env={**os.environ, **slime_cfg.environment},
    )
    checkpoints_volume.commit()
    print(f"Saved torch_dist checkpoint to {save_path}")


# ── Train ─────────────────────────────────────────────────────────────────────

RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265


def _get_cluster_info(slime_cfg) -> tuple[int, str, str, int]:
    """Return (rank, master_addr, my_ip, n_nodes) for this container."""
    if slime_cfg.total_nodes() > 1:
        info = modal.experimental.get_cluster_info()
        ips = info.container_ipv4_ips
        rank = info.rank
        return rank, ips[0], ips[rank], len(ips)
    return 0, "127.0.0.1", "127.0.0.1", 1


def _start_ray_head(my_ip: str, n_nodes: int) -> None:
    """Start Ray head node and wait for all workers to join."""
    subprocess.Popen(
        [
            "ray",
            "start",
            "--head",
            f"--node-ip-address={my_ip}",
            "--dashboard-host=0.0.0.0",
        ]
    )
    for _ in range(30):
        try:
            ray.init(address="auto")
            break
        except ConnectionError:
            time.sleep(1)
    else:
        raise RuntimeError("Ray head node failed to start")

    for _ in range(60):
        alive = [n for n in ray.nodes() if n["Alive"]]
        print(f"Waiting for workers: {len(alive)}/{n_nodes} alive")
        if len(alive) == n_nodes:
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"Timed out waiting for all {n_nodes} Ray nodes to join")


def _resolve_hf_paths(slime_cfg) -> None:
    """Resolve any HF repo IDs in slime_cfg checkpoint fields to local cache paths."""
    from huggingface_hub import snapshot_download

    def _resolve(val: str) -> str:
        return (
            val
            if str(val).startswith("/")
            else snapshot_download(val, local_files_only=True)
        )

    for attr in ("hf_checkpoint", "load", "ref_load", "critic_load"):
        if val := getattr(slime_cfg, attr, None):
            setattr(slime_cfg, attr, _resolve(val))


def _build_train_cmd(slime_cfg) -> str:
    """Build the Ray job entrypoint, sourcing model arch args if slime_model_script is set."""
    train_script = (
        f"{SLIME_ROOT}/{'train_async.py' if slime_cfg.async_mode else 'train.py'}"
    )
    if slime_cfg.slime_model_script:
        # Ray job runner uses /bin/sh; wrap in bash for `source` and array expansion.
        inner = (
            f"source {SLIME_ROOT}/{slime_cfg.slime_model_script} && "
            f"python3 {train_script} ${{MODEL_ARGS[@]}} {shlex.join(slime_cfg.cli_args())}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return f"python3 {train_script} {shlex.join(slime_cfg.cli_args())}"


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}:{slime_cfg.actor_num_gpus_per_node}" if modal_cfg else None,
    volumes=modal_volumes,
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 60 * 60,
    experimental_options={"efa_enabled": True},
)
@(
    modal.experimental.clustered(slime_cfg.total_nodes(), rdma=True)
    if slime_cfg and slime_cfg.total_nodes() > 1
    else lambda fn: fn
)
async def train(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    await asyncio.gather(hf_cache_volume.reload.aio(), data_volume.reload.aio())
    slime_cfg = get_module(experiment).slime

    rank, master_addr, my_ip, n_nodes = _get_cluster_info(slime_cfg)
    os.environ["SLIME_HOST_IP"] = my_ip

    if rank != 0:
        # Worker node: join the Ray cluster and keep the container alive.
        subprocess.Popen(
            [
                "ray",
                "start",
                f"--node-ip-address={my_ip}",
                "--address",
                f"{master_addr}:{RAY_PORT}",
            ]
        )
        while True:
            await asyncio.sleep(10)

    # Head node: start Ray, submit the training job, and stream logs.
    _start_ray_head(my_ip, n_nodes)
    _resolve_hf_paths(slime_cfg)

    # wandb-secret injects WANDB_API_KEY; falls back to local env or disabled.
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key and getattr(slime_cfg, "use_wandb", False):
        slime_cfg.wandb_key = wandb_key

    cmd = _build_train_cmd(slime_cfg)
    runtime_env = {
        "env_vars": {
            "no_proxy": f"127.0.0.1,{master_addr}",
            "MASTER_ADDR": master_addr,
            **slime_cfg.environment,
        }
    }

    client = JobSubmissionClient("http://127.0.0.1:8265")
    job_id = client.submit_job(entrypoint=cmd, runtime_env=runtime_env)
    print(f"Job submitted: {job_id}")
    print(f"Command: {cmd}")

    async with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
        print(f"Ray dashboard: {tunnel.url}")
        async for line in client.tail_job_logs(job_id):
            print(line, end="", flush=True)
