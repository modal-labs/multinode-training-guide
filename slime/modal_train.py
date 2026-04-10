import asyncio
import os
import shlex
import subprocess
import tempfile

import modal
import modal.experimental

from configs import get_module, _CONFIGS_DIR
from configs.base import HF_CACHE_PATH, DATA_PATH, CHECKPOINTS_PATH
from modal_helpers.utils import get_checkpoint_conversion_policy

# ── Experiment (client-side only — feeds decorator params) ────────────────────

experiment = os.environ.get("EXPERIMENT_CONFIG", "")
exp_mod = get_module(experiment) if experiment else None
modal_cfg = exp_mod.modal if exp_mod else None
slime_cfg = exp_mod.slime if exp_mod else None

# ── Image ─────────────────────────────────────────────────────────────────────

SLIME_ROOT = "/root/slime"

image = (
    modal.Image.from_registry(
        "slimerl/slime:nightly-dev-20260329a"
    )  # Please check https://hub.docker.com/r/slimerl/slime/tags for the latest version
    .entrypoint([])
    .add_local_python_source("configs", copy=True)
    .add_local_python_source("modal_helpers", copy=True)
)
if modal_cfg:
    for patch in modal_cfg.patch_files:
        image = image.add_local_file(
            patch, f"/tmp/{os.path.basename(patch)}", copy=True
        )
    if modal_cfg.local_slime:
        image = image.add_local_dir(
            modal_cfg.local_slime,
            remote_path=SLIME_ROOT,
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
        )
    if modal_cfg.image_run_commands:
        image = image.run_commands(*modal_cfg.image_run_commands)

with image.imports():
    from ray.job_submission import JobSubmissionClient
    from modal_helpers.utils import (
        build_train_cmd,
        get_modal_cluster_context,
        prepare_slime_config,
        start_ray_head,
    )

# ── Volumes ───────────────────────────────────────────────────────────────────

hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("slime-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("slime-checkpoints", create_if_missing=True)

modal_volumes = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(DATA_PATH): data_volume,
    str(CHECKPOINTS_PATH): checkpoints_volume,
}

# ── App ──────────────────────────────────────────────────────────────────────

app = modal.App(experiment)

RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265


@app.local_entrypoint()
def list_configs():
    """Print all available experiments."""
    _skip = {"base", "__init__"}
    names = sorted(f.stem for f in _CONFIGS_DIR.glob("*.py") if f.stem not in _skip)
    print("Available experiments:")
    for name in names:
        mod = get_module(name)
        nodes = mod.slime.total_nodes()
        gpu = f"{mod.modal.gpu}:{mod.slime.actor_num_gpus_per_node}"
        mode = "async" if mod.slime.async_mode else "sync"
        print(f"  {name:<40} {nodes} node(s) × {gpu}  ({mode})")


@app.function(
    image=image,
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_model(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Download the model to the HF cache volume."""
    from huggingface_hub import snapshot_download

    slime_cfg = get_module(experiment).slime
    _ = snapshot_download(repo_id=slime_cfg.hf_checkpoint)
    hf_cache_volume.commit()


@app.function(
    image=image,
    volumes={str(DATA_PATH): data_volume},
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def prepare_dataset(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Run the prepare_data() to populate the data volume."""
    slime_cfg = get_module(experiment).slime
    data_volume.reload()
    slime_cfg.prepare_data()
    data_volume.commit()


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}:{slime_cfg.actor_num_gpus_per_node}" if modal_cfg else None,
    volumes=modal_volumes,
    timeout=4 * 60 * 60,
    experimental_options={"efa_enabled": True},
)
@(
    modal.experimental.clustered(
        get_checkpoint_conversion_policy(slime_cfg)[0], rdma=True
    )
    if slime_cfg
    else lambda fn: fn
)
def convert_checkpoint(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Convert HF checkpoint to torch_dist format when megatron_to_hf_mode is raw."""
    from huggingface_hub import snapshot_download

    slime_cfg = get_module(experiment).slime

    if getattr(slime_cfg, "megatron_to_hf_mode", None) == "bridge":
        print(f"Experiment {experiment!r} is in bridge mode — no conversion needed.")
        return

    hf_cache_volume.reload()
    checkpoints_volume.reload()

    hf_path = snapshot_download(slime_cfg.hf_checkpoint, local_files_only=True)
    save_path = str(slime_cfg.ref_load)
    num_nodes, nproc_per_node, extra_args = get_checkpoint_conversion_policy(slime_cfg)
    node_rank, master_addr, _, nnodes = get_modal_cluster_context(num_nodes)

    torchrun_args = [f"--nproc-per-node={nproc_per_node}"]
    if nnodes > 1:
        torchrun_args += [
            f"--nnodes={nnodes}",
            f"--node-rank={node_rank}",
            f"--master-addr={master_addr}",
            "--master-port=12355",
        ]

    # For multi-node, use our wrapper that honours SKIP_RELEASE_RENAME to
    # prevent volume corruption (see modal_helpers/convert_hf_to_torch_dist.py).
    # Single-node uses the upstream script directly.
    import importlib.util
    convert_script = (
        importlib.util.find_spec("modal_helpers.convert_hf_to_torch_dist").origin
        if num_nodes > 1
        else f"{SLIME_ROOT}/tools/convert_hf_to_torch_dist.py"
    )

    cmd = (
        f"source {SLIME_ROOT}/{slime_cfg.slime_model_script} && "
        f"torchrun {' '.join(torchrun_args)} {convert_script} "
        f"${{MODEL_ARGS[@]}} {' '.join(extra_args)} "
        f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
    )

    env = {**os.environ, **slime_cfg.environment}
    if num_nodes > 1:
        env["SKIP_RELEASE_RENAME"] = "1"

    print(
        f"Conversion layout for {experiment!r}: nodes={num_nodes}, "
        f"nproc_per_node={nproc_per_node}, node_rank={node_rank}"
    )
    print(f"Running: bash -c {cmd!r}")
    subprocess.run(["bash", "-c", cmd], check=True, env=env)
    checkpoints_volume.commit()

    if node_rank == 0:
        print(f"Saved torch_dist checkpoint to {save_path}")


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
    if slime_cfg
    else lambda fn: fn
)
async def train(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    await asyncio.gather(
        hf_cache_volume.reload.aio(),
        data_volume.reload.aio(),
        checkpoints_volume.reload.aio(),
    )
    exp_mod = get_module(experiment)
    slime_cfg = exp_mod.slime
    modal_cfg = exp_mod.modal

    rank, master_addr, my_ip, n_nodes = get_modal_cluster_context(
        slime_cfg.total_nodes()
    )

    os.environ["SLIME_HOST_IP"] = my_ip
    os.environ["SGLANG_HOST_IP"] = my_ip
    os.environ["HOST_IP"] = my_ip

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

    # Head node: start Ray, prepare config, submit job, stream logs.
    start_ray_head(my_ip, n_nodes)
    prepare_slime_config(slime_cfg, tempfile.mkdtemp())

    if (wandb_key := os.environ.get("WANDB_API_KEY", "")) and getattr(
        slime_cfg, "use_wandb", False
    ):
        slime_cfg.wandb_key = wandb_key

    cmd = build_train_cmd(slime_cfg, SLIME_ROOT)
    runtime_env = {
        "env_vars": {
            "no_proxy": f"127.0.0.1,{master_addr}",
            "MASTER_ADDR": master_addr,
            **slime_cfg.environment,
        }
    }

    client = JobSubmissionClient("http://127.0.0.1:8265")
    job_id = client.submit_job(entrypoint=cmd, runtime_env=runtime_env)
    nodes = slime_cfg.total_nodes()
    gpu = f"{modal_cfg.gpu}:{slime_cfg.actor_num_gpus_per_node}"
    mode = "async" if slime_cfg.async_mode else "sync"
    print(f"Job submitted: {job_id}")
    print(f"Training {experiment:<40} {nodes} node(s) × {gpu}  ({mode})")
    print(f"Command: {cmd}, runtime_env: {runtime_env}")

    async with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
        print(f"Ray dashboard: {tunnel.url}")
        async for line in client.tail_job_logs(job_id):
            print(line, end="", flush=True)
