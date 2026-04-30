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
miles_cfg = exp_mod.miles if exp_mod else None

# ── Image ─────────────────────────────────────────────────────────────────────

MILES_ROOT = "/root/miles"

image = (
    modal.Image.from_registry("radixark/miles:dev-202604281249")
    .entrypoint([])
    .add_local_python_source("configs", copy=True)
    .add_local_python_source("modal_helpers", copy=True)
)
if modal_cfg:
    for patch in modal_cfg.patch_files:
        image = image.add_local_file(
            patch, f"/tmp/{os.path.basename(patch)}", copy=True
        )
    if modal_cfg.local_miles:
        image = image.add_local_dir(
            modal_cfg.local_miles,
            remote_path=MILES_ROOT,
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
        )
    if modal_cfg.image_run_commands:
        image = image.run_commands(*modal_cfg.image_run_commands)
    # Ensure system libraries (cuDNN, NCCL) take precedence over pip versions
    image = image.env({"LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"})

with image.imports():
    from ray.job_submission import JobSubmissionClient
    from modal_helpers.utils import (
        build_train_cmd,
        get_modal_cluster_context,
        prepare_miles_config,
        resolve_checkpoint_ref,
        start_ray_head,
    )

# ── Volumes ───────────────────────────────────────────────────────────────────

hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("miles-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("miles-checkpoints", create_if_missing=True)

modal_volumes = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(DATA_PATH): data_volume,
    str(CHECKPOINTS_PATH): checkpoints_volume,
}

# ── App ──────────────────────────────────────────────────────────────────────

app = modal.App(experiment)

RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265


def run_config_hook(experiment: str, hook_name: str, mounted_volumes) -> None:
    """Reload mounted volumes, run a MilesConfig hook, then commit them."""
    miles_cfg = get_module(experiment).miles
    for volume in mounted_volumes:
        volume.reload()
    getattr(miles_cfg, hook_name)()
    for volume in mounted_volumes:
        volume.commit()


@app.local_entrypoint()
def list_configs():
    """Print all available experiments."""
    _skip = {"base", "__init__"}
    names = sorted(f.stem for f in _CONFIGS_DIR.glob("*.py") if f.stem not in _skip)
    print("Available experiments:")
    for name in names:
        mod = get_module(name)
        nodes = mod.miles.total_nodes()
        gpu = f"{mod.modal.gpu}:{mod.miles.actor_num_gpus_per_node}"
        mode = "async" if mod.miles.async_mode else "sync"
        print(f"  {name:<40} {nodes} node(s) × {gpu}  ({mode})")


@app.function(
    image=image,
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_model(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Run the experiment's download_model() against the HF cache volume."""
    run_config_hook(experiment, "download_model", (hf_cache_volume,))


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}" if modal_cfg else None,
    volumes=modal_volumes,
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def post_process_model(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Run optional GPU model post-processing for the experiment."""
    run_config_hook(
        experiment,
        "post_process_model",
        (hf_cache_volume, data_volume, checkpoints_volume),
    )


@app.function(
    image=image,
    volumes={str(DATA_PATH): data_volume},
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_data(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Run the experiment's download_data() against the data volume."""
    run_config_hook(experiment, "download_data", (data_volume,))


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}" if modal_cfg else None,
    volumes=modal_volumes,
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def post_process_data(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Run optional GPU data post-processing for the experiment."""
    run_config_hook(
        experiment,
        "post_process_data",
        (hf_cache_volume, data_volume, checkpoints_volume),
    )


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}:{miles_cfg.actor_num_gpus_per_node}" if modal_cfg else None,
    volumes=modal_volumes,
    timeout=4 * 60 * 60,
    experimental_options={"efa_enabled": True},
)
@(
    modal.experimental.clustered(
        get_checkpoint_conversion_policy(miles_cfg)[0], rdma=True
    )
    if miles_cfg
    else lambda fn: fn
)
def convert_hf_to_megatron_checkpoint(
    experiment: str = os.environ.get("EXPERIMENT_CONFIG", ""),
):
    """Convert HF checkpoint to Megatron torch_dist format in raw mode."""
    miles_cfg = get_module(experiment).miles

    if getattr(miles_cfg, "megatron_to_hf_mode", None) == "bridge":
        print(f"Experiment {experiment!r} is in bridge mode — no conversion needed.")
        return

    hf_cache_volume.reload()
    checkpoints_volume.reload()

    conversion_hf_checkpoint = (
        getattr(miles_cfg, "megatron_conversion_hf_checkpoint", None)
        or miles_cfg.hf_checkpoint
    )
    hf_path = resolve_checkpoint_ref(conversion_hf_checkpoint)
    save_path = str(miles_cfg.ref_load)
    num_nodes, nproc_per_node, extra_args = get_checkpoint_conversion_policy(miles_cfg)
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
        else f"{MILES_ROOT}/tools/convert_hf_to_torch_dist.py"
    )

    cmd = (
        f"source {MILES_ROOT}/{miles_cfg.miles_model_script} && "
        f"torchrun {' '.join(torchrun_args)} {convert_script} "
        f"${{MODEL_ARGS[@]}} {' '.join(extra_args)} "
        f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
    )

    env = {**os.environ, **miles_cfg.environment}
    if num_nodes > 1:
        env["SKIP_RELEASE_RENAME"] = "1"

    print(
        f"Conversion layout for {experiment!r}: nodes={num_nodes}, "
        f"nproc_per_node={nproc_per_node}, node_rank={node_rank}"
    )
    print(
        f"Converting HF checkpoint {conversion_hf_checkpoint!r} "
        f"to Megatron torch_dist at {save_path!r}"
    )
    print(f"Running: bash -c {cmd!r}")
    subprocess.run(["bash", "-c", cmd], check=True, env=env)
    checkpoints_volume.commit()

    if node_rank == 0:
        print(f"Saved Megatron torch_dist checkpoint to {save_path}")


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}:{miles_cfg.actor_num_gpus_per_node}" if modal_cfg else None,
    memory=modal_cfg.memory if modal_cfg and modal_cfg.memory else None,
    volumes=modal_volumes,
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 60 * 60,
    experimental_options={"efa_enabled": True},
)
@(
    modal.experimental.clustered(miles_cfg.total_nodes(), rdma=True)
    if miles_cfg
    else lambda fn: fn
)
async def train(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    await asyncio.gather(
        hf_cache_volume.reload.aio(),
        data_volume.reload.aio(),
        checkpoints_volume.reload.aio(),
    )
    exp_mod = get_module(experiment)
    miles_cfg = exp_mod.miles
    modal_cfg = exp_mod.modal

    rank, master_addr, my_ip, n_nodes = get_modal_cluster_context(
        miles_cfg.total_nodes()
    )

    os.environ["MILES_HOST_IP"] = my_ip
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
    prepare_miles_config(miles_cfg, tempfile.mkdtemp())

    if (wandb_key := os.environ.get("WANDB_API_KEY", "")) and getattr(
        miles_cfg, "use_wandb", False
    ):
        miles_cfg.wandb_key = wandb_key

    cmd = build_train_cmd(miles_cfg, MILES_ROOT)
    runtime_env = {
        "env_vars": {
            "no_proxy": f"127.0.0.1,{master_addr}",
            "MASTER_ADDR": master_addr,
            **miles_cfg.environment,
        }
    }

    client = JobSubmissionClient("http://127.0.0.1:8265")
    job_id = client.submit_job(entrypoint=cmd, runtime_env=runtime_env)
    nodes = miles_cfg.total_nodes()
    gpu = f"{modal_cfg.gpu}:{miles_cfg.actor_num_gpus_per_node}"
    mode = "async" if miles_cfg.async_mode else "sync"
    print(f"Job submitted: {job_id}")
    print(f"Training {experiment:<40} {nodes} node(s) × {gpu}  ({mode})")
    print(f"Command: {cmd}, runtime_env: {runtime_env}")

    async with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
        print(f"Ray dashboard: {tunnel.url}")
        async for line in client.tail_job_logs(job_id):
            print(line, end="", flush=True)
