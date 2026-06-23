import asyncio
import os
import subprocess

import modal
import modal.experimental

from configs import get_module, _CONFIGS_DIR
from configs.base import (
    CHECKPOINTS_PATH,
    HF_CACHE_PATH,
    NEMO_RL_ROOT,
    ModalConfig,
)

# ── Experiment (client-side only — feeds decorator params) ────────────────────

experiment = os.environ.get("EXPERIMENT_CONFIG", "")
exp_mod = get_module(experiment) if experiment else None
modal_cfg = exp_mod.modal if exp_mod else None
nemo_rl_cfg = exp_mod.nemo_rl if exp_mod else None

# ── Image ─────────────────────────────────────────────────────────────────────

image = (
    modal.Image.from_registry(
        modal_cfg.docker_image if modal_cfg else ModalConfig.docker_image
    )
    .entrypoint([])
    .add_local_python_source("configs", copy=True)
    .add_local_python_source("modal_helpers", copy=True)
)
if modal_cfg:
    if modal_cfg.local_nemo_rl:
        image = image.add_local_dir(
            modal_cfg.local_nemo_rl,
            remote_path=NEMO_RL_ROOT,
            copy=True,
            ignore=[
                "**/__pycache__",
                "**/*.pyc",
                "**/.git",
                "**/.venv",
                "**/results",
                "**/logs",
            ],
        )
    if modal_cfg.image_run_commands:
        image = image.run_commands(*modal_cfg.image_run_commands)
    if modal_cfg.image_env:
        image = image.env(modal_cfg.image_env)

with image.imports():
    from modal_helpers.utils import (
        build_train_cmd,
        cluster_driver_env,
        get_modal_cluster_context,
        start_ray_head,
        start_ray_worker,
    )

# ── Volumes ───────────────────────────────────────────────────────────────────

hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name(
    "nemo-rl-checkpoints", create_if_missing=True
)

modal_volumes = {
    str(HF_CACHE_PATH): hf_cache_volume,
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
        cfg = mod.nemo_rl
        gpu = f"{mod.modal.gpu}:{cfg.gpus_per_node}"
        print(
            f"  {name:<40} {cfg.num_nodes} node(s) × {gpu}  "
            f"({os.path.basename(cfg.entrypoint)})"
        )


@app.function(
    image=image,
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_model(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Prefetch the recipe's model into the mounted HF cache."""
    from huggingface_hub import snapshot_download

    cfg = get_module(experiment).nemo_rl
    model_id = cfg.model_id()
    if not model_id:
        raise ValueError(
            f"{experiment!r}: set hf_model or overrides['policy.model_name'] to download a model"
        )
    hf_cache_volume.reload()
    print(f"Downloading model {model_id!r} into the HF cache...")
    snapshot_download(model_id)
    hf_cache_volume.commit()


@app.function(
    image=image,
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_data(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    # fetch dataset from huggingface (optional, nemorl will do this automatically)
    cfg = get_module(experiment).nemo_rl
    if not cfg.hf_datasets:
        print(f"{experiment!r} declares no hf_datasets; nothing to prefetch.")
        return

    from datasets import load_dataset

    hf_cache_volume.reload()
    for repo_id in cfg.hf_datasets:
        print(f"Prefetching dataset {repo_id!r}...")
        load_dataset(repo_id)
    hf_cache_volume.commit()



@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}:{nemo_rl_cfg.gpus_per_node}" if modal_cfg else None,
    memory=modal_cfg.memory if modal_cfg and modal_cfg.memory else None,
    cloud=modal_cfg.cloud if modal_cfg and modal_cfg.cloud else None,
    region=modal_cfg.region if modal_cfg and modal_cfg.region else None,
    volumes=modal_volumes,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=24 * 60 * 60,
    experimental_options={"efa_enabled": True},
)
@(
    modal.experimental.clustered(nemo_rl_cfg.total_nodes(), rdma=True)
    if nemo_rl_cfg
    else lambda fn: fn
)
async def train(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    await asyncio.gather(
        hf_cache_volume.reload.aio(),
        checkpoints_volume.reload.aio(),
    )
    exp_mod = get_module(experiment)
    cfg = exp_mod.nemo_rl
    modal_cfg = exp_mod.modal

    rank, head_ip, my_ip, n_nodes, cluster_ips = get_modal_cluster_context(
        cfg.total_nodes()
    )

    # on all ranks that are not the head node, do ray --start and spin
    if rank != 0:
        
        start_ray_worker(head_ip, RAY_PORT, my_ip, cfg.gpus_per_node, rank)
        while True:
            await asyncio.sleep(10)

    # on the head node, start ray and wait for all nodes to join 
    start_ray_head(head_ip, RAY_PORT, n_nodes, cfg.gpus_per_node, node_rank=rank)
    

    cmd = build_train_cmd(cfg, NEMO_RL_ROOT, experiment)
    env = {
        **os.environ,
        "RAY_ADDRESS": f"{head_ip}:{RAY_PORT}",
        **cluster_driver_env(head_ip, cluster_ips),
        **cfg.environment,
    }

    gpu = f"{modal_cfg.gpu}:{cfg.gpus_per_node}"
    print(f"Training {experiment:<40} {n_nodes} node(s) × {gpu}")
    print(f"Command: {cmd}")

    async with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
        print(f"Ray dashboard: {tunnel.url}")
        result = subprocess.run(["bash", "-c", cmd], env=env)

    await checkpoints_volume.commit.aio()
    if result.returncode != 0:
        raise RuntimeError(f"NeMo-RL driver exited with code {result.returncode}")
