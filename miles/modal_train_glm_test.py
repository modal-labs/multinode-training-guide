"""Dedicated 5-layer GLM-5.2 LoRA smoke-test launcher.

glm5_2_744b_a40b_lora_5layer run on 1x8h200

Run:
    uv run modal run miles/modal_train_glm_test.py::download_model
    uv run modal run miles/modal_train_glm_test.py::download_data
    uv run modal run -d miles/modal_train_glm_test.py::train
"""

import asyncio
import os
import tempfile

import modal

from configs import get_module
from configs.base import HF_CACHE_PATH, DATA_PATH, CHECKPOINTS_PATH

_ALLOWED_EXPERIMENTS = {
    "glm5_2_744b_a40b_lora_5layer",
}
EXPERIMENT = os.environ.get("EXPERIMENT_CONFIG", "glm5_2_744b_a40b_lora_5layer")
if EXPERIMENT not in _ALLOWED_EXPERIMENTS:
    raise ValueError(f"This launcher only supports 5-layer GLM diagnostics; got {EXPERIMENT!r}")

exp_mod = get_module(EXPERIMENT)
modal_cfg = exp_mod.modal
miles_cfg = exp_mod.miles

MILES_ROOT = "/root/miles"

image = (
    modal.Image.from_registry(modal_cfg.docker_image)
    .entrypoint([])
    .add_local_python_source("configs", copy=True)
    .add_local_python_source("modal_helpers", copy=True)
)
for patch in modal_cfg.patch_files:
    image = image.add_local_file(
        patch, f"/tmp/{os.path.basename(patch)}", copy=True
    )
if modal_cfg.image_run_commands:
    image = image.run_commands(*modal_cfg.image_run_commands)
if modal_cfg.image_env:
    image = image.env(modal_cfg.image_env)

with image.imports():
    from ray.job_submission import JobSubmissionClient
    from modal_helpers.utils import (
        build_train_cmd,
        prepare_miles_config,
        start_ray_head,
    )

hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("miles-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("miles-checkpoints", create_if_missing=True)

modal_volumes = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(DATA_PATH): data_volume,
    str(CHECKPOINTS_PATH): checkpoints_volume,
}

app = modal.App(f"{EXPERIMENT}-test")

RAY_DASHBOARD_PORT = 8265


def run_config_hook(experiment: str, hook_name: str, mounted_volumes) -> None:
    cfg = get_module(experiment).miles
    for volume in mounted_volumes:
        volume.reload()
    getattr(cfg, hook_name)()
    for volume in mounted_volumes:
        volume.commit()


@app.function(
    image=image,
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_model(experiment: str = EXPERIMENT):
    run_config_hook(experiment, "download_model", (hf_cache_volume,))


@app.function(
    image=image,
    volumes={str(DATA_PATH): data_volume},
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_data(experiment: str = EXPERIMENT):
    run_config_hook(experiment, "download_data", (data_volume,))


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}:{miles_cfg.actor_num_gpus_per_node}",
    memory=modal_cfg.memory if modal_cfg.memory else None,
    cloud=modal_cfg.cloud if modal_cfg.cloud else None,
    region=modal_cfg.region if modal_cfg.region else None,
    volumes=modal_volumes,
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 60 * 60,
)
async def train(experiment: str = EXPERIMENT):
    await asyncio.gather(
        hf_cache_volume.reload.aio(),
        data_volume.reload.aio(),
        checkpoints_volume.reload.aio(),
    )
    exp_mod = get_module(experiment)
    cfg = exp_mod.miles
    my_ip = "127.0.0.1"
    os.environ["MILES_HOST_IP"] = my_ip
    os.environ["SGLANG_HOST_IP"] = my_ip
    os.environ["HOST_IP"] = my_ip

    start_ray_head(my_ip, 1)
    prepare_miles_config(cfg, tempfile.mkdtemp())

    cmd = build_train_cmd(cfg, MILES_ROOT)
    runtime_env = {
        "env_vars": {
            "no_proxy": f"127.0.0.1,{my_ip}",
            "MASTER_ADDR": my_ip,
            **cfg.environment,
        }
    }

    client = JobSubmissionClient("http://127.0.0.1:8265")
    job_id = client.submit_job(entrypoint=cmd, runtime_env=runtime_env)
    print(f"Job submitted: {job_id}")
    print(f"Training {experiment} on 1 node x {exp_mod.modal.gpu}:{cfg.actor_num_gpus_per_node}")
    print(f"Command: {cmd}")

    async with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
        print(f"Ray dashboard: {tunnel.url}")
        async for line in client.tail_job_logs(job_id):
            print(line, end="", flush=True)
