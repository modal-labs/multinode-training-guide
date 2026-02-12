"""Serve a SLIME-trained model with vLLM."""

from pathlib import Path
import aiohttp
import modal

app = modal.App("serve-haiku-model")

MODELS_PATH: Path = Path("/models")
MODEL_PATH = "Qwen3-4B-singlenode-20260211-161245"  # Update this!
BASE_MODEL_NAME = "Qwen3-0.6B"

ITERS_DIR = "iter_0000049"
HF_DIR = "hf"

MODEL_NAME = "slime-qwen"
N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000


# Same volume used in training
checkpoints_volume: modal.Volume = modal.Volume.from_name("grpo-slime-haiku-checkpoints")
hf_cache_vol = modal.Volume.from_name("huggingface-cache")
vllm_cache_vol = modal.Volume.from_name("vllm-cache")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.2",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.2",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

slime_image = (
    modal.Image.from_registry("slimerl/slime:nightly-dev-20260126a")
    .run_commands(
        "uv pip install --system git+https://github.com/huggingface/transformers.git@eebf856",  # 4.54.1
        "uv pip install --system aiohttp",  # For LLM judge reward model
        """sed -i 's/AutoImageProcessor.register(config, None, image_processor, None, exist_ok=True)/AutoImageProcessor.register(config, slow_image_processor_class=image_processor, exist_ok=True)/g' /sgl-workspace/sglang/python/sglang/srt/configs/utils.py""",
        # Fix rope_theta access for transformers 5.x (moved to rope_parameters dict)
        r"""sed -i 's/hf_config\.rope_theta/hf_config.rope_parameters["rope_theta"]/g' /usr/local/lib/python3.12/dist-packages/megatron/bridge/models/glm/glm45_bridge.py""",
        r"""sed -i 's/hf_config\.rope_theta/hf_config.rope_parameters["rope_theta"]/g' /usr/local/lib/python3.12/dist-packages/megatron/bridge/models/qwen/qwen3_bridge.py""",
    )
    .add_local_dir("test-configs", remote_path="/root/test-configs", copy=True)
    .add_local_dir("tools", remote_path="/root/tools", copy=True)
    .entrypoint([])
)

def get_hf_model_path() -> str:
    return f"{MODELS_PATH / MODEL_PATH / HF_DIR}"

def get_megatron_checkpoint_path() -> str:
    return f"{MODELS_PATH / MODEL_PATH / ITERS_DIR}"

@app.function(
    image=slime_image,
    timeout=24 * 60 * 60,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
    volumes={MODELS_PATH.as_posix(): checkpoints_volume},
)
async def convert_checkpoint(
    model_path: str,
    iter_dir: str,
    origin_hf_dir: str = BASE_MODEL_NAME
):
    """Convert Megatron checkpoint to HuggingFace format."""
    from huggingface_hub import snapshot_download
    import subprocess
    
    await checkpoints_volume.reload.aio()

    local_hf_dir = MODELS_PATH / origin_hf_dir

    if not local_hf_dir.exists():
        snapshot_download(repo_id=f"Qwen/{origin_hf_dir}", local_dir=local_hf_dir)
    else:
        print(f"Model {origin_hf_dir} already downloaded.")

    megatron_checkpoint_path = MODELS_PATH / model_path / iter_dir
    output_hf_path = MODELS_PATH / model_path / HF_DIR
    
    subprocess.run(f"PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py --input-dir {megatron_checkpoint_path} --output-dir {output_hf_path} --origin-hf-dir {local_hf_dir}", shell=True, check=True)


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    min_containers=1,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        MODELS_PATH.as_posix(): checkpoints_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    if not (MODELS_PATH / MODEL_PATH / HF_DIR).joinpath("config.json").exists():
        print(f"Converting checkpoint {MODEL_PATH} to HuggingFace format...")
        convert_checkpoint.remote(model_path=MODEL_PATH, iter_dir=ITERS_DIR)
        print(f"Checkpoint {MODEL_PATH}/{ITERS_DIR} converted to HuggingFace format.")

    cmd = [
        "vllm",
        "serve",
        str(MODELS_PATH / MODEL_PATH / HF_DIR),
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--enforce-eager",
        "--tensor-parallel-size",
        str(N_GPU),
        "--reasoning-parser",
        "qwen3"
    ]

    print(" ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)



@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    min_containers=1,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        MODELS_PATH.as_posix(): checkpoints_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve_base():
    import subprocess
    from huggingface_hub import snapshot_download

    base_model_path = MODELS_PATH / BASE_MODEL_NAME

    local_hf_dir = MODELS_PATH / BASE_MODEL_NAME

    if not local_hf_dir.exists():
        snapshot_download(repo_id=f"Qwen/{BASE_MODEL_NAME}", local_dir=local_hf_dir)
    else:
        print(f"Model {BASE_MODEL_NAME} already downloaded.")


    cmd = [
        "vllm",
        "serve",
        str(base_model_path),
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--enforce-eager",
        "--tensor-parallel-size",
        str(N_GPU),
        "--reasoning-parser",
        "qwen3"
    ]

    print(" ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)