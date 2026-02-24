"""Serve SLIME-trained Haiku models with vLLM.

modal deploy eval.serve_haiku_model
"""

from pathlib import Path
import modal
import modal.experimental

from eval.shared import MODEL_CONFIG, ModelConfig, _to_class_name

APP_NAME = "serve-haiku-model"

app = modal.App(APP_NAME)

MODELS_PATH: Path = Path("/models")

HF_DIR = "hf"

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
    .add_local_dir("tools", remote_path="/root/tools", copy=True)
    .entrypoint([])
)

def get_hf_model_path(config: ModelConfig) -> str:
    return f"{MODELS_PATH / config.model_path / HF_DIR}"

def get_megatron_checkpoint_path(config: ModelConfig) -> str:
    return f"{MODELS_PATH / config.model_path / config.iters_dir}"

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
    origin_hf_dir: str
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


CLS_KWARGS = dict(
    image=vllm_image,
    gpu=f"A10G:{N_GPU}",
    scaledown_window=15 * MINUTES,
    startup_timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        MODELS_PATH.as_posix(): checkpoints_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    experimental_options={"flash": "us-east"},
    region="us-east",
    min_containers=1,
)


class _VLLMServerBase:
    """Base class with all vLLM serving logic. Not registered with Modal directly."""

    # Subclasses set this as a class variable
    MODEL_KEY: str

    @modal.enter()
    def setup(self):
        import subprocess

        if self.MODEL_KEY not in MODEL_CONFIG:
            raise ValueError(f"Invalid model name: {self.MODEL_KEY}")

        model_config = MODEL_CONFIG[self.MODEL_KEY]

        if model_config.is_base_model:
            model_dir = self._setup_base_model(model_config)
        else:
            model_dir = self._setup_finetuned_model(model_config)

        cmd = [
            "vllm",
            "serve",
            str(model_dir),
            "--served-model-name",
            model_config.model_name,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--enforce-eager",
            "--tensor-parallel-size",
            str(N_GPU),
            "--reasoning-parser",
            "qwen3",
        ]

        print(" ".join(cmd))
        self._vllm_process = subprocess.Popen(" ".join(cmd), shell=True)

        self._wait_for_port(VLLM_PORT, timeout=600)
        print(f"vLLM ready on port {VLLM_PORT}")

        self.flash_manager = modal.experimental.flash_forward(VLLM_PORT)
        print(f"Flash endpoint ready on port {VLLM_PORT}")

    def _setup_base_model(self, model_config: ModelConfig) -> Path:
        from huggingface_hub import snapshot_download

        local_hf_dir = MODELS_PATH / model_config.model_path

        if not local_hf_dir.exists():
            snapshot_download(repo_id=f"Qwen/{model_config.model_path}", local_dir=local_hf_dir)
        else:
            print(f"Model {model_config.model_path} already downloaded.")

        return local_hf_dir

    def _setup_finetuned_model(self, model_config: ModelConfig) -> Path:
        hf_path = MODELS_PATH / model_config.model_path / HF_DIR

        if not hf_path.joinpath("config.json").exists():
            print(f"Converting checkpoint {model_config.model_path} to HuggingFace format...")
            convert_checkpoint.remote(
                model_path=model_config.model_path,
                iter_dir=model_config.iters_dir,
                origin_hf_dir=model_config.base_model_name,
            )
            checkpoints_volume.reload()
            print(f"Checkpoint {model_config.model_path}/{model_config.iters_dir} converted to HuggingFace format.")

        return hf_path

    def _wait_for_port(self, port: int, timeout: int = 30):
        import socket
        import time

        for _ in range(timeout):
            try:
                socket.create_connection(("localhost", port), timeout=1).close()
                return
            except OSError:
                time.sleep(1)
        raise RuntimeError(f"Server failed to start on port {port}")

    @modal.method()
    def keepalive(self):
        pass

    @modal.exit()
    def cleanup(self):
        if hasattr(self, "flash_manager"):
            self.flash_manager.stop()
            self.flash_manager.close()
        if hasattr(self, "_vllm_process"):
            self._vllm_process.terminate()
            self._vllm_process.wait(timeout=10)


for _model_key in MODEL_CONFIG:
    _cls_name = _to_class_name(_model_key)
    _cls = type(_cls_name, (_VLLMServerBase,), {"MODEL_KEY": _model_key})
    _cls = modal.concurrent(target_inputs=4)(_cls)
    _cls = app.cls(**CLS_KWARGS)(_cls)
    globals()[_cls_name] = _cls
