"""Serve a SLIME-trained model with vLLM."""

import aiohttp
import modal

app = modal.App("serve-slime-model")

# Same volume used in training
checkpoints_volume = modal.Volume.from_name("grpo-slime-example-checkpoints")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

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

# Point to your trained checkpoint
# After training, checkpoints are saved to /models/{model_name}/checkpoints/
# Check your volume to find the exact path
MODEL_PATH = "/models/Qwen3-4B-singlenode-20260206-170445/iter_0000004-hf"  # Update this!
MODEL_NAME = "slime-qwen"
N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000


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
        "/models": checkpoints_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_PATH,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--enforce-eager",
        "--tensor-parallel-size",
        str(N_GPU),
    ]

    print(" ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
async def main(question: str = "What is 25 + 17?", test_timeout=10 * MINUTES):
    url = serve.get_web_url()

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Waiting for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
        print("Server ready!")

        # Adjust prompt format to match your training
        payload = {
            "model": MODEL_NAME,
            "prompt": question,
            "max_tokens": 512,
            "temperature": 0.7,
        }

        async with session.post(
            "/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=1 * MINUTES,
        ) as resp:
            result = await resp.json()
            if "error" in result:
                print(f"Error: {result['error']}")
                return
            answer = result["choices"][0]["text"].strip()
            print(f"Q: {question}")
            print(f"A: {answer}")
