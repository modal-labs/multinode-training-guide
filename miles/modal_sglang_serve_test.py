"""Standalone 1-node SGLang serving test for GLM-5.2 FP8 debugging.

Serves the model on 8xH200 with the same SGLang flags as the miles rollout
engines (minus dp-attention, which needs 32 GPUs), then runs greedy
completions with logprobs to judge output quality directly. Bisect levers
are exposed via --server-args / --env-json so variants (shared-expert TP1,
moe runner, nsa backends, online vs offline quant) don't need code changes.

    uv run modal run --detach miles/modal_sglang_serve_test.py::serve_test
    uv run modal run --detach miles/modal_sglang_serve_test.py::serve_test \
        --env-json '{"SGLANG_SHARED_EXPERT_TP1": "1"}'
"""

import json
import os
import subprocess
import time
import urllib.request

import modal

from configs import get_module
from configs.base import HF_CACHE_PATH

# Reuse the FP8 experiment's image (same sglang build + patches as the
# 64-GPU rollout engines, incl. sglang_fp8_lora_fix and the load-timeout bump).
EXPERIMENT = "glm5_2_744b_a40b_lora_dapo_tilelang_32k_fp8"
modal_cfg = get_module(EXPERIMENT).modal

image = (
    modal.Image.from_registry(modal_cfg.docker_image)
    .entrypoint([])
    .add_local_python_source("configs", copy=True)
    .add_local_python_source("modal_helpers", copy=True)
)
for patch in modal_cfg.patch_files:
    image = image.add_local_file(patch, f"/tmp/{os.path.basename(patch)}", copy=True)
if modal_cfg.image_run_commands:
    image = image.run_commands(*modal_cfg.image_run_commands)
if modal_cfg.image_env:
    image = image.env(modal_cfg.image_env)

hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App("glm52-fp8-serve-test")

PORT = 30000

DEFAULT_SERVER_ARGS = (
    "--model-path zai-org/GLM-5.2-FP8 "
    "--tp-size 8 "
    "--trust-remote-code "
    "--attention-backend nsa "
    "--nsa-decode-backend flashmla_sparse "
    "--nsa-prefill-backend flashmla_sparse "
    "--moe-runner-backend triton "
    "--disable-shared-experts-fusion "
    "--mem-fraction-static 0.80 "
    "--context-length 8192 "
    # Quality test only: skip graph capture (15 min) and its NVLS multicast
    # setup, which Fabric Manager on these hosts cannot provide.
    "--disable-cuda-graph "
    f"--port {PORT} --host 127.0.0.1"
)

# Same rollout-engine env as the miles config.
DEFAULT_ENV = {
    "SGLANG_NSA_FORCE_MLA": "1",
    "INDEXER_ROPE_NEOX_STYLE": "0",
    "NCCL_NVLS_ENABLE": "0",
}

PROMPTS = [
    "The capital of France is",
    (
        "Question: Natalia sold clips to 48 of her friends in April, and then "
        "she sold half as many clips in May. How many clips did Natalia sell "
        "altogether in April and May?\nAnswer:"
    ),
    "def fibonacci(n):\n",
    "1 + 1 = 2, 2 + 2 = 4, 4 + 4 =",
]


def _generate(prompt: str, max_new_tokens: int = 96) -> dict:
    payload = {
        "text": prompt,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        "return_logprob": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


def _post(path: str, payload: dict | None = None) -> str:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/{path}",
        data=json.dumps(payload or {}).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=900).read().decode()


def _run_prompts(phase: str) -> None:
    for prompt in PROMPTS:
        out = _generate(prompt)
        lps = [t[0] for t in out["meta_info"]["output_token_logprobs"]]
        mean_lp = sum(lps) / max(len(lps), 1)
        print("=" * 60)
        print(f"[{phase}] PROMPT: {prompt[:120]!r}")
        print(f"[{phase}] OUTPUT: {out['text'][:400]!r}")
        print(f"[{phase}] greedy mean output logprob: {mean_lp:.3f} over {len(lps)} tokens")


@app.function(
    image=image,
    gpu="H200:8",
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
    timeout=3 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def serve_test(server_args: str = "", env_json: str = "{}", cycle_memory: bool = False):
    hf_cache_volume.reload()
    args = server_args or DEFAULT_SERVER_ARGS
    if cycle_memory:
        # Mirror the miles rollout engines' memory-saver setup so we can
        # exercise the release/resume weight backup path the training loop
        # performs before rollout 0.
        args += " --enable-memory-saver --enable-weights-cpu-backup"
    env = {**os.environ, **DEFAULT_ENV, **json.loads(env_json)}
    print(f"server args: {args}")
    print(f"env overrides: {json.loads(env_json)}")

    proc = subprocess.Popen(f"python -m sglang.launch_server {args}", shell=True, env=env)
    try:
        deadline = time.time() + 45 * 60
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited during startup: {proc.returncode}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health_generate", timeout=5)
                break
            except Exception:
                if time.time() > deadline:
                    raise TimeoutError("server did not become healthy in 45 min")
                time.sleep(10)
        print("server healthy, running prompts")

        _run_prompts("fresh")

        if cycle_memory:
            print("cycling memory occupation (release -> resume)...")
            print(_post("release_memory_occupation")[:200])
            time.sleep(10)
            print(_post("resume_memory_occupation")[:200])
            _run_prompts("after-cycle")

        print("DONE_SERVE_TEST")
    finally:
        proc.terminate()
