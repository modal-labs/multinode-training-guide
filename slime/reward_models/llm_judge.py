"""
LLM-as-a-Judge Reward Model for SLIME GRPO training.

This module provides:
1. LLMJudgeRewardModel class for local/direct use
2. Modal FLASH endpoint for low-latency reward model service

Deploy with: modal deploy reward_models/llm_judge.py
Get flash URL: LLMJudgeFlash._experimental_get_flash_urls()

Usage from SLIME training:
    # Option 1: Direct Claude API calls
    rm = LLMJudgeRewardModel()
    scores = rm(prompts, responses)

    # Option 2: Via deployed Modal flash endpoint (no API key needed in training)
    rm = LLMJudgeRewardModel(
        endpoint_url="https://<flash-url>/score"
    )
    scores = rm(prompts, responses)
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiohttp

import modal
import modal.experimental

# =============================================================================
# Modal App Setup
# =============================================================================

app = modal.App("llm-judge-reward-model")

FLASH_PORT = 8000

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "aiohttp>=3.9.0",
    "pydantic>=2.0.0",
    "fastapi[standard]>=0.115.0",
    "uvicorn>=0.30.0",
)




# =============================================================================
# Shared Prompt Template
# =============================================================================

# TODO(joy): better prompt
HAIKU_JUDGE_PROMPT = """You are a haiku poetry critic. Rate the following haiku response on a scale of 0-10.

A haiku is a three-line poem that follows a specific syllable pattern: 5-7-5.

========================================
Correctness (5 points total):
========================================
- Count the number of lines in the response, delimited by '/'. If the response contains three lines, score 3 points.
- Find the first line of the Haiku. If the first line exists and has 5 syllables, score 1 points.
- Find the second line of the Haiku. If the second line exists and has 7 syllables, score 1 points.
- Find the third line of the Haiku. If the third line exists and has 5 syllables, score 1 points.
Add up the points for each line. The total score is the sum of the points for each line.
========================================
On topic (3 points total):
========================================

- Check if the response about the user request on topic {prompt}.
- If it is, score 3 points.
- If it is not, score 0 points.

========================================
Style and creativity (2 points total):
========================================
- Rate the style and creativity of the haiku on a scale of 0-2.
========================================


Add up the points for each line. The total score is the sum of the points for each line.


User request: {prompt}

Response: {response}

Output ONLY a single number (0-10), nothing else."""


# =============================================================================
# Scoring Logic
# =============================================================================


async def score_single(
    session: aiohttp.ClientSession,
    model: str,
    prompt: str,
    response: str,
) -> float:
    """Score a single response using Claude."""
    judge_prompt = HAIKU_JUDGE_PROMPT.format(prompt=prompt, response=response)

    try:
        async with session.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": judge_prompt}],
            },
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                print(f"Claude API error: {resp.status} - {error_text}")
                return 0.5

            data = await resp.json()
            score_text = data["content"][0]["text"].strip()

            match = re.search(r"(\d+(?:\.\d+)?)", score_text)
            if match:
                score = float(match.group(1))
                return min(max(score / 10.0, 0.0), 1.0)
            return 0.5
    except Exception as e:
        print(f"Error scoring response: {e}")
        return 0.5


# =============================================================================
# Modal Flash Endpoint
# =============================================================================

from pydantic import BaseModel


class ScoreRequest(BaseModel):
    prompts: list[str]
    responses: list[str]


class ScoreResponse(BaseModel):
    scores: list[float]


def create_fastapi_app():
    """Create the FastAPI app for the flash endpoint."""
    from fastapi import FastAPI

    fastapi_app = FastAPI(title="LLM Judge Reward Model", docs_url="/docs")

    @fastapi_app.post("/score", response_model=ScoreResponse)
    async def score_endpoint(request: ScoreRequest):
        """Score prompt/response pairs using Claude as a judge."""
        import aiohttp

        if len(request.prompts) != len(request.responses):
            return {"scores": [], "error": "prompts and responses must have same length"}

        if not request.prompts:
            return {"scores": []}

        model = "claude-sonnet-4-20250514"

        async with aiohttp.ClientSession() as session:
            tasks = [
                score_single(session, model, p, r)
                for p, r in zip(request.prompts, request.responses)
            ]
            scores = await asyncio.gather(*tasks)

        return {"scores": list(scores)}

    @fastapi_app.get("/health")
    def health():
        return {"status": "ok", "model": "claude-sonnet-4-20250514"}

    return fastapi_app


@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("anthropic-secret")],
    min_containers=1,
    scaledown_window=300,
    experimental_options={"flash": "us-east"},
    region="us-east",
)
class LLMJudgeFlash:
    """Modal Flash endpoint for low-latency LLM Judge reward model."""

    @modal.enter()
    def setup(self):
        import threading

        import uvicorn

        self._fastapi_app = create_fastapi_app()

        config = uvicorn.Config(
            self._fastapi_app,
            host="0.0.0.0",
            port=FLASH_PORT,
            log_level="info",
        )
        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        self._wait_ready()
        self.flash_manager = modal.experimental.flash_forward(FLASH_PORT)
        print(f"Flash endpoint ready on port {FLASH_PORT}")

    def _wait_ready(self, timeout: int = 30):
        """Wait for uvicorn to be ready."""
        import socket
        import time

        for _ in range(timeout):
            try:
                socket.create_connection(("localhost", FLASH_PORT), timeout=1).close()
                return
            except OSError:
                time.sleep(1)
        raise RuntimeError(f"Server failed to start on port {FLASH_PORT}")

    @modal.method()
    def keepalive(self):
        """Keepalive method to prevent container from shutting down."""
        pass

    @modal.exit()
    def cleanup(self):
        if hasattr(self, "flash_manager"):
            self.flash_manager.stop()
            self.flash_manager.close()
        if hasattr(self, "_server"):
            self._server.should_exit = True
        if hasattr(self, "_thread"):
            self._thread.join(timeout=5)


# =============================================================================
# Standalone Class (for direct use / SLIME integration)
# =============================================================================


class LLMJudgeRewardModel:
    """LLM-as-a-judge reward model using Claude API for haiku evaluation.

    Can be used directly or via the Modal endpoint.

    Examples:
        # Direct usage
        rm = LLMJudgeRewardModel()
        scores = rm(["Write a haiku"], ["Cherry blossoms fall..."])

        # Via Modal endpoint
        rm = LLMJudgeRewardModel(endpoint_url="https://...")
        scores = rm(["Write a haiku"], ["Cherry blossoms fall..."])
    """

    def __init__(
        self,
    ):
        self.model = "self_hosted"
        self._session: aiohttp.ClientSession | None = None


    async def _get_session(self) -> aiohttp.ClientSession:
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _score_via_endpoint(
        self, prompts: list[str], responses: list[str]
    ) -> list[float]:
        """Score via deployed Modal endpoint."""
        session = await self._get_session()
        async with session.post(
            self.endpoint_url,
            json={"prompts": prompts, "responses": responses},
        ) as resp:
            data = await resp.json()
            return data.get("scores", [0.5] * len(prompts))

    async def _score_direct(
        self, prompts: list[str], responses: list[str]
    ) -> list[float]:
        session = await self._get_session()
        tasks = [
            score_single(session, self.model, p, r)
            for p, r in zip(prompts, responses)
        ]
        return list(await asyncio.gather(*tasks))

    async def score_batch(
        self, prompts: list[str], responses: list[str]
    ) -> list[float]:
        if self.endpoint_url:
            return await self._score_via_endpoint(prompts, responses)
        return await self._score_direct(prompts, responses)

    async def score_response(self, prompt: str, response: str) -> float:
        scores = await self.score_batch([prompt], [response])
        return scores[0]

    def compute_reward(
        self, prompts: list[str], responses: list[str]
    ) -> list[float]:
        """Sync interface for SLIME compatibility."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, self.score_batch(prompts, responses)
                )
                return future.result()
        return asyncio.run(self.score_batch(prompts, responses))

    def __call__(
        self, prompts: list[str], responses: list[str]
    ) -> list[float]:
        return self.compute_reward(prompts, responses)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

# =============================================================================
# Serve LLM Judge Reward Model
# =============================================================================


checkpoint_volume = modal.Volume.from_name("unsloth-checkpoints")
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
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model transfers
)

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_NAME = "qwen3-30b-a3b-instruct"
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
        "/checkpoints": checkpoint_volume,
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
        "--uvicorn-log-level=info",
        MODEL,
        "--served-model-name",
        MODEL_NAME,
        "--port",
        str(VLLM_PORT),
        "--enforce-eager",
        "--tensor-parallel-size",
        str(N_GPU),
        "--max-model-len",
        "8192",
    ]

    print(" ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)
