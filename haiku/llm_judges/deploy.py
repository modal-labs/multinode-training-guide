"""
LLM-as-a-Judge Reward Model for SLIME GRPO training.

Uses CMUdict for syllable counting: https://github.com/cmusphinx/cmudict
Recommended by various packages such as `syllables` and `nltk`.
"""

import asyncio
from enum import Enum
import threading

import aiohttp

import modal
import modal.experimental


# =============================================================================

class JudgeType(str, Enum):
    STRICT = "strict"
    STRICT_LEVELED = "strict_leveled"

ACTIVE_JUDGE_TYPE = JudgeType.STRICT_LEVELED

def get_judge_url(judge_type: JudgeType) -> str:
    if judge_type == JudgeType.STRICT:
        return "https://modal-labs-joy-dev--llm-judge-strict-llmjudge.us-east.modal.direct"
    elif judge_type == JudgeType.STRICT_LEVELED:
        return "https://modal-labs-joy-dev--llm-judge-strict-leveled-llmjudge.us-east.modal.direct"
    else:
        raise ValueError(f"Invalid judge type: {judge_type}")

# =============================================================================
# Modal App Setup
# =============================================================================

app = modal.App(f"llm-judge-{ACTIVE_JUDGE_TYPE.value}")

FLASH_PORT = 8000
VLLM_PORT = 8001

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_NAME = "qwen3-30b-a3b-instruct"
N_GPU = 1
MINUTES = 60

checkpoint_volume = modal.Volume.from_name("unsloth-checkpoints")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)





def _make_judge(judge_type: JudgeType):
    from llm_judges.strict import StrictJudge
    from llm_judges.strict_leveled import StrictLeveledJudge

    judges = {
        JudgeType.STRICT: StrictJudge,
        JudgeType.STRICT_LEVELED: StrictLeveledJudge,
    }
    return judges[judge_type]()

# =============================================================================

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.2",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.2",
        "aiohttp>=3.9.0",
        "pydantic>=2.0.0",
        "fastapi[standard]>=0.115.0",
        "uvicorn>=0.30.0",
        "nltk>=3.8.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_commands(
        "python -c \"import nltk; nltk.download('cmudict'); nltk.download('punkt_tab')\""
    )
    .add_local_dir("llm_judges", "/root/llm_judges")
)

# =============================================================================
# Modal Flash Endpoint
# =============================================================================


def create_fastapi_app(judge_type: JudgeType):
    from fastapi import FastAPI
    from pydantic import BaseModel
    import nltk

    fastapi_app = FastAPI(title="LLM Judge Reward Model", docs_url="/docs")
    cmudict = nltk.corpus.cmudict.dict()
    judge = _make_judge(judge_type)

    class ScoreRequest(BaseModel):
        prompt: str
        response: str

    @fastapi_app.post("/score")
    async def score(request: ScoreRequest) -> float:
        max_retries = 5
        last_error = None

        for attempt in range(max_retries):
            try:
                return await _do_scoring(request)

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    print(
                        f"Scoring failed (attempt {attempt + 1}): {e}, retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise last_error

    async def _do_scoring(request: ScoreRequest) -> float:
        import aiohttp

        prompt = request.prompt
        response_text = request.response

        if prompt is None or response_text is None:
            return None

        async with aiohttp.ClientSession() as session:
            result = await judge.score_single(
                MODEL_NAME, session, prompt, response_text, cmudict
            )

        return float(result)

    @fastapi_app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL_NAME, "judge": judge_type.value}

    return fastapi_app


@app.cls(
    image=image,
    gpu=f"H100:{N_GPU}",
    min_containers=3,
    scaledown_window=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    experimental_options={"flash": "us-east"},
    region="us-east",
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    target_inputs=8
)
class LLMJudge:
    """Modal Flash endpoint combining vLLM + scoring logic in one container.

    Each judge_type value gets its own independent container pool via modal.parameter().
    Deploy with: LLMJudge(judge_type="strict") or LLMJudge(judge_type="balanced")
    """

    @modal.enter()
    def setup(self):
        import subprocess

        import uvicorn


        # Start vLLM on VLLM_PORT (internal)
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
        self._vllm_process = subprocess.Popen(" ".join(cmd), shell=True)

        # Wait for vLLM to be ready
        self._wait_for_port(VLLM_PORT, timeout=600)
        print(f"vLLM ready on port {VLLM_PORT}")

        # Start FastAPI scoring endpoint on FLASH_PORT (exposed)
        self._fastapi_app = create_fastapi_app(ACTIVE_JUDGE_TYPE)
        config = uvicorn.Config(
            self._fastapi_app,
            host="0.0.0.0",
            port=FLASH_PORT,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        self._wait_for_port(FLASH_PORT, timeout=30)
        self.flash_manager = modal.experimental.flash_forward(FLASH_PORT)
        print(f"Flash endpoint ready on port {FLASH_PORT} (judge={ACTIVE_JUDGE_TYPE.value})")

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
        if hasattr(self, "_server"):
            self._server.should_exit = True
        if hasattr(self, "_thread"):
            self._thread.join(timeout=5)
        if hasattr(self, "_vllm_process"):
            self._vllm_process.terminate()
            self._vllm_process.wait(timeout=10)


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
        endpoint_url: str | None = None,
        judge_type: JudgeType = JudgeType.STRICT,
    ):
        self.endpoint_url = endpoint_url
        self.judge = _make_judge(judge_type)
        self._session: aiohttp.ClientSession | None = None
        self._cmudict: dict | None = None

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
            return data.get("scores", [-1.0] * len(prompts))

    def _get_cmudict(self) -> dict:
        if self._cmudict is None:
            import nltk

            self._cmudict = nltk.corpus.cmudict.dict()
        return self._cmudict

    async def _score_direct(
        self, prompts: list[str], responses: list[str]
    ) -> list[float]:
        session = await self._get_session()
        cmudict = self._get_cmudict()
        tasks = [
            self.judge.score_single(MODEL_NAME, session, p, r, cmudict)
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

    def compute_reward(self, prompts: list[str], responses: list[str]) -> list[float]:
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

    def __call__(self, prompts: list[str], responses: list[str]) -> list[float]:
        return self.compute_reward(prompts, responses)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
