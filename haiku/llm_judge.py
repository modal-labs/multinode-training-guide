"""
LLM-as-a-Judge Reward Model for SLIME GRPO training.

Uses CMUdict for syllable counting: https://github.com/cmusphinx/cmudict
Recommended by various packages such as `syllables` and `nltk`.
"""

import asyncio
import re
import aiohttp

import modal
import modal.experimental

# =============================================================================
# Modal App Setup
# =============================================================================

app = modal.App("llm-judge-reward-model")

FLASH_PORT = 8000
VLLM_PORT = 8001

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_NAME = "qwen3-30b-a3b-instruct"
N_GPU = 1
MINUTES = 60

checkpoint_volume = modal.Volume.from_name("unsloth-checkpoints")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

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
)


# =============================================================================
# NLTK Syllable & Sentence Utilities
# =============================================================================


# Ref: https://stackoverflow.com/questions/49581705/using-cmudict-to-count-syllables
# CMU dict stores phonemes; syllables = count of vowel sounds (digits in phoneme)
def lookup_word(word_s, cmudict: dict):
    return cmudict.get(word_s, None)

def count_syllables_for_word(word, cmudict):
    count = 0
    word = word.lower().strip()
    phones = lookup_word(word, cmudict) # this returns a list of matching phonetic rep's
    if phones:                   # if the list isn't empty (the word was found)
        phones0 = phones[0]      #     process the first
        count = len([p for p in phones0 if p[-1].isdigit()]) # count the vowels
        return count
    # Fallback heuristic for words not in CMU dict
    print(f"WARNING: Word {word} not found in CMU dict")
    return -1


def diff_syllables_count(text: str, target_syllables: int, cmudict: dict) -> int:
    """Output the difference between the number of syllables in the text and the target number of syllables."""
    words = re.findall(r"[a-zA-Z]+", text)
    total_syllables = sum(count_syllables_for_word(w, cmudict) for w in words)
    return abs(total_syllables - target_syllables)


def segment_haiku_lines(response: str) -> list[str]:
    if "/" in response:
        lines = [line.strip() for line in response.split("/")]
    elif ". " in response:
        lines = [line.strip() for line in response.split(". ")]
    else:
        lines = [line.strip() for line in response.split("\n")]
    return [line for line in lines if line]


# =============================================================================
# Shared Prompt Template
# =============================================================================


def generate_haiku_judge_prompt(prompt: str, response: str) -> str:
    return f"""You are evaluating a haiku poem.

    Score the response based on the following criteria:
    - 2 points: if the response is relevant to the topic "{prompt}" and evokes meaning and emotion
    - 1 point: if the response is relevant to the topic "{prompt}" but very plain
    - 0 points: if the response is not relevant to the topic "{prompt}"

    --
    **Topic:** {prompt}

    **Response to evaluate:**
    {response}
    ---

    Output ONLY a single number (0-2), nothing else."""


# =============================================================================
# Scoring Logic
# =============================================================================
def score_haiku_structure(response: str, cmudict: dict) -> int:
    # - 2 points: Exactly 3 lines (separated by '/')
    # - 2 point: First line is approximately 5 syllables
    # - 2 point: Second line is approximately 7 syllables
    # - 2 point: Third line is approximately 5 syllables
    lines = segment_haiku_lines(response)
    score = 0
    if len(lines) == 3:
        score += 2

    if len(lines) > 0 and diff_syllables_count(lines[0], 5, cmudict) == 0:
        score += 2
    if len(lines) > 1 and diff_syllables_count(lines[1], 7, cmudict) == 0:
        score += 2
    if len(lines) > 2 and diff_syllables_count(lines[2], 5, cmudict) == 0:
        score += 2

    return score


async def score_haiku_style(
    session: aiohttp.ClientSession,
    prompt: str,
    response: str,
    vllm_base_url: str = f"http://localhost:{VLLM_PORT}",
) -> int:
    judge_prompt = generate_haiku_judge_prompt(prompt, response)

    try:
        async with session.post(
            f"{vllm_base_url}/v1/chat/completions",
            headers={
                "content-type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": judge_prompt}],
                "max_tokens": 100,
            },
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                print(f"vLLM error: {resp.status} - {error_text}")
                return -1

            data = await resp.json()
            print(data)
            score_text = data["choices"][0]["message"]["content"].strip()

            match = re.search(r"(\d+(?:\.\d+)?)", score_text)
            if match:
                score = float(match.group(1))
                print(f"Score: {score}")
                return min(max(score, 0), 2)
            return -1
    except Exception as e:
        print(f"Error scoring response: {e}")
        return -1


async def score_single(
    session: aiohttp.ClientSession,
    prompt: str,
    response: str,
    cmudict: dict,
) -> float:
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

    structure_score = score_haiku_structure(response, cmudict)
    style_score = await score_haiku_style(session, prompt, response) if structure_score == 8 else 0
    print(f"Structure score: {structure_score}, Style score: {style_score}")

    return (structure_score + style_score) / 10


# =============================================================================
# Modal Flash Endpoint
# =============================================================================




def create_fastapi_app():
    from fastapi import FastAPI
    from pydantic import BaseModel
    import nltk

    fastapi_app = FastAPI(title="LLM Judge Reward Model", docs_url="/docs")
    cmudict = nltk.corpus.cmudict.dict()

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
            score = await score_single(session, prompt, response_text, cmudict)

        return float(score)

    @fastapi_app.get("/health")
    def health():
        return {"status": "ok", "model": "claude-sonnet-4-20250514"}

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
class LLMJudgeFlash:
    """Modal Flash endpoint combining vLLM + scoring logic in one container."""

    @modal.enter()
    def setup(self):
        import subprocess
        import threading

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

        self._wait_for_port(FLASH_PORT, timeout=30)
        self.flash_manager = modal.experimental.flash_forward(FLASH_PORT)
        print(f"Flash endpoint ready on port {FLASH_PORT}")

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
    ):
        self.endpoint_url = endpoint_url
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
            return data.get("scores", [-1.0] * len(prompts))

    async def _score_direct(
        self, prompts: list[str], responses: list[str]
    ) -> list[float]:
        session = await self._get_session()
        tasks = [score_single(session, p, r) for p, r in zip(prompts, responses)]
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
