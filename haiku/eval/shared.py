"""Shared constants, endpoint helpers, and query function for haiku eval and playground."""

import asyncio
from dataclasses import dataclass

import httpx

# ---------------------------------------------------------------------------
# Flash URL generation
# ---------------------------------------------------------------------------

APP_NAME = "serve-haiku-model"
ENVIRONMENT = "joy-dev"
FLASH_REGION = "us-east"
WORKSPACE = "modal-labs"


def _to_class_name(model_key: str) -> str:
    """Convert model key like '235b-judge-cl' to '235bJudgeClServer'."""
    return "".join(part.capitalize() for part in model_key.split("-")) + "Server"


def get_flash_url(model_key: str) -> str:
    """Get the Flash endpoint base URL for a given model key."""
    cls_name = _to_class_name(model_key)
    return f"https://{WORKSPACE}-{ENVIRONMENT}--{APP_NAME}-{cls_name.lower()}.{FLASH_REGION}.modal.direct"


def get_model_endpoint(model_key: str) -> str:
    """Return the full chat-completions URL for a given model key."""
    return get_flash_url(model_key) + "/v1/chat/completions"


# ---------------------------------------------------------------------------
# Model registry (single source of truth for both serving and eval/playground)
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    model_path: str
    iters_dir: str
    model_name: str
    model_description: str
    label: str
    base_model_name: str = "Qwen3-4B"
    is_base_model: bool = False

    @property
    def badge(self) -> str:
        return "base" if self.is_base_model else "trained"

    @property
    def flash_url(self) -> str:
        return get_flash_url(self.model_name)


MODEL_CONFIG = {
    "base-model": ModelConfig(
        model_path="Qwen3-4B",
        iters_dir="",
        model_name="base-model",
        model_description="Qwen3-4B Base Model",
        label="Base Model",
        is_base_model=True,
    ),
    "no-llm-model": ModelConfig(
        model_path="2-23-no-llm-Qwen3-4B-20260224-032404",
        iters_dir="iter_0000049",
        model_name="no-llm-model",
        model_description="Qwen3-4B Finetuned with No LLM",
        label="No LLM Judge",
    ),
    "30b-judge": ModelConfig(
        model_path="2_24-30b-Qwen3-4B-20260224-184838",
        iters_dir="iter_0000049",
        model_name="30b-judge-cl",
        model_description="Qwen3-4B Finetuned with 30B Judge using Curriculumn Learning",
        label="30B Judge (CL)",
    ),
    "30b-judge-cl": ModelConfig(
        model_path="2_24-30b-leveled-Qwen3-4B-20260224-180902",
        iters_dir="iter_0000049",
        model_name="30b-judge-cl",
        model_description="Qwen3-4B Finetuned with 30B Judge using Curriculumn Learning",
        label="30B Judge (CL)",
    ),
    "235b-judge": ModelConfig(
        model_path="2_24-235b-Qwen3-4B-20260224-174605",
        iters_dir="iter_0000049",
        model_name="235b-judge",
        model_description="Qwen3-4B Finetuned with 235B Judge",
        label="235B Judge",
    ),
    "235b-judge-cl": ModelConfig(
        model_path="2_23-235b-leveled-Qwen3-4B-20260224-172832",
        iters_dir="iter_0000049",
        model_name="235b-judge-cl",
        model_description="Qwen3-4B Finetuned with 235B Judge using Curriculumn Learning",
        label="235B Judge (CL)",
    ),
}

# Derived views for different consumers
MODELS = {
    key: {"label": c.label, "badge": c.badge}
    for key, c in MODEL_CONFIG.items()
}

# For the UI — list of models with their metadata
MODEL_CHECKPOINTS = [
    {"name": config.model_name, "label": config.label, "badge": config.badge}
    for config in MODEL_CONFIG.values()
]

# Maps model_key -> flash endpoint URL
MODEL_URLS: dict[str, str] = {key: config.flash_url for key, config in MODEL_CONFIG.items()}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODAL_VOCABS = [
    "modal",
    "volume",
    "function",
    "sandbox",
    "flash",
    "inference",
    "train",
]

DEFAULT_CONCURRENCY = 50

EVAL_QUESTIONS = [
    "Write me a haiku about cat.",
    "Write me a haiku about dog.",
    "Write me a haiku about bird.",
    "Write me a haiku about fish.",
    "Write me a haiku about horse.",
    "Write me a haiku about rabbit.",
    "Write me a haiku about snake.",
    "Write me a haiku about tiger.",
    "Write me a haiku about lion.",
    "Write me a haiku about Jason Mancuso.",
    "Write me a haiku about Joy Liu.",
    "Write me a haiku about Modal Labs.",
]

QUICK_PROMPTS = [
    {"emoji": "🐱", "label": "cat", "prompt": "Write me a haiku about cat."},
    {"emoji": "🌊", "label": "ocean", "prompt": "Write me a haiku about the ocean."},
    {"emoji": "🌸", "label": "cherry blossoms", "prompt": "Write me a haiku about cherry blossoms."},
    {"emoji": "💻", "label": "coding", "prompt": "Write me a haiku about coding."},
    {"emoji": "☁️", "label": "Modal", "prompt": "Write me a haiku about Modal."},
    {"emoji": "⚡", "label": "serverless", "prompt": "Write me a haiku about serverless."},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_system_prompt(include_vocab: bool = True) -> str:
    """Build the haiku poet system prompt, optionally including Modal vocabulary."""
    base = "You are a haiku poet. You will be given a prompt and you will need to write a haiku about the prompt."
    if include_vocab:
        vocab_str = ", ".join(MODAL_VOCABS)
        return f"{base} Try to incorporate these words into the haiku if possible: {vocab_str}"
    return base


async def query_model(
    client: httpx.AsyncClient,
    endpoint: str,
    prompt: str,
    *,
    model_name: str = "base-model",
    system_prompt: str | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> str:
    """Send a chat-completion request and return the assistant's reply text.

    Args:
        model_name: The vLLM --served-model-name (matches the model key in MODEL_CONFIG).
    """

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    async def _do_request() -> str:
        try:
            response = await client.post(endpoint, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"ERROR: {e}"

    if semaphore is not None:
        async with semaphore:
            return await _do_request()
    return await _do_request()
