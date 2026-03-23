from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import requests
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from code_utils import code_size_bytes, extract_python_code


class SingleShotCodeAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "single-shot-code-agent"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        api_base: str | None = None,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_sec: float = 90.0,
        **kwargs: Any,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.api_base = (api_base or "http://127.0.0.1:8000/v1").rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return

    def _chat_completion(self, instruction: str) -> dict[str, Any]:
        if not self.model_name:
            raise ValueError("model_name is required for SingleShotCodeAgent.")

        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": "Return only valid Python code."},
                {"role": "user", "content": instruction},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        return response.json()

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        completion = await asyncio.to_thread(self._chat_completion, instruction)
        message = completion["choices"][0]["message"]["content"]
        solution_code = extract_python_code(message)

        local_solution_path = self.logs_dir / "solution.py"
        local_solution_path.write_text(solution_code, encoding="utf-8")
        await environment.upload_file(
            source_path=local_solution_path,
            target_path="/workspace/solution.py",
        )

        usage = completion.get("usage", {})
        context.n_input_tokens = usage.get("prompt_tokens")
        context.n_output_tokens = usage.get("completion_tokens")
        context.metadata = {
            "api_base": self.api_base,
            "response_id": completion.get("id"),
            "candidate_bytes": code_size_bytes(solution_code),
        }
