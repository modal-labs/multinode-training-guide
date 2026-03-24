"""Custom Harbor agent used by the Miles examples."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class SimpleHarborAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "simple-harbor-agent"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        base_url: str,
        request_kwargs: dict | None = None,
        task_mode: str = "hello",
        extra_instruction: str = "",
        submitted_response: str | None = None,
        **kwargs,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.request_kwargs = request_kwargs or {}
        self.task_mode = task_mode
        self.extra_instruction = extra_instruction
        self.submitted_response = submitted_response

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    def _build_messages(self, instruction: str) -> list[dict[str, str]]:
        if self.task_mode == "usaco":
            system = (
                "You are a competitive programmer writing a single file for an evaluator. "
                "Ignore any request for explanations, restatements, pseudocode, or markdown. "
                "Return only raw valid Python 3 code for solution.py."
            )
            user = (
                f"{instruction}\n\n"
                "Solve the task and output only the exact contents of solution.py. "
                "Do not restate the problem. Do not explain your approach. Do not use code fences. "
                f"{self.extra_instruction}"
            ).strip()
        else:
            system = "Return only the exact file contents requested by the user. Do not use code fences."
            user = f"{instruction}\n\nReturn only the contents that should be written to /app/hello.txt.\n{self.extra_instruction}".strip()
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _target_path(self) -> str:
        return "/app/solution.py" if self.task_mode == "usaco" else "/app/hello.txt"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        started = time.time()
        if self.submitted_response is None:
            payload = {
                "model": self.model_name or "model",
                "messages": self._build_messages(instruction),
                **self.request_kwargs,
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=180,
            )
            response.raise_for_status()

            message = response.json()["choices"][0]["message"]["content"]
            content = _strip_code_fence(message)
            raw_response_chars = len(message)
        else:
            content = _strip_code_fence(self.submitted_response)
            raw_response_chars = len(self.submitted_response)

        target_path = self._target_path()

        local_path = self.logs_dir / Path(target_path).name
        local_path.write_text(content)
        await environment.upload_file(local_path, target_path)

        elapsed = time.time() - started
        context.metadata = {
            "task_mode": self.task_mode,
            "model_latency": elapsed,
            "raw_response_chars": raw_response_chars,
            "used_submitted_response": self.submitted_response is not None,
        }
