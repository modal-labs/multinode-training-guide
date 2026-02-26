from __future__ import annotations

import re

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_FUNCTION_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)


def normalize_code(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def extract_function_name_from_code(code: str) -> str:
    match = _FUNCTION_RE.search(normalize_code(code))
    if match is None:
        raise ValueError("No Python function definition found in code field.")
    return match.group(1)


def extract_python_code(text: str) -> str:
    normalized = normalize_code(text).strip()
    if "<|im_start|>assistant" in normalized:
        normalized = normalized.rsplit("<|im_start|>assistant", 1)[-1]
    if "</think>" in normalized:
        normalized = normalized.split("</think>", 1)[1]
    normalized = normalized.replace("<think>", "")
    normalized = normalized.replace("<|im_end|>", "")
    return normalized


def code_size_bytes(code: str) -> int:
    return len(normalize_code(code).encode("utf-8"))
