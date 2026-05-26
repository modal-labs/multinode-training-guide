"""Custom reward model for Windows computer use.

Combines environment reward (task completion) with format reward
(producing valid action tags). The format reward provides strong
GRPO signal even when the model hasn't learned the task yet.
"""

from __future__ import annotations

import re
from typing import Any

from slime.utils.types import Sample

_ACTION_RE = re.compile(r"<action>\s*\w+.*?</action>", re.DOTALL)
_DONE_RE = re.compile(r"<done\s*/?>")
_VERB_RE = re.compile(
    r"<action>\s*(sendkey|type|typeline|wait)\b.*?</action>", re.DOTALL
)


def _format_reward(texts: list[str]) -> float:
    """Compute format reward from model response texts across all turns.

    Returns 0.0 or 0.5-1.0 based on whether ANY turn has valid action
    format. Binary-like signal maximizes GRPO variance.
    """
    if not texts:
        return 0.0

    full = "\n".join(texts)
    if not full.strip():
        return 0.0

    score = 0.0
    if _ACTION_RE.search(full):
        score = 0.5
        if _VERB_RE.search(full):
            score = 0.8
    if _DONE_RE.search(full):
        score = max(score, 0.3)
        score = min(score + 0.2, 1.0)
    return score


async def compute_reward(args: Any, sample: Sample) -> float:
    """Combine environment reward with format reward.

    Environment reward dominates when the model completes the task.
    Format reward provides gradient signal during early training.
    """
    metadata = sample.metadata or {}
    env_reward = float(metadata.get("env_reward", 0.0))

    if env_reward >= 0.5:
        return env_reward

    texts = metadata.get("all_response_texts", [])
    format_reward = _format_reward(texts) if texts else 0.0
    final = max(env_reward, format_reward)
    if final > 0:
        first_text = repr(texts[0][:80]) if texts else "N/A"
        print(f"[RM] env_r={env_reward:.2f} fmt_r={format_reward:.2f} final={final:.2f} n_texts={len(texts)} first={first_text}")
    return final
