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

    Returns 0.0-1.0 based on how well the responses follow the
    expected action format. This creates clear binary GRPO signal.
    """
    full = "\n".join(texts)
    if not full.strip():
        return 0.0

    score = 0.0
    action_matches = _ACTION_RE.findall(full)
    if action_matches:
        score += 0.5
    verb_matches = _VERB_RE.findall(full)
    if verb_matches:
        score += 0.3
    if _DONE_RE.search(full):
        score += 0.2
    return min(score, 1.0)


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
    return max(env_reward, format_reward)
