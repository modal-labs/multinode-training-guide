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
    expected action format. Scores each turn individually for
    granular signal.
    """
    if not texts:
        return 0.0

    n_turns = len(texts)
    turn_scores: list[float] = []
    has_done = False

    for text in texts:
        text = text.strip()
        if not text:
            turn_scores.append(0.0)
            continue
        ts = 0.0
        if _ACTION_RE.search(text):
            ts += 0.6
            if _VERB_RE.search(text):
                ts += 0.3
        elif _DONE_RE.search(text):
            ts += 0.5
            has_done = True
        turn_scores.append(min(ts, 1.0))

    if not turn_scores:
        return 0.0

    avg = sum(turn_scores) / len(turn_scores)
    bonus = 0.1 if has_done else 0.0
    return min(avg + bonus, 1.0)


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
