"""Custom reward model for Windows computer use.

Scales the environment reward to create clear GRPO signal.
The env provides shaping rewards (0.0-0.3 for action quality,
0.5-1.0 for task completion), which we amplify with a multiplier.
"""

from __future__ import annotations

import re
from typing import Any

from slime.utils.types import Sample

_ACTION_CONTENT_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)
_DONE_RE = re.compile(r"<done\s*/?>")

# Valid sendkey arguments that the QEMU HMP understands.
_VALID_KEYS = {
    "ret", "spc", "tab", "esc", "backspace", "delete",
    "up", "down", "left", "right", "home", "end",
    "pgup", "pgdn", "f1", "f2", "f3", "f4", "f5",
    "f6", "f7", "f8", "f9", "f10", "f11", "f12",
    "ctrl-a", "ctrl-c", "ctrl-s", "ctrl-v", "ctrl-x", "ctrl-z",
    "meta_l-r", "meta_l-e", "meta_l-d", "alt-f4", "alt-tab",
}

REWARD_SCALE = 3.0


def _action_quality_reward(texts: list[str]) -> float:
    """Score the quality of actions across all turns.

    Returns 0.0-1.0 based on how well-formed and specific
    the model's actions are. Creates variance between samples
    that all produce action tags but differ in quality.
    """
    if not texts:
        return 0.0

    n_actions = 0
    quality_sum = 0.0
    has_type_with_content = False
    has_valid_sendkey = False
    has_done = False
    multiple_per_turn = 0

    for text in texts:
        text = text.strip()
        if not text:
            continue

        # Extract content between <action>...</action> tags
        action_contents = _ACTION_CONTENT_RE.findall(text)
        if len(action_contents) > 1:
            multiple_per_turn += 1

        for content in action_contents:
            n_actions += 1
            parts = content.strip().split(None, 1)
            verb = parts[0].lower() if parts else ""
            args = parts[1].strip() if len(parts) > 1 else ""

            if verb == "sendkey" and args:
                key = args.replace(" ", "-").lower()
                if key in _VALID_KEYS or len(key) == 1:
                    has_valid_sendkey = True
                    quality_sum += 0.3
                else:
                    quality_sum += 0.1
            elif verb == "type" and args:
                has_type_with_content = True
                quality_sum += 0.3
            elif verb in ("typeline", "wait"):
                quality_sum += 0.2
            elif verb in ("sendkey", "type") and not args:
                quality_sum += 0.05
            else:
                quality_sum += 0.05

        if _DONE_RE.search(text):
            has_done = True

    if n_actions == 0:
        return 0.0

    score = min(quality_sum / max(n_actions, 1), 0.5)
    if has_type_with_content:
        score += 0.15
    if has_valid_sendkey:
        score += 0.15
    if has_done:
        score += 0.1
    if multiple_per_turn == 0 and n_actions > 0:
        score += 0.1

    return min(score, 1.0)


async def compute_reward(args: Any, sample: Sample) -> float:
    """Compute final reward from env + action quality bonus.

    Env reward provides task-completion signal.
    Action quality provides gradient signal for action specificity.
    """
    metadata = sample.metadata or {}
    env_reward = float(metadata.get("env_reward", 0.0))
    texts = metadata.get("all_response_texts", [])
    quality = _action_quality_reward(texts) if texts else 0.0

    # Blend: env reward (task progress) + quality bonus (action specificity)
    raw = env_reward + quality * 0.5
    final = raw * REWARD_SCALE

    if final > 0:
        first_text = repr(texts[0][:80]) if texts else "N/A"
        print(f"[RM] env={env_reward:.2f} qual={quality:.2f} final={final:.2f} n_texts={len(texts)} first={first_text}")
    return final
