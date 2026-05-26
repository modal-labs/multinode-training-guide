"""Custom reward model for Windows computer use.

Uses a per-turn binary scoring approach to create within-group
variance for GRPO. Each turn is scored 1 (valid action with args)
or 0 (gibberish/empty/malformed), and the fraction of good turns
becomes the reward. Since different samples for the same prompt
produce different numbers of valid turns, this creates natural
binary-like variance within GRPO groups.
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


_VALID_VERBS = {"sendkey", "type", "typeline", "wait"}


def _turn_is_good(text: str) -> bool:
    """Binary per-turn check: does this turn contain a valid action?

    Relaxed: any <action> with a recognized verb counts. We want the
    model to first learn the format, THEN learn correct arguments.
    """
    contents = _ACTION_CONTENT_RE.findall(text)
    if not contents:
        return False
    for content in contents:
        parts = content.strip().split(None, 1)
        if not parts:
            continue
        verb = parts[0].lower()
        if verb in _VALID_VERBS:
            return True
    return False


def _turn_is_great(text: str) -> bool:
    """Stricter check: action has valid verb AND meaningful arguments."""
    contents = _ACTION_CONTENT_RE.findall(text)
    if not contents:
        return False
    for content in contents:
        parts = content.strip().split(None, 1)
        if not parts:
            continue
        verb = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""
        if verb == "sendkey" and args:
            key = args.replace(" ", "-").lower()
            if key in _VALID_KEYS or len(key) == 1:
                return True
        elif verb == "type" and args:
            return True
        elif verb in ("typeline", "wait"):
            return True
    return False


async def compute_reward(args: Any, sample: Sample) -> float:
    """Tiered reward: format → quality → task completion.

    Three tiers create natural variance within GRPO groups:
    - Tier 0 (0.0): no valid action tags at all (gibberish)
    - Tier 1 (0.5): has valid action verbs but wrong/missing args
    - Tier 2 (1.0): has valid actions with correct arguments
    - + env_reward for actual task completion
    - + done bonus for signaling completion

    Reward = (env_reward + quality) * 3.0
    """
    metadata = sample.metadata or {}
    env_reward = float(metadata.get("env_reward", 0.0))
    texts = metadata.get("all_response_texts", [])

    if not texts:
        print(f"[RM] env={env_reward:.2f} g=0/0 G=0/0 final=0.00")
        return 0.0

    n_good = sum(1 for t in texts if _turn_is_good(t))
    n_great = sum(1 for t in texts if _turn_is_great(t))
    n_total = len(texts)

    # Tiered quality: good (format) + great (content) bonus
    if n_good == 0:
        quality = 0.0
    else:
        # Base: fraction of turns with valid format
        quality = 0.5 * (n_good / n_total)
        # Bonus: fraction of turns with correct arguments
        quality += 0.5 * (n_great / n_total)

    # Done signal bonus
    has_done = any(_DONE_RE.search(t) for t in texts)
    if has_done:
        quality = min(quality + 0.2, 1.0)

    final = (env_reward + quality) * 3.0

    first_text = repr(texts[0][:80]) if texts else "N/A"
    print(f"[RM] env={env_reward:.2f} g={n_good}/{n_total} G={n_great}/{n_total} qual={quality:.2f} final={final:.2f} first={first_text}")
    return final
