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


def _turn_is_good(text: str) -> bool:
    """Binary per-turn check: does this turn contain a valid action?"""
    contents = _ACTION_CONTENT_RE.findall(text)
    if not contents:
        return False
    # At least one action must have a recognized verb with arguments
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
    """Binary turn-fraction reward for GRPO variance.

    Scores each turn as good (1) or bad (0), then returns the
    fraction of good turns. Different samples naturally produce
    different numbers of good turns for the same prompt, creating
    the within-group variance GRPO needs.

    Adds env_reward for task completion signal.
    """
    metadata = sample.metadata or {}
    env_reward = float(metadata.get("env_reward", 0.0))
    texts = metadata.get("all_response_texts", [])

    if not texts:
        print(f"[RM] env={env_reward:.2f} good=0/0 final=0.00")
        return 0.0

    n_good = sum(1 for t in texts if _turn_is_good(t))
    n_total = len(texts)
    turn_fraction = n_good / n_total if n_total > 0 else 0.0

    # Binary boost: any good turn → 0.5 base, all good → 1.0
    # No good turns → 0.0
    if n_good == 0:
        quality = 0.0
    elif n_good == n_total:
        quality = 1.0
    else:
        quality = 0.3 + 0.7 * turn_fraction

    # Check for done signal
    has_done = any(_DONE_RE.search(t) for t in texts)
    if has_done:
        quality = min(quality + 0.2, 1.0)

    final = (env_reward + quality) * 3.0

    first_text = repr(texts[0][:80]) if texts else "N/A"
    print(f"[RM] env={env_reward:.2f} good={n_good}/{n_total} qual={quality:.2f} final={final:.2f} first={first_text}")
    return final
