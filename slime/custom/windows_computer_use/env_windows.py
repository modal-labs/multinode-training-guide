"""Interactive environment for Windows computer use RL.

Implements the BaseInteractionEnv contract from Slime's VLM multi-turn
rollout. Each step: parse the model's action → execute on Windows VM →
take a screenshot → return as observation.
"""

from __future__ import annotations

import io
import logging
import re
import time

logger = logging.getLogger(__name__)

# Action parsing: <action>VERB ARGS</action> or <done/>
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_DONE_RE = re.compile(r"<done\s*/?>")


def _ppm_to_png(ppm_bytes: bytes) -> bytes:
    """Convert raw PPM screenshot to PNG for the VLM."""
    from PIL import Image

    img = Image.open(io.BytesIO(ppm_bytes))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _parse_action(text: str) -> dict | None:
    """Extract the last action from model output.

    Returns dict with 'verb' and 'args', or None if no action found.
    """
    if _DONE_RE.search(text):
        return {"verb": "done", "args": ""}

    matches = list(_ACTION_RE.finditer(text))
    if not matches:
        return None

    raw = matches[-1].group(1).strip()
    parts = raw.split(None, 1)
    verb = parts[0].lower() if parts else ""
    args = parts[1].strip().strip('"').strip("'") if len(parts) > 1 else ""
    return {"verb": verb, "args": args}


def _execute_action(vm, action: dict) -> str:
    """Execute a parsed action on the Windows VM.

    Returns a short status string.
    """
    verb = action["verb"]
    args = action["args"]

    if verb == "type":
        result = vm.type_text(args)
        return f"Typed {result.get('typed', 0)} characters"
    elif verb == "typeline":
        result = vm.type_line(args)
        return f"Typed line: {args}"
    elif verb == "sendkey":
        key = args.replace(" ", "-")
        vm.sendkey(key)
        return f"Sent key: {key}"
    elif verb == "wait":
        try:
            secs = min(float(args), 10.0)
        except (ValueError, TypeError):
            secs = 2.0
        time.sleep(secs)
        return f"Waited {secs}s"
    elif verb == "done":
        return "Task complete"
    else:
        return f"Unknown action: {verb}"


class WindowsComputerUseEnv:
    """RL environment: control a Windows VM via screenshots + actions.

    Conforms to Slime's BaseInteractionEnv interface:
      - reset() → (observation, info)
      - step(response_text) → (observation, done, info)
      - format_observation(observation) → chat message dict
      - close()
    """

    def __init__(self, *, target_text: str, vm=None, sandbox=None):
        self.target_text = target_text
        self.vm = vm
        self.sandbox = sandbox
        self.turn = 0
        self.last_reward: float = 0.0

    def reset(self) -> tuple[dict, dict]:
        self.turn = 0
        self.last_reward = 0.0

        screenshot = self._take_screenshot()
        obs = {
            "obs_str": (
                "You are looking at a Windows desktop. "
                "Use <action>...</action> tags to interact. "
                "Available actions: type, sendkey, wait, typeline. "
                "When done, output <done/>."
            ),
            "multi_modal_data": {"image": [screenshot]} if screenshot else {},
        }
        return obs, {}

    def step(self, response_text: str) -> tuple[dict, bool, dict]:
        self.turn += 1
        action = _parse_action(response_text)

        if action is None:
            screenshot = self._take_screenshot()
            obs = {
                "obs_str": "No valid action found. Use <action>VERB ARGS</action>.",
                "multi_modal_data": {"image": [screenshot]} if screenshot else {},
            }
            return obs, False, {}

        if action["verb"] == "done":
            reward = self._compute_reward()
            self.last_reward = reward
            return {}, True, {"reward": reward}

        status = _execute_action(self.vm, action)
        logger.info("Turn %d: %s → %s", self.turn, action, status)

        time.sleep(1.5)
        screenshot = self._take_screenshot()

        obs = {
            "obs_str": status,
            "multi_modal_data": {"image": [screenshot]} if screenshot else {},
        }
        return obs, False, {}

    def format_observation(self, observation: dict) -> dict:
        """Convert observation to a chat message with image content."""
        observation = observation or {}
        content: list[dict] = []

        multimodal = observation.get("multi_modal_data") or {}
        for _, images in multimodal.items():
            for image in images:
                content.append({"type": "image", "image": image})

        obs_text = observation.get("obs_str", "")
        if obs_text:
            content.append({"type": "text", "text": obs_text})

        return {"role": "user", "content": content}

    def close(self):
        if self.sandbox is not None:
            try:
                self.sandbox.terminate()
            except Exception:
                pass

    def _take_screenshot(self) -> bytes | None:
        """Take a screenshot and convert PPM → PNG."""
        if self.vm is None:
            return None
        raw = self.vm.screenshot()
        if raw is None:
            return None
        try:
            return _ppm_to_png(raw)
        except Exception:
            logger.warning("Failed to convert screenshot to PNG")
            return None

    def _compute_reward(self) -> float:
        """Check if the target file was created with the correct content."""
        if self.vm is None:
            return 0.0
        try:
            result = self.vm.exec(
                'powershell -Command "Get-Content C:\\output.txt -Raw"',
                timeout=10,
            )
            content = result.get("stdout", "").strip()
            if content == self.target_text:
                return 1.0
            # Partial credit for partial match
            if self.target_text.lower() in content.lower():
                return 0.5
            if len(content) > 0:
                return 0.2
            return 0.0
        except Exception:
            return 0.0


# --------------------------------------------------------------------------
# Slime env factory — called by the rollout to instantiate the env
# --------------------------------------------------------------------------


def build_env(sample=None, args=None, **_) -> WindowsComputerUseEnv:
    """Build a WindowsComputerUseEnv for a rollout sample.

    Called by the Slime VLM multi-turn rollout. Creates a fresh Windows
    sandbox for each sample.
    """
    from custom.windows_computer_use.sandbox_manager import boot_and_login

    target_text = ""
    if sample is not None:
        metadata = getattr(sample, "metadata", None) or {}
        target_text = metadata.get("target", "")
        if not target_text and hasattr(sample, "label"):
            target_text = sample.label or ""

    sb, vm = boot_and_login(timeout=1800)

    return WindowsComputerUseEnv(
        target_text=target_text,
        vm=vm,
        sandbox=sb,
    )
