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


# ── Reward checkers ───────────────────────────────────────────────────────────


def _check_exact_match(content: str, target: str) -> float:
    content = content.strip()
    if content == target:
        return 1.0
    if target.lower() in content.lower():
        return 0.5
    if len(content) > 0:
        return 0.2
    return 0.0


def _check_date_format(content: str, _target: str) -> float:
    content = content.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", content):
        return 1.0
    if re.search(r"\d{4}-\d{2}-\d{2}", content):
        return 0.5
    if len(content) > 0:
        return 0.1
    return 0.0


def _check_has_windows_dirs(content: str, _target: str) -> float:
    content_lower = content.lower()
    known = ["windows", "users", "program files"]
    found = sum(1 for d in known if d in content_lower)
    if found >= 2:
        return 1.0
    if found >= 1:
        return 0.5
    if len(content.strip()) > 0:
        return 0.1
    return 0.0


def _check_non_empty(content: str, _target: str) -> float:
    return 1.0 if len(content.strip()) > 0 else 0.0


def _check_has_step1_step2(content: str, _target: str) -> float:
    has1 = "step1" in content
    has2 = "step2" in content
    if has1 and has2:
        return 1.0
    if has1 or has2:
        return 0.5
    if len(content.strip()) > 0:
        return 0.1
    return 0.0


CHECKERS = {
    "exact_match": _check_exact_match,
    "date_format": _check_date_format,
    "has_windows_dirs": _check_has_windows_dirs,
    "non_empty": _check_non_empty,
    "has_step1_step2": _check_has_step1_step2,
}


class WindowsComputerUseEnv:
    """RL environment: control a Windows VM via screenshots + actions.

    Conforms to Slime's BaseInteractionEnv interface:
      - reset() → (observation, info)
      - step(response_text) → (observation, done, info)
      - format_observation(observation) → chat message dict
      - close()
    """

    def __init__(
        self,
        *,
        target_text: str,
        output_path: str = "C:/output.txt",
        checker: str = "exact_match",
        vm=None,
        sandbox=None,
    ):
        self.target_text = target_text
        self.output_path = output_path
        self.checker_name = checker
        self.vm = vm
        self.sandbox = sandbox
        self.turn = 0
        self.last_reward: float = 0.0
        self.valid_action_count = 0
        self.done_signaled = False
        self.action_verbs_used: set[str] = set()
        self.partial_format_count = 0

    def reset(self) -> tuple[dict, dict]:
        self.turn = 0
        self.last_reward = 0.0
        self.valid_action_count = 0
        self.done_signaled = False
        self.action_verbs_used = set()
        self.partial_format_count = 0

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
            if "<action" in response_text or "<done" in response_text:
                self.partial_format_count += 1
            screenshot = self._take_screenshot()
            obs = {
                "obs_str": "No valid action found. Use <action>VERB ARGS</action>.",
                "multi_modal_data": {"image": [screenshot]} if screenshot else {},
            }
            return obs, False, {}

        if action["verb"] == "done":
            self.done_signaled = True
            reward = self._compute_reward()
            self.last_reward = reward
            return {}, True, {"reward": reward}

        self.valid_action_count += 1
        self.action_verbs_used.add(action["verb"])
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
        from PIL import Image as _PILImage

        observation = observation or {}
        content: list[dict] = []

        multimodal = observation.get("multi_modal_data") or {}
        for _, images in multimodal.items():
            for image in images:
                if isinstance(image, bytes):
                    image = _PILImage.open(io.BytesIO(image))
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
        """Compute reward with shaping for action quality.

        Task-completion reward (0.5-1.0) dominates when the file is correct.
        Shaping reward (0.0-0.3) provides gradient signal when the model
        hasn't completed the task but is producing valid actions.
        """
        task_reward = 0.0
        if self.vm is not None:
            try:
                content = self.vm.read_guest_file(self.output_path, timeout=10)
                if content is not None:
                    checker_fn = CHECKERS.get(self.checker_name, _check_exact_match)
                    task_reward = checker_fn(content, self.target_text)
            except Exception:
                pass

        if task_reward >= 0.5:
            return task_reward

        shaping = 0.0
        if self.partial_format_count > 0:
            shaping += min(self.partial_format_count * 0.03, 0.06)
        if self.valid_action_count > 0:
            shaping += min(self.valid_action_count * 0.05, 0.15)
        relevant = {"sendkey", "type", "typeline", "wait"}
        if self.action_verbs_used & relevant:
            shaping += 0.1
        if self.done_signaled:
            shaping += 0.1
        return min(shaping, 0.4)


# --------------------------------------------------------------------------
# Slime env factory — called by the rollout to instantiate the env
# --------------------------------------------------------------------------


def _parse_target(raw_target: str) -> tuple[str, str, str]:
    """Parse the target field which may be JSON-encoded task metadata.

    Returns (target_text, output_path, checker).
    """
    import json as _json

    if raw_target.startswith("{"):
        try:
            payload = _json.loads(raw_target)
            return (
                payload.get("text", ""),
                payload.get("output_path", "C:/output.txt"),
                payload.get("checker", "exact_match"),
            )
        except _json.JSONDecodeError:
            pass
    return raw_target, "C:/output.txt", "exact_match"


def build_env(sample=None, args=None, **_) -> WindowsComputerUseEnv:
    """Build a WindowsComputerUseEnv for a rollout sample.

    Called by the Slime VLM multi-turn rollout. Creates a fresh Windows
    sandbox for each sample.
    """
    from custom.windows_computer_use.sandbox_manager import boot_and_login

    target_text = ""
    output_path = "C:/output.txt"
    checker = "exact_match"

    if sample is not None:
        # First try sample.label (set by label_key in config)
        raw_target = ""
        if hasattr(sample, "label") and sample.label:
            raw_target = sample.label
        else:
            metadata = getattr(sample, "metadata", None) or {}
            raw_target = metadata.get("target", "")

        target_text, output_path, checker = _parse_target(raw_target)

    sb, vm = boot_and_login(timeout=1800)

    return WindowsComputerUseEnv(
        target_text=target_text,
        output_path=output_path,
        checker=checker,
        vm=vm,
        sandbox=sb,
    )
