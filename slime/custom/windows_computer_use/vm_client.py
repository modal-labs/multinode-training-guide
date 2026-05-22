"""Minimal HTTP client for controlling a Windows VM via the RPC server.

Vendored from windows-sandboxes/client.py so we don't need that repo
as a dependency inside the Slime container. Talks to the RPC server
(server.py) running alongside QEMU in a Modal sandbox.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class WindowsVM:
    """Client for a Windows VM running in a Modal sandbox."""

    def __init__(self, host: str, port: int = 8765):
        self.base_url = f"https://{host}"
        self._timeout = 30

    @classmethod
    def from_sandbox(cls, sb) -> "WindowsVM":
        tunnels = sb.tunnels()
        for port_num, tunnel in tunnels.items():
            if port_num == 8765:
                return cls(tunnel.host)
        raise RuntimeError(
            f"No tunnel found for RPC port 8765. "
            f"Available ports: {list(tunnels.keys())}"
        )

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())

    def _post(self, path: str, data: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        payload = json.dumps(data or {}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())

    def _post_long(
        self, path: str, data: dict | None = None, timeout: int = 300
    ) -> dict:
        url = f"{self.base_url}{path}"
        payload = json.dumps(data or {}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())

    # ----- Status -----

    def health(self) -> bool:
        try:
            return self._get("/health").get("ok", False)
        except Exception:
            return False

    def wait_ready(self, timeout: int = 120):
        start = time.time()
        while time.time() - start < timeout:
            if self.health():
                return
            time.sleep(2)
        raise TimeoutError(f"RPC server not ready after {timeout}s")

    # ----- Keyboard -----

    def sendkey(self, key: str) -> dict:
        return self._post("/sendkey", {"key": key})

    def type_text(self, text: str, delay: float = 0.12) -> dict:
        return self._post("/type", {"text": text, "delay": delay})

    def type_line(self, text: str, delay: float = 0.12) -> dict:
        return self._post("/type", {"text": text, "delay": delay, "enter": True})

    def send_keys(self, keys: list[str], delay: float = 0.3) -> dict:
        return self._post("/keys", {"keys": keys, "delay": delay})

    # ----- Screenshot -----

    def screenshot(self) -> bytes | None:
        url = f"{self.base_url}/screenshot"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.read()
        except Exception:
            return None

    def screenshot_size(self) -> int:
        return self._get("/screenshot/size").get("size", 0)

    def wait_for_screen(self, timeout: int = 300, min_size: int = 500_000) -> bool:
        start = time.time()
        time.sleep(20)
        while time.time() - start < timeout:
            size = self.screenshot_size()
            if size > min_size:
                return True
            time.sleep(10)
        return False

    # ----- Shell -----

    def exec(self, command: str, timeout: int = 30) -> dict:
        return self._post_long(
            "/exec", {"command": command, "timeout": timeout}, timeout=timeout + 10
        )

    # ----- Lifecycle -----

    def login(self, password: str):
        self.sendkey("ctrl-alt-delete")
        time.sleep(8)
        self.type_text(password, delay=0.15)
        self.sendkey("ret")
        time.sleep(25)
        self.sendkey("meta_l-r")
        time.sleep(3)
        self.type_line("powershell")
        time.sleep(5)
        self.type_line("powercfg /change monitor-timeout-ac 0")
        time.sleep(1)
        self.type_line("powercfg /change standby-timeout-ac 0")
        time.sleep(1)

    def shutdown(self) -> dict:
        return self._post("/shutdown")
