"""Manage Windows VM sandboxes for RL rollouts.

Each rollout gets its own sandbox with a copy-on-write disk overlay so
that the base Windows image is never modified.
"""

from __future__ import annotations

import base64
import textwrap

import modal

from custom.windows_computer_use.vm_client import WindowsVM

VOLUME_NAME = "windows-qemu-disk"
NOVNC_PORT = 6080
RPC_PORT = 8765
ADMIN_PASSWORD = "P@ssw0rd123"

# Sandbox image: QEMU + deps (matches windows-sandboxes)
sandbox_image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "qemu-system-x86",
    "qemu-utils",
    "ovmf",
    "kmod",
    "socat",
    "python3-websockify",
    "genisoimage",
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# RPC server code — vendored inline so we don't need the windows-sandboxes repo.
# This runs inside the sandbox alongside QEMU.
_SERVER_PY = textwrap.dedent(r'''
"""RPC server for controlling the Windows VM via QEMU HMP."""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import subprocess
import os
import base64
import time
import threading

HMP_SOCK = "/tmp/qemu-hmp.sock"
PID_FILE = "/tmp/qemu.pid"
SCREENSHOT_PATH = "/tmp/screen.ppm"

_SHIFT_MAP = {
    "!": "shift-1", "@": "shift-2", "#": "shift-3", "$": "shift-4",
    "%": "shift-5", "^": "shift-6", "&": "shift-7", "*": "shift-8",
    "(": "shift-9", ")": "shift-0", "_": "shift-minus", "+": "shift-equal",
    "{": "shift-bracket_left", "}": "shift-bracket_right",
    "|": "shift-backslash", ":": "shift-semicolon", '"': "shift-apostrophe",
    "<": "shift-comma", ">": "shift-dot", "?": "shift-slash",
    "~": "shift-grave_accent",
}
for _c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    _SHIFT_MAP[_c] = f"shift-{_c.lower()}"

_NORMAL_MAP = {
    " ": "spc", "-": "minus", "=": "equal", "[": "bracket_left",
    "]": "bracket_right", "\\": "backslash", ";": "semicolon",
    "'": "apostrophe", ",": "comma", ".": "dot", "/": "slash",
    "`": "grave_accent",
}


def char_to_key(c):
    if c in _SHIFT_MAP:
        return _SHIFT_MAP[c]
    if c in _NORMAL_MAP:
        return _NORMAL_MAP[c]
    if c.isalnum():
        return c
    return None


_hmp_lock = threading.Lock()


def hmp_send(command):
    with _hmp_lock:
        try:
            result = subprocess.run(
                ["socat", "-", f"UNIX-CONNECT:{HMP_SOCK}"],
                input=f"{command}\n",
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "timeout"
        except Exception as e:
            return f"error: {e}"


def sendkey(key):
    return hmp_send(f"sendkey {key}")


def type_text(text, delay=0.12):
    typed = 0
    skipped = []
    for c in text:
        key = char_to_key(c)
        if key is None:
            skipped.append(c)
            continue
        sendkey(key)
        typed += 1
        time.sleep(delay)
    return {"typed": typed, "skipped": skipped, "total": len(text)}


def type_line(text, delay=0.12):
    result = type_text(text, delay)
    sendkey("ret")
    return result


def qemu_running():
    try:
        result = subprocess.run(
            ["pgrep", "-c", "qemu-system"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() != "0"
    except Exception:
        return False


def qemu_pid():
    try:
        with open(PID_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return None


def take_screenshot():
    hmp_send(f"screendump {SCREENSHOT_PATH}")
    time.sleep(0.5)
    try:
        with open(SCREENSHOT_PATH, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None


def screenshot_size():
    data = take_screenshot()
    return len(data) if data else 0


class VMHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        if self.path == "/status":
            self._json_response({
                "qemu_running": qemu_running(),
                "qemu_pid": qemu_pid(),
                "hmp_socket": os.path.exists(HMP_SOCK),
            })
        elif self.path == "/screenshot":
            data = take_screenshot()
            if data is None:
                self._json_response({"error": "no screenshot"}, 500)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/x-portable-pixmap")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/screenshot/size":
            self._json_response({"size": screenshot_size()})
        elif self.path == "/health":
            self._json_response({"ok": True})
        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self):
        body = self._read_body()

        if self.path == "/hmp":
            result = hmp_send(body.get("command", ""))
            self._json_response({"result": result})
        elif self.path == "/sendkey":
            key = body.get("key", "")
            result = sendkey(key)
            self._json_response({"result": result, "key": key})
        elif self.path == "/type":
            text = body.get("text", "")
            delay = body.get("delay", 0.12)
            press_enter = body.get("enter", False)
            if press_enter:
                result = type_line(text, delay)
            else:
                result = type_text(text, delay)
            self._json_response(result)
        elif self.path == "/keys":
            keys = body.get("keys", [])
            delay = body.get("delay", 0.3)
            for key in keys:
                sendkey(key)
                time.sleep(delay)
            self._json_response({"sent": len(keys)})
        elif self.path == "/shutdown":
            hmp_send("system_powerdown")
            self._json_response({"result": "shutdown sent"})
        elif self.path == "/reset":
            hmp_send("system_reset")
            self._json_response({"result": "reset sent"})
        elif self.path == "/exec":
            cmd = body.get("command", "")
            timeout = body.get("timeout", 30)
            try:
                result = subprocess.run(
                    ["bash", "-c", cmd],
                    capture_output=True, text=True, timeout=timeout,
                )
                self._json_response({
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                })
            except subprocess.TimeoutExpired:
                self._json_response({"error": "timeout"}, 504)
            except Exception as e:
                self._json_response({"error": str(e)}, 500)
        elif self.path == "/guest-file":
            # Read a file from the Windows guest via its HTTP file server
            import urllib.request, urllib.error
            guest_path = body.get("path", "")
            timeout = body.get("timeout", 10)
            url = f"http://127.0.0.1:9999/{guest_path}"
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    content = resp.read().decode("utf-8", errors="replace")
                self._json_response({"content": content, "path": guest_path})
            except urllib.error.HTTPError as e:
                self._json_response({"error": f"HTTP {e.code}", "path": guest_path}, e.code)
            except Exception as e:
                self._json_response({"error": str(e), "path": guest_path}, 500)
        else:
            self._json_response({"error": "not found"}, 404)


def main():
    port = int(os.environ.get("RPC_PORT", "8765"))
    server = HTTPServer(("0.0.0.0", port), VMHandler)
    print(f"VM RPC server listening on :{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
''').lstrip()


# PowerShell HTTP file server that runs inside the Windows guest.
# Written to a floppy image and copied during login.
_FILESERVER_PS1 = r"""$h=[System.Net.HttpListener]::new()
$h.Prefixes.Add("http://*:9999/")
$h.Start()
while($true){
    $c=$h.GetContext()
    $p=$c.Request.Url.LocalPath.TrimStart('/')
    try{
        $b=[IO.File]::ReadAllBytes($p)
        $c.Response.OutputStream.Write($b,0,$b.Length)
    }catch{
        $c.Response.StatusCode=404
    }
    $c.Response.Close()
}
"""


# Entrypoint that boots QEMU from a COW overlay disk
_ENTRYPOINT = r"""#!/bin/bash
set -e

BASE_DISK="/vol/windows-disk.qcow2"
OVMF_VARS_SRC="/vol/ovmf_vars.fd"

# Create copy-on-write overlay so we never modify the base disk
OVERLAY="/tmp/rollout_disk.qcow2"
qemu-img create -f qcow2 -b "$BASE_DISK" -F qcow2 "$OVERLAY"
echo "Created COW overlay: $OVERLAY (backing: $BASE_DISK)"

# Copy OVMF vars so each VM gets its own UEFI state
cp "$OVMF_VARS_SRC" /tmp/ovmf_vars.fd

# Create a small ISO image with utility scripts for the guest
ISO="/tmp/scripts.iso"
mkdir -p /tmp/isodir
cp /sandbox/fileserver.ps1 /tmp/isodir/
genisoimage -quiet -o $ISO -J -R /tmp/isodir
echo "Created ISO with guest scripts"

QEMU_ARGS=(
    -enable-kvm -m 4096 -cpu host -smp 4
    -drive "if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd"
    -drive "if=pflash,format=raw,file=/tmp/ovmf_vars.fd"
    -drive "file=$OVERLAY,format=qcow2,if=ide"
    -display none
    -vnc "127.0.0.1:0"
    -monitor "unix:/tmp/qemu-hmp.sock,server,nowait"
    -daemonize -pidfile /tmp/qemu.pid
    -usb -device usb-tablet
    -vga std
    -nic "user,hostfwd=tcp:127.0.0.1:9999-:9999"
    -drive "file=$ISO,media=cdrom,readonly=on"
)

echo "=== Starting QEMU ==="
qemu-system-x86_64 "${QEMU_ARGS[@]}"

PID=$(cat /tmp/qemu.pid)
echo "QEMU PID: $PID"

python3 /sandbox/server.py &
echo "RPC server started on :8765"

echo "=== READY ==="
while kill -0 $PID 2>/dev/null; do sleep 10; done
echo "QEMU exited."
"""


def _write_b64(sb, path: str, content: bytes):
    b64 = base64.b64encode(content).decode()
    p = sb.exec(
        "bash", "-c", f'mkdir -p "$(dirname {path})"; echo "{b64}" | base64 -d > {path}'
    )
    p.wait()


def create_rollout_sandbox(timeout: int = 3600) -> tuple[modal.Sandbox, WindowsVM]:
    """Create a Windows VM sandbox for a single RL rollout.

    Uses a COW overlay so the base disk image is never modified.

    Returns:
        (sandbox, vm_client) tuple
    """
    app_ref = modal.App.lookup("windows-computer-use-rl", create_if_missing=True)

    sb = modal.Sandbox.create(
        app=app_ref,
        image=sandbox_image,
        cpu=4,
        memory=8192,
        timeout=timeout,
        experimental_options={"vm_runtime": True},
        encrypted_ports=[RPC_PORT],
        volumes={"/vol": vol},
    )

    # Upload the RPC server and guest file server script
    _write_b64(sb, "/sandbox/server.py", _SERVER_PY.encode())
    _write_b64(sb, "/sandbox/fileserver.ps1", _FILESERVER_PS1.encode())

    # Write and run the entrypoint
    _write_b64(sb, "/sandbox/entrypoint.sh", _ENTRYPOINT.encode())
    sb.exec("bash", "-c", "bash /sandbox/entrypoint.sh")

    # Create and connect client
    vm = WindowsVM.from_sandbox(sb)
    vm.wait_ready(timeout=120)

    return sb, vm


def boot_and_login(timeout: int = 3600, retries: int = 2) -> tuple[modal.Sandbox, WindowsVM]:
    """Create a sandbox, boot Windows, login, and return ready-to-use VM."""
    import time as _time

    for attempt in range(retries + 1):
        t0 = _time.time()
        sb = None
        try:
            print(f"[boot_and_login] Creating sandbox (attempt {attempt+1}/{retries+1})...")
            sb, vm = create_rollout_sandbox(timeout=timeout)
            print(f"[boot_and_login] Sandbox created in {_time.time()-t0:.0f}s, waiting for screen...")
            vm.wait_for_screen(timeout=300)
            print(f"[boot_and_login] Screen ready at {_time.time()-t0:.0f}s, logging in...")
            vm.login(ADMIN_PASSWORD)
            print(f"[boot_and_login] Login done at {_time.time()-t0:.0f}s, setting up file server...")
            vm.setup_file_server()
            print(f"[boot_and_login] File server ready. Total boot time: {_time.time()-t0:.0f}s")
            return sb, vm
        except Exception as e:
            print(f"[boot_and_login] Attempt {attempt+1} failed: {type(e).__name__}: {e}")
            if sb is not None:
                try:
                    sb.terminate()
                except Exception:
                    pass
            if attempt >= retries:
                raise
            _time.sleep(5)
    raise RuntimeError("boot_and_login: all retries exhausted")
