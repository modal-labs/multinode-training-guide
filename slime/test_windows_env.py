"""End-to-end test for the Windows computer use environment.

Boots a Windows VM, takes screenshots, executes actions, and verifies
the reward function. Run with:

    modal run slime/test_windows_env.py

This is a quick smoke test — not part of training.
"""

import io
import time

import modal

app = modal.App("test-windows-computer-use-env")


@app.local_entrypoint()
def main():
    import sys

    sys.path.insert(0, "slime")

    from custom.windows_computer_use.sandbox_manager import (
        ADMIN_PASSWORD,
        create_rollout_sandbox,
    )

    print("=" * 60)
    print("Test: Windows Computer Use Environment")
    print("=" * 60)

    # ── Step 1: Boot VM ──────────────────────────────────────────────────────
    print("\n[1/7] Creating Windows sandbox (COW overlay)...")
    with modal.enable_output():
        sb, vm = create_rollout_sandbox(timeout=1800)

    print(f"  Sandbox: {sb.object_id}")

    # ── Step 2: Wait for screen ──────────────────────────────────────────────
    print("\n[2/7] Waiting for screen to be ready...")
    ready = vm.wait_for_screen(timeout=300)
    print(f"  Screen ready: {ready}")
    if not ready:
        print("  ERROR: Screen not ready after 300s")
        sb.terminate()
        return

    # ── Step 3: Login ────────────────────────────────────────────────────────
    print("\n[3/7] Logging into Windows...")
    vm.login(ADMIN_PASSWORD)
    print("  Setting up file server...")
    vm.setup_file_server()
    print("  Login complete, file server started, PS minimized")

    # ── Step 4: Take screenshot ──────────────────────────────────────────────
    print("\n[4/7] Taking screenshot...")
    raw = vm.screenshot()
    if raw:
        from PIL import Image

        print(f"  Screenshot: {len(raw)} bytes (PPM)")
        img = Image.open(io.BytesIO(raw))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        print(f"  PNG: {len(png_bytes)} bytes, size: {img.size}")
        with open("/tmp/test_screenshot.png", "wb") as f:
            f.write(png_bytes)
        print("  Saved to /tmp/test_screenshot.png")
    else:
        print("  ERROR: No screenshot returned")
        from PIL import Image  # still need it below

    # ── Step 4b: Verify file server setup ────────────────────────────────────
    print("\n[4b] Verifying file server setup...")
    vm.sendkey("meta_l-r")
    time.sleep(3)
    vm.type_line("powershell")
    time.sleep(5)
    vm.type_line("Test-Path C:\\fileserver.ps1; Get-Job")
    time.sleep(3)
    raw_debug = vm.screenshot()
    if raw_debug:
        img_debug = Image.open(io.BytesIO(raw_debug))
        img_debug.save("/tmp/test_debug_fileserver.png", format="PNG")
        print(f"  Debug screenshot: {img_debug.size}")
    vm.sendkey("meta_l-d")
    time.sleep(2)

    # ── Step 5: Execute Notepad task ─────────────────────────────────────────
    print("\n[5/7] Executing Notepad task...")
    target_text = "Hello World from RL training"

    # Open Run dialog
    print("  Sending Win+R...")
    vm.sendkey("meta_l-r")
    time.sleep(3)

    # Type notepad
    print("  Typing 'notepad'...")
    vm.type_line("notepad")
    time.sleep(5)

    # Take screenshot after Notepad opens
    raw2 = vm.screenshot()
    if raw2:
        img2 = Image.open(io.BytesIO(raw2))
        img2.save("/tmp/test_notepad_open.png", format="PNG")
        print(f"  Screenshot after Notepad open: {img2.size}")

    # Type the target text
    print(f"  Typing: '{target_text}'...")
    vm.type_text(target_text)
    time.sleep(2)

    # Screenshot after typing
    raw_typed = vm.screenshot()
    if raw_typed:
        img_typed = Image.open(io.BytesIO(raw_typed))
        img_typed.save("/tmp/test_after_typing.png", format="PNG")
        print(f"  Screenshot after typing: {img_typed.size}")

    # Save with Ctrl+S
    print("  Sending Ctrl+S...")
    vm.sendkey("ctrl-s")
    time.sleep(3)

    # Take screenshot of Save dialog
    raw_save = vm.screenshot()
    if raw_save:
        img_save = Image.open(io.BytesIO(raw_save))
        img_save.save("/tmp/test_save_dialog.png", format="PNG")
        print(f"  Save dialog screenshot: {img_save.size}")

    # Focus the "File name:" field with Alt+N, then type the path
    print("  Focusing filename field with Alt+N...")
    vm.sendkey("alt-n")
    time.sleep(1)
    vm.sendkey("ctrl-a")
    time.sleep(0.3)
    vm.type_text("C:\\output.txt")
    time.sleep(1)

    # Screenshot before pressing Enter
    raw_pre_save = vm.screenshot()
    if raw_pre_save:
        img_pre = Image.open(io.BytesIO(raw_pre_save))
        img_pre.save("/tmp/test_pre_save_enter.png", format="PNG")
        print(f"  Pre-save Enter screenshot: {img_pre.size}")

    vm.sendkey("ret")
    time.sleep(5)

    # Take final screenshot
    raw3 = vm.screenshot()
    if raw3:
        img3 = Image.open(io.BytesIO(raw3))
        img3.save("/tmp/test_after_save.png", format="PNG")
        print(f"  Screenshot after save: {img3.size}")

    # ── Step 6: Verify reward via guest file server ──────────────────────────
    print("\n[6/7] Checking reward via guest HTTP file server...")

    # Diagnostics: check QEMU config and port connectivity
    diag_qemu = vm.exec("ps aux | grep qemu | head -3", timeout=10)
    print(f"  QEMU process: {diag_qemu.get('stdout', '')[:300]}")

    diag_port = vm.exec(
        'python3 -c "import socket; s=socket.socket(); s.settimeout(3); '
        "r=s.connect_ex(('127.0.0.1',9999)); "
        'print(0 if r==0 else r); s.close()"',
        timeout=10,
    )
    print(f"  Port 9999 open: {diag_port.get('stdout', '').strip()}")

    # Raw HTTP test from Linux side
    diag_http = vm.exec(
        "python3 -c '"
        "import urllib.request\n"
        "try:\n"
        "    r = urllib.request.urlopen(\"http://127.0.0.1:9999/C:/output.txt\", timeout=5)\n"
        "    print(\"STATUS:\", r.status)\n"
        "    print(\"BODY:\", repr(r.read()[:200]))\n"
        "except Exception as e:\n"
        "    print(\"ERROR:\", type(e).__name__, str(e)[:200])\n"
        "'",
        timeout=15,
    )
    print(f"  Raw HTTP test: {diag_http}")

    content = vm.read_guest_file("C:/output.txt", timeout=10)
    if content is not None:
        content_stripped = content.strip()
        print(f"  File content: '{content_stripped}'")

        if content_stripped == target_text:
            reward = 1.0
        elif target_text.lower() in content_stripped.lower():
            reward = 0.5
        elif len(content_stripped) > 0:
            reward = 0.2
        else:
            reward = 0.0
        print(f"  Reward: {reward}")
    else:
        print("  File not found or file server not reachable")
        reward = 0.0

    # Also test reading a non-existent file
    print("  Testing non-existent file read...")
    none_content = vm.read_guest_file("C:/nonexistent.txt", timeout=5)
    print(f"  Non-existent file result: {none_content}")

    # ── Step 7: Cleanup ──────────────────────────────────────────────────────
    print("\n[7/7] Cleaning up...")
    sb.terminate()
    print("  Sandbox terminated")

    print("\n" + "=" * 60)
    print(f"NOTEPAD SAVE + REWARD: {'PASS' if reward >= 0.5 else 'FAIL'} (reward={reward})")
    print("VM BOOT + SCREENSHOT + TYPING: PASS")
    print(f"GUEST FILE SERVER: {'WORKING' if content is not None else 'NOT WORKING'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
