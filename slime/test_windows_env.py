"""End-to-end test for the Windows computer use environment.

Boots a Windows VM, executes multiple task types, and verifies that
different reward levels are produced. Run with:

    modal run slime/test_windows_env.py

This is a smoke test — not part of training.
"""

import io
import time

import modal

app = modal.App("test-windows-computer-use-env")


@app.local_entrypoint()
def main():
    import sys

    sys.path.insert(0, "slime")

    from custom.windows_computer_use.env_windows import CHECKERS
    from custom.windows_computer_use.sandbox_manager import (
        ADMIN_PASSWORD,
        create_rollout_sandbox,
    )

    print("=" * 60)
    print("Test: Windows Computer Use Environment — Multi-Task")
    print("=" * 60)

    # ── Unit test: reward checkers ────────────────────────────────────────────
    print("\n[0/8] Unit testing reward checkers...")
    assert CHECKERS["exact_match"]("Hello World", "Hello World") == 1.0
    assert CHECKERS["exact_match"]("Hello World\n", "Hello World") == 1.0
    assert CHECKERS["exact_match"]("hello world", "Hello World") == 0.5
    assert CHECKERS["exact_match"]("something", "Hello World") == 0.2
    assert CHECKERS["exact_match"]("", "Hello World") == 0.0
    assert CHECKERS["date_format"]("2026-05-22", "") == 1.0
    assert CHECKERS["date_format"]("Today is 2026-05-22 ok", "") == 0.5
    assert CHECKERS["date_format"]("no date here", "") == 0.1
    assert CHECKERS["has_windows_dirs"]("Windows\nUsers\nProgram Files", "") == 1.0
    assert CHECKERS["has_windows_dirs"]("Windows only", "") == 0.5
    assert CHECKERS["non_empty"]("anything", "") == 1.0
    assert CHECKERS["non_empty"]("  ", "") == 0.0
    assert CHECKERS["has_step1_step2"]("step1\n step2", "") == 1.0
    assert CHECKERS["has_step1_step2"]("step1 only", "") == 0.5
    print("  All checker unit tests passed")

    # ── Step 1: Boot VM ──────────────────────────────────────────────────────
    print("\n[1/8] Creating Windows sandbox (COW overlay)...")
    with modal.enable_output():
        sb, vm = create_rollout_sandbox(timeout=1800)

    print(f"  Sandbox: {sb.object_id}")

    # ── Step 2: Wait for screen ──────────────────────────────────────────────
    print("\n[2/8] Waiting for screen to be ready...")
    ready = vm.wait_for_screen(timeout=300)
    print(f"  Screen ready: {ready}")
    if not ready:
        print("  ERROR: Screen not ready after 300s")
        sb.terminate()
        return

    # ── Step 3: Login ────────────────────────────────────────────────────────
    print("\n[3/8] Logging into Windows...")
    vm.login(ADMIN_PASSWORD)
    print("  Setting up file server...")
    vm.setup_file_server()
    print("  Login complete, file server started")

    # ── Step 4: Take screenshot ──────────────────────────────────────────────
    print("\n[4/8] Taking screenshot...")
    raw = vm.screenshot()
    if raw:
        from PIL import Image

        print(f"  Screenshot: {len(raw)} bytes (PPM)")
        img = Image.open(io.BytesIO(raw))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        print(f"  PNG: {len(png_bytes)} bytes, size: {img.size}")
    else:
        print("  WARNING: No screenshot")

    rewards = {}

    # ── Task A: Simple Notepad save (Level 1 — should succeed) ───────────────
    print("\n[5/8] Task A: Simple Notepad save...")
    target_a = "Hello World from RL training"
    vm.sendkey("meta_l-r")
    time.sleep(3)
    vm.type_line("notepad")
    time.sleep(5)
    vm.type_text(target_a)
    time.sleep(2)
    vm.sendkey("ctrl-s")
    time.sleep(3)
    vm.sendkey("alt-n")
    time.sleep(1)
    vm.sendkey("ctrl-a")
    time.sleep(0.3)
    vm.type_text("C:\\output.txt")
    time.sleep(1)
    vm.sendkey("ret")
    time.sleep(5)

    content_a = vm.read_guest_file("C:/output.txt", timeout=10)
    if content_a is not None:
        reward_a = CHECKERS["exact_match"](content_a, target_a)
        print(f"  Content: '{content_a.strip()}'")
    else:
        reward_a = 0.0
        print("  File not found")
    rewards["notepad_simple"] = reward_a
    print(f"  Reward: {reward_a}")

    # Close Notepad
    vm.sendkey("alt-F4")
    time.sleep(2)

    # ── Task B: Check reward for wrong file path (Level 2 — should fail) ─────
    print("\n[6/8] Task B: Reading non-existent file (simulating wrong path)...")
    content_b = vm.read_guest_file("C:/notes.txt", timeout=5)
    if content_b is not None:
        reward_b = CHECKERS["exact_match"](content_b, "Meeting notes")
    else:
        reward_b = 0.0
    rewards["wrong_path"] = reward_b
    print(f"  Reward: {reward_b} (expected 0.0 — file doesn't exist)")

    # ── Task C: PowerShell file creation (Level 3) ───────────────────────────
    print("\n[7/8] Task C: PowerShell file creation...")
    vm.sendkey("meta_l-r")
    time.sleep(3)
    vm.type_line("powershell")
    time.sleep(5)
    # Write content via PowerShell
    vm.type_line("Set-Content C:\\ps_output.txt 'Hello from PowerShell'")
    time.sleep(3)
    vm.sendkey("meta_l-d")
    time.sleep(2)

    content_c = vm.read_guest_file("C:/ps_output.txt", timeout=10)
    if content_c is not None:
        reward_c = CHECKERS["exact_match"](content_c, "Hello from PowerShell")
        print(f"  Content: '{content_c.strip()}'")
    else:
        reward_c = 0.0
        print("  File not found")
    rewards["powershell"] = reward_c
    print(f"  Reward: {reward_c}")

    # ── Task D: Partial match test (should give 0.5 or 0.2) ─────────────────
    print("\n[7b/8] Task D: Partial match check...")
    # The file at C:\output.txt has "Hello World from RL training"
    # Check against a target that only partially matches
    if content_a is not None:
        reward_d_partial = CHECKERS["exact_match"](content_a, "Hello World")
        reward_d_wrong = CHECKERS["exact_match"](content_a, "Completely different text")
        print(f"  Partial target 'Hello World': reward={reward_d_partial}")
        print(f"  Wrong target: reward={reward_d_wrong}")
        rewards["partial_match"] = reward_d_partial
        rewards["wrong_content"] = reward_d_wrong
    else:
        rewards["partial_match"] = 0.0
        rewards["wrong_content"] = 0.0

    # ── Step 8: Cleanup + Summary ────────────────────────────────────────────
    print("\n[8/8] Cleaning up...")
    sb.terminate()
    print("  Sandbox terminated")

    print("\n" + "=" * 60)
    print("REWARD SUMMARY:")
    for task_name, reward in rewards.items():
        status = "PASS" if reward > 0 else "ZERO"
        print(f"  {task_name:20s}: reward={reward:.1f}  [{status}]")
    unique_rewards = set(rewards.values())
    print(f"\nUnique reward values: {sorted(unique_rewards)}")
    varying = len(unique_rewards) > 1
    print(f"Reward signal varies: {'YES' if varying else 'NO'}")
    print("=" * 60)

    if not varying:
        print("WARNING: All rewards are the same — tasks may be too easy/hard")


if __name__ == "__main__":
    main()
