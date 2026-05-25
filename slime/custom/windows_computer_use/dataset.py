"""Dataset generation for Windows computer use RL.

Creates prompt/target pairs across multiple task types with varying
difficulty. Each task produces a verifiable file at a known path.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SYSTEM_PROMPT = """\
You are an AI agent controlling a Windows computer. You can see screenshots of \
the desktop and interact with it using actions.

Available actions (use XML tags):
- <action>sendkey KEY</action> — Send a key or key combo (e.g., meta_l-r, ctrl-s, ret, tab)
- <action>type TEXT</action> — Type text into the active window
- <action>typeline TEXT</action> — Type text and press Enter
- <action>wait SECONDS</action> — Wait for an operation to complete (max 10s)
- <done/> — Signal that the task is complete

Think step by step about what you see on screen and what action to take next.\
"""

# ── Task definitions ──────────────────────────────────────────────────────────
# Each task has a type, difficulty, prompt builder, expected output file,
# and expected content (or a checker function name).

# --- Level 1: Simple Notepad save (type text, save to C:\output.txt) ---
NOTEPAD_SENTENCES = [
    "Hello World",
    "The quick brown fox jumps over the lazy dog",
    "Modal makes cloud computing simple and fast",
    "Reinforcement learning trains agents through rewards",
    "Windows Server 2022 is running in a virtual machine",
    "This file was created by an AI agent",
    "Computer use is the future of AI interaction",
    "QEMU provides hardware virtualization for this sandbox",
    "Screenshots help the model understand the screen",
    "Type carefully and save the file when done",
    "Python is a great programming language",
    "Machine learning models improve with training",
    "GPU clusters accelerate distributed training",
    "The reward signal guides the learning process",
    "Multi-turn rollouts enable complex task completion",
    "Every journey begins with a single step",
    "Practice makes perfect in any endeavor",
    "Artificial intelligence is transforming technology",
    "Cloud computing enables scalable infrastructure",
    "Data is the new oil of the digital economy",
    "Open source software powers the modern world",
    "Innovation requires both creativity and persistence",
    "The best way to predict the future is to create it",
    "Knowledge is power when applied with wisdom",
    "Collaboration multiplies individual capabilities",
    "Simplicity is the ultimate sophistication",
    "Continuous improvement leads to excellence",
    "Feedback loops accelerate the learning process",
    "Automation frees humans for creative work",
    "The whole is greater than the sum of its parts",
]

# --- Level 2: Notepad with specific filename (not just output.txt) ---
NOTEPAD_FILENAMES = [
    ("notes.txt", "Meeting notes: discuss Q3 roadmap and hiring plan"),
    ("readme.txt", "This project implements RL for computer use tasks"),
    ("hello.txt", "Hello from the AI agent running on Windows"),
    ("log.txt", "2026-05-22: Training run started successfully"),
    ("config.txt", "max_turns=10\ntemperature=0.8\nmodel=qwen3-vl-2b"),
    ("todo.txt", "1. Open Notepad\n2. Type this list\n3. Save the file"),
    ("report.txt", "Summary: All 7 test steps passed with reward 1.0"),
    ("data.csv", "name,score\nalice,95\nbob,87\ncharlie,92"),
]

# --- Level 3: PowerShell file creation (use command line to create file) ---
POWERSHELL_TASKS = [
    {
        "description": "Use PowerShell to write 'Hello from PowerShell' to C:\\output.txt",
        "expected": "Hello from PowerShell",
    },
    {
        "description": (
            "Use PowerShell to get today's date and write it to C:\\output.txt "
            "in the format YYYY-MM-DD (hint: Get-Date -Format yyyy-MM-dd)"
        ),
        "checker": "date_format",
    },
    {
        "description": (
            "Use PowerShell to list the files in C:\\ and save the output to C:\\output.txt "
            "(hint: dir C:\\ > C:\\output.txt)"
        ),
        "checker": "has_windows_dirs",
    },
    {
        "description": (
            "Use PowerShell to compute 7 * 13 and write only the result to C:\\output.txt "
            "(hint: Set-Content C:\\output.txt (7*13))"
        ),
        "expected": "91",
    },
    {
        "description": (
            "Use PowerShell to get the computer name and write it to C:\\output.txt "
            "(hint: $env:COMPUTERNAME | Out-File C:\\output.txt)"
        ),
        "checker": "non_empty",
    },
    {
        "description": (
            "Use PowerShell to create a file C:\\output.txt containing the numbers "
            "1 through 5, one per line (hint: 1..5 | Out-File C:\\output.txt)"
        ),
        "expected": "1\n2\n3\n4\n5",
    },
]

# --- Level 4: Multi-step tasks (create directory then save file) ---
MULTISTEP_TASKS = [
    {
        "description": (
            "Create a folder C:\\work using PowerShell (mkdir C:\\work), "
            "then open Notepad, type 'Project initialized', and save it as "
            "C:\\work\\status.txt"
        ),
        "output_path": "C:/work/status.txt",
        "expected": "Project initialized",
    },
    {
        "description": (
            "Open Notepad, type 'First file', save it as C:\\first.txt. "
            "Then open a new Notepad (Win+R, notepad), type 'Second file', "
            "and save it as C:\\output.txt"
        ),
        "output_path": "C:/output.txt",
        "expected": "Second file",
    },
    {
        "description": (
            "Use PowerShell to create C:\\output.txt containing 'step1'. "
            "Then append ' step2' to the same file "
            "(hint: Add-Content C:\\output.txt ' step2'). "
            "The final file should contain 'step1' on the first line "
            "and ' step2' on the second."
        ),
        "output_path": "C:/output.txt",
        "checker": "has_step1_step2",
    },
]


def _make_notepad_prompt(sentence: str) -> dict:
    return {
        "task_type": "notepad_simple",
        "difficulty": 1,
        "prompt": (
            f"Open Notepad on this Windows computer, type exactly the following text, "
            f"and save it as C:\\output.txt:\n\n"
            f'"{sentence}"\n\n'
            f"Steps: 1) Open Run dialog (Win+R), 2) Launch notepad, "
            f"3) Type the text, 4) Save with Ctrl+S, choose the filename, "
            f"5) Signal <done/>"
        ),
        "target": sentence,
        "output_path": "C:/output.txt",
        "checker": "exact_match",
    }


def _make_notepad_filename_prompt(filename: str, content: str) -> dict:
    return {
        "task_type": "notepad_filename",
        "difficulty": 2,
        "prompt": (
            f"Open Notepad, type the following text, and save it as C:\\{filename}:\n\n"
            f'"{content}"\n\n'
            f"Make sure to save to C:\\{filename} (not the default location)."
        ),
        "target": content,
        "output_path": f"C:/{filename}",
        "checker": "exact_match",
    }


def _make_powershell_prompt(task: dict) -> dict:
    return {
        "task_type": "powershell",
        "difficulty": 3,
        "prompt": (
            f"{task['description']}\n\n"
            f"Open PowerShell (Win+R, then type 'powershell'), run the command, "
            f"then signal <done/>."
        ),
        "target": task.get("expected", ""),
        "output_path": "C:/output.txt",
        "checker": task.get("checker", "exact_match"),
    }


def _make_multistep_prompt(task: dict) -> dict:
    return {
        "task_type": "multistep",
        "difficulty": 4,
        "prompt": (
            f"{task['description']}\n\n"
            f"Complete all steps in order, then signal <done/>."
        ),
        "target": task.get("expected", ""),
        "output_path": task.get("output_path", "C:/output.txt"),
        "checker": task.get("checker", "exact_match"),
    }


def _build_all_tasks() -> list[dict]:
    """Build the full task list across all difficulty levels."""
    tasks = []
    for sentence in NOTEPAD_SENTENCES:
        tasks.append(_make_notepad_prompt(sentence))
    for filename, content in NOTEPAD_FILENAMES:
        tasks.append(_make_notepad_filename_prompt(filename, content))
    for ps_task in POWERSHELL_TASKS:
        tasks.append(_make_powershell_prompt(ps_task))
    for ms_task in MULTISTEP_TASKS:
        tasks.append(_make_multistep_prompt(ms_task))
    return tasks


def generate_dataset(output_dir: str) -> None:
    """Generate train/test parquet files with multi-task data."""
    import pandas as pd

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tasks = _build_all_tasks()
    random.seed(42)
    random.shuffle(tasks)

    split_idx = max(1, int(len(tasks) * 0.8))
    train_tasks = tasks[:split_idx]
    test_tasks = tasks[split_idx:]

    def make_records(task_list):
        records = []
        for t in task_list:
            # Encode task metadata into the target field as JSON so that
            # build_env can extract output_path/checker regardless of
            # how Slime propagates extra parquet columns.
            target_payload = json.dumps(
                {
                    "text": t["target"],
                    "output_path": t["output_path"],
                    "checker": t["checker"],
                    "task_type": t["task_type"],
                    "difficulty": t["difficulty"],
                }
            )
            records.append(
                {
                    "prompt": t["prompt"],
                    "target": target_payload,
                    "messages": json.dumps(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": t["prompt"]},
                        ]
                    ),
                }
            )
        return records

    train_df = pd.DataFrame(make_records(train_tasks))
    test_df = pd.DataFrame(make_records(test_tasks))

    train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
    test_df.to_parquet(f"{output_dir}/test.parquet", index=False)

    print(f"Generated {len(train_df)} train + {len(test_df)} test samples")
    for d in range(1, 5):
        n = len([t for t in tasks if t["difficulty"] == d])
        print(f"  Difficulty {d}: {n} tasks")


if __name__ == "__main__":
    generate_dataset("/tmp/windows_computer_use")
