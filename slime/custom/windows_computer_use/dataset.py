"""Dataset generation for Windows computer use RL.

Creates prompt/target pairs for the Notepad file-saving task.
Each prompt instructs the model to open Notepad, type a specific
sentence, and save it as C:\\output.txt.
"""

from __future__ import annotations

import json
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

# Sentences varying in length and complexity
SENTENCES = [
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
    "Curiosity is the engine of achievement",
    "Testing in production builds resilience",
    "Infrastructure as code enables reproducibility",
    "Distributed systems require careful coordination",
    "Observability helps debug complex systems",
    "Containers provide portable deployment units",
    "Serverless computing abstracts infrastructure",
    "Version control tracks changes over time",
    "APIs enable communication between services",
    "Monitoring and alerting prevent outages",
    "Good documentation saves future developers time",
    "Security should be built into every layer",
    "Performance optimization requires measurement",
    "Scalability planning prevents growing pains",
    "Resilience engineering prepares for failure",
    "The cloud enables global scale deployment",
    "Microservices decompose monolithic applications",
    "Event-driven architecture improves responsiveness",
    "Caching reduces latency and improves throughput",
    "Load balancing distributes traffic evenly",
]


def _make_prompt(sentence: str) -> str:
    return (
        f"Open Notepad on this Windows computer, type exactly the following text, "
        f"and save it as C:\\output.txt:\n\n"
        f'"{sentence}"\n\n'
        f"Steps: 1) Open Run dialog (Win+R), 2) Launch notepad, "
        f"3) Type the text, 4) Save with Ctrl+S, choose the filename, 5) Signal <done/>"
    )


def generate_dataset(output_dir: str) -> None:
    """Generate train/test parquet files for the Notepad task."""
    import pandas as pd

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 80/20 train/test split
    split_idx = int(len(SENTENCES) * 0.8)
    train_sentences = SENTENCES[:split_idx]
    test_sentences = SENTENCES[split_idx:]

    def make_records(sentences):
        records = []
        for s in sentences:
            records.append(
                {
                    "prompt": _make_prompt(s),
                    "target": s,
                    "messages": json.dumps(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": _make_prompt(s)},
                        ]
                    ),
                }
            )
        return records

    train_df = pd.DataFrame(make_records(train_sentences))
    test_df = pd.DataFrame(make_records(test_sentences))

    train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
    test_df.to_parquet(f"{output_dir}/test.parquet", index=False)

    print(
        f"Generated {len(train_df)} train + {len(test_df)} test samples → {output_dir}"
    )


if __name__ == "__main__":
    generate_dataset("/data/windows_computer_use")
