from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from code_utils import code_size_bytes, extract_function_name_from_code, normalize_code

MBPP_REPO_ID = "Muennighoff/mbpp"
MBPP_JSONL_PATH = "data/mbpp.jsonl"


@dataclass
class MbppRecord:
    task_id: int
    text: str
    code: str
    test_setup_code: str
    test_list: list[str]
    challenge_test_list: list[str]
    function_name: str


def _load_mbpp_records(
    repo_id: str = MBPP_REPO_ID, dataset_file: str = MBPP_JSONL_PATH
) -> list[MbppRecord]:
    mbpp_path = hf_hub_download(repo_id=repo_id, filename=dataset_file, repo_type="dataset")
    rows = [json.loads(line) for line in Path(mbpp_path).read_text(encoding="utf-8").splitlines() if line.strip()]

    records: list[MbppRecord] = []
    for row in rows:
        code = normalize_code(row["code"])
        records.append(
            MbppRecord(
                task_id=int(row["task_id"]),
                text=row["text"].strip(),
                code=code,
                test_setup_code=normalize_code(row.get("test_setup_code", "")),
                test_list=list(row.get("test_list", [])),
                challenge_test_list=list(row.get("challenge_test_list", [])),
                function_name=extract_function_name_from_code(code),
            )
        )
    records.sort(key=lambda item: item.task_id)
    return records


def _build_instruction(record: MbppRecord) -> str:
    return textwrap.dedent(
        f"""\
        You are solving a Python code-golf programming task.

        Task:
        {record.text}

        You must define a function named `{record.function_name}`.
        Write only valid Python code to `/workspace/solution.py`.
        """
    ).strip() + "\n"


def _build_task_toml(record: MbppRecord) -> str:
    tags = json.dumps(["mbpp", "python", "code-golf"])
    return textwrap.dedent(
        f"""\
        version = "1.0"

        [metadata]
        author_name = "MBPP"
        author_email = "unknown"
        difficulty = "medium"
        category = "coding"
        tags = {tags}
        source = "mbpp"
        task_id = "{record.task_id}"
        function_name = "{record.function_name}"

        [verifier]
        timeout_sec = 180.0

        [agent]
        timeout_sec = 180.0

        [environment]
        build_timeout_sec = 300.0
        cpus = 1
        memory_mb = 2048
        storage_mb = 2048
        gpus = 0
        allow_internet = false
        mcp_servers = []

        [verifier.env]

        [solution.env]
        """
    )


def _build_test_sh() -> str:
    return textwrap.dedent(
        """\
        #!/bin/bash
        set -euo pipefail

        mkdir -p /logs/verifier
        if python3 "$(dirname "$0")/verify.py"; then
          exit 0
        fi

        if [ ! -f /logs/verifier/reward.json ]; then
          echo '{"reward": 0.0, "pass_rate": 0.0, "passed": 0, "total": 0}' > /logs/verifier/reward.json
        fi
        """
    )


def _build_verify_py(record: MbppRecord) -> str:
    all_tests = record.test_list + record.challenge_test_list
    return textwrap.dedent(
        f"""\
        import json
        import traceback
        from pathlib import Path

        TASK_ID = {record.task_id}
        FUNCTION_NAME = {record.function_name!r}
        TEST_SETUP_CODE = {record.test_setup_code!r}
        TEST_LIST = {json.dumps(all_tests)}

        SOLUTION_PATH = Path("/workspace/solution.py")
        REWARD_JSON = Path("/logs/verifier/reward.json")
        DETAILS_JSON = Path("/logs/verifier/details.json")


        def _write_outputs(reward: dict, details: dict) -> None:
            REWARD_JSON.parent.mkdir(parents=True, exist_ok=True)
            REWARD_JSON.write_text(json.dumps(reward), encoding="utf-8")
            DETAILS_JSON.write_text(json.dumps(details, indent=2), encoding="utf-8")


        def _load_solution() -> tuple[dict, str]:
            source = SOLUTION_PATH.read_text(encoding="utf-8")
            namespace: dict = {{}}
            exec(compile(source, str(SOLUTION_PATH), "exec"), namespace, namespace)
            return namespace, source


        def _run_tests(namespace: dict) -> tuple[int, int, list[dict]]:
            runtime = dict(namespace)
            if TEST_SETUP_CODE.strip():
                exec(TEST_SETUP_CODE, runtime, runtime)

            passed = 0
            failures: list[dict] = []
            for test_expr in TEST_LIST:
                try:
                    exec(test_expr, runtime, runtime)
                    passed += 1
                except Exception as exc:
                    failures.append({{"test": test_expr, "error": repr(exc)}})
            return passed, len(TEST_LIST), failures


        def main() -> int:
            details = {{"task_id": TASK_ID, "function_name": FUNCTION_NAME}}
            try:
                namespace, source = _load_solution()
                passed, total, failures = _run_tests(namespace)
                pass_rate = float(passed) / float(total) if total else 0.0
                reward = 1.0 if total > 0 and passed == total else 0.0
                details.update(
                    {{
                        "source_bytes": len(source.encode("utf-8")),
                        "passed": passed,
                        "total": total,
                        "failures": failures,
                    }}
                )
                _write_outputs(
                    {{
                        "reward": reward,
                        "pass_rate": pass_rate,
                        "passed": passed,
                        "total": total,
                    }},
                    details,
                )
                return 0
            except Exception as exc:
                details["exception"] = repr(exc)
                details["traceback"] = traceback.format_exc()
                _write_outputs(
                    {{
                        "reward": 0.0,
                        "pass_rate": 0.0,
                        "passed": 0,
                        "total": len(TEST_LIST),
                    }},
                    details,
                )
                return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )


def _build_solution_sh(record: MbppRecord) -> str:
    return textwrap.dedent(
        f"""\
        #!/bin/bash
        set -euo pipefail

        cat > /workspace/solution.py <<'PY'
        {record.code}
        PY
        """
    )


def _write_harbor_task(task_dir: Path, record: MbppRecord) -> None:
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "solution").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)

    (task_dir / "instruction.md").write_text(_build_instruction(record), encoding="utf-8")
    (task_dir / "task.toml").write_text(_build_task_toml(record), encoding="utf-8")
    (task_dir / "environment" / "Dockerfile").write_text(
        "FROM python:3.11-slim\n\nWORKDIR /workspace\n", encoding="utf-8"
    )

    solve_path = task_dir / "solution" / "solve.sh"
    solve_path.write_text(_build_solution_sh(record), encoding="utf-8")
    solve_path.chmod(0o755)

    test_path = task_dir / "tests" / "test.sh"
    test_path.write_text(_build_test_sh(), encoding="utf-8")
    test_path.chmod(0o755)

    verify_path = task_dir / "tests" / "verify.py"
    verify_path.write_text(_build_verify_py(record), encoding="utf-8")


def _build_slime_row(record: MbppRecord, harbor_task_rel: str) -> dict[str, Any]:
    prompt = textwrap.dedent(
        f"""\
        Solve the following Python task.

        Task:
        {record.text}

        Required function name: `{record.function_name}`.

        Output only Python code. Do not include Markdown fences.
        """
    ).strip()
    label_payload = {
        "task_id": record.task_id,
        "function_name": record.function_name,
        "harbor_task_rel": harbor_task_rel,
        "reference_bytes": code_size_bytes(record.code),
    }
    return {
        "task_id": record.task_id,
        "messages": [
            {"role": "system", "content": "You write short Python code that passes tests."},
            {"role": "user", "content": prompt},
        ],
        "label": json.dumps(label_payload, separators=(",", ":")),
    }


def convert_mbpp_to_harbor_and_slime(
    output_root: Path,
    train_size: int = 900,
    limit: int | None = None,
) -> dict[str, Any]:
    records = _load_mbpp_records()
    if limit is not None:
        records = records[:limit]

    if not records:
        raise ValueError("No MBPP records available for conversion.")

    tasks_root = output_root / "tasks"
    slime_root = output_root / "slime"
    tasks_root.mkdir(parents=True, exist_ok=True)
    slime_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for record in records:
        task_name = f"mbpp_{record.task_id:04d}"
        task_dir = tasks_root / task_name
        _write_harbor_task(task_dir, record)
        rows.append(_build_slime_row(record, f"tasks/{task_name}"))

    train_size = max(1, min(train_size, len(rows) - 1))
    train_rows = rows[:train_size]
    eval_rows = rows[train_size:]

    pd.DataFrame(train_rows).to_parquet(slime_root / "train.parquet", index=False)
    pd.DataFrame(eval_rows).to_parquet(slime_root / "test.parquet", index=False)

    manifest = {
        "dataset": MBPP_REPO_ID,
        "tasks": len(rows),
        "train": len(train_rows),
        "eval": len(eval_rows),
        "output_root": str(output_root),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert MBPP to Harbor + Slime dataset format.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=900)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    summary = convert_mbpp_to_harbor_and_slime(
        output_root=args.output_root,
        train_size=args.train_size,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
