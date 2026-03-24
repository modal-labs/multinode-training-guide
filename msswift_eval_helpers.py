import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

TRAIN_FILENAME = "train.jsonl"
EVAL_FILENAME = "eval.jsonl"
METADATA_FILENAME = "metadata.json"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def normalize_for_compare(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", normalize_text(text))


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_prompt_messages(system_message: str, prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    return messages


def build_train_record(
    system_message: str,
    prompt: str,
    chosen: str,
    rejected: str,
) -> Dict[str, Any]:
    messages = build_prompt_messages(system_message, prompt)
    messages.append({"role": "assistant", "content": chosen})
    return {"messages": messages, "rejected_response": rejected}


def build_eval_record(
    row_id: str,
    system_message: str,
    prompt: str,
    chosen: str,
    rejected: str,
) -> Dict[str, Any]:
    return {
        "id": row_id,
        "messages": build_prompt_messages(system_message, prompt),
        "reference_response": chosen,
        "rejected_response": rejected,
    }


def run_root(checkpoints_dir: str, run_prefix: str, run_id: str) -> str:
    return os.path.join(checkpoints_dir, f"{run_prefix}_{run_id}")


def latest_numbered_dir(parent_dir: str, prefix: str) -> str:
    if not os.path.isdir(parent_dir):
        raise FileNotFoundError(f"Directory does not exist: {parent_dir}")

    def _sort_key(name: str) -> int:
        suffix = name[len(prefix) :]
        try:
            return int(suffix)
        except ValueError:
            return -1

    candidates = [
        name
        for name in os.listdir(parent_dir)
        if name.startswith(prefix) and os.path.isdir(os.path.join(parent_dir, name))
    ]
    if not candidates:
        raise FileNotFoundError(f"No directories with prefix {prefix!r} under {parent_dir}")
    candidates.sort(key=_sort_key)
    return os.path.join(parent_dir, candidates[-1])


def latest_checkpoint_dir(run_dir: str) -> str:
    return latest_numbered_dir(run_dir, "checkpoint-")


def export_dir_name(checkpoint_dir: str) -> str:
    return f"{checkpoint_dir}-hf-merged"


def eval_dir(base_dir: str, surface: str) -> str:
    return os.path.join(base_dir, "eval", surface)


def score_rows_by_id(path: str) -> Dict[str, Dict[str, Any]]:
    return {row["id"]: row for row in read_jsonl(path)}


def _token_delta_summary(a: List[float], b: List[float]) -> Dict[str, Any]:
    if len(a) != len(b):
        return {
            "length_match": False,
            "a_length": len(a),
            "b_length": len(b),
            "mean_abs_delta": None,
            "max_abs_delta": None,
        }
    if not a:
        return {
            "length_match": True,
            "a_length": 0,
            "b_length": 0,
            "mean_abs_delta": 0.0,
            "max_abs_delta": 0.0,
        }
    deltas = [abs(x - y) for x, y in zip(a, b)]
    return {
        "length_match": True,
        "a_length": len(a),
        "b_length": len(b),
        "mean_abs_delta": sum(deltas) / len(deltas),
        "max_abs_delta": max(deltas),
    }


def build_parity_report(
    megatron_rows: Dict[str, Dict[str, Any]],
    hf_rows: Dict[str, Dict[str, Any]],
    sglang_rows: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    common_ids = sorted(set(megatron_rows) & set(hf_rows) & set(sglang_rows))
    per_example = []
    hf_vs_megatron = []
    sglang_vs_hf = []
    sglang_vs_megatron = []
    hf_vs_sglang_generations = 0

    for row_id in common_ids:
        mg = megatron_rows[row_id]
        hf = hf_rows[row_id]
        sg = sglang_rows[row_id]

        mg_hf_delta = _token_delta_summary(
            mg.get("response_token_logprobs", []),
            hf.get("response_token_logprobs", []),
        )
        hf_sg_delta = _token_delta_summary(
            hf.get("response_token_logprobs", []),
            sg.get("response_token_logprobs", []),
        )
        mg_sg_delta = _token_delta_summary(
            mg.get("response_token_logprobs", []),
            sg.get("response_token_logprobs", []),
        )
        hf_vs_megatron.append(mg_hf_delta["mean_abs_delta"] or 0.0)
        sglang_vs_hf.append(hf_sg_delta["mean_abs_delta"] or 0.0)
        sglang_vs_megatron.append(mg_sg_delta["mean_abs_delta"] or 0.0)

        hf_generation = normalize_for_compare(hf.get("generated_text"))
        sg_generation = normalize_for_compare(sg.get("generated_text"))
        generation_match = bool(hf_generation) and hf_generation == sg_generation
        hf_vs_sglang_generations += int(generation_match)

        per_example.append(
            {
                "id": row_id,
                "reference_response": mg.get("reference_response", ""),
                "megatron_sequence_mean_logprob": mg.get("sequence_mean_logprob"),
                "hf_sequence_mean_logprob": hf.get("sequence_mean_logprob"),
                "sglang_sequence_mean_logprob": sg.get("sequence_mean_logprob"),
                "hf_vs_megatron": mg_hf_delta,
                "sglang_vs_hf": hf_sg_delta,
                "sglang_vs_megatron": mg_sg_delta,
                "hf_generated_text": hf.get("generated_text"),
                "sglang_generated_text": sg.get("generated_text"),
                "hf_vs_sglang_generation_match": generation_match,
            }
        )

    total = len(common_ids)

    def _avg(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    return {
        "num_examples": total,
        "example_ids": common_ids,
        "summary": {
            "hf_vs_megatron_mean_abs_logprob_delta": _avg(hf_vs_megatron),
            "sglang_vs_hf_mean_abs_logprob_delta": _avg(sglang_vs_hf),
            "sglang_vs_megatron_mean_abs_logprob_delta": _avg(sglang_vs_megatron),
            "hf_vs_sglang_generation_exact_match_rate": (
                hf_vs_sglang_generations / total if total else None
            ),
        },
        "per_example": per_example,
    }
