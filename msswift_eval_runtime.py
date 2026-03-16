import os
import signal
import subprocess
import time
from typing import Any, Dict, List, Optional

import requests
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from msswift_eval_helpers import (
    build_parity_report,
    normalize_text,
    read_jsonl,
    score_rows_by_id,
    write_json,
    write_jsonl,
)


def _tokenizer_from_model(model_dir: str):
    try:
        return AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=False,
            local_files_only=True,
        )
    except Exception:
        return AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=False,
            token=os.environ.get("HF_TOKEN"),
        )


def _chat_tokens(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> List[int]:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return token_ids


def _response_tokens(tokenizer, response_text: str) -> List[int]:
    return tokenizer.encode(response_text, add_special_tokens=False)


def hf_score_and_generate(
    model_dir: str,
    tokenizer_dir: str,
    eval_dataset: str,
    output_dir: str,
    max_samples: Optional[int],
    max_new_tokens: int,
    adapter_dir: Optional[str] = None,
) -> Dict[str, Any]:
    tokenizer = _tokenizer_from_model(tokenizer_dir)
    config = AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code=False,
        local_files_only=True,
    )
    if not getattr(config, "_name_or_path", None):
        config._name_or_path = model_dir
    if getattr(config, "name_or_path", None) is None:
        config.name_or_path = model_dir
    model_cls = AutoModelForCausalLM
    architectures = getattr(config, "architectures", None) or []
    if architectures:
        architecture = architectures[0]
        model_cls = getattr(transformers, architecture, None)
        if model_cls is None and architecture == "Glm4MoeForCausalLM":
            from transformers.models.glm4_moe.modeling_glm4_moe import (
                Glm4MoeForCausalLM,
            )

            model_cls = Glm4MoeForCausalLM
        if model_cls is None:
            model_cls = AutoModelForCausalLM

    try:
        model = model_cls.from_pretrained(
            model_dir,
            config=config,
            trust_remote_code=False,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except Exception:
        model = model_cls.from_pretrained(
            model_dir,
            config=config,
            trust_remote_code=True,
            local_files_only=False,
            token=os.environ.get("HF_TOKEN"),
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    if adapter_dir is not None:
        from peft import PeftModel

        print(f"Loading PEFT adapter from {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)
        print("Loaded PEFT adapter")
    model = model.eval()
    input_device = next(model.parameters()).device
    eval_rows = read_jsonl(eval_dataset)
    if max_samples is not None:
        eval_rows = eval_rows[:max_samples]

    results: List[Dict[str, Any]] = []
    total_rows = len(eval_rows)
    for idx, row in enumerate(eval_rows, start=1):
        prompt_ids = _chat_tokens(tokenizer, row["messages"], add_generation_prompt=True)
        response_token_ids = _response_tokens(tokenizer, row["reference_response"])
        if not response_token_ids:
            continue
        full_ids = prompt_ids + response_token_ids

        full_tensor = torch.tensor([full_ids], device=input_device, dtype=torch.long)
        prompt_tensor = torch.tensor([prompt_ids], device=input_device, dtype=torch.long)
        full_attention_mask = torch.ones_like(full_tensor, device=input_device)
        prompt_attention_mask = torch.ones_like(prompt_tensor, device=input_device)

        with torch.inference_mode():
            logits = model(
                input_ids=full_tensor,
                attention_mask=full_attention_mask,
                use_cache=False,
            ).logits[0, :-1].float()
            if max_new_tokens > 0:
                generated = model.generate(
                    input_ids=prompt_tensor,
                    attention_mask=prompt_attention_mask,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )[0]
            else:
                generated = prompt_tensor[0]

        target_ids = torch.tensor(full_ids[1:], device=logits.device, dtype=torch.long)
        token_logprobs = torch.log_softmax(logits, dim=-1).gather(
            -1, target_ids.unsqueeze(-1)
        ).squeeze(-1)
        response_start = len(prompt_ids) - 1
        response_token_logprobs = token_logprobs[response_start:].cpu().tolist()
        generated_ids = generated[len(prompt_ids) :].cpu().tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        results.append(
            {
                "id": row["id"],
                "messages": row["messages"],
                "reference_response": row["reference_response"],
                "response_token_ids": response_token_ids,
                "response_token_logprobs": response_token_logprobs,
                "sequence_sum_logprob": sum(response_token_logprobs),
                "sequence_mean_logprob": (
                    sum(response_token_logprobs) / len(response_token_logprobs)
                ),
                "generated_token_ids": generated_ids,
                "generated_text": normalize_text(generated_text),
            }
        )
        if idx == 1 or idx == total_rows or idx % 8 == 0:
            print(f"HF eval progress: {idx}/{total_rows} examples")

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "per_example.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")
    write_jsonl(results_path, results)
    write_json(
        summary_path,
        {
            "num_examples": len(results),
            "mean_sequence_logprob": (
                sum(row["sequence_mean_logprob"] for row in results) / len(results)
                if results
                else None
            ),
            "results_path": results_path,
        },
    )
    return {"results_path": results_path, "summary_path": summary_path, "count": len(results)}


def _wait_for_server(base_url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_error: Optional[Exception] = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            if response.ok:
                return
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"SGLang server at {base_url} did not become ready: {last_error}")


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _extract_requested_token_logprob(
    score_payload: Dict[str, Any], target_token_id: int
) -> float:
    meta_info = score_payload.get("meta_info", {})
    candidate_groups = meta_info.get("output_token_ids_logprobs") or []
    for position_group in reversed(candidate_groups):
        if not position_group:
            continue
        for candidate in position_group:
            if (
                isinstance(candidate, (list, tuple))
                and len(candidate) >= 2
                and candidate[0] is not None
                and candidate[1] == target_token_id
            ):
                return float(candidate[0])

    candidate_groups = meta_info.get("input_token_ids_logprobs") or []
    for position_group in reversed(candidate_groups):
        if not position_group:
            continue
        for candidate in position_group:
            if (
                isinstance(candidate, (list, tuple))
                and len(candidate) >= 2
                and candidate[0] is not None
                and candidate[1] == target_token_id
            ):
                return float(candidate[0])

    raise ValueError(
        f"Could not find requested token logprob for token_id={target_token_id}: "
        f"{score_payload}"
    )


def sglang_score_and_generate(
    model_dir: str,
    tokenizer_dir: str,
    eval_dataset: str,
    output_dir: str,
    max_samples: Optional[int],
    max_new_tokens: int,
    tp_size: int,
    port: int = 30000,
    startup_timeout_s: int = 900,
    server_extra_args: Optional[List[str]] = None,
    lora_name: Optional[str] = None,
) -> Dict[str, Any]:
    tokenizer = _tokenizer_from_model(tokenizer_dir)
    eval_rows = read_jsonl(eval_dataset)
    if max_samples is not None:
        eval_rows = eval_rows[:max_samples]
    os.makedirs(output_dir, exist_ok=True)

    base_url = f"http://127.0.0.1:{port}"
    command = [
        "sglang",
        "serve",
        "--model-path",
        model_dir,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tp-size",
        str(tp_size),
        "--trust-remote-code",
    ]
    if server_extra_args:
        command.extend(server_extra_args)

    process = subprocess.Popen(command)
    try:
        _wait_for_server(base_url, startup_timeout_s)
        results: List[Dict[str, Any]] = []
        total_rows = len(eval_rows)
        for idx, row in enumerate(eval_rows, start=1):
            prompt_ids = _chat_tokens(tokenizer, row["messages"], add_generation_prompt=True)
            response_token_ids = _response_tokens(tokenizer, row["reference_response"])
            if not response_token_ids:
                continue
            response_logprobs = []
            for token_idx, target_token_id in enumerate(response_token_ids):
                context_ids = prompt_ids + response_token_ids[:token_idx]
                score_response = requests.post(
                    f"{base_url}/generate",
                    json={
                        "input_ids": context_ids,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                        "return_logprob": True,
                        "return_text_in_logprobs": True,
                        "token_ids_logprob": [target_token_id],
                        **({"lora_path": lora_name} if lora_name else {}),
                    },
                    timeout=300,
                )
                score_response.raise_for_status()
                score_payload = score_response.json()
                if idx == 1 and token_idx == 0:
                    write_json(
                        os.path.join(output_dir, "sglang_first_score_payload.json"),
                        score_payload,
                    )
                response_logprobs.append(
                    _extract_requested_token_logprob(score_payload, target_token_id)
                )

            generated_ids = []
            generated_text = ""
            generated_token_logprobs = []
            if max_new_tokens > 0:
                gen_response = requests.post(
                    f"{base_url}/generate",
                    json={
                        "input_ids": prompt_ids,
                        "sampling_params": {
                            "temperature": 0,
                            "top_p": 1.0,
                            "max_new_tokens": max_new_tokens,
                        },
                        "return_logprob": True,
                        "return_text_in_logprobs": True,
                        **({"lora_path": lora_name} if lora_name else {}),
                    },
                    timeout=300,
                )
                gen_response.raise_for_status()
                gen_payload = gen_response.json()
                generated_ids = gen_payload.get("output_ids", [])
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                generated_token_logprobs = [
                    item[0] for item in gen_payload["meta_info"]["output_token_logprobs"]
                ]

            results.append(
                {
                    "id": row["id"],
                    "messages": row["messages"],
                    "reference_response": row["reference_response"],
                    "response_token_ids": response_token_ids,
                    "response_token_logprobs": response_logprobs,
                    "sequence_sum_logprob": sum(response_logprobs),
                    "sequence_mean_logprob": (
                        sum(response_logprobs) / len(response_logprobs)
                    ),
                    "generated_token_ids": generated_ids,
                    "generated_text": normalize_text(generated_text),
                    "generated_token_logprobs": generated_token_logprobs,
                }
            )
            if idx == 1 or idx == total_rows or idx % 8 == 0:
                print(f"SGLang eval progress: {idx}/{total_rows} examples")

        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "per_example.jsonl")
        summary_path = os.path.join(output_dir, "summary.json")
        write_jsonl(results_path, results)
        write_json(
            summary_path,
            {
                "num_examples": len(results),
                "mean_sequence_logprob": (
                    sum(row["sequence_mean_logprob"] for row in results) / len(results)
                    if results
                    else None
                ),
                "results_path": results_path,
            },
        )
        return {"results_path": results_path, "summary_path": summary_path, "count": len(results)}
    finally:
        _terminate_process(process)


def write_parity_report_from_paths(
    megatron_results_path: str,
    hf_results_path: str,
    sglang_results_path: str,
    output_path: str,
) -> Dict[str, Any]:
    report = build_parity_report(
        score_rows_by_id(megatron_results_path),
        score_rows_by_id(hf_results_path),
        score_rows_by_id(sglang_results_path),
    )
    write_json(output_path, report)
    return report
