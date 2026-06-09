from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false

import argparse
import base64
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import tinker
from tinker import types


RESULT_TIMEOUT_SECONDS = 20 * 60


def tensor_int(values: list[int]) -> types.TensorData:
    return types.TensorData.from_numpy(np.asarray(values, dtype=np.int64))


def tensor_float(values: list[float]) -> types.TensorData:
    return types.TensorData.from_numpy(np.asarray(values, dtype=np.float32))


def resolve(future, label: str):
    return future.result(timeout=RESULT_TIMEOUT_SECONDS)


def validate_finite(result, label: str) -> dict[str, float]:
    total = 0.0
    count = 0
    for output in result.loss_fn_outputs:
        for tensor in output.values():
            for value in tensor.data:
                scalar = float(value)
                if not math.isfinite(scalar):
                    raise RuntimeError(f"Non-finite {label} value: {scalar}")
                total += scalar
                count += 1
    for key, value in (result.metrics or {}).items():
        scalar = float(value)
        if not math.isfinite(scalar):
            raise RuntimeError(f"Non-finite {label} metric {key}: {scalar}")
    return {"loss_sum": total, "loss_values": float(count)}


def validate_optimizer_metrics(result, label: str) -> None:
    for key, value in (result.metrics or {}).items():
        scalar = float(value)
        if not math.isfinite(scalar):
            raise RuntimeError(f"Non-finite {label} metric {key}: {scalar}")


def optimizer() -> types.AdamParams:
    return types.AdamParams(learning_rate=1e-6, beta1=0.9, beta2=0.95, eps=1e-8)


def completion_datum(tokenizer, prompt: str, completion: str) -> types.Datum:
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    tokens = prompt_tokens + completion_tokens
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs={
            "weights": tensor_float(weights[1:]),
            "target_tokens": tensor_int(tokens[1:]),
        },
    )


def chat_datum(tokenizer, user: str, assistant: str) -> types.Datum:
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=True,
            add_generation_prompt=True,
        )
        full_tokens = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(prompt_tokens, dict):
            prompt_tokens = prompt_tokens["input_ids"]
        if isinstance(full_tokens, dict):
            full_tokens = full_tokens["input_ids"]
        if len(full_tokens) > len(prompt_tokens):
            weights = [0.0] * len(prompt_tokens) + [1.0] * (
                len(full_tokens) - len(prompt_tokens)
            )
            return types.Datum(
                model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
                loss_fn_inputs={
                    "weights": tensor_float(weights[1:]),
                    "target_tokens": tensor_int(full_tokens[1:]),
                },
            )
    return completion_datum(
        tokenizer,
        f"<|user|>\n{user}\n<|assistant|>\n",
        assistant,
    )


def rollout_datum(
    prompt_tokens: list[int],
    response_tokens: list[int],
    old_logprobs: list[float],
    advantage: float,
) -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput.from_ints(prompt_tokens + response_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tensor_int(response_tokens),
            "weights": tensor_float([1.0] * len(response_tokens)),
            "logprobs": tensor_float(old_logprobs),
            "advantages": tensor_float([advantage] * len(response_tokens)),
        },
    )


def sample_rollouts(sampler, tokenizer, prompt: str, advantage: float) -> list[types.Datum]:
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    sample = resolve(
        sampler.sample(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=8,
                temperature=0.0,
                top_k=1,
            ),
        ),
        "sample_rollout",
    )
    data = []
    for sequence in sample.sequences:
        response_tokens = list(sequence.tokens)
        if not response_tokens:
            continue
        if sequence.logprobs is None:
            old_logprobs = [0.0] * len(response_tokens)
        else:
            old_logprobs = [float(value) for value in sequence.logprobs]
        data.append(rollout_datum(prompt_tokens, response_tokens, old_logprobs, advantage))
    if not data:
        raise RuntimeError("Sampler produced no rollout tokens")
    return data


def compute_logprobs(sampler, model_input: types.ModelInput) -> list[float | None]:
    if hasattr(sampler, "compute_logprobs"):
        return list(resolve(sampler.compute_logprobs(model_input), "compute_logprobs"))
    if hasattr(sampler, "compute_logprobs_async"):
        import asyncio

        return list(asyncio.run(sampler.compute_logprobs_async(model_input)))
    raise RuntimeError("Sampling client does not expose compute_logprobs")


class CookbookSmokeRunner:
    def __init__(self, base_url: str, model_name: str, lora_rank: int):
        self.service_client = tinker.ServiceClient(base_url=base_url, api_key="tml-dummy")
        self.training_client = self.service_client.create_lora_training_client(
            base_model=model_name,
            rank=lora_rank,
        )
        self.tokenizer = self.training_client.get_tokenizer()
        self.model_name = model_name

    def run_cross_entropy_step(self, datum: types.Datum, label: str) -> dict[str, float]:
        forward = resolve(
            self.training_client.forward_backward([datum], "cross_entropy"),
            f"{label}_forward_backward",
        )
        metrics = validate_finite(forward, label)
        optim = resolve(self.training_client.optim_step(optimizer()), f"{label}_optim")
        validate_optimizer_metrics(optim, f"{label}_optim")
        return metrics

    def run_importance_sampling_step(
        self, data: list[types.Datum], label: str
    ) -> dict[str, float]:
        forward = resolve(
            self.training_client.forward_backward(data, "importance_sampling"),
            f"{label}_forward_backward",
        )
        metrics = validate_finite(forward, label)
        optim = resolve(self.training_client.optim_step(optimizer()), f"{label}_optim")
        validate_optimizer_metrics(optim, f"{label}_optim")
        return metrics

    def sl_loop(self) -> dict[str, float | str]:
        metrics = self.run_cross_entropy_step(
            completion_datum(self.tokenizer, "Question: 17 + 25\nAnswer:", " 42"),
            "sl_loop",
        )
        state_path = resolve(
            self.training_client.save_state(name="cookbook_sl_loop_state"),
            "sl_loop_save_state",
        ).path
        sampler_path = resolve(
            self.training_client.save_weights_for_sampler(name="cookbook_sl_loop_sampler"),
            "sl_loop_save_sampler",
        ).path
        return {**metrics, "state_path": state_path, "sampler_path": sampler_path}

    def chat_sl(self) -> dict[str, float]:
        return self.run_cross_entropy_step(
            chat_datum(
                self.tokenizer,
                "Give a one-word greeting.",
                "Hello",
            ),
            "chat_sl",
        )

    def rl_loop(self) -> dict[str, float]:
        sampler = self.training_client.save_weights_and_get_sampling_client()
        data = sample_rollouts(sampler, self.tokenizer, "Question: 6 * 7\nAnswer:", 1.0)
        return self.run_importance_sampling_step(data, "rl_loop")

    def math_rl(self) -> dict[str, float]:
        sampler = self.training_client.save_weights_and_get_sampling_client()
        data = sample_rollouts(
            sampler,
            self.tokenizer,
            "Solve the math problem. Return only the integer.\nQuestion: 13 + 29\nAnswer:",
            1.0,
        )
        return self.run_importance_sampling_step(data, "math_rl")

    def code_rl(self) -> dict[str, float | str]:
        sampler = self.training_client.save_weights_and_get_sampling_client()
        data = sample_rollouts(
            sampler,
            self.tokenizer,
            "Write Python code that prints 42.\n```python\n",
            0.5,
        )
        metrics = self.run_importance_sampling_step(data, "code_rl")
        return {
            **metrics,
            "external_dependency": "sandbox not exercised; model-side RL path ran",
        }

    def search_tool(self) -> dict[str, float | str]:
        sampler = self.training_client.save_weights_and_get_sampling_client()
        data = sample_rollouts(
            sampler,
            self.tokenizer,
            "Question: What planet is known as the Red Planet?\n"
            "<search>Red Planet</search>\nObservation: Mars is known as the Red Planet.\nAnswer:",
            1.0,
        )
        metrics = self.run_importance_sampling_step(data, "search_tool")
        return {
            **metrics,
            "external_dependency": "Chroma retrieval not exercised; transcript RL path ran",
        }

    def preference_rlhf(self) -> dict[str, float | str]:
        reward_client = self.service_client.create_lora_training_client(
            base_model=self.model_name,
            rank=4,
        )
        reward_tokenizer = reward_client.get_tokenizer()
        reward_datum = completion_datum(
            reward_tokenizer,
            "Choose the better answer.\nA: helpful\nB: harmful\nAnswer:",
            " A",
        )
        forward = resolve(
            reward_client.forward_backward([reward_datum], "cross_entropy"),
            "rlhf_reward_forward_backward",
        )
        metrics = validate_finite(forward, "rlhf_reward")
        resolve(reward_client.optim_step(optimizer()), "rlhf_reward_optim")
        reward_sampler_path = resolve(
            reward_client.save_weights_for_sampler(name="cookbook_rlhf_reward_sampler"),
            "rlhf_reward_save_sampler",
        ).path
        reward_sampler = self.service_client.create_sampling_client(model_path=reward_sampler_path)
        preference = resolve(
            reward_sampler.sample(
                prompt=types.ModelInput.from_ints(
                    reward_tokenizer.encode(
                        "Choose the better answer.\nA: helpful\nB: harmful\nAnswer:",
                        add_special_tokens=True,
                    )
                ),
                num_samples=1,
                sampling_params=types.SamplingParams(max_tokens=1, temperature=0.0, top_k=1),
            ),
            "rlhf_reward_sample",
        )
        policy_sampler = self.training_client.save_weights_and_get_sampling_client()
        data = sample_rollouts(policy_sampler, self.tokenizer, "Say something helpful:", 0.5)
        policy_metrics = self.run_importance_sampling_step(data, "rlhf_policy_rl")
        return {
            **metrics,
            **{f"policy_{key}": value for key, value in policy_metrics.items()},
            "reward_sampler_path": reward_sampler_path,
            "reward_tokens": str(list(preference.sequences[0].tokens)),
        }

    def preference_dpo(self) -> dict[str, float]:
        import torch
        import torch.nn.functional as functional

        reference_client = self.training_client.save_weights_and_get_sampling_client()
        chosen = completion_datum(self.tokenizer, "Question: 1 + 1\nAnswer:", " 2")
        rejected = completion_datum(self.tokenizer, "Question: 1 + 1\nAnswer:", " 3")
        data = [chosen, rejected]
        full_sequences = [
            datum.model_input.append_int(int(datum.loss_fn_inputs["target_tokens"].data[-1]))
            for datum in data
        ]
        reference_logprobs = [
            torch.tensor([value if value is not None else 0.0 for value in compute_logprobs(reference_client, seq)[1:]])
            for seq in full_sequences
        ]

        def weighted_dot(logprobs, weights, reference):
            width = min(len(logprobs), len(weights), len(reference))
            return (
                torch.dot(logprobs[-width:].float(), weights[-width:].float()),
                torch.dot(reference[-width:].float(), weights[-width:].float()),
            )

        def dpo_loss_fn(loss_data: list[types.Datum], logprobs_list):
            chosen_weights = torch.tensor(loss_data[0].loss_fn_inputs["weights"].data)
            rejected_weights = torch.tensor(loss_data[1].loss_fn_inputs["weights"].data)
            chosen_logprob, chosen_ref = weighted_dot(
                logprobs_list[0],
                chosen_weights,
                reference_logprobs[0],
            )
            rejected_logprob, rejected_ref = weighted_dot(
                logprobs_list[1],
                rejected_weights,
                reference_logprobs[1],
            )
            margin = (chosen_logprob - chosen_ref) - (rejected_logprob - rejected_ref)
            loss = -functional.logsigmoid(0.1 * margin)
            return loss, {
                "dpo_loss": float(loss.detach().cpu()),
                "dpo_margin": float(margin.detach().cpu()),
            }

        forward = resolve(
            self.training_client.forward_backward_custom(data, dpo_loss_fn),
            "dpo_forward_backward_custom",
        )
        metrics = validate_finite(forward, "dpo")
        resolve(self.training_client.optim_step(optimizer()), "dpo_optim")
        return metrics

    def distillation_on_policy(self) -> dict[str, float | str]:
        teacher = self.training_client.save_weights_and_get_sampling_client()
        prompt_tokens = self.tokenizer.encode("Question: 2 + 2\nAnswer:", add_special_tokens=True)
        sample = resolve(
            teacher.sample(
                prompt=types.ModelInput.from_ints(prompt_tokens),
                num_samples=1,
                sampling_params=types.SamplingParams(max_tokens=4, temperature=0.0, top_k=1),
            ),
            "distillation_sample",
        )
        response_tokens = list(sample.sequences[0].tokens)
        full_input = types.ModelInput.from_ints(prompt_tokens + response_tokens)
        teacher_logprobs = compute_logprobs(teacher, full_input)[1:]
        if sample.sequences[0].logprobs is None:
            sampled_logprobs = [0.0] * len(response_tokens)
        else:
            sampled_logprobs = [float(value) for value in sample.sequences[0].logprobs]
        usable_teacher = [
            float(value) if value is not None else sampled_logprobs[index]
            for index, value in enumerate(teacher_logprobs[-len(response_tokens) :])
        ]
        reverse_kl = [
            sampled - teacher_value
            for sampled, teacher_value in zip(sampled_logprobs, usable_teacher, strict=True)
        ]
        advantage = -sum(reverse_kl) / max(1, len(reverse_kl))
        data = [rollout_datum(prompt_tokens, response_tokens, sampled_logprobs, advantage)]
        metrics = self.run_importance_sampling_step(data, "distillation")
        return {**metrics, "teacher_logprobs": str(len(teacher_logprobs))}

    def vlm_classifier(self) -> dict[str, float]:
        prompt_tokens = self.tokenizer.encode(
            "Classify this image as red or blue: ",
            add_special_tokens=True,
        )
        target_tokens = self.tokenizer.encode(" red", add_special_tokens=False)
        png_1x1 = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        )
        model_input = types.ModelInput(
            chunks=[
                types.EncodedTextChunk(tokens=prompt_tokens),
                types.ImageChunk(data=png_1x1, format="png", expected_tokens=1),
                types.EncodedTextChunk(tokens=target_tokens[:-1]),
            ]
        )
        datum = types.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": tensor_int(target_tokens),
                "weights": tensor_float([1.0] * len(target_tokens)),
            },
        )
        forward = resolve(
            self.training_client.forward_backward([datum], "cross_entropy"),
            "vlm_forward_backward",
        )
        metrics = validate_finite(forward, "vlm")
        resolve(self.training_client.optim_step(optimizer()), "vlm_optim")
        return metrics


EXAMPLES = [
    ("sl_loop", "expected_supported"),
    ("rl_loop", "expected_supported"),
    ("chat_sl", "expected_supported"),
    ("math_rl", "expected_supported"),
    ("code_rl", "expected_partial"),
    ("preference_dpo", "expected_partial"),
    ("preference_rlhf", "expected_partial"),
    ("distillation_on_policy", "expected_partial"),
    ("search_tool", "expected_partial"),
    ("vlm_classifier", "expected_partial"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--results-path", type=Path, default=None)
    args = parser.parse_args()

    runner = CookbookSmokeRunner(args.base_url, args.model_name, args.lora_rank)
    results: list[dict[str, object]] = []
    unexpected_failures = 0
    if args.results_path is not None:
        args.results_path.parent.mkdir(parents=True, exist_ok=True)

    for example, expectation in EXAMPLES:
        started = time.monotonic()
        try:
            payload = getattr(runner, example)()
            status = "PASS"
            error = ""
            details = payload
        except Exception as exc:
            details = {"traceback_tail": traceback.format_exc().splitlines()[-8:]}
            if expectation == "expected_supported":
                status = "FAIL_UNEXPECTED"
                unexpected_failures += 1
            else:
                status = "FAIL_EXPECTED"
            error = f"{type(exc).__name__}: {exc}"
        record = {
            "example": example,
            "expectation": expectation,
            "status": status,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": error,
            "details": details,
        }
        results.append(record)
        line = json.dumps(record, sort_keys=True)
        print(f"cookbook_result={line}", flush=True)
        if args.results_path is not None:
            with args.results_path.open("a") as output:
                output.write(line + "\n")

    summary = {
        "passed": sum(1 for item in results if item["status"] == "PASS"),
        "expected_failures": sum(
            1 for item in results if item["status"] == "FAIL_EXPECTED"
        ),
        "unexpected_failures": unexpected_failures,
    }
    print(f"cookbook_summary={json.dumps(summary, sort_keys=True)}", flush=True)
    if unexpected_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
