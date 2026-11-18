from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import math
import numpy as np
import tinker
from tinker import types


RESULT_TIMEOUT_SECONDS = 20 * 60

EXAMPLES = [
    ("What is 17 + 25?", " 42"),
    ("What is 9 * 8?", " 72"),
    ("What is 144 / 12?", " 12"),
    ("What is 31 - 14?", " 17"),
]


def tensor_int(values: list[int]) -> types.TensorData:
    return types.TensorData.from_numpy(np.asarray(values, dtype=np.int64))


def tensor_float(values: list[float]) -> types.TensorData:
    return types.TensorData.from_numpy(np.asarray(values, dtype=np.float32))


def make_datum(tokenizer, question: str, answer: str) -> types.Datum:
    prompt = f"Solve the arithmetic problem. Answer with only the integer.\nQuestion: {question}\nAnswer:"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
    tokens = prompt_tokens + answer_tokens
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(answer_tokens)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs={
            "weights": tensor_float(weights[1:]),
            "target_tokens": tensor_int(tokens[1:]),
        },
    )


def resolve(future, label: str):
    return future.result(timeout=RESULT_TIMEOUT_SECONDS)


def sum_loss(result, label: str) -> float:
    total = 0.0
    for output in result.loss_fn_outputs:
        for value in output["elementwise_loss"].data:
            loss = float(value)
            if not math.isfinite(loss):
                raise RuntimeError(f"Non-finite {label} loss value: {loss}")
            total += loss
    return total


def validate_optimizer_metrics(result, label: str) -> None:
    for key, value in (result.metrics or {}).items():
        metric = float(value)
        if not math.isfinite(metric):
            raise RuntimeError(f"Non-finite {label} metric {key}: {metric}")


def eval_loss(training_client, batch: list[types.Datum]) -> float:
    forward = resolve(
        training_client.forward_backward(batch, "cross_entropy"), "sft_eval"
    )
    return sum_loss(forward, "sft_eval")


def sample_arithmetic(sampler, tokenizer, label: str) -> str:
    prompt = "Solve the arithmetic problem. Answer with only the integer.\nQuestion: What is 6 * 7?\nAnswer:"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    sample = sampler.sample(
        prompt=types.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=16, temperature=0.0, top_k=1),
    )
    sample = resolve(sample, label)
    return tokenizer.decode(list(sample.sequences[0].tokens), skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    args = parser.parse_args()

    service_client = tinker.ServiceClient(base_url=args.base_url, api_key="tml-dummy")
    training_client = service_client.create_lora_training_client(
        base_model=args.model_name,
        rank=args.lora_rank,
    )
    tokenizer = training_client.get_tokenizer()
    batch = [make_datum(tokenizer, question, answer) for question, answer in EXAMPLES]
    eval_batch = [make_datum(tokenizer, "What is 6 * 7?", " 42")]
    optimizer = types.AdamParams(
        learning_rate=args.learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    for step in range(args.steps):
        forward = resolve(
            training_client.forward_backward(batch, "cross_entropy"), f"sft_step_{step}"
        )
        optim = resolve(training_client.optim_step(optimizer), f"sft_optim_step_{step}")
        validate_optimizer_metrics(optim, f"sft_optim_step_{step}")
        print(
            f"sft_step={step} loss={sum_loss(forward, f'sft_step_{step}'):.4f}",
            flush=True,
        )

    state_path = resolve(
        training_client.save_state(name=f"sft_state_step_{args.steps}"),
        "sft_save_state",
    ).path
    print(f"sft_state_checkpoint={state_path}", flush=True)
    current_loss = eval_loss(training_client, eval_batch)
    print(f"sft_eval_loss={current_loss:.4f}", flush=True)

    sampler_path = resolve(
        training_client.save_weights_for_sampler(name=f"sft_sampler_step_{args.steps}"),
        "sft_save_sampler",
    ).path
    print(f"sft_sampler_checkpoint={sampler_path}", flush=True)
    sampler = service_client.create_sampling_client(model_path=sampler_path)
    completion = sample_arithmetic(sampler, tokenizer, "sft_sampler_eval")
    print(f"sft_sampler_eval_sample={completion!r}", flush=True)


if __name__ == "__main__":
    main()
