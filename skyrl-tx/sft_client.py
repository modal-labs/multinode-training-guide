from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import numpy as np
import tinker
from tinker import types


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


def sum_loss(result) -> float:
    total = 0.0
    for output in result.loss_fn_outputs:
        total += float(sum(output["elementwise_loss"].data))
    return total


def eval_loss(training_client, batch: list[types.Datum]) -> float:
    return sum_loss(training_client.forward_backward(batch, "cross_entropy").result())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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
        beta2=0.999,
        eps=1e-8,
    )

    for step in range(args.steps):
        forward = training_client.forward_backward(batch, "cross_entropy").result()
        training_client.optim_step(optimizer).result()
        print(f"sft_step={step} loss={sum_loss(forward):.4f}", flush=True)

    state_path = training_client.save_state(name=f"sft_state_step_{args.steps}").result().path
    print(f"sft_state_checkpoint={state_path}", flush=True)
    restored_client = service_client.create_training_client_from_state(state_path)
    restored_loss = eval_loss(restored_client, eval_batch)
    print(f"sft_restored_eval_loss={restored_loss:.4f}", flush=True)

    sampler_path = training_client.save_weights_for_sampler(name=f"sft_sampler_step_{args.steps}").result().path
    print(f"sft_sampler_checkpoint={sampler_path}", flush=True)
    sampler = training_client.save_weights_and_get_sampling_client(name="sft_smoke")
    prompt = "Solve the arithmetic problem. Answer with only the integer.\nQuestion: What is 6 * 7?\nAnswer:"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    sample = sampler.sample(
        prompt=types.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=16, temperature=0.0, top_k=1),
    ).result()
    completion = tokenizer.decode(list(sample.sequences[0].tokens), skip_special_tokens=True)
    print(f"sft_sample={completion!r}", flush=True)


if __name__ == "__main__":
    main()
