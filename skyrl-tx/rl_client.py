from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import re

import numpy as np
import tinker
from tinker import types


STRICT_INTEGER_RE = re.compile(r"-?\d+")
PROMPTS = [
    ("What is 13 + 29?", "42"),
    ("What is 7 * 9?", "63"),
    ("What is 81 / 9?", "9"),
    ("What is 50 - 18?", "32"),
]


def tensor_int(values: list[int]) -> types.TensorData:
    return types.TensorData.from_numpy(np.asarray(values, dtype=np.int64))


def tensor_float(values: list[float]) -> types.TensorData:
    return types.TensorData.from_numpy(np.asarray(values, dtype=np.float32))


def extract_integer(text: str) -> str | None:
    matches = STRICT_INTEGER_RE.findall(text.replace(",", ""))
    return matches[-1] if matches else None


def reward_for(response: str, answer: str) -> float:
    prediction = extract_integer(response)
    if prediction == answer:
        return 1.0
    if prediction is not None:
        return 0.1
    return 0.0


def make_prompt(question: str) -> str:
    return (
        "Solve the arithmetic problem. Answer with only the final integer.\n"
        f"Question: {question}\nAnswer:"
    )


def make_policy_datum(
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


def build_rollout_batch(training_client, tokenizer, samples_per_prompt: int, step: int) -> tuple[list[types.Datum], list[float]]:
    sampler = training_client.save_weights_and_get_sampling_client(name=f"rl_step_{step}")
    data: list[types.Datum] = []
    rewards: list[float] = []

    for offset, (question, answer) in enumerate(PROMPTS):
        prompt_tokens = tokenizer.encode(make_prompt(question), add_special_tokens=True)
        sample = sampler.sample(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=samples_per_prompt,
            sampling_params=types.SamplingParams(
                max_tokens=24,
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                seed=step * 1000 + offset,
            ),
        ).result()
        for sequence in sample.sequences:
            response_tokens = list(sequence.tokens)
            if not response_tokens:
                continue
            response = tokenizer.decode(response_tokens, skip_special_tokens=True)
            reward = reward_for(response, answer)
            if sequence.logprobs is None:
                old_logprobs = [0.0] * len(response_tokens)
            else:
                old_logprobs = [float(logprob) for logprob in sequence.logprobs]
            data.append(make_policy_datum(prompt_tokens, response_tokens, old_logprobs, reward))
            rewards.append(reward)

    if not data:
        raise RuntimeError("No rollout tokens were sampled; cannot run RL update")
    return data, rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()

    service_client = tinker.ServiceClient(base_url=args.base_url, api_key="tml-dummy")
    training_client = service_client.create_lora_training_client(
        base_model=args.model_name,
        rank=args.lora_rank,
    )
    tokenizer = training_client.get_tokenizer()
    optimizer = types.AdamParams(
        learning_rate=args.learning_rate,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    )

    data: list[types.Datum] = []
    rewards: list[float] = []
    for step in range(args.steps):
        data, rewards = build_rollout_batch(training_client, tokenizer, args.samples_per_prompt, step)
        forward = training_client.forward_backward(
            data,
            "ppo",
            {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
        ).result()
        training_client.optim_step(optimizer).result()
        mean_reward = sum(rewards) / len(rewards)
        print(
            f"rl_step={step} mean_reward={mean_reward:.3f} trajectories={len(data)} "
            f"loss_outputs={len(forward.loss_fn_outputs)}",
            flush=True,
        )

    state_path = training_client.save_state(name=f"rl_state_step_{args.steps}").result().path
    print(f"rl_state_checkpoint={state_path}", flush=True)
    restored_client = service_client.create_training_client_from_state(state_path)
    restored_forward = restored_client.forward_backward(
        data,
        "ppo",
        {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
    ).result()
    print(f"rl_restored_eval_loss_outputs={len(restored_forward.loss_fn_outputs)}", flush=True)

    sampler_path = training_client.save_weights_for_sampler(name=f"rl_sampler_step_{args.steps}").result().path
    print(f"rl_sampler_checkpoint={sampler_path}", flush=True)
    eval_data, eval_rewards = build_rollout_batch(training_client, tokenizer, args.samples_per_prompt, args.steps)
    mean_eval_reward = sum(eval_rewards) / len(eval_rewards)
    print(f"rl_eval mean_reward={mean_eval_reward:.3f} trajectories={len(eval_data)}", flush=True)


if __name__ == "__main__":
    main()
