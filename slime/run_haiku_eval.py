import argparse
import asyncio
import json
import re
from pathlib import Path

import httpx

ENDPOINT = "https://modal-labs-joy-dev--serve-slime-model-serve.modal.run/v1/completions"
EVAL_FILE = Path(__file__).parent / "auto_math_eval.jsonl"
CONCURRENCY = 50
DEFAULT_MODEL = "slime-qwen"



questions = [
    "Write me a haiku about a cat",
    "Write me a haiku about a dog",
    "Write me a haiku about a bird",
    "Write me a haiku about a fish",
    "Write me a haiku about a horse",
    "Write me a haiku about a rabbit",
    "Write me a haiku about a snake",
    "Write me a haiku about a tiger",
    "Write me a haiku about a lion",
]



async def query_model(
    client: httpx.AsyncClient, semaphore: asyncio.Semaphore, prompt: str, model: str
) -> str:
    async with semaphore:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": ["\n", "Solve", "Question"],
            "stream": False,
        }
        try:
            response = await client.post(ENDPOINT, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["text"].strip()
        except Exception as e:
            return f"ERROR: {e}"


async def eval_problem(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    question: str,
    model: str,
) -> bool:
    response = await query_model(client, semaphore, question, model)


    print("=" * 70)
    print(f"Question: {question}")
    print(f"Response: {response}")
    print("=" * 70)




async def run_eval(model: str = DEFAULT_MODEL):
    print(f"Model: {model}")
    print(f"Loaded {len(questions)} questions")
    print(f"Running with concurrency={CONCURRENCY}\n")
    print("=" * 70)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    progress = {"done": 0, "total": len(questions)}

    async with httpx.AsyncClient() as client:
        tasks = [
            eval_problem(client, semaphore, question, model)
            for question in questions
        ]
        results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["slime-qwen"],
        help="Model to evaluate: slime-qwen",
    )
    args = parser.parse_args()
    asyncio.run(run_eval(model=args.model))
