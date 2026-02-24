"""Haiku eval script — queries a served model checkpoint and scores haiku structure."""

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import httpx
import modal
import nltk

from eval.shared import (
    DEFAULT_CONCURRENCY,
    EVAL_QUESTIONS,
    SERVED_MODEL_NAME,
    get_model_endpoint,
    query_model,
)
from llm_judges.nlp import score_haiku_structure

EVALS_PATH = "/opt/evals"


@dataclass(frozen=True)
class EvalResult:
    question: str
    response: str
    passed: bool


async def eval_problem(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    question: str,
    endpoint: str,
    cmudict: dict,
) -> EvalResult:
    response = await query_model(client, endpoint, question, semaphore=semaphore)
    structure_score = score_haiku_structure(response, cmudict)
    print(f"Structure score: {structure_score}")

    print("=" * 70)
    print(f"Question: {question}")
    print(f"Response: {response}")
    print("=" * 70)

    passed = structure_score >= 0.75
    print(f"Passed: {passed}")

    return EvalResult(question=question, response=response, passed=passed)


async def run_eval(
    step: str = "base",
    file_path: str | None = None,
):
    if file_path is None:
        file_path = f"{EVALS_PATH}/{SERVED_MODEL_NAME}_eval.jsonl"

    endpoint = get_model_endpoint(step)
    cmudict = nltk.corpus.cmudict.dict()

    print(f"Step: {step}")
    print(f"Endpoint: {endpoint}")
    print(f"Loaded {len(EVAL_QUESTIONS)} questions")
    print(f"Running with concurrency={DEFAULT_CONCURRENCY}\n")
    print("=" * 70)

    semaphore = asyncio.Semaphore(DEFAULT_CONCURRENCY)

    async with httpx.AsyncClient() as client:
        tasks = [
            eval_problem(client, semaphore, question, endpoint, cmudict)
            for question in EVAL_QUESTIONS
        ]
        results = await asyncio.gather(*tasks)

    with open(file_path, "w") as f:
        for result in results:
            f.write(json.dumps(asdict(result)) + "\n")

    success_rate = sum(result.passed for result in results) / len(results)
    print(f"Success rate: {success_rate}")

    # Save results to a Modal Dict
    eval_dict = modal.Dict.from_name("haiku-eval-results", create_if_missing=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    key = f"{SERVED_MODEL_NAME}/{step}/{timestamp}"
    eval_dict[key] = {
        "step": step,
        "timestamp": timestamp,
        "success_rate": success_rate,
        "results": [asdict(r) for r in results],
    }
    print(f"Saved eval results to Modal Dict 'haiku-eval-results' with key '{key}'")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run haiku structure eval against a served checkpoint.")
    parser.add_argument(
        "--step",
        default="base",
        help="Checkpoint step name (e.g. base, step-9, step-19, step-29, step-31)",
    )
    args = parser.parse_args()
    asyncio.run(run_eval(step=args.step))
