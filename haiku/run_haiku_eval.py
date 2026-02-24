import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
import nltk
import httpx

from llm_judges.nlp import score_haiku_structure

# RL'ed Model
# ENDPOINT = "https://modal-labs-joy-dev--serve-slime-model-serve.modal.run/v1/chat/completions"

# Base model
ENDPOINT = "https://modal-labs-joy-dev--serve-haiku-model-serve-base.modal.run/v1/chat/completions"

CONCURRENCY = 50
DEFAULT_MODEL = "slime-qwen"
MODEL_NAME = "qwen3-4b-haiku"

EVALS_PATH = "/opt/evals"

questions = [
    "Write me a haiku about cat.",
    "Write me a haiku about dog.",
    "Write me a haiku about bird.",
    "Write me a haiku about fish.",
    "Write me a haiku about horse.",
    "Write me a haiku about rabbit.",
    "Write me a haiku about snake.",
    "Write me a haiku about tiger.",
    "Write me a haiku about lion.",
    "Write me a haiku about Jason Mancuso.",
    "Write me a haiku about Joy Liu.",
    "Write me a haiku about Modal Labs.",
]


async def query_model(
    client: httpx.AsyncClient, semaphore: asyncio.Semaphore, prompt: str, model: str
) -> str:
    async with semaphore:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            response = await client.post(ENDPOINT, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"ERROR: {e}"
@dataclass(frozen=True)
class EvalResult:
    question: str
    response: str
    passed: bool

async def eval_problem(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    question: str,
    model: str,
    cmudict: dict,
) -> EvalResult:
    response = await query_model(client, semaphore, question, model)
    structure_score = score_haiku_structure(response, cmudict)
    print(f"Structure score: {structure_score}")

    print("=" * 70)
    print(f"Question: {question}")
    print(f"Response: {response}")
    print("=" * 70)

    passed = structure_score >= 0.75
    print(f"Passed: {passed}")

    return EvalResult(question=question, response=response, passed=passed)



async def run_eval(model: str = DEFAULT_MODEL, file_path: str = f"{EVALS_PATH}/{MODEL_NAME}_eval.jsonl"):
    cmudict = nltk.corpus.cmudict.dict()

    print(f"Model: {model}")
    print(f"Loaded {len(questions)} questions")
    print(f"Running with concurrency={CONCURRENCY}\n")
    print("=" * 70)

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient() as client:
        tasks = [
            eval_problem(client, semaphore, question, model, cmudict)
            for question in questions
        ]
        results = await asyncio.gather(*tasks)

    with open(file_path, "w") as f:
        for result in results:
            f.write(json.dumps(asdict(result)) + "\n")
    
    print(f"Success rate: {sum(result.passed for result in results) / len(results)}")
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
