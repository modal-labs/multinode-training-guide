import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from serve_haiku_model import TRAINED_MODELS
from llm_judges.base import MODAL_VOCABS, HaikuJudge
import nltk
import httpx


# RL'ed Model
# ENDPOINT = "https://modal-labs-joy-dev--serve-slime-model-serve.modal.run/v1/chat/completions"

def get_endpoint(step_name: str):
    # https://modal-labs-joy-dev--serve-haiku-model-serve-step-9.modal.run
    return f"https://modal-labs-joy-dev--serve-haiku-model-serve-{step_name}.modal.run/v1/chat/completions"


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
]


async def query_model(
    client: httpx.AsyncClient, semaphore: asyncio.Semaphore, prompt: str, model: str, step_name: str
) -> str:
    async with semaphore:
        payload = {
            "model": model,
            "messages": [
                {
                    "content": "You are a haiku poet. You will be given a prompt and you will need to write a haiku about the prompt. Try to incorproate these words into the haiku if possible: " + ", ".join(MODAL_VOCABS),
                    "role": "system"
                },
                {
                    "content": prompt,
                    "role": "user"
                }
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            response = await client.post(get_endpoint(step_name), json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"ERROR: {e}"
@dataclass(frozen=True)
class EvalResult:
    step_name: str
    question: str
    response: str
    passed: bool
    score: float

async def eval_problem(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    question: str,
    model: str,
    cmudict: dict,
    step_name: str,
) -> EvalResult:
    response = await query_model(client, semaphore, question, model, step_name)
    structure_score = HaikuJudge.score_haiku_structure(response, cmudict)
    print(f"Structure score: {structure_score}")

    print("=" * 70)
    print(f"Question: {question}")
    print(f"Response: {response}")
    print("=" * 70)

    passed = structure_score >= 0.75
    print(f"Passed: {passed}")

    return EvalResult(step_name=step_name, question=question, response=response, passed=passed, score=structure_score)



async def run_eval(model: str = DEFAULT_MODEL, file_path: str = f"{EVALS_PATH}/{MODEL_NAME}_eval.jsonl"):
    cmudict = nltk.corpus.cmudict.dict()

    print(f"Model: {model}")
    print(f"Loaded {len(questions)} questions")
    print(f"Running with concurrency={CONCURRENCY}\n")
    print("=" * 70)

    semaphore = asyncio.Semaphore(CONCURRENCY)

    results = []
    for step_name in ["base"] + list(TRAINED_MODELS.keys()):
        print(f"Evaluating {step_name}")
        async with httpx.AsyncClient() as client:
            tasks = [
                eval_problem(client, semaphore, question, model, cmudict, step_name)
                for question in questions
            ]
            step_results = await asyncio.gather(*tasks)
            results.extend(step_results)
        with open(file_path, "w") as f:
            for result in results:
                f.write(json.dumps(asdict(result)) + "\n")
        
        print(f"[{step_name}] Success rate: {sum(result.passed for result in step_results) / len(step_results)}")
        print(f"[{step_name}] Average score: {sum(result.score for result in step_results) / len(step_results)}")
        print("*" * 70)
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
