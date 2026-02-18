"""
Abstract base class for haiku LLM judges.

Provides shared structure and style scoring. Subclasses implement
score_single to define their own weighting strategies.
"""

import re
from abc import ABC, abstractmethod

import aiohttp

from llm_judges.deploy import VLLM_PORT
from llm_judges.nlp import diff_syllables_count, segment_haiku_lines



# =============================================================================
# Shared Prompt Template
# =============================================================================

MODAL_VOCABS = [
    "modal",
    "volume"
    "function",
    "sandbox",
    "flash",
    "inference",
    "train",
]

def generate_haiku_judge_prompt(prompt: str, response: str, label: str) -> str:
    modal_vocab_str = ", ".join(MODAL_VOCABS)

    return f"""You are evaluating a haiku poem.

    Score the response based on the following criteria:
    
    Relevance (5 points total)
    - 5 points: if the central theme and punchline of the haiku is "{prompt}"
    - 3 points: if the response directly discusses "{prompt}" but it is not the central theme
    - 2 points: if the response is relevant to the topic "{prompt}" but very plain
    - 0 points: if the response is not relevant to the topic "{prompt}"

    Poetic quality (5 points total)
    - 5 points: if the response makes sense, can be considered a poetic haiku, with a clear theme and punchline
    - 3 point: if the response makes sense, but is not very poetic
    - 1 point: if the response doesn't make sense
    - 0 points: if the response is not poetic and incoherent

    Uses Modal vocabulary (5 points total): (modal vocab: {modal_vocab_str})
    - 5 points: if the response uses the above words in a way that is coherent and relevant to the topic "{prompt}"
    - 3 points: if the response uses the above words in a way that is not relevant to the topic "{prompt}"
    - 0 points: if the response does not use the above words

    Better than the existing poem (5 points total):
    Given the existing poem, score the response by comparing its quality to the existing poem:
    {label}
    - 5 points: if the response is better than the poem "{label}".
    - 3 points: if the response is equal in quality to the poem "{label}".
    - 0 points: if the response is worse than the poem "{label}".

    Add up the scores from the above criteria to get the total score.

    --
    **Topic:** {prompt}

    **Response to evaluate:**
    {response}
    ---

    Output ONLY a single number (0-20), nothing else."""


class HaikuJudge(ABC):
    """Abstract base class for haiku judges.

    Shared scoring:
        - score_haiku_structure: 0-8 based on line count and syllable accuracy
        - score_haiku_style: 0-2 based on LLM evaluation of relevance and emotion

    Subclasses implement score_single to combine these into a final [0, 1] score.
    """

    MAX_STRUCTURE_SCORE = 1
    MAX_STYLE_SCORE = 20

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this judge, used as the Modal app/deployment name."""
        ...

    @staticmethod
    def score_syllable_line(diff: int, allow_off_by_one: bool = False) -> float:
        """Score a single line's syllable count: 1 for exact, 0.5 for off-by-1, 0 otherwise."""
        if diff == 0:
            return 1
        elif diff == 1:
            return 0.5 if allow_off_by_one else 0
        return 0

    @staticmethod
    def score_haiku_structure(response: str, cmudict: dict, allow_off_by_one: bool = False) -> float:
        """Score haiku structure (0-1): 1/4 for 3 lines + up to 1/4 per line for syllables."""
        lines = segment_haiku_lines(response)
        score = 0.0
        fractional_multiplier = 0.25

        if len(lines) == 3:
            score += fractional_multiplier

        if len(lines) > 0:
            score += HaikuJudge.score_syllable_line(
                diff_syllables_count(lines[0], 5, cmudict), allow_off_by_one
            ) * fractional_multiplier
        if len(lines) > 1:
            score += HaikuJudge.score_syllable_line(
                diff_syllables_count(lines[1], 7, cmudict), allow_off_by_one
            ) * fractional_multiplier
        if len(lines) > 2:
            score += HaikuJudge.score_syllable_line(
                diff_syllables_count(lines[2], 5, cmudict), allow_off_by_one
            ) * fractional_multiplier

        return score

    @staticmethod
    async def score_haiku_style(
        model_name: str,
        session: aiohttp.ClientSession,
        prompt: str,
        response: str,
        label: str,
        vllm_base_url: str = f"http://localhost:{VLLM_PORT}",
    ) -> float:
        """Score haiku style via LLM judge (0-1), or 0 on error."""
        judge_prompt = generate_haiku_judge_prompt(prompt, response, label)

        try:
            async with session.post(
                f"{vllm_base_url}/v1/chat/completions",
                headers={"content-type": "application/json"},
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": judge_prompt}],
                    "max_tokens": 100,
                },
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"vLLM error: {resp.status} - {error_text}")
                    return 0

                data = await resp.json()
                score_text = data["choices"][0]["message"]["content"].strip()
                print(f"Scored {response} with score {score_text}")

                match = re.search(r"(\d+(?:\.\d+)?)", score_text)
                if match:
                    score = float(match.group(1))
                    return min(max(score, 0), 10) / 10
                return 0
        except Exception as e:
            print(f"Error scoring response: {e}")
            return 0

    @abstractmethod
    async def score_single(
        self,
        model_name: str,
        session: aiohttp.ClientSession,
        prompt: str,
        response: str,
        label: str,
        cmudict: dict,
    ) -> float:
        """Score a single haiku. Returns a normalized score in [0, 1]."""
        ...

