"""
Haiku LLM judge — scores structure (syllable counting) and style (LLM evaluation).

Structure scoring uses CMUdict for syllable counting.
Style scoring uses a local vLLM instance to evaluate relevance, poetic quality, etc.
"""

import re

import aiohttp

from llm_judges.deploy import VLLM_PORT
from llm_judges.nlp import score_haiku_structure


MODAL_VOCABS = [
    "modal",
    "volume",
    "function",
    "sandbox",
    "flash",
    "inference",
    "train",
]


def _build_judge_prompt(prompt: str, response: str, label: str = "") -> tuple[str, int]:
    """Build the LLM judge prompt. Returns (prompt_text, max_score)."""
    modal_vocab_str = ", ".join(MODAL_VOCABS)

    max_score = 15  # relevance(5) + poetic(5) + modal vocab(5)

    text = f"""You are evaluating a haiku poem.

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
"""

    if label:
        max_score = 20
        text += f"""
    Better than the existing poem (5 points total):
    Given the existing poem, score the response by comparing its quality to the existing poem:
    {label}
    - 5 points: if the response is better than the poem "{label}".
    - 3 points: if the response is equal in quality to the poem "{label}".
    - 0 points: if the response is worse than the poem "{label}".
"""

    prereq_score = max_score - 5
    text += f"""
    Uses Modal vocabulary (5 points total): (modal vocab: {modal_vocab_str})
    - 5 points: if the response uses the above words in a way that is coherent and relevant to the topic "{prompt}"
    - 3 points: if the response uses the above words in a way that is not relevant to the topic "{prompt}"
    - 0 points: if the response does not use the above words
    DO NOT GIVE ANY POINTS TO USE MODAL VOCABULARY IF THE POEM ITSELF DOES NOT ALREADY ACHIEVE A SCORE OF {prereq_score} OR HIGHER

    Add up the scores from the above criteria to get the total score.

    --
    **Topic:** {prompt}

    **Response to evaluate:**
    {response}
    ---

    Output ONLY a single number (0-{max_score}), nothing else."""

    return text, max_score


class HaikuJudge:
    """Scores haikus on structure (syllable counting) and style (LLM evaluation).

    Args:
        gate_style_on_structure: If True, only evaluate style when structure
            score is perfect (1.0). If False, always evaluate style.
    """

    def __init__(self, gate_style_on_structure: bool = True):
        self.gate_style_on_structure = gate_style_on_structure

    async def score_style(
        self,
        model_name: str,
        session: aiohttp.ClientSession,
        prompt: str,
        response: str,
        label: str = "",
        vllm_base_url: str = f"http://localhost:{VLLM_PORT}",
    ) -> float:
        """Score haiku style via LLM judge, normalized to [0, 1]."""
        judge_prompt, max_score = _build_judge_prompt(prompt, response, label)

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
                    return min(max(score, 0), max_score) / max_score
                return 0
        except Exception as e:
            print(f"Error scoring response: {e}")
            return 0

    async def score_single(
        self,
        model_name: str,
        session: aiohttp.ClientSession,
        prompt: str,
        response: str,
        cmudict: dict,
        label: str = "",
    ) -> float:
        """Score a single haiku. Returns a score in [0, 2]."""
        structure_score = score_haiku_structure(response, cmudict)

        style_score = 0.0
        if not self.gate_style_on_structure or structure_score >= 1.0:
            style_score = await self.score_style(
                model_name, session, prompt, response, label
            )
            style_score = max(style_score, 0.0)

        total = structure_score + style_score
        print(f"[HaikuJudge] structure={structure_score}, style={style_score}, gated={self.gate_style_on_structure}")
        return total
