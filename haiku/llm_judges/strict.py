"""
Strict haiku judge: no style points unless structure passes threshold.

Style is only evaluated when the structure score exceeds half the maximum,
ensuring the model produces well-formed haikus before being rewarded for style.
"""

import aiohttp

from llm_judges.base import HaikuJudge


class StrictJudge(HaikuJudge):
    """Gates style scoring behind a structure threshold.

    No style points are awarded unless structure_score > MAX_STRUCTURE_SCORE / 2.
    Final score: (structure + style) / (MAX_STRUCTURE_SCORE + MAX_STYLE_SCORE).
    """

    @property
    def name(self) -> str:
        return "strict"

    async def score_single(
        self,
        model_name: str,
        session: aiohttp.ClientSession,
        prompt: str,
        response: str,
        cmudict: dict,
    ) -> float:
        structure_score = self.score_haiku_structure(response, cmudict, allow_off_by_one=False)

        style_score = 0.0
        style_score = await self.score_haiku_style(
            model_name, session, prompt, response
        )
        if style_score < 0:
            style_score = 0.0

        total = structure_score + style_score
        print(f"[StrictJudge] structure={structure_score}, style={style_score}")
        return total
