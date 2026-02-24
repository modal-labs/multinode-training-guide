import re

import nltk
from nltk.corpus import cmudict as nltk_cmudict

from slime.utils.types import Sample

_cmudict = None


def _get_cmudict() -> dict:
    global _cmudict
    if _cmudict is None:
        nltk.download("cmudict", quiet=True)
        _cmudict = dict(nltk_cmudict.dict())
    return _cmudict


def lookup_word(word_s, cmudict: dict):
    return cmudict.get(word_s, None)


def is_acronym(word: str) -> bool:
    return word.isupper() and 2 <= len(word) <= 6 and word.isalpha()


def count_syllables_for_word(word, cmudict):
    original_word = word
    word = word.lower().strip()

    phones = lookup_word(word, cmudict)
    if phones:
        return len([p for p in phones[0] if p[-1].isdigit()])

    if is_acronym(original_word):
        total = 0
        for c in original_word.lower():
            if c == "w":
                total += 3  # "dub-ul-you"
            else:
                total += 1
        return total

    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def diff_syllables_count(text: str, target_syllables: int, cmudict: dict) -> int:
    words = re.findall(r"[a-zA-Z]+", text)
    total_syllables = sum(count_syllables_for_word(w, cmudict) for w in words)
    return abs(total_syllables - target_syllables)


def segment_haiku_lines(response: str) -> list[str]:
    if "/" in response:
        lines = [line.strip() for line in response.split("/")]
    elif ". " in response:
        lines = [line.strip() for line in response.split(". ")]
    else:
        lines = [line.strip() for line in response.split("\n")]
    return [line for line in lines if line]


def score_syllable_line(diff: int, allow_off_by_one: bool) -> float:
    if diff == 0:
        return 1.0
    if diff == 1:
        return 1.0 if allow_off_by_one else 0.5
    return 0.0


def score_haiku_structure(response: str, cmudict: dict, allow_off_by_one: bool = False) -> float:
    """Score haiku structure (0-1): 1/4 for 3 lines + up to 1/4 per line for syllables."""
    lines = segment_haiku_lines(response)
    score = 0.0
    fractional_multiplier = 0.25

    if len(lines) == 3:
        score += fractional_multiplier

    targets = [5, 7, 5]
    for i, target in enumerate(targets):
        if i < len(lines):
            diff = diff_syllables_count(lines[i], target, cmudict)
            score += score_syllable_line(diff, allow_off_by_one) * fractional_multiplier

    return score


async def haiku_rm(args, sample: Sample, **kwargs) -> float:
    cmudict = _get_cmudict()
    allow_off_by_one = getattr(args, "haiku_allow_off_by_one", False)
    return score_haiku_structure(sample.response, cmudict, allow_off_by_one=allow_off_by_one)
