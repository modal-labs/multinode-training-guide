"""
NLTK Syllable & Sentence Utilities.
"""

import re


# Ref: https://stackoverflow.com/questions/49581705/using-cmudict-to-count-syllables
# CMU dict stores phonemes; syllables = count of vowel sounds (digits in phoneme)
def lookup_word(word_s, cmudict: dict):
    return cmudict.get(word_s, None)


def is_acronym(word: str) -> bool:
    # I specificaly added this because of the word 'GPU'.
    return word.isupper() and 2 <= len(word) <= 6 and word.isalpha()


def count_syllables_for_word(word, cmudict):
    original_word = word
    word = word.lower().strip()
    
    # Check CMU dict first
    phones = lookup_word(word, cmudict)
    if phones:
        phones0 = phones[0]
        return len([p for p in phones0 if p[-1].isdigit()])
    
    # Handle acronyms: count syllables per letter
    if is_acronym(original_word):
        total = 0
        for c in original_word.lower():
            if c == 'w':
                total += 3  # "dub-ul-you"
            else:
                total += 1
        return total
    
    # Fallback heuristic for unknown words
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)

def diff_syllables_count(text: str, target_syllables: int, cmudict: dict) -> int:
    """Output the difference between the number of syllables in the text and the target number of syllables."""
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
