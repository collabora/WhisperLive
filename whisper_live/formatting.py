"""
Post-processing pipeline for improving transcription readability.

Applies optional formatting rules:
- Capitalize first letter of each sentence
- Normalize spoken numbers ("twenty one" -> "21")
- Basic punctuation cleanup (collapse spaces, strip trailing whitespace)
"""

import re

# Spoken-number word values
_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}

_NUMBER_WORDS = set(_ONES) | set(_TENS) | set(_SCALES) | {"and"}


def _words_to_number(words):
    """Convert a list of spoken-number words to an integer, or None on failure."""
    if not words:
        return None
    # Filter out 'and'
    tokens = [w for w in words if w != "and"]
    if not tokens:
        return None
    try:
        result = 0
        current = 0
        for tok in tokens:
            if tok in _ONES:
                current += _ONES[tok]
            elif tok in _TENS:
                current += _TENS[tok]
            elif tok == "hundred":
                current = (current or 1) * 100
            elif tok in _SCALES:
                current = (current or 1) * _SCALES[tok]
                result += current
                current = 0
            else:
                return None
        return result + current
    except Exception:
        return None


def _replace_spoken_numbers(text):
    """Replace sequences of spoken number words with digits."""
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        lower = words[i].lower().rstrip(".,!?;:")
        trailing_punct = words[i][len(lower):]
        if lower in _NUMBER_WORDS and lower != "and":
            # Collect consecutive number words
            num_words = []
            j = i
            while j < len(words):
                w = words[j].lower().rstrip(".,!?;:")
                if w in _NUMBER_WORDS:
                    num_words.append(w)
                    j += 1
                else:
                    break
            # Grab trailing punctuation from last number word
            last_raw = words[j - 1]
            last_clean = last_raw.lower().rstrip(".,!?;:")
            end_punct = last_raw[len(last_clean):]

            val = _words_to_number(num_words)
            if val is not None:
                out.append(str(val) + end_punct)
                i = j
            else:
                out.append(words[i])
                i += 1
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def _capitalize_sentences(text):
    """Capitalize the first letter after sentence-ending punctuation."""
    # Capitalize start of string
    if text:
        text = text[0].upper() + text[1:]
    # Capitalize after . ! ?
    text = re.sub(r'([.!?])\s+([a-z])', lambda m: m.group(1) + " " + m.group(2).upper(), text)
    return text


def _collapse_whitespace(text):
    """Collapse multiple spaces into one and strip."""
    return re.sub(r' {2,}', ' ', text).strip()


def format_transcript(text, capitalize=True, numbers=True):
    """Apply smart formatting to a transcription text.

    Args:
        text: Raw transcription text.
        capitalize: Capitalize sentence starts.
        numbers: Convert spoken numbers to digits.

    Returns:
        Formatted text string.
    """
    if not text:
        return text
    text = _collapse_whitespace(text)
    if numbers:
        text = _replace_spoken_numbers(text)
    if capitalize:
        text = _capitalize_sentences(text)
    return text
