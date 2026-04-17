"""
Post-processing pipeline for improving transcription readability.

Applies optional formatting rules:
- Capitalize first letter of each sentence
- Normalize spoken numbers ("twenty one" -> "21")
- Basic punctuation cleanup (collapse spaces, strip trailing whitespace)
- Smart formatting: dates, times, currency, percentages, ordinals
- Find & replace: custom term substitutions via regex or plain text
"""

import re
from typing import List, Tuple, Optional

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


def format_transcript(text, capitalize=True, numbers=True, smart=False):
    """Apply smart formatting to a transcription text.

    Args:
        text: Raw transcription text.
        capitalize: Capitalize sentence starts.
        numbers: Convert spoken numbers to digits.
        smart: Apply smart formatting (dates, times, currency, etc.).

    Returns:
        Formatted text string.
    """
    if not text:
        return text
    text = _collapse_whitespace(text)
    if numbers:
        text = _replace_spoken_numbers(text)
    if smart:
        text = smart_format(text)
    if capitalize:
        text = _capitalize_sentences(text)
    return text


# ---------------------------------------------------------------------------
# Smart Formatting — dates, times, currency, percentages, ordinals
# ---------------------------------------------------------------------------

_MONTHS = {
    "january": "January", "february": "February", "march": "March",
    "april": "April", "may": "May", "june": "June", "july": "July",
    "august": "August", "september": "September", "october": "October",
    "november": "November", "december": "December",
}

_ORDINAL_MAP = {
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
    "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
    "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
    "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th",
    "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
    "nineteenth": "19th", "twentieth": "20th", "thirtieth": "30th",
}

_CURRENCY_MAP = {
    "dollars": "$", "dollar": "$", "bucks": "$",
    "euros": "€", "euro": "€",
    "pounds": "£", "pound": "£",
    "yen": "¥",
}


def _format_currency(text):
    """Convert 'fifty dollars' → '$50', 'twenty euros' → '€20'."""
    pattern = re.compile(
        r'\b(\d+)\s+(' + '|'.join(_CURRENCY_MAP.keys()) + r')\b',
        re.IGNORECASE,
    )

    def _replace(m):
        amount = m.group(1)
        currency = _CURRENCY_MAP[m.group(2).lower()]
        return f"{currency}{amount}"

    return pattern.sub(_replace, text)


def _format_percentages(text):
    """Convert 'fifty percent' or '50 percent' → '50%'."""
    text = re.sub(r'\b(\d+)\s+percent\b', r'\1%', text, flags=re.IGNORECASE)
    return text


def _format_times(text):
    """Convert spoken times: 'three thirty p m' → '3:30 PM'."""
    # "X thirty/fifteen/forty-five AM/PM"
    time_pattern = re.compile(
        r'\b(\d{1,2})\s+(o\'?clock|(?:fifteen|thirty|forty[\s-]?five))\s*(a\s*m|p\s*m)?\b',
        re.IGNORECASE,
    )
    minute_map = {"fifteen": "15", "thirty": "30", "forty five": "45",
                  "forty-five": "45", "fortyfive": "45"}

    def _replace_time(m):
        hour = m.group(1)
        minute_word = m.group(2).lower().replace(" ", "")
        period = m.group(3)
        if minute_word == "o'clock" or minute_word == "oclock":
            minutes = "00"
        else:
            minutes = minute_map.get(minute_word, minute_word)
        result = f"{hour}:{minutes}"
        if period:
            result += f" {period.replace(' ', '').upper()}"
        return result

    return time_pattern.sub(_replace_time, text)


def _format_ordinals(text):
    """Convert spoken ordinals to numeric: 'twenty first' → '21st'."""
    for word, num in _ORDINAL_MAP.items():
        text = re.sub(r'\b' + word + r'\b', num, text, flags=re.IGNORECASE)
    return text


def _format_dates(text):
    """Convert 'January fifteenth twenty twenty three' patterns."""
    month_pattern = '|'.join(_MONTHS.keys())
    # "Month Day" pattern (month + number)
    text = re.sub(
        r'\b(' + month_pattern + r')\s+(\d{1,2}(?:st|nd|rd|th)?)\b',
        lambda m: _MONTHS[m.group(1).lower()] + " " + m.group(2),
        text,
        flags=re.IGNORECASE,
    )
    return text


def smart_format(text):
    """Apply all smart formatting rules.

    Converts spoken dates, times, currency, percentages, and ordinals
    to their written/numeric forms.

    Args:
        text: Input text.

    Returns:
        Formatted text.
    """
    if not text:
        return text
    text = _format_ordinals(text)
    text = _format_currency(text)
    text = _format_percentages(text)
    text = _format_times(text)
    text = _format_dates(text)
    return text


# ---------------------------------------------------------------------------
# Find & Replace — custom term substitutions
# ---------------------------------------------------------------------------


def find_and_replace(
    text: str,
    replacements: List[Tuple[str, str]],
    use_regex: bool = False,
    case_sensitive: bool = False,
) -> str:
    """Apply a list of find-and-replace substitutions to text.

    Args:
        text: Input text.
        replacements: List of (find, replace) tuples.
        use_regex: If True, treat find patterns as regex.
        case_sensitive: If True, matches are case-sensitive.

    Returns:
        Text with all substitutions applied.
    """
    if not text or not replacements:
        return text

    flags = 0 if case_sensitive else re.IGNORECASE

    for find, replace_with in replacements:
        if use_regex:
            text = re.sub(find, replace_with, text, flags=flags)
        else:
            # Escape the find pattern for literal matching
            pattern = re.escape(find)
            text = re.sub(pattern, replace_with, text, flags=flags)

    return text
