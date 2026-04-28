"""
Profanity filter for transcription output.

Masks or removes profane words from transcription text using a built-in
word list. Supports configurable masking characters and partial/full masking
modes.
"""

import re
from typing import Optional, Set

# Common English profanity word list (kept minimal and non-exhaustive)
_DEFAULT_PROFANITY = {
    "ass", "asshole", "bastard", "bitch", "bullshit", "cock", "crap",
    "damn", "dick", "fuck", "fucking", "fucker", "goddamn", "hell",
    "motherfucker", "motherfucking", "piss", "shit", "shitty", "whore",
}


def _build_pattern(words: Set[str]) -> re.Pattern:
    """Build a compiled regex that matches any of the given words (case-insensitive)."""
    escaped = sorted((re.escape(w) for w in words), key=len, reverse=True)
    return re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)


_DEFAULT_PATTERN = _build_pattern(_DEFAULT_PROFANITY)


def _mask_word(word: str, mask_char: str = "*", mode: str = "partial") -> str:
    """Mask a profane word.

    Args:
        word: The matched profane word.
        mask_char: Character used for masking.
        mode: "partial" keeps first and last character, "full" masks entirely.

    Returns:
        Masked version of the word.
    """
    if mode == "full" or len(word) <= 2:
        return mask_char * len(word)
    return word[0] + mask_char * (len(word) - 2) + word[-1]


def filter_profanity(
    text: str,
    mode: str = "partial",
    mask_char: str = "*",
    custom_words: Optional[Set[str]] = None,
    extra_words: Optional[Set[str]] = None,
) -> str:
    """Filter profanity from text by masking offensive words.

    Args:
        text: Input text to filter.
        mode: Masking mode - "partial" (f**k), "full" (****), or "remove".
        mask_char: Character used for masking. Default "*".
        custom_words: If provided, replaces the default profanity list entirely.
        extra_words: If provided, adds to the default profanity list.

    Returns:
        Text with profanity masked or removed.
    """
    if not text:
        return text

    if custom_words is not None:
        pattern = _build_pattern(custom_words)
    elif extra_words:
        pattern = _build_pattern(_DEFAULT_PROFANITY | extra_words)
    else:
        pattern = _DEFAULT_PATTERN

    if mode == "remove":
        # Remove the word and collapse extra spaces
        result = pattern.sub("", text)
        return re.sub(r' {2,}', ' ', result).strip()

    def _replacer(match):
        return _mask_word(match.group(0), mask_char=mask_char, mode=mode)

    return pattern.sub(_replacer, text)


def get_default_profanity_words() -> Set[str]:
    """Return a copy of the default profanity word set."""
    return set(_DEFAULT_PROFANITY)
