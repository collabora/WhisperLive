"""
PII (Personally Identifiable Information) redaction for transcription output.

Uses regex patterns to detect and redact common PII types:
- Social Security Numbers (SSN)
- Credit/debit card numbers
- Phone numbers (US formats)
- Email addresses
- IP addresses

All redaction is applied as a post-processing step on transcription text.
"""

import re
from typing import List, Optional, Set


# Redaction placeholder per PII type
_REDACTION_MAP = {
    "ssn": "[SSN_REDACTED]",
    "credit_card": "[CARD_REDACTED]",
    "phone": "[PHONE_REDACTED]",
    "email": "[EMAIL_REDACTED]",
    "ip_address": "[IP_REDACTED]",
}

# Regex patterns for PII detection
_PATTERNS = {
    "ssn": re.compile(
        r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
    ),
    "credit_card": re.compile(
        r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
    ),
    "phone": re.compile(
        r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
    ),
    "email": re.compile(
        r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    ),
    "ip_address": re.compile(
        r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
    ),
}

ALL_PII_TYPES = set(_PATTERNS.keys())

# Apply in this order: longer/more-specific patterns first
_APPLICATION_ORDER = ["credit_card", "ssn", "ip_address", "email", "phone"]


def redact_pii(
    text: str,
    pii_types: Optional[Set[str]] = None,
    custom_patterns: Optional[dict] = None,
) -> str:
    """Redact PII from text using regex patterns.

    Args:
        text: Input text to redact.
        pii_types: Set of PII types to redact. If None, redacts all types.
            Valid types: "ssn", "credit_card", "phone", "email", "ip_address".
        custom_patterns: Optional dict mapping label -> (compiled regex, replacement string).

    Returns:
        Text with PII replaced by redaction placeholders.
    """
    if not text:
        return text

    types_to_check = pii_types if pii_types is not None else ALL_PII_TYPES

    # Apply in fixed order: longer patterns first to prevent partial matches
    for pii_type in _APPLICATION_ORDER:
        if pii_type in types_to_check and pii_type in _PATTERNS:
            text = _PATTERNS[pii_type].sub(_REDACTION_MAP[pii_type], text)

    if custom_patterns:
        for label, (pattern, replacement) in custom_patterns.items():
            text = pattern.sub(replacement, text)

    return text


def get_supported_pii_types() -> List[str]:
    """Return list of supported PII type names."""
    return sorted(ALL_PII_TYPES)
