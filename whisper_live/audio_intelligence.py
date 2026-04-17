"""
Audio intelligence pipeline for post-transcription analysis.

Provides lightweight, zero-dependency analysis modules:
- Sentiment analysis (keyword/heuristic-based)
- Topic detection (keyword frequency)
- Entity extraction (regex-based named entities)
- Extractive summarization (sentence scoring)

These are designed as a fast baseline. For production NLP, integrate
spaCy, transformers, or a dedicated NLP API.
"""

import re
import math
from collections import Counter
from typing import Dict, List, Optional, Any


# --- Sentiment Analysis ---

_POSITIVE_WORDS = {
    "good", "great", "excellent", "wonderful", "amazing", "fantastic",
    "happy", "love", "best", "perfect", "awesome", "outstanding",
    "beautiful", "brilliant", "superb", "pleased", "delighted",
    "impressive", "terrific", "incredible", "thank", "thanks",
    "appreciate", "glad", "enjoy", "success", "successful",
}

_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "worst", "hate",
    "angry", "sad", "poor", "disappointed", "frustrating", "annoying",
    "ugly", "stupid", "fail", "failed", "failure", "wrong",
    "problem", "issue", "error", "broken", "sucks", "unfortunately",
    "unhappy", "difficult", "confused", "complaint", "complain",
}


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of text using keyword scoring.

    Returns:
        Dict with 'label' ("positive", "negative", "neutral"),
        'score' (float -1.0 to 1.0), 'positive_count', 'negative_count'.
    """
    if not text:
        return {"label": "neutral", "score": 0.0, "positive_count": 0, "negative_count": 0}

    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    total = pos + neg

    if total == 0:
        score = 0.0
    else:
        score = (pos - neg) / total

    if score > 0.1:
        label = "positive"
    elif score < -0.1:
        label = "negative"
    else:
        label = "neutral"

    return {"label": label, "score": round(score, 3), "positive_count": pos, "negative_count": neg}


# --- Topic Detection ---

# Common English stop words to exclude
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "and", "but", "or", "nor", "not", "so", "for", "yet", "to", "of", "in",
    "on", "at", "by", "with", "from", "up", "about", "into", "through",
    "during", "before", "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "because", "as", "until", "while", "what",
    "which", "who", "whom", "if", "also", "well", "much", "many", "any",
    "like", "even", "still", "get", "got", "go", "going", "make", "made",
    "say", "said", "know", "think", "see", "come", "take", "want", "give",
    "use", "find", "tell", "ask", "seem", "feel", "try", "leave", "call",
}


def detect_topics(text: str, top_n: int = 5, min_word_length: int = 4) -> List[Dict[str, Any]]:
    """Detect topics by word frequency analysis.

    Args:
        text: Input text to analyze.
        top_n: Number of top topics to return.
        min_word_length: Minimum word length to consider.

    Returns:
        List of dicts with 'topic' (word) and 'count'.
    """
    if not text:
        return []

    words = re.findall(r'\b[a-z]+\b', text.lower())
    filtered = [w for w in words if len(w) >= min_word_length and w not in _STOP_WORDS]
    counts = Counter(filtered)
    return [{"topic": word, "count": count} for word, count in counts.most_common(top_n)]


# --- Entity Extraction ---

_ENTITY_PATTERNS = {
    "date": re.compile(
        r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|'
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
        r'\s+\d{1,2}(?:,?\s+\d{4})?)\b',
        re.IGNORECASE,
    ),
    "time": re.compile(
        r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b'
    ),
    "money": re.compile(
        r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
    ),
    "percentage": re.compile(
        r'\b\d+(?:\.\d+)?%'
    ),
    "url": re.compile(
        r'https?://[^\s<>\"\']+|www\.[^\s<>\"\']+',
        re.IGNORECASE,
    ),
}


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities from text using regex patterns.

    Returns:
        Dict mapping entity type to list of matched strings.
    """
    if not text:
        return {}

    entities = {}
    for entity_type, pattern in _ENTITY_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            entities[entity_type] = list(set(matches))
    return entities


# --- Extractive Summarization ---

def summarize(text: str, num_sentences: int = 3) -> str:
    """Produce an extractive summary by scoring sentences.

    Scores each sentence by sum of its word frequencies (TF-based).

    Args:
        text: Input text to summarize.
        num_sentences: Number of sentences to include in summary.

    Returns:
        Summary string composed of top-scoring sentences in original order.
    """
    if not text:
        return ""

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= num_sentences:
        return text.strip()

    # Build word frequency table
    words = re.findall(r'\b[a-z]+\b', text.lower())
    freq = Counter(w for w in words if w not in _STOP_WORDS)

    # Score each sentence
    scored = []
    for i, sent in enumerate(sentences):
        sent_words = re.findall(r'\b[a-z]+\b', sent.lower())
        score = sum(freq.get(w, 0) for w in sent_words)
        # Normalize by sentence length to avoid bias toward long sentences
        if sent_words:
            score /= math.sqrt(len(sent_words))
        scored.append((i, score, sent))

    # Select top sentences and sort by original position
    scored.sort(key=lambda x: x[1], reverse=True)
    top = sorted(scored[:num_sentences], key=lambda x: x[0])
    return " ".join(item[2] for item in top)


# --- Pipeline ---

def analyze_transcript(
    text: str,
    sentiment: bool = True,
    topics: bool = True,
    entities: bool = True,
    summary: bool = True,
    summary_sentences: int = 3,
    topic_count: int = 5,
) -> Dict[str, Any]:
    """Run the full audio intelligence pipeline on a transcript.

    Args:
        text: Complete transcript text.
        sentiment: Include sentiment analysis.
        topics: Include topic detection.
        entities: Include entity extraction.
        summary: Include extractive summarization.
        summary_sentences: Number of sentences in summary.
        topic_count: Number of topics to detect.

    Returns:
        Dict with results for each enabled analysis type.
    """
    result: Dict[str, Any] = {}

    if sentiment:
        result["sentiment"] = analyze_sentiment(text)
    if topics:
        result["topics"] = detect_topics(text, top_n=topic_count)
    if entities:
        result["entities"] = extract_entities(text)
    if summary:
        result["summary"] = summarize(text, num_sentences=summary_sentences)

    return result
