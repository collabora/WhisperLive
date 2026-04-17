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


# --- Auto-Highlights (Key Phrases) ---

def extract_highlights(text: str, max_highlights: int = 10, min_phrase_length: int = 2) -> List[Dict[str, Any]]:
    """Extract key phrases / highlights from a transcript.

    Uses TF-based scoring on multi-word phrases (bigrams and trigrams)
    to identify the most important phrases.

    Args:
        text: Input transcript text.
        max_highlights: Maximum number of key phrases to return.
        min_phrase_length: Minimum number of words in a phrase (2=bigrams, 3=trigrams).

    Returns:
        List of dicts with 'text' (the phrase), 'count', and 'rank'.
    """
    if not text:
        return []

    words = re.findall(r'\b[a-z]+\b', text.lower())
    # Filter stop words
    filtered = [(i, w) for i, w in enumerate(words) if w not in _STOP_WORDS and len(w) >= 3]

    # Build n-grams (bigrams and trigrams)
    phrases = Counter()
    for idx in range(len(filtered)):
        for n in range(min_phrase_length, 4):  # 2-grams and 3-grams
            window = []
            pos = idx
            while len(window) < n and pos < len(filtered):
                # Only consecutive words (within 2 positions in original)
                if window and filtered[pos][0] - filtered[len(window) - 1 + idx][0] > 2:
                    break
                window.append(filtered[pos][1])
                pos += 1
            if len(window) == n:
                phrase = " ".join(window)
                phrases[phrase] += 1

    # Filter phrases that appear at least twice
    significant = [(phrase, count) for phrase, count in phrases.items() if count >= 2]
    significant.sort(key=lambda x: x[1], reverse=True)

    highlights = []
    for rank, (phrase, count) in enumerate(significant[:max_highlights], 1):
        highlights.append({"text": phrase, "count": count, "rank": rank})

    # If not enough multi-word phrases, fall back to single important words
    if len(highlights) < max_highlights:
        word_freq = Counter(w for _, w in filtered)
        for word, count in word_freq.most_common(max_highlights - len(highlights)):
            if count >= 2 and not any(word in h["text"] for h in highlights):
                highlights.append({"text": word, "count": count, "rank": len(highlights) + 1})

    return highlights[:max_highlights]


# --- Auto-Chapters ---

def generate_chapters(
    segments: List[Dict[str, Any]],
    max_chapter_duration: float = 300.0,
    min_chapter_duration: float = 30.0,
) -> List[Dict[str, Any]]:
    """Generate auto-chapters from transcription segments.

    Groups segments into chapters based on time gaps and topic shifts.
    Each chapter gets an auto-generated title from its key phrases.

    Args:
        segments: List of segment dicts with 'text', 'start', 'end'.
        max_chapter_duration: Maximum chapter length in seconds.
        min_chapter_duration: Minimum chapter length in seconds.

    Returns:
        List of chapter dicts with 'start', 'end', 'title', 'summary', 'text'.
    """
    if not segments:
        return []

    chapters = []
    current_segs = [segments[0]]

    for i in range(1, len(segments)):
        seg = segments[i]
        prev = segments[i - 1]
        gap = seg.get("start", 0) - prev.get("end", 0)
        chapter_duration = seg.get("end", 0) - current_segs[0].get("start", 0)

        # Break on long pause or max duration
        should_break = False
        if gap >= 3.0 and chapter_duration >= min_chapter_duration:
            should_break = True
        if chapter_duration >= max_chapter_duration:
            should_break = True

        if should_break:
            chapters.append(_build_chapter(current_segs))
            current_segs = [seg]
        else:
            current_segs.append(seg)

    if current_segs:
        chapters.append(_build_chapter(current_segs))

    return chapters


def _build_chapter(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a chapter from a group of segments."""
    text = " ".join(s.get("text", "").strip() for s in segments)
    # Generate title from top keywords
    topics = detect_topics(text, top_n=3, min_word_length=3)
    if topics:
        title = " ".join(t["topic"].capitalize() for t in topics[:3])
    else:
        # Fallback: first few words
        words = text.split()[:5]
        title = " ".join(words) + ("..." if len(text.split()) > 5 else "")

    summary_text = summarize(text, num_sentences=1) if len(text.split()) > 20 else text

    return {
        "start": segments[0].get("start", 0.0),
        "end": segments[-1].get("end", 0.0),
        "title": title,
        "summary": summary_text,
        "text": text,
    }


# --- Filler Word Removal ---

_FILLER_WORDS = {
    "um", "uh", "uhm", "umm", "hmm", "hm", "mm",
    "er", "err", "ah", "ahh", "eh",
    "like",  # Only when used as filler
    "you know", "i mean", "sort of", "kind of",
    "basically", "actually", "literally", "right",
}

# Multi-word fillers (order: longest first for greedy matching)
_MULTI_WORD_FILLERS = sorted(
    [f for f in _FILLER_WORDS if " " in f],
    key=len, reverse=True,
)
_SINGLE_WORD_FILLERS = {f for f in _FILLER_WORDS if " " not in f}


def remove_filler_words(text: str, aggressive: bool = False) -> str:
    """Remove filler words from transcript text.

    Args:
        text: Input transcript text.
        aggressive: If True, also removes borderline fillers like
            'like', 'basically', 'actually', 'literally', 'right'
            which can be real words in context.

    Returns:
        Cleaned text with filler words removed.
    """
    if not text:
        return text

    # Conservative set (always remove)
    conservative = {"um", "uh", "uhm", "umm", "hmm", "hm", "mm", "er", "err", "ah", "ahh", "eh"}
    conservative_multi = {"you know", "i mean", "sort of", "kind of"}

    if aggressive:
        single = _SINGLE_WORD_FILLERS
        multi = _MULTI_WORD_FILLERS
    else:
        single = conservative
        multi = sorted(conservative_multi, key=len, reverse=True)

    # Remove multi-word fillers first
    result = text
    for filler in multi:
        pattern = re.compile(r'\b' + re.escape(filler) + r'\b[,]?\s*', re.IGNORECASE)
        result = pattern.sub('', result)

    # Remove single-word fillers
    for filler in single:
        pattern = re.compile(r'\b' + re.escape(filler) + r'\b[,]?\s*', re.IGNORECASE)
        result = pattern.sub('', result)

    # Clean up double spaces and leading/trailing whitespace
    result = re.sub(r' {2,}', ' ', result).strip()
    # Fix capitalization after removal at sentence start
    result = re.sub(r'^([a-z])', lambda m: m.group(1).upper(), result)

    return result


# --- Custom Spelling Hints ---

def apply_spelling_hints(text: str, hints: Dict[str, str]) -> str:
    """Apply custom spelling corrections to transcript text.

    Unlike find & replace, this is specifically for correcting ASR
    mis-spellings of proper nouns, technical terms, etc.

    Args:
        text: Input transcript text.
        hints: Dict mapping wrong spelling → correct spelling.
            Example: {"cube ernestes": "Kubernetes", "pie torch": "PyTorch"}

    Returns:
        Text with spelling corrections applied.
    """
    if not text or not hints:
        return text

    for wrong, correct in hints.items():
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        text = pattern.sub(correct, text)

    return text


# --- Pipeline ---

def analyze_transcript(
    text: str,
    sentiment: bool = True,
    topics: bool = True,
    entities: bool = True,
    summary: bool = True,
    highlights: bool = False,
    summary_sentences: int = 3,
    topic_count: int = 5,
    max_highlights: int = 10,
) -> Dict[str, Any]:
    """Run the full audio intelligence pipeline on a transcript.

    Args:
        text: Complete transcript text.
        sentiment: Include sentiment analysis.
        topics: Include topic detection.
        entities: Include entity extraction.
        summary: Include extractive summarization.
        highlights: Include auto-highlights (key phrases).
        summary_sentences: Number of sentences in summary.
        topic_count: Number of topics to detect.
        max_highlights: Maximum key phrases to extract.

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
    if highlights:
        result["highlights"] = extract_highlights(text, max_highlights=max_highlights)

    return result
