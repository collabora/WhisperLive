"""
Utterance detection and automatic paragraph segmentation.

Utterance detection identifies natural speech boundaries — where a speaker
finishes a thought vs. a brief pause. This goes beyond simple silence/VAD
by considering pause duration, sentence-ending punctuation, and segment length.

Paragraph segmentation groups utterances into coherent paragraphs based on
longer pauses, topic shifts, or speaker changes.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Utterance:
    """A single utterance (a complete thought or sentence)."""
    text: str
    start: float  # seconds
    end: float  # seconds
    speaker: Optional[str] = None
    is_final: bool = True  # False if still being built

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = {"text": self.text, "start": self.start, "end": self.end,
             "is_final": self.is_final}
        if self.speaker:
            d["speaker"] = self.speaker
        return d


@dataclass
class Paragraph:
    """A paragraph is a group of related utterances."""
    utterances: List[Utterance] = field(default_factory=list)
    speaker: Optional[str] = None

    @property
    def text(self) -> str:
        return " ".join(u.text for u in self.utterances)

    @property
    def start(self) -> float:
        return self.utterances[0].start if self.utterances else 0.0

    @property
    def end(self) -> float:
        return self.utterances[-1].end if self.utterances else 0.0

    def to_dict(self) -> dict:
        d = {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "sentences": [u.to_dict() for u in self.utterances],
            "num_sentences": len(self.utterances),
        }
        if self.speaker:
            d["speaker"] = self.speaker
        return d


# Sentence-ending punctuation
_SENTENCE_END = re.compile(r'[.!?]\s*$')


class UtteranceDetector:
    """Detects utterance boundaries from a stream of transcription segments.

    Uses a combination of:
    - Pause duration between segments
    - Sentence-ending punctuation
    - Maximum utterance length
    """

    def __init__(
        self,
        min_pause_seconds: float = 0.7,
        max_utterance_seconds: float = 30.0,
        sentence_end_pause_seconds: float = 0.3,
    ):
        """
        Args:
            min_pause_seconds: Minimum silence gap to force an utterance break.
            max_utterance_seconds: Maximum duration before forcing a break.
            sentence_end_pause_seconds: If a segment ends with punctuation and
                the gap is at least this long, break the utterance.
        """
        self.min_pause = min_pause_seconds
        self.max_duration = max_utterance_seconds
        self.sentence_end_pause = sentence_end_pause_seconds
        self._buffer_text = ""
        self._buffer_start = 0.0
        self._buffer_end = 0.0
        self._buffer_speaker = None

    def reset(self):
        """Reset the detector state."""
        self._buffer_text = ""
        self._buffer_start = 0.0
        self._buffer_end = 0.0
        self._buffer_speaker = None

    def process_segment(
        self,
        text: str,
        start: float,
        end: float,
        speaker: Optional[str] = None,
    ) -> List[Utterance]:
        """Process a transcription segment and return any completed utterances.

        Args:
            text: Segment text.
            start: Segment start time in seconds.
            end: Segment end time in seconds.
            speaker: Optional speaker label.

        Returns:
            List of completed Utterance objects (may be empty).
        """
        completed = []

        if not self._buffer_text:
            # Start a new utterance
            self._buffer_text = text.strip()
            self._buffer_start = start
            self._buffer_end = end
            self._buffer_speaker = speaker
            return completed

        gap = start - self._buffer_end
        duration = end - self._buffer_start

        # Check for utterance break conditions
        should_break = False

        # 1. Long pause always breaks
        if gap >= self.min_pause:
            should_break = True

        # 2. Sentence-ending punctuation + short pause
        if gap >= self.sentence_end_pause and _SENTENCE_END.search(self._buffer_text):
            should_break = True

        # 3. Speaker change
        if speaker and self._buffer_speaker and speaker != self._buffer_speaker:
            should_break = True

        # 4. Maximum duration exceeded
        if duration > self.max_duration:
            should_break = True

        if should_break:
            # Emit the buffered utterance
            completed.append(Utterance(
                text=self._buffer_text,
                start=self._buffer_start,
                end=self._buffer_end,
                speaker=self._buffer_speaker,
                is_final=True,
            ))
            # Start new buffer
            self._buffer_text = text.strip()
            self._buffer_start = start
            self._buffer_end = end
            self._buffer_speaker = speaker
        else:
            # Extend the current utterance
            self._buffer_text += " " + text.strip()
            self._buffer_end = end
            if speaker:
                self._buffer_speaker = speaker

        return completed

    def flush(self) -> Optional[Utterance]:
        """Flush any remaining buffered text as a final utterance."""
        if self._buffer_text:
            utterance = Utterance(
                text=self._buffer_text,
                start=self._buffer_start,
                end=self._buffer_end,
                speaker=self._buffer_speaker,
                is_final=True,
            )
            self.reset()
            return utterance
        return None


class ParagraphSegmenter:
    """Groups utterances into paragraphs based on pauses and speaker changes.

    A new paragraph starts when:
    - There is a long pause between utterances
    - The speaker changes (if diarization is available)
    - A maximum number of sentences is reached
    """

    def __init__(
        self,
        paragraph_pause_seconds: float = 2.0,
        max_sentences_per_paragraph: int = 8,
        split_on_speaker_change: bool = True,
    ):
        """
        Args:
            paragraph_pause_seconds: Minimum gap between utterances to start
                a new paragraph.
            max_sentences_per_paragraph: Force a paragraph break after this
                many sentences.
            split_on_speaker_change: Start a new paragraph when the speaker
                changes.
        """
        self.paragraph_pause = paragraph_pause_seconds
        self.max_sentences = max_sentences_per_paragraph
        self.split_on_speaker = split_on_speaker_change

    def segment(self, utterances: List[Utterance]) -> List[Paragraph]:
        """Segment a list of utterances into paragraphs.

        Args:
            utterances: Ordered list of utterances.

        Returns:
            List of Paragraph objects.
        """
        if not utterances:
            return []

        paragraphs = []
        current = Paragraph(
            utterances=[utterances[0]],
            speaker=utterances[0].speaker,
        )

        for i in range(1, len(utterances)):
            utt = utterances[i]
            prev = utterances[i - 1]
            gap = utt.start - prev.end

            should_break = False

            # Long pause
            if gap >= self.paragraph_pause:
                should_break = True

            # Speaker change
            if (self.split_on_speaker and utt.speaker and prev.speaker
                    and utt.speaker != prev.speaker):
                should_break = True

            # Max sentences
            if len(current.utterances) >= self.max_sentences:
                should_break = True

            if should_break:
                paragraphs.append(current)
                current = Paragraph(
                    utterances=[utt],
                    speaker=utt.speaker,
                )
            else:
                current.utterances.append(utt)

        paragraphs.append(current)
        return paragraphs


def detect_utterances_from_segments(
    segments: List[dict],
    min_pause: float = 0.7,
    max_duration: float = 30.0,
) -> List[Utterance]:
    """Convenience function: convert a list of segment dicts to utterances.

    Each segment dict should have 'text', 'start', 'end', and optionally 'speaker'.
    """
    detector = UtteranceDetector(
        min_pause_seconds=min_pause,
        max_utterance_seconds=max_duration,
    )
    utterances = []
    for seg in segments:
        completed = detector.process_segment(
            text=seg.get("text", ""),
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            speaker=seg.get("speaker"),
        )
        utterances.extend(completed)
    final = detector.flush()
    if final:
        utterances.append(final)
    return utterances


def segment_into_paragraphs(
    segments: List[dict],
    paragraph_pause: float = 2.0,
    max_sentences: int = 8,
) -> List[dict]:
    """Convenience function: segments → utterances → paragraphs as dicts.

    Returns list of paragraph dicts with 'text', 'start', 'end', 'sentences'.
    """
    utterances = detect_utterances_from_segments(segments)
    segmenter = ParagraphSegmenter(
        paragraph_pause_seconds=paragraph_pause,
        max_sentences_per_paragraph=max_sentences,
    )
    paragraphs = segmenter.segment(utterances)
    return [p.to_dict() for p in paragraphs]
