"""
Multi-model ensemble transcription.

Runs multiple Whisper models in parallel and merges their outputs using
configurable strategies (voting, confidence-weighted, longest).
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A transcription segment from a single model."""
    start: float
    end: float
    text: str
    confidence: float = 1.0
    model_name: str = ""


@dataclass
class EnsembleResult:
    """Merged result from multiple models."""
    segments: list = field(default_factory=list)
    text: str = ""
    model_results: dict = field(default_factory=dict)
    strategy: str = ""


class EnsembleTranscriber:
    """
    Run multiple Whisper models and merge results.

    Strategies:
    - "longest": Use output from the model producing the most text
    - "confidence": Use output from the model with highest avg confidence
    - "voting": Character-level majority voting across aligned outputs
    """

    def __init__(self, models=None, strategy="confidence", max_workers=None):
        """
        Args:
            models: Dict mapping model_name → callable(audio) → list[Segment]
            strategy: Merging strategy ("longest", "confidence", "voting")
            max_workers: Thread pool size (default: number of models)
        """
        self._models = models or {}
        self._strategy = strategy
        self._max_workers = max_workers
        self._lock = threading.Lock()

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        if value not in ("longest", "confidence", "voting"):
            raise ValueError(f"Unknown strategy: {value}")
        self._strategy = value

    def add_model(self, name, transcribe_fn):
        """Register a model.

        Args:
            name: Model identifier
            transcribe_fn: Callable that takes audio (numpy array) and returns
                          list of Segment objects
        """
        with self._lock:
            self._models[name] = transcribe_fn

    def remove_model(self, name):
        """Remove a model."""
        with self._lock:
            self._models.pop(name, None)

    @property
    def model_names(self):
        return list(self._models.keys())

    def transcribe(self, audio):
        """
        Transcribe audio with all models and merge results.

        Args:
            audio: Audio data (numpy array, 16kHz mono float32)

        Returns:
            EnsembleResult with merged segments and per-model results
        """
        if not self._models:
            return EnsembleResult(strategy=self._strategy)

        model_results = {}
        workers = self._max_workers or len(self._models)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for name, fn in self._models.items():
                futures[executor.submit(self._run_model, name, fn, audio)] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    segments = future.result()
                    model_results[name] = segments
                except Exception as e:
                    logger.error(f"Model '{name}' failed: {e}")
                    model_results[name] = []

        merged = self._merge(model_results)
        merged.model_results = model_results
        merged.strategy = self._strategy
        return merged

    def _run_model(self, name, fn, audio):
        """Run a single model's transcription."""
        logger.debug(f"Running model: {name}")
        segments = fn(audio)
        for s in segments:
            s.model_name = name
        return segments

    def _merge(self, model_results):
        """Merge results using the configured strategy."""
        if self._strategy == "longest":
            return self._merge_longest(model_results)
        elif self._strategy == "confidence":
            return self._merge_confidence(model_results)
        elif self._strategy == "voting":
            return self._merge_voting(model_results)
        else:
            raise ValueError(f"Unknown strategy: {self._strategy}")

    def _merge_longest(self, model_results):
        """Pick the model output with the most total text."""
        best_name = ""
        best_len = -1
        for name, segments in model_results.items():
            total = sum(len(s.text) for s in segments)
            if total > best_len:
                best_len = total
                best_name = name

        if not best_name:
            return EnsembleResult()

        segments = model_results[best_name]
        text = " ".join(s.text.strip() for s in segments)
        return EnsembleResult(segments=segments, text=text)

    def _merge_confidence(self, model_results):
        """Pick the model output with the highest average confidence."""
        best_name = ""
        best_conf = -1.0
        for name, segments in model_results.items():
            if not segments:
                continue
            avg_conf = sum(s.confidence for s in segments) / len(segments)
            if avg_conf > best_conf:
                best_conf = avg_conf
                best_name = name

        if not best_name:
            return EnsembleResult()

        segments = model_results[best_name]
        text = " ".join(s.text.strip() for s in segments)
        return EnsembleResult(segments=segments, text=text)

    def _merge_voting(self, model_results):
        """Character-level majority voting from aligned full texts."""
        texts = []
        for segments in model_results.values():
            full = " ".join(s.text.strip() for s in segments)
            if full:
                texts.append(full)

        if not texts:
            return EnsembleResult()
        if len(texts) == 1:
            return EnsembleResult(text=texts[0])

        max_len = max(len(t) for t in texts)
        result_chars = []
        for i in range(max_len):
            chars = {}
            for t in texts:
                if i < len(t):
                    c = t[i]
                    chars[c] = chars.get(c, 0) + 1
            if chars:
                winner = max(chars, key=chars.get)
                result_chars.append(winner)

        merged_text = "".join(result_chars).strip()
        return EnsembleResult(text=merged_text)


def create_faster_whisper_model_fn(model_size, device="cpu", compute_type="int8"):
    """
    Create a transcribe function wrapping a faster-whisper model.

    Args:
        model_size: Model name/path (e.g., "tiny", "base", "small")
        device: "cpu" or "cuda"
        compute_type: "int8", "float16", "float32"

    Returns:
        Callable that takes audio → list[Segment]
    """
    def transcribe_fn(audio):
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        raw_segments, info = model.transcribe(audio)
        segments = []
        for s in raw_segments:
            segments.append(Segment(
                start=s.start,
                end=s.end,
                text=s.text,
                confidence=getattr(s, "avg_logprob", 0.0),
                model_name=model_size,
            ))
        return segments

    return transcribe_fn
