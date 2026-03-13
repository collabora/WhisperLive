"""
Batch inference scheduler for WhisperLive.

Replaces the per-session SINGLE_MODEL_LOCK with a queue-based batch system.
Multiple sessions submit audio to a central queue; a single dedicated thread
collects pending requests and runs them as a GPU batch via CTranslate2's
batched encode() + generate() API.

For batch_size=1, falls back to standard transcriber.transcribe() for
identical behavior to the non-batched path.

Usage:
    Enable via ``--batch_inference`` CLI flag. The batch worker is lazily
    started after the first client connects and the shared model is loaded.

Thread safety:
    - ``queue.Queue`` is stdlib thread-safe.
    - Each ``BatchRequest.future`` (``threading.Event``) is written by the
      batch worker BEFORE ``.set()``, read by the session thread AFTER
      ``.wait()`` — no data race.
    - Only the batch worker thread touches the GPU model — zero lock
      contention between session threads.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Dict, List, Optional

import numpy as np

from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.vad import (
    VadOptions,
    collect_chunks,
    get_speech_timestamps,
)

from whisper_live.transcriber.transcriber_faster_whisper import (
    Segment,
    TranscriptionInfo,
    get_compression_ratio,
    get_suppressed_tokens,
)


@dataclass
class BatchRequest:
    """A single inference request submitted by a session thread.

    The session thread creates this, calls ``BatchInferenceWorker.submit()``,
    then blocks on ``future.wait()``.  The batch worker fills ``result``
    and/or ``error``, then signals ``future.set()``.

    Attributes:
        audio: Raw audio samples (float32, 16 kHz mono).
        language: ISO language code or None for auto-detection.
        task: ``"transcribe"`` or ``"translate"``.
        initial_prompt: Optional prompt for Whisper conditioning.
        use_vad: Whether to apply Voice Activity Detection.
        vad_parameters: Parameters forwarded to ``VadOptions``.
        future: Event signaled when the result is ready.
        result: List of ``Segment`` objects (filled by worker).
        info: ``TranscriptionInfo`` metadata (filled by worker).
        error: Exception instance if processing failed.
    """
    audio: np.ndarray
    language: Optional[str] = None
    task: str = "transcribe"
    initial_prompt: Optional[str] = None
    use_vad: bool = True
    vad_parameters: Optional[Dict] = None
    # Signaling
    future: threading.Event = field(default_factory=threading.Event)
    # Results (filled by batch worker)
    result: Optional[Any] = None
    info: Optional[Any] = None
    error: Optional[Exception] = None


class BatchInferenceWorker:
    """Central batch inference scheduler for the faster_whisper backend.

    Owns a single daemon thread that is the **only** thread touching the GPU
    model.  Per-session transcription threads submit ``BatchRequest`` objects
    and block on ``future.wait()`` instead of competing for
    ``SINGLE_MODEL_LOCK``.

    The worker loop:

    1. Blocks until the first request arrives from the queue.
    2. Waits up to ``batch_window_ms`` for additional requests (up to
       ``max_batch_size``).
    3. Processes the collected batch:
       - **batch_size == 1**: delegates to ``transcriber.transcribe()`` for
         identical behavior to the non-batched path.
       - **batch_size > 1**: runs a custom batched GPU path using
         CTranslate2's ``encode()`` + ``generate()`` APIs.

    Args:
        transcriber: The shared ``WhisperModel`` instance.
        max_batch_size: Maximum number of requests per batch.
        batch_window_ms: Maximum time (ms) to wait for the batch to fill
            after the first request arrives.
    """

    def __init__(
        self,
        transcriber,
        max_batch_size: int = 8,
        batch_window_ms: int = 50,
    ):
        self.transcriber = transcriber
        self.max_batch_size = max_batch_size
        self.batch_window_ms = batch_window_ms
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the background batch worker thread."""
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        logging.info(
            f"[BatchInference] Started (max_batch={self.max_batch_size}, "
            f"window={self.batch_window_ms}ms)"
        )

    def stop(self):
        """Signal the worker to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, request: BatchRequest):
        """Submit an inference request to the batch queue.

        Args:
            request: The ``BatchRequest`` to enqueue.  The caller should
                then call ``request.future.wait()`` to block until the
                result is ready.
        """
        self._queue.put(request)

    # -------------------------------------------------------------------------
    # Worker loop
    # -------------------------------------------------------------------------

    def _worker_loop(self):
        """Main loop: collect requests into batches and process them."""
        while not self._stop_event.is_set():
            batch: List[BatchRequest] = []

            # Block until first request arrives
            try:
                first = self._queue.get(timeout=0.5)
                batch.append(first)
            except queue.Empty:
                continue

            # Collect more requests within the batch window
            deadline = time.monotonic() + (self.batch_window_ms / 1000.0)
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                    batch.append(item)
                except queue.Empty:
                    break

            # Process the collected batch
            try:
                self._process_batch(batch)
            except Exception as e:
                logging.error(f"[BatchInference] Batch processing error: {e}")
                for req in batch:
                    if not req.future.is_set():
                        req.error = e
                        req.future.set()

    # -------------------------------------------------------------------------
    # Batch processing
    # -------------------------------------------------------------------------

    def _process_batch(self, batch: List[BatchRequest]):
        """Dispatch to single or multi-item processing."""
        if len(batch) == 1:
            self._process_single(batch[0])
            return

        logging.info(f"[BatchInference] Processing batch of {len(batch)}")
        self._process_multi(batch)

    def _process_single(self, req: BatchRequest):
        """Process a single request using standard ``transcriber.transcribe()``.

        This path is used when only one request is available in the batch
        window, ensuring identical behavior to the non-batched code path.
        """
        try:
            result, info = self.transcriber.transcribe(
                req.audio,
                language=req.language,
                task=req.task,
                initial_prompt=req.initial_prompt,
                vad_filter=req.use_vad,
                vad_parameters=req.vad_parameters if req.use_vad else None,
            )
            # Materialize the generator into a list
            req.result = list(result) if result is not None else []
            req.info = info
        except Exception as e:
            req.error = e
        finally:
            req.future.set()

    def _process_multi(self, batch: List[BatchRequest]):
        """Batched GPU path: encode + generate for multiple sessions at once.

        Pipeline:
            1. Per-item CPU preprocessing (VAD filtering + mel feature extraction)
            2. Batch GPU encode — single ``transcriber.encode()`` call
            3. Per-item prompt construction (handles different languages/tasks)
            4. Batch GPU generate — single ``transcriber.model.generate()`` call
            5. Per-item segment parsing and result dispatch
        """
        # Step 1: Per-item CPU preprocessing (VAD + feature extraction)
        preprocessed = []
        for req in batch:
            try:
                audio = req.audio
                speech_chunks = None

                if req.use_vad:
                    vad_params = req.vad_parameters or {}
                    vad_opts = VadOptions(**vad_params) if isinstance(vad_params, dict) else vad_params
                    speech_chunks = get_speech_timestamps(audio, vad_opts)
                    if speech_chunks:
                        audio_chunks, _ = collect_chunks(audio, speech_chunks)
                        audio = np.concatenate(audio_chunks, axis=0) if audio_chunks else audio

                if audio.shape[0] == 0:
                    # No speech detected — return empty result immediately
                    req.result = []
                    req.info = self._make_info(req, 0.0, 0.0)
                    req.future.set()
                    continue

                duration = audio.shape[0] / self.transcriber.feature_extractor.sampling_rate
                features = self.transcriber.feature_extractor(audio)
                features = pad_or_trim(features)  # -> [n_mels, 3000]
                preprocessed.append((req, features, audio, duration, speech_chunks))
            except Exception as e:
                req.error = e
                req.future.set()

        if not preprocessed:
            return

        try:
            # Step 2: Batch GPU encode
            feature_batch = np.stack([p[1] for p in preprocessed])  # [B, n_mels, 3000]
            encoder_output = self.transcriber.encode(feature_batch)

            # Step 3: Build per-item prompts (handles different languages/tasks)
            tokenizers_list = []
            prompts = []
            resolved_languages = []

            for i, (req, features, audio, duration, speech_chunks) in enumerate(preprocessed):
                lang = req.language
                # If language unknown, detect from encoder output
                if lang is None:
                    try:
                        lang_results = self.transcriber.model.detect_language(encoder_output)
                        if lang_results and len(lang_results) > i:
                            detected = lang_results[i]
                            if detected:
                                lang = detected[0][0].strip("<|>")
                    except Exception:
                        lang = "en"  # fallback

                resolved_languages.append(lang or "en")

                tokenizer = Tokenizer(
                    self.transcriber.hf_tokenizer,
                    self.transcriber.model.is_multilingual,
                    task=req.task,
                    language=lang or "en",
                )

                previous_tokens = []
                if req.initial_prompt:
                    previous_tokens = tokenizer.encode(" " + req.initial_prompt.strip())

                prompt = self.transcriber.get_prompt(
                    tokenizer,
                    previous_tokens=previous_tokens,
                    without_timestamps=False,
                )
                tokenizers_list.append(tokenizer)
                prompts.append(prompt)

            # Step 4: Batch GPU generate
            suppress_tokens = get_suppressed_tokens(tokenizers_list[0], [-1])

            results = self.transcriber.model.generate(
                encoder_output,
                prompts,
                beam_size=5,
                patience=1,
                length_penalty=1,
                max_length=self.transcriber.max_length,
                suppress_blank=True,
                suppress_tokens=suppress_tokens,
                return_scores=True,
                return_no_speech_prob=True,
                sampling_temperature=0.0,
                repetition_penalty=1,
                no_repeat_ngram_size=0,
            )

            # Step 5: Per-item segment parsing and result dispatch
            for i, (req, features, audio, duration, speech_chunks) in enumerate(preprocessed):
                try:
                    tokenizer = tokenizers_list[i]
                    gen_result = results[i]

                    tokens = gen_result.sequences_ids[0]
                    seq_len = len(tokens)
                    cum_logprob = gen_result.scores[0] * seq_len
                    avg_logprob = cum_logprob / (seq_len + 1) if seq_len > 0 else 0.0

                    segment_size = int(ceil(duration) * self.transcriber.frames_per_second)

                    subsegments, _, _ = self.transcriber._split_segments_by_timestamps(
                        tokenizer=tokenizer,
                        tokens=tokens,
                        time_offset=0,
                        segment_size=segment_size,
                        segment_duration=duration,
                        seek=0,
                    )

                    segments = []
                    for seg_idx, subseg in enumerate(subsegments):
                        text = tokenizer.decode(subseg["tokens"]).strip()
                        if not text:
                            continue
                        segments.append(Segment(
                            id=seg_idx,
                            seek=subseg.get("seek", 0),
                            start=subseg["start"],
                            end=subseg["end"],
                            text=text,
                            tokens=subseg["tokens"],
                            avg_logprob=avg_logprob,
                            compression_ratio=get_compression_ratio(text),
                            no_speech_prob=gen_result.no_speech_prob,
                            words=None,
                            temperature=0.0,
                        ))

                    req.result = segments
                    req.info = self._make_info(
                        req, duration, duration,
                        language=resolved_languages[i],
                    )
                except Exception as e:
                    req.error = e
                finally:
                    req.future.set()

        except Exception as e:
            logging.error(f"[BatchInference] GPU batch error: {e}")
            for req, *_ in preprocessed:
                if not req.future.is_set():
                    req.error = e
                    req.future.set()

    def _make_info(self, req, duration, duration_after_vad, language=None):
        """Build a ``TranscriptionInfo`` for the given request."""
        return TranscriptionInfo(
            language=language or req.language or "en",
            language_probability=1.0,
            duration=duration,
            duration_after_vad=duration_after_vad,
            all_language_probs=None,
            transcription_options=None,
            vad_options=None,
        )
