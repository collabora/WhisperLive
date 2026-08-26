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

from whisper_live import metrics as wl_metrics
from whisper_live.transcriber.transcriber_faster_whisper import (
    Segment,
    TranscriptionInfo,
    get_compression_ratio,
    get_suppressed_tokens,
    restore_speech_timestamps,
)


FALLBACK_TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


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
        hotwords: Optional phrases to bias decoding toward.
        word_timestamps: Whether to compute per-word timings; such requests
            take the single path, the batched path has no word alignment.
        future: Event signaled when the result is ready.
        result: List of ``Segment`` objects (filled by worker).
        info: ``TranscriptionInfo`` metadata (filled by worker).
        error: Exception instance if processing failed.
        abandoned: Set by the session thread when it stopped waiting; the
            worker skips such requests instead of spending GPU time on them.
        features: Mel spectrogram, filled by ``prepare_features``.
        speech_chunks: VAD speech timestamps, filled by ``prepare_features``.
        speech_duration: Seconds of audio left after VAD, filled by
            ``prepare_features``.  ``None`` means the request has not been
            preprocessed yet, ``0.0`` means VAD found no speech.
        submitted_at: ``time.monotonic()`` stamp set by ``submit()``.
    """
    audio: np.ndarray
    language: Optional[str] = None
    task: str = "transcribe"
    initial_prompt: Optional[str] = None
    use_vad: bool = True
    vad_parameters: Optional[Dict] = None
    word_timestamps: bool = False
    hotwords: Optional[str] = None
    client_uid: Optional[str] = None
    # Signaling
    future: threading.Event = field(default_factory=threading.Event)
    # Preprocessing (filled by prepare_features, in the session thread or the worker)
    features: Optional[np.ndarray] = None
    speech_chunks: Optional[list] = None
    speech_duration: Optional[float] = None
    submitted_at: float = 0.0
    # Results (filled by batch worker)
    result: Optional[Any] = None
    info: Optional[Any] = None
    error: Optional[Exception] = None
    abandoned: bool = False


def prepare_features(transcriber, request: BatchRequest):
    """Run VAD and mel extraction for one request and store them on it.

    Called from the session thread so the batch worker only does GPU work.
    ``feature_extractor`` is stateless numpy and the silero VAD runs through
    one thread safe onnxruntime session, so parallel calls are fine.
    """
    sampling_rate = transcriber.feature_extractor.sampling_rate
    audio = request.audio
    speech_chunks = None

    if request.use_vad:
        vad_params = request.vad_parameters or {}
        vad_opts = VadOptions(**vad_params) if isinstance(vad_params, dict) else vad_params
        speech_chunks = get_speech_timestamps(audio, vad_opts)
        if speech_chunks:
            audio_chunks, _ = collect_chunks(audio, speech_chunks)
            audio = np.concatenate(audio_chunks, axis=0) if audio_chunks else audio

    request.speech_chunks = speech_chunks

    if audio.shape[0] == 0:
        request.speech_duration = 0.0
        return

    request.speech_duration = audio.shape[0] / sampling_rate
    features = transcriber.feature_extractor(audio)
    request.features = pad_or_trim(features)  # -> [n_mels, 3000]


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
        beam_size: Beam width for the first (temperature 0) decode of every
            item on both paths. Fallback decodes at higher temperatures sample
            with a beam of 1.
        temperature_fallback: Re-decode items that fail the compression ratio
            or log probability check at rising temperatures, like
            ``transcriber.transcribe``. Off keeps the temperature 0 result.
        max_queue_wait_s: Measured queue wait above which ``overloaded()``
            reports True so the server can turn new clients away.
    """

    QUEUE_WAIT_EMA_ALPHA = 0.2

    def __init__(
        self,
        transcriber,
        max_batch_size: int = 16,
        batch_window_ms: int = 50,
        max_queue_wait_s: float = 2.0,
        beam_size: int = 5,
        temperature_fallback: bool = True,
    ):
        self.transcriber = transcriber
        self.beam_size = beam_size
        self.temperature_fallback = temperature_fallback
        self.max_batch_size = max_batch_size
        self.batch_window_ms = batch_window_ms
        self.max_queue_wait_s = max_queue_wait_s
        self.queue_wait_s = 0.0
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
        request.submitted_at = time.monotonic()
        self._queue.put(request)

    def overloaded(self) -> bool:
        """Whether requests are waiting longer in the queue than allowed."""
        return self.queue_wait_s > self.max_queue_wait_s

    def _record_queue_wait(self, request: BatchRequest):
        waited = time.monotonic() - request.submitted_at
        self.queue_wait_s = (
            self.QUEUE_WAIT_EMA_ALPHA * waited
            + (1 - self.QUEUE_WAIT_EMA_ALPHA) * self.queue_wait_s
        )

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
            except queue.Empty:
                self.queue_wait_s = 0.0
                continue
            self._record_queue_wait(first)
            batch.append(first)

            # Collect more requests within the batch window
            deadline = time.monotonic() + (self.batch_window_ms / 1000.0)
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                self._record_queue_wait(item)
                batch.append(item)

            batch = [req for req in batch if not req.abandoned]
            if not batch:
                continue

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
        """Dispatch to single or multi-item processing.

        The multi-item path encodes a single 30s mel window per item and has
        no word alignment, so audio longer than 30s and requests wanting word
        timestamps go through ``transcriber.transcribe`` one at a time.
        """
        if len(batch) == 1:
            self._process_single(batch[0])
            return

        window_samples = 30 * self.transcriber.feature_extractor.sampling_rate

        def needs_single_path(req):
            return req.audio.shape[0] > window_samples or req.word_timestamps

        single_items = [r for r in batch if needs_single_path(r)]
        short_items = [r for r in batch if not needs_single_path(r)]

        for req in single_items:
            self._process_single(req)

        if len(short_items) == 1:
            self._process_single(short_items[0])
        elif short_items:
            logging.info(f"[BatchInference] Processing batch of {len(short_items)}")
            self._process_multi(short_items)

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
                hotwords=req.hotwords,
                word_timestamps=req.word_timestamps,
                beam_size=self.beam_size,
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
            1. Per-item CPU preprocessing, unless the session thread already
               ran ``prepare_features``
            2. Batch GPU encode — single ``transcriber.encode()`` call
            3. Per-item prompt construction (handles different languages/tasks)
            4. Batch GPU generate — single ``transcriber.model.generate()`` call
            5. Per-item segment parsing and result dispatch
        """
        # Step 1: Per-item CPU preprocessing (VAD + feature extraction)
        sampling_rate = self.transcriber.feature_extractor.sampling_rate
        preprocessed = []
        for req in batch:
            try:
                if req.speech_duration is None:
                    prepare_features(self.transcriber, req)

                if req.speech_duration == 0:
                    # No speech detected — return empty result immediately
                    req.result = []
                    req.info = self._make_info(req, 0.0, 0.0)
                    req.future.set()
                    continue

                full_duration = req.audio.shape[0] / sampling_rate
                preprocessed.append(
                    (req, req.features, req.speech_duration, full_duration, req.speech_chunks)
                )
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

            for i, (req, *_) in enumerate(preprocessed):
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
                    hotwords=req.hotwords,
                )
                tokenizers_list.append(tokenizer)
                prompts.append(prompt)

            # Step 4: Batch GPU generate with per-item temperature fallback.
            # Mirrors faster_whisper.transcribe()'s fallback loop. Items that
            # pass quality thresholds at lower temperature keep their result;
            # only failed items are re-decoded at the next temperature.
            suppress_tokens = get_suppressed_tokens(tokenizers_list[0], [-1])

            temperatures = FALLBACK_TEMPERATURES if self.temperature_fallback else [0.0]
            comp_thresh = 2.4
            logprob_thresh = -1.0
            no_speech_thresh = 0.6

            n = len(preprocessed)
            final_results = [None] * n  # tuples of (gen_result, avg_logprob, used_temp, is_silence)
            pending_indices = list(range(n))

            for temp in temperatures:
                if not pending_indices:
                    break

                if len(pending_indices) == n:
                    sub_encoder = encoder_output
                else:
                    # Re-encode features for just the pending items to get
                    # an encoder_output of the right batch dimension.
                    sub_feature_batch = np.stack(
                        [preprocessed[i][1] for i in pending_indices]
                    )
                    sub_encoder = self.transcriber.encode(sub_feature_batch)
                sub_prompts = [prompts[i] for i in pending_indices]

                gen_kwargs = dict(
                    beam_size=self.beam_size if temp == 0.0 else 1,
                    patience=1,
                    length_penalty=1,
                    max_length=self.transcriber.max_length,
                    suppress_blank=True,
                    suppress_tokens=suppress_tokens,
                    return_scores=True,
                    return_no_speech_prob=True,
                    sampling_temperature=temp,
                    repetition_penalty=1,
                    no_repeat_ngram_size=0,
                )
                batch_results = self.transcriber.model.generate(
                    sub_encoder, sub_prompts, **gen_kwargs
                )

                next_pending = []
                for j, idx in enumerate(pending_indices):
                    gen_result = batch_results[j]
                    tokens = gen_result.sequences_ids[0]
                    seq_len = len(tokens)
                    cum_logprob = gen_result.scores[0] * seq_len
                    avg_logprob = cum_logprob / (seq_len + 1) if seq_len > 0 else 0.0
                    raw_text = tokenizers_list[idx].decode(tokens).strip()
                    comp_ratio = get_compression_ratio(raw_text) if raw_text else 0.0

                    bad = (
                        comp_ratio > comp_thresh
                        or avg_logprob < logprob_thresh
                    )
                    # High no_speech + low logprob -> treat as silence, accept empty.
                    is_silence = (
                        gen_result.no_speech_prob > no_speech_thresh
                        and avg_logprob < logprob_thresh
                    )

                    if not bad or is_silence or temp == temperatures[-1]:
                        final_results[idx] = (gen_result, avg_logprob, temp, is_silence)
                    else:
                        next_pending.append(idx)

                if next_pending:
                    wl_metrics.track_batch_fallback(len(next_pending))
                pending_indices = next_pending

            # Step 5: Per-item segment parsing and result dispatch
            for i, (req, features, duration, full_duration, speech_chunks) in enumerate(preprocessed):
                try:
                    tokenizer = tokenizers_list[i]
                    gen_result, avg_logprob, used_temp, is_silence = final_results[i]

                    tokens = [] if is_silence else gen_result.sequences_ids[0]
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
                            temperature=used_temp,
                        ))

                    if speech_chunks:
                        segments = list(restore_speech_timestamps(segments, speech_chunks, sampling_rate))

                    req.result = segments
                    req.info = self._make_info(
                        req, full_duration, duration,
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
