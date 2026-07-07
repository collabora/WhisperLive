import json
import logging
import threading

import numpy as np
from pywhispercpp.model import Model

from whisper_live.backend.base import ServeClientBase


class WhisperCppSegment:
    """Lightweight adapter that exposes a ``pywhispercpp`` segment with the
    attribute names WhisperLive's :class:`ServeClientBase` reads.

    ``pywhispercpp`` returns ``Segment`` objects with ``t0``/``t1`` timestamps in
    centiseconds (1/100 s) and a ``text`` field. WhisperLive's ``update_segments``
    (via ``get_segment_start`` / ``get_segment_end`` / ``get_segment_no_speech_prob``)
    expects ``start`` / ``end`` in seconds and a ``no_speech_prob``.
    """

    __slots__ = ("text", "start", "end", "no_speech_prob")

    def __init__(self, text, start, end, no_speech_prob=0.0):
        self.text = text
        self.start = start
        self.end = end
        self.no_speech_prob = no_speech_prob


class ServeClientWhisperCpp(ServeClientBase):
    """WhisperLive backend backed by whisper.cpp (ggml) through ``pywhispercpp``.

    Unlike the CTranslate2-based ``faster_whisper`` backend, whisper.cpp ships GPU
    backends that work on AMD (Vulkan / ROCm-HIP), Intel and Apple hardware, so
    this backend gives real GPU-accelerated transcription on those GPUs. Whether
    the GPU is actually used depends on how ``pywhispercpp`` was built
    (e.g. ``GGML_VULKAN=1 pip install pywhispercpp``); on a CPU-only build it
    transparently runs on the CPU.
    """

    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()

    # WhisperLive model aliases that do not match a ggml model name 1:1.
    MODEL_ALIASES = {"turbo": "large-v3-turbo"}

    def __init__(
        self,
        websocket,
        task="transcribe",
        device=None,
        language=None,
        client_uid=None,
        model="small.en",
        initial_prompt=None,
        vad_parameters=None,
        use_vad=True,
        single_model=False,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=10,
        cache_path="~/.cache/whisper-live/",
        translation_queue=None,
        diarization=None,
        word_timestamps=False,
        **kwargs,
    ):
        """
        Initialize a ServeClientWhisperCpp instance.

        The whisper.cpp model is loaded, a transcription thread is started, and a
        "SERVER_READY" message is sent to the client to indicate the server is ready.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            task (str, optional): The task type, e.g. "transcribe" or "translate". Defaults to "transcribe".
            device (str, optional): "cpu" forces CPU; anything else (or None) uses the GPU
                backend the pywhispercpp build was compiled with. Defaults to None.
            language (str, optional): Language code for transcription. Auto-detected when None
                (unless the model is English-only). Defaults to None.
            client_uid (str, optional): A unique identifier for the client. Defaults to None.
            model (str, optional): A ggml model name (e.g. "base.en", "large-v3-turbo") or a path
                to a local ggml ``.bin`` file. Defaults to "small.en".
            initial_prompt (str, optional): Prompt for whisper inference. Defaults to None.
            use_vad (bool, optional): Accepted for API parity; whisper.cpp's own VAD is not wired
                up here, so WhisperLive's buffering/clip logic handles silence. Defaults to True.
            single_model (bool, optional): Share a single model instance across clients. Defaults to False.
            send_last_n_segments (int, optional): Number of recent segments to send. Defaults to 10.
            no_speech_thresh (float, optional): No-speech probability threshold. Defaults to 0.45.
            clip_audio (bool, optional): Whether to clip audio with no valid segments. Defaults to False.
            same_output_threshold (int, optional): Repeated outputs before finalizing a segment. Defaults to 10.
        """
        super().__init__(
            client_uid,
            websocket,
            send_last_n_segments,
            no_speech_thresh,
            clip_audio,
            same_output_threshold,
            translation_queue,
            diarization,
            word_timestamps,
        )

        model = model or "small.en"
        self.model_size_or_path = self.MODEL_ALIASES.get(model, model)
        # English-only models only decode English; otherwise honor the requested language
        # (None => auto-detect on the first chunk).
        self.language = "en" if isinstance(model, str) and model.endswith("en") else language
        self.task = task or "transcribe"
        self.initial_prompt = initial_prompt
        self.use_vad = use_vad
        self.cache_path = cache_path

        # Use the GPU backend that pywhispercpp was built with unless CPU is explicitly requested.
        self.use_gpu = not (isinstance(device, str) and device.lower() == "cpu")

        try:
            if single_model and ServeClientWhisperCpp.SINGLE_MODEL is not None:
                self.transcriber = ServeClientWhisperCpp.SINGLE_MODEL
            else:
                self.create_model()
                if single_model:
                    ServeClientWhisperCpp.SINGLE_MODEL = self.transcriber
        except Exception as e:
            logging.error(f"Failed to load whisper.cpp model: {e}")
            self.websocket.send(json.dumps({
                "uid": self.client_uid,
                "status": "ERROR",
                "message": f"Failed to load model: {str(self.model_size_or_path)}"
            }))
            self.websocket.close()
            return

        # threading
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()
        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.SERVER_READY,
            "backend": "whisper_cpp"
        }))
        logging.info(
            f"Running whisper.cpp backend (model={self.model_size_or_path}, use_gpu={self.use_gpu}, "
            f"task={self.task}, language={self.language})"
        )

    def create_model(self):
        """Instantiate the pywhispercpp model and set it as the transcriber.

        ``model`` may be a built-in ggml model name (downloaded on first use) or a
        path to a local ``.bin`` file. Leaving ``redirect_whispercpp_logs_to`` at its
        default lets whisper.cpp print its backend/device line (e.g. the Vulkan or
        ROCm device) once at load time.
        """
        logging.info(f"Loading whisper.cpp model: {self.model_size_or_path}")
        self.transcriber = Model(
            self.model_size_or_path,
            context_params={"use_gpu": self.use_gpu},
            print_progress=False,
            print_realtime=False,
        )

    def detect_language(self, input_sample):
        """Detect and set the transcription language from an audio chunk."""
        try:
            (language, prob), _ = self.transcriber.auto_detect_language(input_sample)
        except Exception as e:
            logging.warning(f"whisper.cpp language detection failed ({e}); defaulting to 'en'")
            language, prob = "en", 0.0
        if language:
            self.language = language
            logging.info(f"Detected language {language} with probability {prob}")
            try:
                self.websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "language": self.language,
                    "language_prob": float(prob),
                }))
            except Exception as e:
                logging.error(f"[ERROR]: Sending language to client: {e}")

    def transcribe_audio(self, input_sample):
        """
        Transcribe an audio sample with whisper.cpp and adapt the segments.

        Args:
            input_sample (np.ndarray): float32 mono audio at 16 kHz.

        Returns:
            list[WhisperCppSegment]: segments with the fields WhisperLive expects.
        """
        input_sample = np.ascontiguousarray(input_sample, dtype=np.float32)

        if ServeClientWhisperCpp.SINGLE_MODEL:
            ServeClientWhisperCpp.SINGLE_MODEL_LOCK.acquire()
        try:
            if self.language is None:
                self.detect_language(input_sample)

            params = {
                "language": self.language or "",
                "translate": self.task == "translate",
                "print_progress": False,
                "print_realtime": False,
            }
            if self.initial_prompt is not None:
                params["initial_prompt"] = self.initial_prompt

            segments = self.transcriber.transcribe(input_sample, **params)
        finally:
            if ServeClientWhisperCpp.SINGLE_MODEL:
                ServeClientWhisperCpp.SINGLE_MODEL_LOCK.release()

        return self._to_whisperlive_segments(segments)

    @staticmethod
    def _to_whisperlive_segments(segments):
        """Convert pywhispercpp ``Segment`` objects (t0/t1 in centiseconds) to
        objects exposing ``start``/``end`` in seconds and a ``no_speech_prob``."""
        adapted = []
        for seg in segments:
            text = seg.text
            if not text:
                continue
            # faster_whisper segments carry a leading space; match that so the
            # streaming concatenation in base.update_segments keeps words separated.
            if not text.startswith(" "):
                text = " " + text
            adapted.append(WhisperCppSegment(
                text=text,
                start=seg.t0 / 100.0,
                end=seg.t1 / 100.0,
                no_speech_prob=0.0,
            ))
        return adapted

    def handle_transcription_output(self, result, duration):
        """
        Handle the transcription output, updating the transcript and sending data to the client.

        Args:
            result (list): The list of segments from whisper.cpp inference.
            duration (float): Duration of the transcribed audio chunk.
        """
        segments = []
        if len(result):
            self.t_start = None
            last_segment = self.update_segments(result, duration)
            segments = self.prepare_segments(last_segment)

        if len(segments):
            self.send_transcription_to_client(segments)
