"""
MLX Whisper Backend for WhisperLive

This backend uses mlx-whisper for optimized transcription on Apple Silicon (M1/M2/M3).
MLX leverages Apple's Neural Engine and GPU for fast, efficient inference.
"""

import json
import logging
import threading
from typing import Optional

from whisper_live.transcriber.transcriber_mlx import WhisperMLX
from whisper_live.backend.base import ServeClientBase


class ServeClientMLXWhisper(ServeClientBase):
    """
    Backend implementation for MLX Whisper on Apple Silicon.

    This backend provides hardware-accelerated transcription using Apple's MLX
    framework, optimized for M1/M2/M3 chips with Neural Engine and GPU support.
    """

    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()

    def __init__(
        self,
        websocket,
        task: str = "transcribe",
        language: Optional[str] = None,
        client_uid: Optional[str] = None,
        model: str = "small.en",
        initial_prompt: Optional[str] = None,
        vad_parameters: Optional[dict] = None,
        use_vad: bool = True,
        single_model: bool = False,
        send_last_n_segments: int = 10,
        no_speech_thresh: float = 0.45,
        clip_audio: bool = False,
        same_output_threshold: int = 7,
        translation_queue=None,
    ):
        """
        Initialize MLX Whisper backend.

        Args:
            websocket: WebSocket connection to the client
            task (str): "transcribe" or "translate"
            language (str, optional): Language code (e.g., "en", "es")
            client_uid (str, optional): Unique client identifier
            model (str): Model size or HuggingFace repo ID
            initial_prompt (str, optional): Initial prompt for transcription
            vad_parameters (dict, optional): VAD parameters (not used in MLX)
            use_vad (bool): Whether to use VAD (not used in MLX)
            single_model (bool): Share one model across all clients
            send_last_n_segments (int): Number of recent segments to send
            no_speech_thresh (float): Threshold for filtering silent segments
            clip_audio (bool): Whether to clip audio with no valid segments
            same_output_threshold (int): Threshold for repeated output filtering
            translation_queue: Queue for translation (if enabled)
        """
        super().__init__(
            client_uid,
            websocket,
            send_last_n_segments,
            no_speech_thresh,
            clip_audio,
            same_output_threshold,
            translation_queue,
        )

        self.model_name = model
        self.language = "en" if model.endswith(".en") else language
        self.task = task
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters or {"onset": 0.5}
        self.use_vad = use_vad

        logging.info(f"Initializing MLX Whisper backend")
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Language: {self.language}")
        logging.info(f"Task: {self.task}")
        logging.info(f"Using Apple Neural Engine and GPU acceleration")

        # Initialize model
        try:
            if single_model:
                if ServeClientMLXWhisper.SINGLE_MODEL is None:
                    self.create_model()
                    ServeClientMLXWhisper.SINGLE_MODEL = self.transcriber
                else:
                    self.transcriber = ServeClientMLXWhisper.SINGLE_MODEL
                    logging.info("Using shared MLX model instance")
            else:
                self.create_model()
        except Exception as e:
            logging.error(f"Failed to load MLX Whisper model: {e}")
            self.websocket.send(
                json.dumps({
                    "uid": self.client_uid,
                    "status": "ERROR",
                    "message": f"Failed to load MLX Whisper model: {str(e)}. "
                               f"Make sure you're running on Apple Silicon (M1/M2/M3) "
                               f"and have mlx-whisper installed.",
                })
            )
            self.websocket.close()
            return

        # Start transcription thread
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()

        # Send ready message to client
        self.websocket.send(
            json.dumps({
                "uid": self.client_uid,
                "message": self.SERVER_READY,
                "backend": "mlx_whisper",
            })
        )
        logging.info(f"[MLX] Client {self.client_uid} initialized successfully")

    def create_model(self):
        """
        Initialize the MLX Whisper model.

        This method loads the specified model using MLX for hardware acceleration
        on Apple Silicon devices.
        """
        logging.info(f"Loading MLX Whisper model: {self.model_name}")
        self.transcriber = WhisperMLX(
            model_name=self.model_name,
            path_or_hf_repo=None,
        )
        logging.info("MLX Whisper model loaded successfully")

    def transcribe_audio(self, input_sample):
        """
        Transcribe audio using MLX Whisper.

        Args:
            input_sample (np.ndarray): Audio data as numpy array (16kHz)

        Returns:
            List[MLXSegment]: List of transcribed segments with timing information
        """
        # Acquire lock if using shared model
        if ServeClientMLXWhisper.SINGLE_MODEL:
            ServeClientMLXWhisper.SINGLE_MODEL_LOCK.acquire()

        try:
            result = self.transcriber.transcribe(
                input_sample,
                language=self.language,
                task=self.task,
                initial_prompt=self.initial_prompt,
                vad_filter=self.use_vad,
                vad_parameters=self.vad_parameters,
            )

            # Auto-detect language if not set
            if self.language is None and len(result) > 0:
                detected_lang, prob = self.transcriber.detect_language(input_sample)
                self.set_language(detected_lang)
                logging.info(f"Detected language: {detected_lang} (probability: {prob:.2f})")

            return result

        except Exception as e:
            logging.error(f"MLX transcription failed: {e}")
            raise

        finally:
            # Release lock if using shared model
            if ServeClientMLXWhisper.SINGLE_MODEL:
                ServeClientMLXWhisper.SINGLE_MODEL_LOCK.release()

    def handle_transcription_output(self, result, duration):
        """
        Process and send transcription results to the client.

        Args:
            result: Transcription result from transcribe_audio()
            duration (float): Duration of the audio chunk in seconds
        """
        segments = []

        if len(result):
            last_segment = self.update_segments(result, duration)
            segments = self.prepare_segments(last_segment)

        if len(segments):
            self.send_transcription_to_client(segments)

    def set_language(self, language: str):
        """
        Set the transcription language.

        Args:
            language (str): Language code (e.g., "en", "es", "fr")
        """
        self.language = language
        logging.info(f"Language set to: {language}")

    def cleanup(self):
        """
        Clean up resources when client disconnects.
        """
        super().cleanup()
        logging.info(f"[MLX] Client {self.client_uid} cleaned up")
