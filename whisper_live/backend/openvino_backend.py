import json
import logging
import threading
import time
import numpy as np
import soundfile as sf
from pathlib import Path

from openvino import Core
from whisper_live.backend.base import ServeClientBase
from whisper_live.transcriber.transcriber_openvino import WhisperOpenVINO


class ServeClientOpenVINO(ServeClientBase):
    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()

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
        cpu_threads=0,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=2,
        cache_path=None,
    ):
        """
        Initialize a ServeClient instance.
        The Whisper model is initialized based on the client's language and device availability.
        The transcription thread is started upon initialization. A "SERVER_READY" message is sent
        to the client to indicate that the server is ready.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            task (str, optional): The task type, e.g., "transcribe." Defaults to "transcribe".
            device (str, optional): The device type for Whisper, "cuda" or "cpu". Defaults to None.
            language (str, optional): The language for transcription. Defaults to None.
            client_uid (str, optional): A unique identifier for the client. Defaults to None.
            model (str, optional): Huggingface model_id for a valid OpenVINO model.
            initial_prompt (str, optional): Prompt for whisper inference. Defaults to None.
            single_model (bool, optional): Whether to instantiate a new model for each client connection. Defaults to False.
            cpu_threads (int, optional): Number of CPU threads for OpenVINO inference. 0 means auto. Defaults to 0.
            send_last_n_segments (int, optional): Number of most recent segments to send to the client. Defaults to 10.
            no_speech_thresh (float, optional): Segments with no speech probability above this threshold will be discarded. Defaults to 0.45.
            clip_audio (bool, optional): Whether to clip audio with no valid segments. Defaults to False.
            same_output_threshold (int, optional): Number of repeated outputs before considering it as a valid segment. Defaults to 3 (optimized for OpenVINO's fast inference).
            cache_path (str, optional): Path to OpenVINO model cache directory. Defaults to None.
        """
        super().__init__(
            client_uid,
            websocket,
            send_last_n_segments,
            no_speech_thresh,
            clip_audio,
            same_output_threshold,
        )
        # Handle language: None for auto-detection, otherwise format for OpenVINO
        if language is None:
            self.language = None  # Auto-detect language
        else:
            self.language = language if language.startswith("<|") else f"<|{language}|>"

        self.task = "transcribe" if task is None else task
        self.initial_prompt = initial_prompt

        # Store VAD parameters for potential future use
        self.use_vad = use_vad
        self.vad_parameters = vad_parameters
        self.cpu_threads = cpu_threads
        self.cache_path = cache_path

        self.clip_audio = True

        core = Core()
        available_devices = core.available_devices
        if 'GPU' in available_devices:
            selected_device = 'GPU'
        else:
            gpu_devices = [d for d in available_devices if d.startswith('GPU')]
            selected_device = gpu_devices[0] if gpu_devices else 'CPU'
        self.device = selected_device


        if single_model:
            if ServeClientOpenVINO.SINGLE_MODEL is None:
                self.create_model(model)
                ServeClientOpenVINO.SINGLE_MODEL = self.transcriber
            else:
                self.transcriber = ServeClientOpenVINO.SINGLE_MODEL
        else:
            self.create_model(model)

        # threading
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()

        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.SERVER_READY,
            "backend": "openvino"
        }))

        logging.info(f"Using OpenVINO device: {self.device}")
        logging.info(f"Running OpenVINO backend with language: {self.language} and task: {self.task}")

    def create_model(self, model_id):
        """
        Instantiates a new model, sets it as the transcriber and performs warmup.
        """
        self.transcriber = WhisperOpenVINO(
            model_id,
            device=self.device,
            language=self.language,
            task=self.task,
            initial_prompt=self.initial_prompt,
            cpu_threads=self.cpu_threads,
            cache_path=self.cache_path
        )
        # Perform warmup to trigger model compilation before first real inference
        # self.warmup()

    def warmup(self, warmup_steps=3):
        """
        Warmup OpenVINO WhisperPipeline to trigger model compilation.

        The first inference with OpenVINO includes model compilation overhead,
        which can take 2-5 seconds. This warmup eliminates that delay from the
        first real transcription by triggering compilation during initialization.

        Args:
            warmup_steps (int): Number of warmup inferences to perform. Default: 3.
        """
        logging.info("[OpenVINO] Warming up pipeline...")

        # Load real audio sample for warmup: 3-second JFK speech excerpt
        audio_path = Path(__file__).resolve().parents[2] / "assets" / "jfk_3s.flac"

        try:
            warmup_audio, sample_rate = sf.read(audio_path)
            # Convert to mono if stereo
            if len(warmup_audio.shape) > 1:
                warmup_audio = warmup_audio.mean(axis=1)
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                from scipy.signal import resample
                warmup_audio = resample(warmup_audio, int(len(warmup_audio) * 16000 / sample_rate))
            warmup_audio = warmup_audio.astype(np.float32)
            logging.info(f"[OpenVINO] Using real audio sample for warmup: {audio_path}")
        except Exception as e:
            logging.warning(f"[OpenVINO] Failed to load warmup audio, using silence: {e}")
            warmup_audio = np.zeros(16000, dtype=np.float32)

        for i in range(warmup_steps):
            try:
                start_time = time.time()
                _ = self.transcriber.transcribe(warmup_audio)
                duration = time.time() - start_time
                logging.info(f"[OpenVINO] Warmup step {i+1}/{warmup_steps} completed in {duration:.2f}s")
            except Exception as e:
                logging.warning(f"[OpenVINO] Warmup step {i+1} failed: {e}")

        logging.info("[OpenVINO] Warmup complete. Model compiled and ready for inference.")

    def speech_to_text(self):
        """
        Process an audio stream in an infinite loop, continuously transcribing the speech.

        Optimized version for OpenVINO backend with reduced latency:
        - Lower buffer threshold (0.5s instead of 1.0s)
        - Faster polling intervals (0.02-0.05s instead of 0.1-0.25s)
        - Similar to TensorRT backend optimizations

        This method continuously receives audio frames, performs real-time transcription,
        and sends transcribed segments to the client via a WebSocket connection.
        """
        while True:
            if self.exit:
                logging.info("[OpenVINO] Exiting speech to text thread")
                break

            if self.frames_np is None:
                time.sleep(0.02)  # Faster polling: 20ms instead of no wait
                continue

            if self.clip_audio:
                self.clip_audio_if_no_valid_segment()

            input_bytes, duration = self.get_audio_chunk_for_processing()

            # Optimized: 0.3s threshold (vs 1.0s in base, 0.4s in TensorRT)
            if duration < 0.3:
                time.sleep(0.03)  # Reduced from 0.1s to 0.01s
                continue

            try:
                input_sample = input_bytes.copy()
                logging.debug(f"[OpenVINO] Processing audio with duration: {duration:.2f}s")
                result = self.transcribe_audio(input_sample)

                # Handle cases where VAD filtered all audio
                if result is None or self.language is None:
                    self.timestamp_offset += duration
                    time.sleep(0.03)  # Reduced from 0.25s to 0.03s
                    continue

                self.handle_transcription_output(result, duration)

            except Exception as e:
                logging.error(f"[OpenVINO ERROR]: Failed to transcribe audio chunk: {e}")
                time.sleep(0.01)

    def transcribe_audio(self, input_sample):
        """
        Transcribes the provided audio sample using the configured transcriber instance.

        If the language has not been set, it updates the session's language based on the transcription
        information.

        Args:
            input_sample (np.array): The audio chunk to be transcribed. This should be a NumPy
                                    array representing the audio data.

        Returns:
            The transcription result from the transcriber. The exact format of this result
            depends on the implementation of the `transcriber.transcribe` method but typically
            includes the transcribed text.
        """
        if ServeClientOpenVINO.SINGLE_MODEL:
            ServeClientOpenVINO.SINGLE_MODEL_LOCK.acquire()
        result = self.transcriber.transcribe(input_sample)
        if ServeClientOpenVINO.SINGLE_MODEL:
            ServeClientOpenVINO.SINGLE_MODEL_LOCK.release()
        return result

    def handle_transcription_output(self, result, duration):
        """
        Handle the transcription output, updating the transcript and sending data to the client.

        Args:
            result (str): The result from whisper inference i.e. the list of segments.
            duration (float): Duration of the transcribed audio chunk.
        """
        segments = []
        if len(result):
            self.t_start = None
            last_segment = self.update_segments(result, duration)
            segments = self.prepare_segments(last_segment)

        if len(segments):
            self.send_transcription_to_client(segments)
