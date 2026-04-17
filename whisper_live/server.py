import os
import time
import threading
import collections
import queue
import json
import functools
import logging
import signal
import shutil
import tempfile
from typing import Optional, List
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import PlainTextResponse, JSONResponse, StreamingResponse
import uvicorn
from faster_whisper import WhisperModel
import torch

from enum import Enum

from whisper_live import metrics as wl_metrics
from typing import List, Optional
import numpy as np
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed
from whisper_live.vad import VoiceActivityDetector
from whisper_live.backend.base import ServeClientBase

logging.basicConfig(level=logging.INFO)


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production deployments."""

    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Attach extra fields if present
        for attr in ("request_id", "client_uid", "client_ip", "endpoint"):
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)
        return json.dumps(log_entry)


def configure_logging(json_logs: bool = False, level: int = logging.INFO):
    """Configure root logger. Use json_logs=True for production/CloudWatch."""
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        ))
    root.handlers.clear()
    root.addHandler(handler)

class ClientManager:
    def __init__(self, max_clients=4, max_connection_time=600):
        """
        Initializes the ClientManager with specified limits on client connections and connection durations.

        Args:
            max_clients (int, optional): The maximum number of simultaneous client connections allowed. Defaults to 4.
            max_connection_time (int, optional): The maximum duration (in seconds) a client can stay connected. Defaults
                                                 to 600 seconds (10 minutes).
        """
        self.clients = {}
        self.start_times = {}
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time
        self.lock = threading.Lock()

    def add_client(self, websocket, client):
        """
        Adds a client and their connection start time to the tracking dictionaries.

        Args:
            websocket: The websocket associated with the client to add.
            client: The client object to be added and tracked.
        """
        with self.lock:
            self.clients[websocket] = client
            self.start_times[websocket] = time.time()

    def get_client(self, websocket):
        """
        Retrieves a client associated with the given websocket.

        Args:
            websocket: The websocket associated with the client to retrieve.

        Returns:
            The client object if found, False otherwise.
        """
        with self.lock:
            if websocket in self.clients:
                return self.clients[websocket]
            return False

    def remove_client(self, websocket):
        """
        Removes a client and their connection start time from the tracking dictionaries. Performs cleanup on the
        client if necessary.

        Args:
            websocket: The websocket associated with the client to be removed.
        """
        with self.lock:
            client = self.clients.pop(websocket, None)
            self.start_times.pop(websocket, None)
        if client:
            client.cleanup()

    def get_wait_time(self):
        """
        Calculates the estimated wait time for new clients based on the remaining connection times of current clients.

        Returns:
            The estimated wait time in minutes for new clients to connect. Returns 0 if there are available slots.
        """
        with self.lock:
            wait_time = None
            for start_time in self.start_times.values():
                current_client_time_remaining = self.max_connection_time - (time.time() - start_time)
                if wait_time is None or current_client_time_remaining < wait_time:
                    wait_time = current_client_time_remaining
        return wait_time / 60 if wait_time is not None else 0

    def is_server_full(self, websocket, options):
        """
        Checks if the server is at its maximum client capacity and sends a wait message to the client if necessary.

        Args:
            websocket: The websocket of the client attempting to connect.
            options: A dictionary of options that may include the client's unique identifier.

        Returns:
            True if the server is full, False otherwise.
        """
        with self.lock:
            if len(self.clients) >= self.max_clients:
                wait_time = None
                for start_time in self.start_times.values():
                    remaining = self.max_connection_time - (time.time() - start_time)
                    if wait_time is None or remaining < wait_time:
                        wait_time = remaining
                wait_time_minutes = wait_time / 60 if wait_time is not None else 0
                response = {"uid": options["uid"], "status": "WAIT", "message": wait_time_minutes}
                websocket.send(json.dumps(response))
                return True
            return False

    def is_client_timeout(self, websocket):
        """
        Checks if a client has exceeded the maximum allowed connection time and disconnects them if so, issuing a warning.

        Args:
            websocket: The websocket associated with the client to check.

        Returns:
            True if the client's connection time has exceeded the maximum limit, False otherwise.
        """
        with self.lock:
            elapsed_time = time.time() - self.start_times[websocket]
            client = self.clients.get(websocket)
        if elapsed_time >= self.max_connection_time and client:
            client.disconnect()
            logging.warning(f"Client with uid '{client.client_uid}' disconnected due to overtime.")
            return True
        return False


class BackendType(Enum):
    FASTER_WHISPER = "faster_whisper"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"

    @staticmethod
    def valid_types() -> List[str]:
        return [backend_type.value for backend_type in BackendType]

    @staticmethod
    def is_valid(backend: str) -> bool:
        return backend in BackendType.valid_types()

    def is_faster_whisper(self) -> bool:
        return self == BackendType.FASTER_WHISPER

    def is_tensorrt(self) -> bool:
        return self == BackendType.TENSORRT
    
    def is_openvino(self) -> bool:
        return self == BackendType.OPENVINO


class TranscriptionServer:
    RATE = 16000

    def __init__(self):
        self.client_manager = None
        self.no_voice_activity_chunks = 0
        self.use_vad = True
        self.single_model = False
        self.batch_config = None
        self.raw_pcm_input = False
        self.plugin_registry = None
        self._model_cache = None
        self._noise_reducer = None
        self._shutting_down = False
        self._ws_server = None

    def initialize_client(
        self, websocket, options, faster_whisper_custom_model_path,
        whisper_tensorrt_path, trt_multilingual, trt_py_session=False,
    ):
        client: Optional[ServeClientBase] = None

        # Check if client wants translation
        enable_translation = options.get("enable_translation", False)
        
        # Create translation queue if translation is enabled
        translation_queue = None
        translation_client = None
        translation_thread = None
        
        if enable_translation:
            target_language = options.get("target_language", "fr")
            translation_queue = queue.Queue(maxsize=ServeClientBase.MAX_TRANSLATION_QUEUE_SIZE)
            from whisper_live.backend.translation_backend import ServeClientTranslation
            translation_client = ServeClientTranslation(
                client_uid=options["uid"],
                websocket=websocket,
                translation_queue=translation_queue,
                target_language=target_language,
                send_last_n_segments=options.get("send_last_n_segments", 10)
            )
            
            # Start translation thread
            translation_thread = threading.Thread(
                target=translation_client.speech_to_text,
                daemon=True
            )
            translation_thread.start()
            
            logging.info(f"Translation enabled for client {options['uid']} with target language: {target_language}")

        if self.backend.is_tensorrt():
            try:
                from whisper_live.backend.trt_backend import ServeClientTensorRT
                client = ServeClientTensorRT(
                    websocket,
                    multilingual=trt_multilingual,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=whisper_tensorrt_path,
                    single_model=self.single_model,
                    use_py_session=trt_py_session,
                    send_last_n_segments=options.get("send_last_n_segments", 10),
                    no_speech_thresh=options.get("no_speech_thresh", 0.45),
                    clip_audio=options.get("clip_audio", False),
                    same_output_threshold=options.get("same_output_threshold", 10),
                )
                logging.info("Running TensorRT backend.")
            except Exception as e:
                logging.error(f"TensorRT-LLM not supported: {e}")
                self.client_uid = options["uid"]
                websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "status": "WARNING",
                    "message": "TensorRT-LLM not supported on Server yet. "
                               "Reverting to available backend: 'faster_whisper'"
                }))
                self.backend = BackendType.FASTER_WHISPER
        
        if self.backend.is_openvino():
            try:
                from whisper_live.backend.openvino_backend import ServeClientOpenVINO
                client = ServeClientOpenVINO(
                    websocket,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=options["model"],
                    single_model=self.single_model,
                    send_last_n_segments=options.get("send_last_n_segments", 10),
                    no_speech_thresh=options.get("no_speech_thresh", 0.45),
                    clip_audio=options.get("clip_audio", False),
                    same_output_threshold=options.get("same_output_threshold", 10),
                )
                logging.info("Running OpenVINO backend.")
            except Exception as e:
                logging.error(f"OpenVINO not supported: {e}")
                self.backend = BackendType.FASTER_WHISPER
                self.client_uid = options["uid"]
                websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "status": "WARNING",
                    "message": "OpenVINO not supported on Server yet. "
                                "Reverting to available backend: 'faster_whisper'"
                }))

        try:
            if self.backend.is_faster_whisper():
                from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper
                # model is of the form namespace/repo_name and not a filesystem path
                if faster_whisper_custom_model_path is not None:
                    logging.info(f"Using custom model {faster_whisper_custom_model_path}")
                    options["model"] = faster_whisper_custom_model_path
                client = ServeClientFasterWhisper(
                    websocket,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=options["model"],
                    initial_prompt=options.get("initial_prompt"),
                    vad_parameters=options.get("vad_parameters"),
                    use_vad=self.use_vad,
                    single_model=self.single_model,
                    send_last_n_segments=options.get("send_last_n_segments", 10),
                    no_speech_thresh=options.get("no_speech_thresh", 0.45),
                    clip_audio=options.get("clip_audio", False),
                    same_output_threshold=options.get("same_output_threshold", 10),
                    cache_path=self.cache_path,
                    translation_queue=translation_queue,
                    word_timestamps=options.get("word_timestamps", False),
                    hotwords=options.get("hotwords"),
                    diarization=self._create_diarizer(options),
                    smart_formatting=options.get("smart_formatting", False),
                    pii_redaction=self._parse_pii_option(options),
                    profanity_filter=self._parse_profanity_option(options),
                )

                logging.info("Running faster_whisper backend.")

                if self.plugin_registry:
                    client.plugin_registry = self.plugin_registry

                # Start batch inference worker on first client (after model is loaded)
                if (self.batch_config is not None
                        and ServeClientFasterWhisper.BATCH_WORKER is None
                        and ServeClientFasterWhisper.SINGLE_MODEL is not None):
                    from whisper_live.batch_inference import BatchInferenceWorker
                    worker = BatchInferenceWorker(
                        transcriber=ServeClientFasterWhisper.SINGLE_MODEL,
                        **self.batch_config,
                    )
                    worker.start()
                    ServeClientFasterWhisper.BATCH_WORKER = worker
        except Exception as e:
            logging.error(e)
            return

        if client is None:
            raise ValueError(f"Backend type {self.backend.value} not recognised or not handled.")

        if translation_client:
            client.translation_client = translation_client
            client.translation_thread = translation_thread

        self.client_manager.add_client(websocket, client)

    def _create_diarizer(self, options):
        """Create a SpeakerDiarizer if the client requested diarization.

        Returns:
            SpeakerDiarizer or None
        """
        if not options.get("enable_diarization", False):
            return None
        try:
            from whisper_live.diarization import SpeakerDiarizer
            diarizer = SpeakerDiarizer(
                similarity_threshold=options.get("diarization_threshold", 0.55),
                max_speakers=options.get("max_speakers", 10),
                hf_token=options.get("hf_token"),
            )
            # Enroll known speakers if provided
            known_refs = options.get("known_speaker_refs")
            if known_refs and isinstance(known_refs, dict):
                diarizer.enroll_speakers_from_files(known_refs)
            return diarizer
        except ImportError:
            logging.warning("pyannote.audio not installed; diarization disabled")
            return None

    def _parse_pii_option(self, options):
        """Parse PII redaction option from client handshake.

        Returns:
            set of PII type strings, or None if disabled.
        """
        pii = options.get("pii_redaction")
        if not pii:
            return None
        if pii is True or pii == "all":
            from whisper_live.pii_redaction import ALL_PII_TYPES
            return ALL_PII_TYPES
        if isinstance(pii, list):
            return set(pii)
        if isinstance(pii, str):
            return {t.strip() for t in pii.split(",") if t.strip()}
        return None

    def _parse_profanity_option(self, options):
        """Parse profanity filter option from client handshake.

        Returns:
            dict of kwargs for filter_profanity(), or None if disabled.
        """
        pf = options.get("profanity_filter")
        if not pf:
            return None
        if pf is True:
            return {"mode": "partial"}
        if isinstance(pf, str):
            if pf in ("partial", "full", "remove"):
                return {"mode": pf}
            return {"mode": "partial"}
        if isinstance(pf, dict):
            result = {"mode": pf.get("mode", "partial")}
            if "mask_char" in pf:
                result["mask_char"] = pf["mask_char"]
            if "extra_words" in pf:
                result["extra_words"] = set(pf["extra_words"])
            if "custom_words" in pf:
                result["custom_words"] = set(pf["custom_words"])
            return result
        return None

    def get_audio_from_websocket(self, websocket):
        """
        Receives audio buffer from websocket and creates a numpy array out of it.

        Args:
            websocket: The websocket to receive audio from.

        Returns:
            A numpy array containing the audio.
        """
        frame_data = websocket.recv()
        if frame_data == b"END_OF_AUDIO":
            return False
        if self.raw_pcm_input:
            audio_np = np.frombuffer(frame_data, dtype=np.int16)
            audio_np = audio_np.astype(np.float32) / 32768.0
        else:
            audio_np = np.frombuffer(frame_data, dtype=np.float32)
        if self._noise_reducer is not None:
            audio_np = self._noise_reducer.reduce(audio_np)
        return audio_np

    def handle_new_connection(self, websocket, faster_whisper_custom_model_path,
                              whisper_tensorrt_path, trt_multilingual, trt_py_session=False):
        try:
            logging.info("New client connected")
            options = websocket.recv()
            options = json.loads(options)

            self.use_vad = options.get('use_vad')
            if self.client_manager.is_server_full(websocket, options):
                wl_metrics.track_connection_rejected(reason="full")
                websocket.close()
                return False  # Indicates that the connection should not continue

            if self.backend.is_tensorrt():
                self.vad_detector = VoiceActivityDetector(frame_rate=self.RATE)
            self.initialize_client(websocket, options, faster_whisper_custom_model_path,
                                   whisper_tensorrt_path, trt_multilingual, trt_py_session=trt_py_session)
            wl_metrics.track_connection_opened()
            return True
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON from client")
            return False
        except ConnectionClosed:
            logging.info("Connection closed by client")
            return False
        except Exception as e:
            logging.error(f"Error during new connection initialization: {str(e)}")
            return False

    def process_audio_frames(self, websocket):
        frame_np = self.get_audio_from_websocket(websocket)
        client = self.client_manager.get_client(websocket)
        if frame_np is False:
            if self.backend.is_tensorrt():
                client.set_eos(True)
            return False

        if self.backend.is_tensorrt():
            voice_active = self.voice_activity(websocket, frame_np)
            if voice_active:
                self.no_voice_activity_chunks = 0
                client.set_eos(False)
            if self.use_vad and not voice_active:
                return True

        client.add_frames(frame_np)
        return True

    def recv_audio(self,
                   websocket,   
                   backend: BackendType = BackendType.FASTER_WHISPER,
                   faster_whisper_custom_model_path=None,
                   whisper_tensorrt_path=None,
                   trt_multilingual=False,
                   trt_py_session=False):
        """
        Receive audio chunks from a client in an infinite loop.

        Continuously receives audio frames from a connected client
        over a WebSocket connection. It processes the audio frames using a
        voice activity detection (VAD) model to determine if they contain speech
        or not. If the audio frame contains speech, it is added to the client's
        audio data for ASR.
        If the maximum number of clients is reached, the method sends a
        "WAIT" status to the client, indicating that they should wait
        until a slot is available.
        If a client's connection exceeds the maximum allowed time, it will
        be disconnected, and the client's resources will be cleaned up.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            backend (str): The backend to run the server with.
            faster_whisper_custom_model_path (str): path to custom faster whisper model.
            whisper_tensorrt_path (str): Required for tensorrt backend.
            trt_multilingual(bool): Only used for tensorrt, True if multilingual model.

        Raises:
            Exception: If there is an error during the audio frame processing.
        """
        self.backend = backend
        if not self.handle_new_connection(websocket, faster_whisper_custom_model_path,
                                          whisper_tensorrt_path, trt_multilingual, trt_py_session=trt_py_session):
            return

        try:
            while not self.client_manager.is_client_timeout(websocket):
                if not self.process_audio_frames(websocket):
                    break
        except ConnectionClosed:
            logging.info("Connection closed by client")
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
        finally:
            if self.client_manager.get_client(websocket):
                self.cleanup(websocket)
                websocket.close()
            wl_metrics.track_connection_closed()
            del websocket

    def _stream_transcription(self, file, language, prompt, temperature,
                              timestamp_granularities, hotwords,
                              faster_whisper_custom_model_path):
        """Return a StreamingResponse that yields SSE events per segment."""

        async def _sse_generator():
            tmp_path = None
            try:
                suffix = os.path.splitext(file.filename)[1] or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    shutil.copyfileobj(file.file, tmp)
                    tmp_path = tmp.name

                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                model_name = faster_whisper_custom_model_path or "small"
                transcriber = WhisperModel(model_name, device=device, compute_type=compute_type)
                segments, info = transcriber.transcribe(
                    tmp_path,
                    language=language,
                    initial_prompt=prompt,
                    temperature=temperature,
                    vad_filter=False,
                    word_timestamps=(timestamp_granularities and "word" in timestamp_granularities),
                    hotwords=hotwords,
                )

                # Emit metadata event with detected language info
                meta = {
                    "type": "metadata",
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "duration": info.duration,
                }
                yield f"data: {json.dumps(meta)}\n\n"

                for seg in segments:
                    seg_dict = {
                        "id": seg.id,
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip(),
                    }
                    if timestamp_granularities and "word" in timestamp_granularities:
                        seg_dict["words"] = [
                            {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                            for w in seg.words
                        ]
                    yield f"data: {json.dumps(seg_dict)}\n\n"

                yield "data: [DONE]\n\n"
                wl_metrics.track_rest_request(endpoint="transcriptions_stream", status=200)
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                wl_metrics.track_rest_request(endpoint="transcriptions_stream", status=500)
                wl_metrics.track_error("rest_stream")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        return StreamingResponse(_sse_generator(), media_type="text/event-stream")

    def _multichannel_transcribe(self, file, language, prompt, temperature,
                                  timestamp_granularities, hotwords,
                                  faster_whisper_custom_model_path,
                                  channel_labels_str):
        """Transcribe each audio channel independently and return merged results."""
        from whisper_live.multichannel import detect_channels_from_wav, split_channels, merge_channel_segments
        import soundfile as sf

        tmp_path = None
        try:
            suffix = os.path.splitext(file.filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name

            # Read audio with soundfile for proper channel handling
            audio_data, sample_rate = sf.read(tmp_path, dtype="float32")
            if audio_data.ndim == 1:
                num_channels = 1
                channel_arrays = [audio_data]
            else:
                num_channels = audio_data.shape[1]
                channel_arrays = [audio_data[:, ch] for ch in range(num_channels)]

            labels = None
            if channel_labels_str:
                labels = [l.strip() for l in channel_labels_str.split(",")]

            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            model_name = faster_whisper_custom_model_path or "small"
            transcriber = WhisperModel(model_name, device=device, compute_type=compute_type)

            all_channel_segments = []
            for ch_idx, ch_audio in enumerate(channel_arrays):
                # Write each channel to a temp mono file
                ch_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                sf.write(ch_tmp.name, ch_audio, sample_rate)
                ch_tmp.close()

                segments, info = transcriber.transcribe(
                    ch_tmp.name,
                    language=language,
                    initial_prompt=prompt,
                    temperature=temperature,
                    vad_filter=False,
                    word_timestamps=(timestamp_granularities and "word" in timestamp_granularities),
                    hotwords=hotwords,
                )

                ch_segs = []
                for seg in segments:
                    seg_dict = {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip(),
                    }
                    if timestamp_granularities and "word" in timestamp_granularities:
                        seg_dict["words"] = [
                            {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                            for w in seg.words
                        ]
                    ch_segs.append(seg_dict)

                all_channel_segments.append(ch_segs)
                os.unlink(ch_tmp.name)

            merged = merge_channel_segments(all_channel_segments, labels)
            wl_metrics.track_rest_request(endpoint="transcriptions_multichannel", status=200)
            return {
                "channels": num_channels,
                "segments": merged,
                "text": " ".join(seg["text"] for seg in merged),
            }
        except Exception as e:
            wl_metrics.track_rest_request(endpoint="transcriptions_multichannel", status=500)
            wl_metrics.track_error("multichannel")
            return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def run(self,
            host,
            port=9090,
            backend="tensorrt",
            faster_whisper_custom_model_path=None,
            whisper_tensorrt_path=None,
            trt_multilingual=False,
            trt_py_session=False,
            single_model=False,
            max_clients=4,
            max_connection_time=600,
            cache_path="~/.cache/whisper-live/",
            rest_port=8000,
            enable_rest=False,
            cors_origins: Optional[str] = None,
            batch_enabled=False,
            batch_max_size=8,
            batch_window_ms=50,
            raw_pcm_input=False,
            api_key: Optional[str] = None,
            rate_limit_rpm: int = 0,
            metrics_port: int = 0,
            plugin_registry=None,
            noise_reduction: Optional[str] = None,
            json_logs: bool = False,
            storage_backend: str = "local",
            storage_bucket: Optional[str] = None,
            storage_prefix: str = "whisperlive/",
            data_retention_days: int = 0,
            user_store_path: Optional[str] = None,
            jwt_jwks_url: Optional[str] = None,
            jwt_secret: Optional[str] = None,
            jwt_audience: Optional[str] = None,
            jwt_issuer: Optional[str] = None):
        """
        Run the transcription server.

        Args:
            host (str): The host address to bind the server.
            port (int): The port number to bind the server.
            batch_enabled (bool): Enable cross-client GPU batch inference for
                the faster_whisper backend. When enabled, ``single_model`` is
                forced to True and a ``BatchInferenceWorker`` is started after
                the first client connects. Defaults to False.
            batch_max_size (int): Maximum number of requests per GPU batch.
                Defaults to 8.
            batch_window_ms (int): Maximum time in milliseconds to wait for
                the batch to fill after the first request arrives. Defaults
                to 50.
            json_logs (bool): Emit structured JSON logs for CloudWatch/ELK.
                Defaults to False.
            storage_backend (str): "local" or "s3". Defaults to "local".
            storage_bucket (str): S3 bucket name (required if storage_backend="s3").
            storage_prefix (str): S3 key prefix. Defaults to "whisperlive/".
            data_retention_days (int): Auto-delete data older than this. 0 = disabled.
        """
        configure_logging(json_logs=json_logs)

        # Initialize storage backend
        from whisper_live.storage import create_storage
        storage_kwargs = {}
        if storage_backend == "s3":
            if not storage_bucket:
                raise ValueError("storage_bucket is required when storage_backend='s3'")
            storage_kwargs["bucket"] = storage_bucket
            storage_kwargs["prefix"] = storage_prefix
        self._storage = create_storage(storage_backend, **storage_kwargs)

        # Initialize user management / ACL
        self._user_store = None
        self._jwt_validator = None
        if user_store_path:
            from whisper_live.acl import UserStore
            self._user_store = UserStore(path=user_store_path)
            logging.info(f"User store loaded from {user_store_path}")
        if jwt_jwks_url or jwt_secret:
            from whisper_live.acl import JWTValidator
            self._jwt_validator = JWTValidator(
                jwks_url=jwt_jwks_url,
                secret=jwt_secret,
                audience=jwt_audience,
                issuer=jwt_issuer,
            )
            logging.info("JWT validation enabled")
        self.cache_path = cache_path
        self.raw_pcm_input = raw_pcm_input
        self.plugin_registry = plugin_registry

        if max_clients < 1:
            raise ValueError(f"max_clients must be >= 1, got {max_clients}")
        if max_connection_time <= 0:
            raise ValueError(f"max_connection_time must be > 0, got {max_connection_time}")
        if batch_enabled and batch_max_size < 1:
            raise ValueError(f"batch_max_size must be >= 1, got {batch_max_size}")
        if batch_enabled and batch_window_ms < 0:
            raise ValueError(f"batch_window_ms must be >= 0, got {batch_window_ms}")

        self.client_manager = ClientManager(max_clients, max_connection_time)
        if faster_whisper_custom_model_path is not None and not os.path.exists(faster_whisper_custom_model_path):
            if "/" not in faster_whisper_custom_model_path:
                raise ValueError(f"Custom faster_whisper model '{faster_whisper_custom_model_path}' is not a valid path or HuggingFace model.")
        if whisper_tensorrt_path is not None and not os.path.exists(whisper_tensorrt_path):
            raise ValueError(f"TensorRT model '{whisper_tensorrt_path}' is not a valid path.")

        # Batch inference config
        if batch_enabled:
            single_model = True  # Batch mode requires shared model
            self.batch_config = {
                'max_batch_size': batch_max_size,
                'batch_window_ms': batch_window_ms,
            }
            logging.info(f"Batch inference enabled (max_batch={batch_max_size}, window={batch_window_ms}ms)")
        else:
            self.batch_config = None

        if single_model:
            if faster_whisper_custom_model_path or whisper_tensorrt_path:
                logging.info("Custom model option was provided. Switching to single model mode.")
                self.single_model = True
                # TODO: load model initially
            else:
                logging.info("Single model mode currently only works with custom models.")
        if not BackendType.is_valid(backend):
            raise ValueError(f"{backend} is not a valid backend type. Choose backend from {BackendType.valid_types()}")

        # Start Prometheus metrics endpoint if port is specified
        if metrics_port > 0:
            wl_metrics.start_metrics_server(metrics_port)

        # Initialize model cache for REST API hot-swap
        from whisper_live.model_cache import ModelCache
        self._model_cache = ModelCache(
            max_models=3,
            default_model=faster_whisper_custom_model_path or "small",
        )

        # Initialize noise reduction if requested
        if noise_reduction:
            from whisper_live.noise_reduction import NoiseReducer
            self._noise_reducer = NoiseReducer(mode=noise_reduction)
            logging.info(f"Noise reduction enabled: {noise_reduction} mode")

        # New OpenAI-compatible REST API (toggleable via enable_rest boolean)
        if enable_rest:
            app = FastAPI(
                title="WhisperLive OpenAI-Compatible API",
                description=(
                    "Real-time and batch speech-to-text transcription API. "
                    "Compatible with the OpenAI Audio API format. "
                    "Supports streaming (SSE), webhooks, multi-channel audio, "
                    "speaker diarization, PII redaction, profanity filtering, "
                    "and audio intelligence analysis."
                ),
                version="0.9.0",
                docs_url="/docs",
                redoc_url="/redoc",
                openapi_url="/openapi.json",
            )
            origins = [o.strip() for o in cors_origins.split(',')] if cors_origins else []
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
                allow_headers=["*"],  # Allows all headers
            )

            # Authentication middleware
            if self._user_store or api_key:
                @app.middleware("http")
                async def _check_auth(request: Request, call_next):
                    # Skip auth for health, docs, and static
                    path = request.url.path
                    if path in ("/health", "/docs", "/redoc", "/openapi.json", "/"):
                        return await call_next(request)

                    auth = request.headers.get("Authorization", "")
                    token = auth.replace("Bearer ", "") if auth.startswith("Bearer ") else ""

                    # Try user store first
                    if self._user_store and token:
                        user = self._user_store.authenticate(token)
                        if user:
                            # Check per-user rate limit
                            if not self._user_store.check_rate_limit(user.user_id):
                                return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)
                            # Check quota
                            if not self._user_store.check_quota(user.user_id):
                                return JSONResponse({"error": "Monthly quota exceeded"}, status_code=429)
                            # Attach user to request state
                            request.state.user = user
                            return await call_next(request)

                    # Try JWT
                    if self._jwt_validator and token:
                        claims = self._jwt_validator.validate(token)
                        if claims:
                            request.state.jwt_claims = claims
                            return await call_next(request)

                    # Fall back to simple API key
                    if api_key and token == api_key:
                        return await call_next(request)

                    return JSONResponse({"error": "Invalid or missing API key"}, status_code=401)

            # Optional rate limiting (requests per minute per client IP)
            if rate_limit_rpm > 0:
                _rate_lock = threading.Lock()
                _rate_buckets: dict = {}  # ip -> deque of timestamps

                @app.middleware("http")
                async def _rate_limit(request: Request, call_next):
                    client_ip = request.client.host if request.client else "unknown"
                    now = time.time()
                    with _rate_lock:
                        bucket = _rate_buckets.setdefault(client_ip, collections.deque())
                        # Discard entries older than 60s
                        while bucket and bucket[0] < now - 60:
                            bucket.popleft()
                        if len(bucket) >= rate_limit_rpm:
                            return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)
                        bucket.append(now)
                    return await call_next(request)

            # Security headers middleware (HSTS, CSP, etc.)
            @app.middleware("http")
            async def _security_headers(request: Request, call_next):
                response = await call_next(request)
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
                response.headers["X-Content-Type-Options"] = "nosniff"
                response.headers["X-Frame-Options"] = "DENY"
                response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
                return response

            # Reject oversized uploads (default 500 MB)
            max_upload_bytes = int(os.environ.get("WHISPER_MAX_UPLOAD_BYTES", 500 * 1024 * 1024))

            @app.middleware("http")
            async def _check_upload_size(request: Request, call_next):
                content_length = request.headers.get("content-length")
                if content_length and int(content_length) > max_upload_bytes:
                    return JSONResponse(
                        {"error": f"Upload too large. Maximum size: {max_upload_bytes // (1024*1024)} MB"},
                        status_code=413,
                    )
                return await call_next(request)


            @app.get("/health", tags=["System"],
                     summary="Health check",
                     description="Returns server health status, connected client count, GPU availability, and model readiness.")
            async def health_check():
                client_count = len(self.client_manager.clients) if self.client_manager else 0
                max_clients = self.client_manager.max_clients if self.client_manager else 0

                gpu_available = torch.cuda.is_available()
                gpu_info = None
                if gpu_available:
                    gpu_info = {
                        "device_count": torch.cuda.device_count(),
                        "device_name": torch.cuda.get_device_name(0),
                    }
                    try:
                        mem = torch.cuda.mem_get_info(0)
                        gpu_info["free_vram_mb"] = round(mem[0] / 1024 / 1024)
                        gpu_info["total_vram_mb"] = round(mem[1] / 1024 / 1024)
                    except Exception:
                        pass

                models_loaded = self._model_cache.list_loaded() if self._model_cache else []

                status = "draining" if self._shutting_down else "ok"
                status_code = 503 if self._shutting_down else 200

                body = {
                    "status": status,
                    "clients": client_count,
                    "max_clients": max_clients,
                    "gpu_available": gpu_available,
                    "gpu": gpu_info,
                    "models_loaded": models_loaded,
                    "noise_reduction": self._noise_reducer is not None,
                }
                return JSONResponse(body, status_code=status_code)

            @app.get("/v1/plugins", tags=["System"],
                     summary="List plugins",
                     description="Returns the list of registered post-processing plugins.")
            async def list_plugins():
                if self.plugin_registry:
                    return {"plugins": self.plugin_registry.list_plugins()}
                return {"plugins": []}

            @app.get("/v1/models", tags=["System"],
                     summary="List loaded models",
                     description="Returns currently cached models available for hot-swap.")
            async def list_models():
                if self._model_cache:
                    return {"models": self._model_cache.list_loaded()}
                return {"models": []}

            @app.post("/v1/audio/transcriptions", tags=["Transcription"],
                      summary="Transcribe audio file",
                      description="Transcribe an audio file. Supports JSON, text, SRT, VTT, "
                                  "verbose_json response formats. Optional SSE streaming, "
                                  "webhook callbacks, and multi-channel mode.")
            async def transcribe(
                file: UploadFile,
                model: str = Form(default="whisper-1"),
                language: Optional[str] = Form(default=None),
                prompt: Optional[str] = Form(default=None),
                response_format: str = Form(default="json"),
                temperature: float = Form(default=0.0),
                timestamp_granularities: Optional[List[str]] = Form(default=None),
                # Stubs for unsupported OpenAI params
                chunking_strategy: Optional[str] = Form(default=None),
                include: Optional[List[str]] = Form(default=None),
                known_speaker_names: Optional[List[str]] = Form(default=None),
                known_speaker_references: Optional[List[str]] = Form(default=None),
                stream: bool = Form(default=False),
                hotwords: Optional[str] = Form(default=None),
                callback_url: Optional[str] = Form(default=None),
                multichannel: bool = Form(default=False),
                channel_labels: Optional[str] = Form(default=None),
            ):
                if stream:
                    return self._stream_transcription(
                        file, language, prompt, temperature,
                        timestamp_granularities, hotwords,
                        faster_whisper_custom_model_path,
                    )

                if callback_url:
                    import uuid as _uuid
                    import urllib.request
                    job_id = str(_uuid.uuid4())
                    suffix = os.path.splitext(file.filename)[1] or ".wav"
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    shutil.copyfileobj(file.file, tmp)
                    tmp_path = tmp.name
                    tmp.close()

                    def _run_async_job():
                        try:
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            compute_type = "float16" if device == "cuda" else "int8"
                            model_name = faster_whisper_custom_model_path or "small"
                            transcriber = WhisperModel(model_name, device=device, compute_type=compute_type)
                            segments, info = transcriber.transcribe(
                                tmp_path,
                                language=language,
                                initial_prompt=prompt,
                                temperature=temperature,
                                vad_filter=False,
                                word_timestamps=(timestamp_granularities and "word" in timestamp_granularities),
                                hotwords=hotwords,
                            )
                            text = " ".join([s.text.strip() for s in segments])
                            payload = json.dumps({"job_id": job_id, "text": text}).encode()
                            req = urllib.request.Request(
                                callback_url,
                                data=payload,
                                headers={"Content-Type": "application/json"},
                                method="POST",
                            )
                            urllib.request.urlopen(req, timeout=30)
                            wl_metrics.track_rest_request(endpoint="transcriptions_async", status=200)
                        except Exception as e:
                            logging.error(f"Async job {job_id} failed: {e}")
                            wl_metrics.track_error("async_job")
                            try:
                                err_payload = json.dumps({"job_id": job_id, "error": str(e)}).encode()
                                req = urllib.request.Request(
                                    callback_url,
                                    data=err_payload,
                                    headers={"Content-Type": "application/json"},
                                    method="POST",
                                )
                                urllib.request.urlopen(req, timeout=30)
                            except Exception:
                                logging.error(f"Failed to send error callback for job {job_id}")
                        finally:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)

                    threading.Thread(target=_run_async_job, daemon=True).start()
                    return JSONResponse({"job_id": job_id, "status": "processing"}, status_code=202)

                if multichannel:
                    return self._multichannel_transcribe(
                        file, language, prompt, temperature,
                        timestamp_granularities, hotwords,
                        faster_whisper_custom_model_path,
                        channel_labels,
                    )

                ignored_params = []
                if chunking_strategy:
                    ignored_params.append(f"chunking_strategy='{chunking_strategy}'")
                if known_speaker_names:
                    ignored_params.append("known_speaker_names")
                if known_speaker_references:
                    ignored_params.append("known_speaker_references")
                if include:
                    ignored_params.append(f"include={include}")
                if ignored_params:
                    logging.warning(f"Unsupported OpenAI params ignored: {', '.join(ignored_params)}")

                supported_formats = ["json", "text", "srt", "verbose_json", "vtt"]
                if response_format not in supported_formats:
                    wl_metrics.track_rest_request(endpoint="transcriptions", status=400)
                    return JSONResponse({"error": f"Unsupported response_format. Supported: {supported_formats}"}, status_code=400)

                if model != "whisper-1":
                    logging.info(f"Model '{model}' requested via REST API.")
                # Use model cache for hot-swap: accept model names like
                # "whisper-1", "small", "medium", "large-v3", etc.
                _model_map = {"whisper-1": "small"}
                resolved_model = _model_map.get(model, model)
                if faster_whisper_custom_model_path:
                    resolved_model = faster_whisper_custom_model_path

                try:
                    suffix = os.path.splitext(file.filename)[1] or ".wav"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        shutil.copyfileobj(file.file, tmp)
                        tmp_path = tmp.name

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    compute_type = "float16" if device == "cuda" else "int8"

                    transcriber = self._model_cache.get(resolved_model, device=device, compute_type=compute_type)
                    segments, info = transcriber.transcribe(
                        tmp_path,
                        language=language,
                        initial_prompt=prompt,
                        temperature=temperature,
                        vad_filter=False,
                        word_timestamps=(timestamp_granularities and "word" in timestamp_granularities),
                        hotwords=hotwords,
                    )

                    text = " ".join([s.text.strip() for s in segments])
                    os.unlink(tmp_path)

                    if response_format == "text":
                        wl_metrics.track_rest_request(endpoint="transcriptions", status=200)
                        return PlainTextResponse(text)
                    elif response_format == "json":
                        wl_metrics.track_rest_request(endpoint="transcriptions", status=200)
                        return {
                            "text": text,
                            "language": info.language,
                            "language_probability": info.language_probability,
                        }
                    elif response_format == "verbose_json":
                        verbose = {
                            "task": "transcribe",
                            "language": info.language,
                            "language_probability": info.language_probability,
                            "duration": info.duration,
                            "text": text,
                            "segments": []
                        }
                        for seg in segments:
                            seg_dict = {
                                "id": seg.id,
                                "seek": seg.seek,
                                "start": seg.start,
                                "end": seg.end,
                                "text": seg.text.strip(),
                                "tokens": seg.tokens,
                                "temperature": seg.temperature,
                                "avg_logprob": seg.avg_logprob,
                                "compression_ratio": seg.compression_ratio,
                                "no_speech_prob": seg.no_speech_prob
                            }
                            if timestamp_granularities and "word" in timestamp_granularities:
                                seg_dict["words"] = [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in seg.words]
                            verbose["segments"].append(seg_dict)
                        wl_metrics.track_rest_request(endpoint="transcriptions", status=200)
                        return verbose
                    elif response_format in ["srt", "vtt"]:
                        output = []
                        for i, seg in enumerate(segments, 1):
                            start = f"{int(seg.start // 3600):02}:{int((seg.start % 3600) // 60):02}:{seg.start % 60:06.3f}"
                            end = f"{int(seg.end // 3600):02}:{int((seg.end % 3600) // 60):02}:{seg.end % 60:06.3f}"
                            if response_format == "srt":
                                output.append(f"{i}\n{start.replace('.', ',')} --> {end.replace('.', ',')}\n{seg.text.strip()}\n")
                            else:  # vtt
                                output.append(f"{start} --> {end}\n{seg.text.strip()}\n")
                        wl_metrics.track_rest_request(endpoint="transcriptions", status=200)
                        return PlainTextResponse("\n".join(output))
                except Exception as e:
                    wl_metrics.track_rest_request(endpoint="transcriptions", status=500)
                    wl_metrics.track_error("rest_transcription")
                    return JSONResponse({"error": str(e)}, status_code=500)

            @app.post("/v1/audio/intelligence", tags=["Intelligence"],
                      summary="Analyze transcript",
                      description="Transcribe audio and perform NLP analysis: "
                                  "sentiment, topic detection, entity extraction, and summarization.")
            async def intelligence_endpoint(
                file: UploadFile,
                model: str = Form(default="whisper-1"),
                language: Optional[str] = Form(default=None),
                prompt: Optional[str] = Form(default=None),
                temperature: float = Form(default=0.0),
                sentiment: bool = Form(default=True),
                topics: bool = Form(default=True),
                entities: bool = Form(default=True),
                summary: bool = Form(default=True),
                summary_sentences: int = Form(default=3),
                topic_count: int = Form(default=5),
            ):
                from whisper_live.audio_intelligence import analyze_transcript
                try:
                    suffix = os.path.splitext(file.filename)[1] or ".wav"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        shutil.copyfileobj(file.file, tmp)
                        tmp_path = tmp.name

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    compute_type = "float16" if device == "cuda" else "int8"
                    model_name = faster_whisper_custom_model_path or "small"
                    transcriber = WhisperModel(model_name, device=device, compute_type=compute_type)
                    segments, info = transcriber.transcribe(
                        tmp_path,
                        language=language,
                        initial_prompt=prompt,
                        temperature=temperature,
                        vad_filter=False,
                    )
                    text = " ".join([s.text.strip() for s in segments])
                    os.unlink(tmp_path)

                    analysis = analyze_transcript(
                        text,
                        sentiment=sentiment,
                        topics=topics,
                        entities=entities,
                        summary=summary,
                        summary_sentences=summary_sentences,
                        topic_count=topic_count,
                    )
                    analysis["text"] = text
                    analysis["language"] = info.language
                    analysis["language_probability"] = info.language_probability
                    analysis["duration"] = info.duration
                    wl_metrics.track_rest_request(endpoint="intelligence", status=200)
                    return analysis
                except Exception as e:
                    wl_metrics.track_rest_request(endpoint="intelligence", status=500)
                    wl_metrics.track_error("intelligence")
                    return JSONResponse({"error": str(e)}, status_code=500)

            # --- Data management / GDPR endpoints ---

            @app.get("/v1/jobs/{job_id}", tags=["Data Management"],
                     summary="Get job result",
                     description="Retrieve stored transcription result for a job.")
            async def get_job_result(job_id: str):
                result = self._storage.get_result(job_id)
                if result is None:
                    return JSONResponse({"error": "Job not found"}, status_code=404)
                return result

            @app.delete("/v1/jobs/{job_id}", tags=["Data Management"],
                        summary="Delete job data",
                        description="Delete all stored audio and result data for a job.")
            async def delete_job(job_id: str):
                self._storage.delete_job(job_id)
                return {"status": "deleted", "job_id": job_id}

            @app.delete("/v1/users/{user_id}/data", tags=["Data Management"],
                        summary="GDPR: Delete all user data",
                        description="Delete all audio and result data associated with a user. "
                                    "Implements GDPR right to deletion.")
            async def delete_user_data(user_id: str):
                count = self._storage.delete_all_for_user(user_id)
                return {"status": "deleted", "user_id": user_id, "files_deleted": count}

            # Periodic data retention cleanup
            if data_retention_days > 0:
                retention_seconds = data_retention_days * 86400

                def _retention_cleanup_loop():
                    while not self._shutting_down:
                        try:
                            deleted = self._storage.cleanup_expired(retention_seconds)
                            if deleted:
                                logging.info(f"Data retention cleanup: deleted {deleted} expired files")
                        except Exception as e:
                            logging.error(f"Data retention cleanup error: {e}")
                        # Run every hour
                        for _ in range(3600):
                            if self._shutting_down:
                                break
                            time.sleep(1)

                threading.Thread(target=_retention_cleanup_loop, daemon=True).start()
                logging.info(f"Data retention enabled: {data_retention_days} days")

            # --- Admin / User management endpoints ---
            if self._user_store:
                from whisper_live.acl import Role

                def _require_admin(request: Request):
                    user = getattr(request.state, "user", None)
                    if not user or not Role(user.role).can_admin():
                        return None
                    return user

                @app.get("/v1/admin/users", tags=["Admin"],
                         summary="List all users",
                         description="Admin-only. Returns all registered users.")
                async def list_users(request: Request):
                    if not _require_admin(request):
                        return JSONResponse({"error": "Admin access required"}, status_code=403)
                    users = self._user_store.list_users()
                    # Redact key hashes
                    for u in users:
                        u.pop("api_key_hash", None)
                    return {"users": users}

                @app.post("/v1/admin/users", tags=["Admin"],
                          summary="Create a user",
                          description="Admin-only. Create a new user with API key.")
                async def create_user(
                    request: Request,
                    name: str = Form(...),
                    role: str = Form(default="user"),
                    rate_limit_rpm: int = Form(default=60),
                    quota_minutes: int = Form(default=0),
                ):
                    if not _require_admin(request):
                        return JSONResponse({"error": "Admin access required"}, status_code=403)
                    try:
                        user_role = Role(role)
                    except ValueError:
                        return JSONResponse({"error": f"Invalid role. Must be: admin, user, readonly"}, status_code=400)
                    user, api_key = self._user_store.create_user(
                        name=name, role=user_role,
                        rate_limit_rpm=rate_limit_rpm,
                        quota_minutes=quota_minutes,
                    )
                    return {
                        "user_id": user.user_id,
                        "name": user.name,
                        "role": user.role.value,
                        "api_key": api_key,  # Only shown once at creation
                        "rate_limit_rpm": user.rate_limit_rpm,
                        "quota_minutes": user.quota_minutes,
                    }

                @app.patch("/v1/admin/users/{user_id}", tags=["Admin"],
                           summary="Update a user",
                           description="Admin-only. Update user settings.")
                async def update_user(
                    request: Request,
                    user_id: str,
                    name: Optional[str] = Form(default=None),
                    role: Optional[str] = Form(default=None),
                    rate_limit_rpm: Optional[int] = Form(default=None),
                    quota_minutes: Optional[int] = Form(default=None),
                    enabled: Optional[bool] = Form(default=None),
                ):
                    if not _require_admin(request):
                        return JSONResponse({"error": "Admin access required"}, status_code=403)
                    kwargs = {}
                    if name is not None:
                        kwargs["name"] = name
                    if role is not None:
                        kwargs["role"] = role
                    if rate_limit_rpm is not None:
                        kwargs["rate_limit_rpm"] = rate_limit_rpm
                    if quota_minutes is not None:
                        kwargs["quota_minutes"] = quota_minutes
                    if enabled is not None:
                        kwargs["enabled"] = enabled
                    user = self._user_store.update_user(user_id, **kwargs)
                    if not user:
                        return JSONResponse({"error": "User not found"}, status_code=404)
                    return {"status": "updated", "user_id": user_id}

                @app.delete("/v1/admin/users/{user_id}", tags=["Admin"],
                            summary="Delete a user",
                            description="Admin-only. Delete a user and revoke their API key.")
                async def delete_user(request: Request, user_id: str):
                    if not _require_admin(request):
                        return JSONResponse({"error": "Admin access required"}, status_code=403)
                    if self._user_store.delete_user(user_id):
                        return {"status": "deleted", "user_id": user_id}
                    return JSONResponse({"error": "User not found"}, status_code=404)

                @app.post("/v1/admin/users/{user_id}/rotate-key", tags=["Admin"],
                          summary="Rotate API key",
                          description="Admin-only. Generate a new API key for a user. Old key is invalidated.")
                async def rotate_key(request: Request, user_id: str):
                    if not _require_admin(request):
                        return JSONResponse({"error": "Admin access required"}, status_code=403)
                    new_key = self._user_store.rotate_key(user_id)
                    if not new_key:
                        return JSONResponse({"error": "User not found"}, status_code=404)
                    return {"user_id": user_id, "api_key": new_key}

            # Serve the web transcription UI from the web/ directory
            web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
            if os.path.isdir(web_dir):
                @app.get("/", response_class=HTMLResponse)
                async def serve_index():
                    index_path = os.path.join(web_dir, "index.html")
                    with open(index_path) as f:
                        return f.read()

                app.mount("/static", StaticFiles(directory=web_dir), name="static-web")
                logging.info(f"📄 Web transcription UI available at http://0.0.0.0:{rest_port}/")

            threading.Thread(
                target=uvicorn.run,
                args=(app,),
                kwargs={"host": "0.0.0.0", "port": rest_port, "log_level": "info"},
                daemon=True
            ).start()
            logging.info(f"✅ OpenAI-Compatible API started on http://0.0.0.0:{rest_port}")

        # Original WebSocket server (always supported)
        extra_ws_kwargs = {}
        if api_key:
            def _ws_auth(path, request_headers):
                auth = request_headers.get("Authorization", "")
                token_param = None
                # Check query string for token parameter
                if "?" in path:
                    from urllib.parse import urlparse, parse_qs
                    parsed = urlparse(path)
                    token_param = parse_qs(parsed.query).get("token", [None])[0]
                if auth == f"Bearer {api_key}" or token_param == api_key:
                    return None  # Allow connection
                wl_metrics.track_connection_rejected(reason="auth")
                return (401, [("Content-Type", "text/plain")], b"Unauthorized\n")
            extra_ws_kwargs["process_request"] = _ws_auth

        with serve(
            functools.partial(
                self.recv_audio,
                backend=BackendType(backend),
                faster_whisper_custom_model_path=faster_whisper_custom_model_path,
                whisper_tensorrt_path=whisper_tensorrt_path,
                trt_multilingual=trt_multilingual,
                trt_py_session=trt_py_session,
            ),
            host,
            port,
            **extra_ws_kwargs,
        ) as server:
            self._ws_server = server

            def _graceful_shutdown(signum, frame):
                sig_name = signal.Signals(signum).name
                logging.info(f"Received {sig_name} — initiating graceful shutdown")
                self._shutting_down = True
                # Stop accepting new connections
                server.shutdown()
                # Drain existing clients
                if self.client_manager:
                    with self.client_manager.lock:
                        clients = list(self.client_manager.clients.items())
                    for ws, client in clients:
                        try:
                            client.disconnect()
                            ws.close()
                        except Exception:
                            pass
                logging.info("Graceful shutdown complete")

            signal.signal(signal.SIGTERM, _graceful_shutdown)
            signal.signal(signal.SIGINT, _graceful_shutdown)
            server.serve_forever()

    def voice_activity(self, websocket, frame_np):
        """
        Evaluates the voice activity in a given audio frame and manages the state of voice activity detection.

        This method uses the configured voice activity detection (VAD) model to assess whether the given audio frame
        contains speech. If the VAD model detects no voice activity for more than three consecutive frames,
        it sets an end-of-speech (EOS) flag for the associated client. This method aims to efficiently manage
        speech detection to improve subsequent processing steps.

        Args:
            websocket: The websocket associated with the current client. Used to retrieve the client object
                    from the client manager for state management.
            frame_np (numpy.ndarray): The audio frame to be analyzed. This should be a NumPy array containing
                                    the audio data for the current frame.

        Returns:
            bool: True if voice activity is detected in the current frame, False otherwise. When returning False
                after detecting no voice activity for more than three consecutive frames, it also triggers the
                end-of-speech (EOS) flag for the client.
        """
        if not self.vad_detector(frame_np):
            self.no_voice_activity_chunks += 1
            if self.no_voice_activity_chunks > 3:
                client = self.client_manager.get_client(websocket)
                if not client.eos:
                    client.set_eos(True)
                time.sleep(0.1)    # Sleep 100m; wait some voice activity.
            return False
        return True

    def cleanup(self, websocket):
        """
        Cleans up resources associated with a given client's websocket.

        Args:
            websocket: The websocket associated with the client to be cleaned up.
        """
        client = self.client_manager.get_client(websocket)
        if client:
            if hasattr(client, 'translation_client') and client.translation_client:
                client.translation_client.cleanup()
                
            # Wait for translation thread to finish
            if hasattr(client, 'translation_thread') and client.translation_thread:
                client.translation_thread.join(timeout=2.0)
            self.client_manager.remove_client(websocket)