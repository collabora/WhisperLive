import os
import time
import threading
import queue
import json
import functools
import logging
import csv   # 🟢 added for CSV logging
from enum import Enum
from typing import List, Optional

import numpy as np
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed
from whisper_live.vad import VoiceActivityDetector
from whisper_live.backend.base import ServeClientBase

logging.basicConfig(level=logging.INFO)

# 🟢 CSV file path for latency logging
SERVER_CSV = "server_latency.csv"

def log_server_latency(model, latency):
    """Append latency measurement to CSV."""
    file_exists = os.path.exists(SERVER_CSV)
    with open(SERVER_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model", "latency_seconds"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), model, latency])


class ClientManager:
    def __init__(self, max_clients=4, max_connection_time=600):
        self.clients = {}
        self.start_times = {}
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time

    def add_client(self, websocket, client):
        self.clients[websocket] = client
        self.start_times[websocket] = time.time()

    def get_client(self, websocket):
        return self.clients.get(websocket, False)

    def remove_client(self, websocket):
        client = self.clients.pop(websocket, None)
        if client:
            client.cleanup()
        self.start_times.pop(websocket, None)

    def get_wait_time(self):
        wait_time = None
        for start_time in self.start_times.values():
            current_client_time_remaining = self.max_connection_time - (time.time() - start_time)
            if wait_time is None or current_client_time_remaining < wait_time:
                wait_time = current_client_time_remaining
        return wait_time / 60 if wait_time is not None else 0

    def is_server_full(self, websocket, options):
        if len(self.clients) >= self.max_clients:
            wait_time = self.get_wait_time()
            response = {"uid": options["uid"], "status": "WAIT", "message": wait_time}
            websocket.send(json.dumps(response))
            return True
        return False

    def is_client_timeout(self, websocket):
        elapsed_time = time.time() - self.start_times[websocket]
        if elapsed_time >= self.max_connection_time:
            self.clients[websocket].disconnect()
            logging.warning(f"Client with uid '{self.clients[websocket].client_uid}' disconnected due to overtime.")
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
        self.model_name = None   # 🟢 track model name

    def initialize_client(
        self, websocket, options, faster_whisper_custom_model_path,
        whisper_tensorrt_path, trt_multilingual, trt_py_session=False,
    ):
        client: Optional[ServeClientBase] = None
        self.model_name = options.get("model", "unknown")  # 🟢 store model name

        if self.backend.is_faster_whisper():
            from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper
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
                translation_queue=None
            )
            logging.info("Running faster_whisper backend.")

        self.client_manager.add_client(websocket, client)

    def get_audio_from_websocket(self, websocket):
        frame_data = websocket.recv()
        if frame_data == b"END_OF_AUDIO":
            return False
        return np.frombuffer(frame_data, dtype=np.float32)

    def handle_new_connection(self, websocket, faster_whisper_custom_model_path,
                              whisper_tensorrt_path, trt_multilingual, trt_py_session=False):
        try:
            logging.info("New client connected")
            options = websocket.recv()
            options = json.loads(options)
            self.use_vad = options.get('use_vad')
            if self.client_manager.is_server_full(websocket, options):
                websocket.close()
                return False
            self.initialize_client(websocket, options, faster_whisper_custom_model_path,
                                   whisper_tensorrt_path, trt_multilingual, trt_py_session=trt_py_session)
            return True
        except Exception as e:
            logging.error(f"Error during new connection initialization: {str(e)}")
            return False

    def process_audio_frames(self, websocket):
        frame_np = self.get_audio_from_websocket(websocket)
        client = self.client_manager.get_client(websocket)
        if frame_np is False:
            return False

        # 🟢 measure latency for this chunk
        start_time = time.time()
        client.add_frames(frame_np)
        end_time = time.time()
        latency = end_time - start_time
        logging.info(f"[SERVER LATENCY] Processed chunk in {latency:.3f}s")
        if self.model_name:
            log_server_latency(self.model_name, latency)

        return True

    def recv_audio(self, websocket, backend: BackendType = BackendType.FASTER_WHISPER,
                   faster_whisper_custom_model_path=None,
                   whisper_tensorrt_path=None,
                   trt_multilingual=False,
                   trt_py_session=False):
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
            del websocket

    def run(self, host, port=9090, backend="faster_whisper",
            faster_whisper_custom_model_path=None,
            whisper_tensorrt_path=None,
            trt_multilingual=False,
            trt_py_session=False,
            single_model=False,
            max_clients=4,
            max_connection_time=600,
            cache_path="~/.cache/whisper-live/"):
        self.cache_path = cache_path
        self.client_manager = ClientManager(max_clients, max_connection_time)
        if not BackendType.is_valid(backend):
            raise ValueError(f"{backend} is not a valid backend type. Choose from {BackendType.valid_types()}")
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
            port
        ) as server:
            server.serve_forever()

    def cleanup(self, websocket):
        client = self.client_manager.get_client(websocket)
        if client:
            self.client_manager.remove_client(websocket)


if __name__ == "__main__":
    logging.info("🚀 Starting WhisperLive server...")
    server = TranscriptionServer()
    server.run(
        host="127.0.0.1",
        port=9090,
        backend="faster_whisper",
        max_clients=2,
        max_connection_time=600,
    )
