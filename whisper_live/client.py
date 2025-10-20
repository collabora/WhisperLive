#!/usr/bin/env python3
import os
import wave
import numpy as np
import pyaudio
import threading
import json
import websocket
import uuid
import time
import argparse
import whisper_live.utils as utils


class Client:
    """Handles communication with a server using WebSocket."""
    INSTANCES = {}
    END_OF_AUDIO = "END_OF_AUDIO"

    def __init__(
        self,
        host=None,
        port=None,
        lang=None,
        translate=False,
        model="small",
        srt_file_path="output.srt",
        use_vad=True,
        use_wss=False,
        log_transcription=True,
        send_last_n_segments=3,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=3,
        transcription_callback=None,
        enable_translation=False,
        target_language="fr",
        translation_callback=None,
        translation_srt_file_path="output_translated.srt",
    ):
        self.recording = False
        self.task = "translate" if translate else "transcribe"
        self.uid = str(uuid.uuid4())
        self.waiting = False
        self.last_response_received = None
        self.disconnect_if_no_response_for = 15
        self.language = lang
        self.model = model
        self.server_error = False
        self.srt_file_path = srt_file_path
        self.use_vad = use_vad
        self.use_wss = use_wss
        self.last_segment = None
        self.last_received_segment = None
        self.log_transcription = log_transcription
        self.send_last_n_segments = send_last_n_segments
        self.no_speech_thresh = no_speech_thresh
        self.clip_audio = clip_audio
        self.same_output_threshold = same_output_threshold
        self.transcription_callback = transcription_callback
        self.enable_translation = enable_translation
        self.target_language = target_language
        self.translation_callback = translation_callback
        self.translation_srt_file_path = translation_srt_file_path
        self.last_translated_segment = None

        # latency stats
        self.audio_start_time = None
        self.latencies = []

        if host is not None and port is not None:
            socket_protocol = 'wss' if self.use_wss else "ws"
            socket_url = f"{socket_protocol}://{host}:{port}"
            self.client_socket = websocket.WebSocketApp(
                socket_url,
                on_open=lambda ws: self.on_open(ws),
                on_message=lambda ws, message: self.on_message(ws, message),
                on_error=lambda ws, error: self.on_error(ws, error),
                on_close=lambda ws, close_status_code, close_msg: self.on_close(ws, close_status_code, close_msg),
            )
        else:
            print("[ERROR]: No host or port specified.")
            return

        Client.INSTANCES[self.uid] = self

        self.ws_thread = threading.Thread(target=self.client_socket.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        self.transcript = []
        self.translated_transcript = []
        print("[INFO]: * recording")

    # ---------- WebSocket Handlers ----------
    def handle_status_messages(self, message_data):
        status = message_data["status"]
        if status == "WAIT":
            self.waiting = True
            print(f"[INFO]: Server is full. Estimated wait time {round(message_data['message'])} minutes.")
        elif status == "ERROR":
            print(f"Message from Server: {message_data['message']}")
            self.server_error = True
        elif status == "WARNING":
            print(f"Message from Server: {message_data['message']}")

    def process_segments(self, segments, translated=False):
        text = []
        now = time.time()   # 🕒 arrival time of message

        for i, seg in enumerate(segments):
            if not text or text[-1] != seg["text"]:
                text.append(seg["text"].strip())
                if i == len(segments) - 1 and not seg.get("completed", False):
                    self.last_segment = seg
                elif self.server_backend == "faster_whisper" and seg.get("completed", False):
                    if translated:
                        if not self.translated_transcript or float(seg['start']) >= float(self.translated_transcript[-1]['end']):
                            self.translated_transcript.append(seg)
                    else:
                        if not self.transcript or float(seg['start']) >= float(self.transcript[-1]['end']):
                            self.transcript.append(seg)

        # Latency measurement
        if segments:
            seg_end = float(segments[-1]["end"])
            if self.audio_start_time:
                wall_latency = now - self.audio_start_time
                self.latencies.append(wall_latency)
                print(f"[LATENCY] Segment: {segments[-1]['text']!r} | "
                      f"AudioEnd={seg_end:.2f}s | Latency={wall_latency:.2f}s")

        if not translated:
            if self.last_received_segment is None or self.last_received_segment != segments[-1]["text"]:
                self.last_response_received = time.time()
                self.last_received_segment = segments[-1]["text"]

        if translated:
            if self.translation_callback:
                try:
                    self.translation_callback(" ".join(text), segments)
                except Exception as e:
                    print(f"[WARN] translation_callback raised: {e}")
        else:
            if self.transcription_callback:
                try:
                    self.transcription_callback(" ".join(text), segments)
                except Exception as e:
                    print(f"[WARN] transcription_callback raised: {e}")

        if self.log_transcription:
            original_text = [seg["text"] for seg in self.transcript[-4:]]
            if self.last_segment and self.last_segment["text"] not in original_text:
                original_text.append(self.last_segment["text"])
            utils.clear_screen()
            utils.print_transcript(original_text)
            if self.enable_translation:
                print(f"\n\nTRANSLATION to {self.target_language}:")
                utils.print_transcript([seg["text"] for seg in self.translated_transcript[-4:]], translated=True)

    def on_message(self, ws, message):
        message = json.loads(message)
        if self.uid != message.get("uid"):
            print("[ERROR]: invalid client uid")
            return
        if "status" in message:
            self.handle_status_messages(message)
            return
        if message.get("message") == "DISCONNECT":
            print("[INFO]: Server disconnected due to overtime.")
            self.recording = False
        if message.get("message") == "SERVER_READY":
            self.last_response_received = time.time()
            self.recording = True
            self.server_backend = message["backend"]
            print(f"[INFO]: Server Running with backend {self.server_backend}")
            return
        if "language" in message:
            print(f"[INFO]: Server detected language {message['language']} (p={message['language_prob']})")
            return
        if "segments" in message:
            self.process_segments(message["segments"])
        if "translated_segments" in message:
            self.process_segments(message["translated_segments"], translated=True)

    def on_error(self, ws, error):
        print(f"[ERROR] WebSocket Error: {error}")
        self.server_error = True

    def on_close(self, ws, close_status_code, close_msg):
        print(f"[INFO]: Websocket connection closed: {close_status_code}: {close_msg}")
        self.recording = False
        self.waiting = False
        if self.latencies:
            avg = sum(self.latencies) / len(self.latencies)
            print(f"[SUMMARY] Avg latency={avg:.2f}s | Min={min(self.latencies):.2f}s | Max={max(self.latencies):.2f}s")

    def on_open(self, ws):
        print("[INFO]: Opened connection")
        ws.send(json.dumps({
            "uid": self.uid,
            "language": self.language,
            "task": self.task,
            "model": self.model,
            "use_vad": self.use_vad,
            "send_last_n_segments": self.send_last_n_segments,
            "no_speech_thresh": self.no_speech_thresh,
            "clip_audio": self.clip_audio,
            "same_output_threshold": self.same_output_threshold,
            "enable_translation": self.enable_translation,
            "target_language": self.target_language,
        }))

    def send_packet_to_server(self, message):
        try:
            self.client_socket.send(message, websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            print(e)

    def close_websocket(self):
        try:
            self.client_socket.close()
            self.ws_thread.join()
        except Exception as e:
            print("[ERROR]: Error closing WebSocket:", e)

    def write_srt_file(self, output_path="output.srt"):
        if self.server_backend == "faster_whisper":
            if not self.transcript and self.last_segment:
                self.transcript.append(self.last_segment)
            elif self.last_segment and self.transcript[-1]["text"] != self.last_segment["text"]:
                self.transcript.append(self.last_segment)
            utils.create_srt_file(self.transcript, output_path)
        if self.enable_translation:
            utils.create_srt_file(self.translated_transcript, self.translation_srt_file_path)


class TranscriptionTeeClient:
    """Send audio from mic/file to multiple clients"""
    def __init__(self, clients, save_output_recording=False, output_recording_filename="./output_recording.wav", mute_audio_playback=False):
        self.clients = clients
        self.chunk = 4096
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.p = pyaudio.PyAudio()
        try:
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
            )
        except OSError as error:
            print(f"[WARN]: Unable to access microphone. {error}")
            self.stream = None

    def __call__(self, audio=None):
        print("[INFO]: Waiting for server ready ...")
        for client in self.clients:
            while not client.recording:
                if client.waiting or client.server_error:
                    self.close_all_clients()
                    return
        print("[INFO]: Server Ready!")
        if audio:
            self.play_file(audio)
        else:
            self.record()

    def close_all_clients(self):
        for client in self.clients:
            client.close_websocket()

    def multicast_packet(self, packet, unconditional=False):
        for client in self.clients:
            if unconditional or client.recording:
                client.send_packet_to_server(packet)

    def play_file(self, filename):
        with wave.open(filename, "rb") as wavfile:
            while any(client.recording for client in self.clients):
                data = wavfile.readframes(self.chunk)
                if not data:
                    break
                audio_array = self.bytes_to_float_array(data)
                self.multicast_packet(audio_array.tobytes())
            for client in self.clients:
                client.write_srt_file()
            self.multicast_packet(Client.END_OF_AUDIO.encode('utf-8'), True)
            self.close_all_clients()

    def record(self):
        if self.stream is None:
            print("[ERROR]: No microphone stream available.")
            return
        print("[INFO]: Recording from microphone...")

        # mark audio start
        for client in self.clients:
            client.audio_start_time = time.time()

        try:
            while any(client.recording for client in self.clients):
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                audio_array = self.bytes_to_float_array(data)
                self.multicast_packet(audio_array.tobytes())
        except KeyboardInterrupt:
            print("[INFO]: Keyboard interrupt – stopping.")
        finally:
            self.multicast_packet(Client.END_OF_AUDIO.encode('utf-8'), True)
            for client in self.clients:
                client.write_srt_file()
            self.close_all_clients()
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.p.terminate()

    @staticmethod
    def bytes_to_float_array(audio_bytes):
        raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
        return raw_data.astype(np.float32) / 32768.0


class TranscriptionClient(TranscriptionTeeClient):
    def __init__(self, host, port, lang=None, translate=False, model="small",
                 use_vad=True, use_wss=False, save_output_recording=False,
                 output_recording_filename="./output_recording.wav", output_transcription_path="./output.srt",
                 log_transcription=True, mute_audio_playback=False,
                 send_last_n_segments=10, no_speech_thresh=0.45,
                 clip_audio=False, same_output_threshold=10,
                 transcription_callback=None, enable_translation=False,
                 target_language="fr", translation_callback=None,
                 translation_srt_file_path="./output_translated.srt"):

        self.client = Client(
            host=host,
            port=port,
            lang=lang,
            translate=translate,
            model=model,
            srt_file_path=output_transcription_path,
            use_vad=use_vad,
            use_wss=use_wss,
            log_transcription=log_transcription,
            send_last_n_segments=send_last_n_segments,
            no_speech_thresh=no_speech_thresh,
            clip_audio=clip_audio,
            same_output_threshold=same_output_threshold,
            transcription_callback=transcription_callback,
            enable_translation=enable_translation,
            target_language=target_language,
            translation_callback=translation_callback,
            translation_srt_file_path=translation_srt_file_path,
        )

        # Validate SRT/WAV filenames
        if save_output_recording and not output_recording_filename.endswith(".wav"):
            raise ValueError("output_recording_filename must end with .wav")
        if not output_transcription_path.endswith(".srt"):
            raise ValueError("output_transcription_path must end with .srt")
        if not translation_srt_file_path.endswith(".srt"):
            raise ValueError("translation_srt_file_path must end with .srt")

        TranscriptionTeeClient.__init__(
            self, [self.client],
            save_output_recording=save_output_recording,
            output_recording_filename=output_recording_filename,
            mute_audio_playback=mute_audio_playback
        )
        print("[INFO]: TranscriptionClient initialized and ready")
