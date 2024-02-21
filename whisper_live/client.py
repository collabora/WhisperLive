import os
import wave

import numpy as np
import pyaudio
import threading
import json
import websocket
import uuid
import time
import ffmpeg
import whisper_live.utils as utils


class Client:
    """
    Handles audio recording, streaming, and communication with a server using WebSocket.
    """
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
        use_vad=True
    ):
        """
        Initializes a Client instance for audio recording and streaming to a server.

        If host and port are not provided, the WebSocket connection will not be established.
        When translate is True, the task will be set to "translate" instead of "transcribe".
        he audio recording starts immediately upon initialization.

        Args:
            host (str): The hostname or IP address of the server.
            port (int): The port number for the WebSocket server.
            lang (str, optional): The selected language for transcription. Default is None.
            translate (bool, optional): Specifies if the task is translation. Default is False.
        """
        self.chunk = 4096
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 60000
        self.recording = False
        self.task = "transcribe"
        self.uid = str(uuid.uuid4())
        self.waiting = False
        self.last_response_recieved = None
        self.disconnect_if_no_response_for = 15
        self.language = lang
        self.model = model
        self.server_error = False
        self.srt_file_path = srt_file_path
        self.use_vad = use_vad
        self.last_recieved_segment = None

        if translate:
            self.task = "translate"

        self.timestamp_offset = 0.0
        self.audio_bytes = None
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

        if host is not None and port is not None:
            socket_url = f"ws://{host}:{port}"
            self.client_socket = websocket.WebSocketApp(
                socket_url,
                on_open=lambda ws: self.on_open(ws),
                on_message=lambda ws, message: self.on_message(ws, message),
                on_error=lambda ws, error: self.on_error(ws, error),
                on_close=lambda ws, close_status_code, close_msg: self.on_close(
                    ws, close_status_code, close_msg
                ),
            )
        else:
            print("[ERROR]: No host or port specified.")
            return

        Client.INSTANCES[self.uid] = self

        # start websocket client in a thread
        self.ws_thread = threading.Thread(target=self.client_socket.run_forever)
        self.ws_thread.setDaemon(True)
        self.ws_thread.start()

        self.frames = b""
        self.transcript = []
        print("[INFO]: * recording")

    def handle_status_messages(self, message_data):
        """Handles server status messages."""
        status = message_data["status"]
        if status == "WAIT":
            self.waiting = True
            print(f"[INFO]: Server is full. Estimated wait time {round(message_data['message'])} minutes.")
        elif status == "ERROR":
            print(f"Message from Server: {message_data['message']}")
            self.server_error = True
        elif status == "WARNING":
            print(f"Message from Server: {message_data['message']}")

    def process_segments(self, segments):
        """Processes transcript segments."""
        text = []
        for i, seg in enumerate(segments):
            if not text or text[-1] != seg["text"]:
                text.append(seg["text"])
                if i == len(segments) - 1:
                    self.last_segment = seg
                elif (self.server_backend == "faster_whisper" and
                      (not self.transcript or
                        float(seg['start']) >= float(self.transcript[-1]['end']))):
                    self.transcript.append(seg)
        # update last received segment and last valild responsne time
        if self.last_recieved_segment is None or self.last_recieved_segment != segments[-1]["text"]:
            self.last_response_recieved = time.time()
            self.last_recieved_segment = segments[-1]["text"]

        # Truncate to last 3 entries for brevity.
        text = text[-3:]
        utils.clear_screen()
        utils.print_transcript(text)

    def on_message(self, ws, message):
        """
        Callback function called when a message is received from the server.

        It updates various attributes of the client based on the received message, including
        recording status, language detection, and server messages. If a disconnect message
        is received, it sets the recording status to False.

        Args:
            ws (websocket.WebSocketApp): The WebSocket client instance.
            message (str): The received message from the server.

        """
        message = json.loads(message)

        if self.uid != message.get("uid"):
            print("[ERROR]: invalid client uid")
            return

        if "status" in message.keys():
            self.handle_status_messages(message)
            return

        if "message" in message.keys() and message["message"] == "DISCONNECT":
            print("[INFO]: Server disconnected due to overtime.")
            self.recording = False

        if "message" in message.keys() and message["message"] == "SERVER_READY":
            self.last_response_recieved = time.time()
            self.recording = True
            self.server_backend = message["backend"]
            print(f"[INFO]: Server Running with backend {self.server_backend}")
            return

        if "language" in message.keys():
            self.language = message.get("language")
            lang_prob = message.get("language_prob")
            print(
                f"[INFO]: Server detected language {self.language} with probability {lang_prob}"
            )
            return

        if "segments" in message.keys():
            self.process_segments(message["segments"])

    def on_error(self, ws, error):
        print(f"[ERROR] WebSocket Error: {error}")
        self.server_error = True
        self.error_message = error

    def on_close(self, ws, close_status_code, close_msg):
        print(f"[INFO]: Websocket connection closed: {close_status_code}: {close_msg}")
        self.recording = False
        self.server_error = False
        self.waiting = False

    def on_open(self, ws):
        """
        Callback function called when the WebSocket connection is successfully opened.

        Sends an initial configuration message to the server, including client UID,
        language selection, and task type.

        Args:
            ws (websocket.WebSocketApp): The WebSocket client instance.

        """
        print("[INFO]: Opened connection")
        ws.send(
            json.dumps(
                {
                    "uid": self.uid,
                    "language": self.language,
                    "task": self.task,
                    "model": self.model,
                    "use_vad": self.use_vad
                }
            )
        )

    @staticmethod
    def bytes_to_float_array(audio_bytes):
        """
        Convert audio data from bytes to a NumPy float array.

        It assumes that the audio data is in 16-bit PCM format. The audio data is normalized to
        have values between -1 and 1.

        Args:
            audio_bytes (bytes): Audio data in bytes.

        Returns:
            np.ndarray: A NumPy array containing the audio data as float values normalized between -1 and 1.
        """
        raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
        return raw_data.astype(np.float32) / 32768.0

    def send_packet_to_server(self, message):
        """
        Send an audio packet to the server using WebSocket.

        Args:
            message (bytes): The audio data packet in bytes to be sent to the server.

        """
        try:
            self.client_socket.send(message, websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            print(e)

    def play_file(self, filename):
        """
        Play an audio file and send it to the server for processing.

        Reads an audio file, plays it through the audio output, and simultaneously sends
        the audio data to the server for processing. It uses PyAudio to create an audio
        stream for playback. The audio data is read from the file in chunks, converted to
        floating-point format, and sent to the server using WebSocket communication.
        This method is typically used when you want to process pre-recorded audio and send it
        to the server in real-time.

        Args:
            filename (str): The path to the audio file to be played and sent to the server.
        """

        # read audio and create pyaudio stream
        with wave.open(filename, "rb") as wavfile:
            self.stream = self.p.open(
                format=self.p.get_format_from_width(wavfile.getsampwidth()),
                channels=wavfile.getnchannels(),
                rate=wavfile.getframerate(),
                input=True,
                output=True,
                frames_per_buffer=self.chunk,
            )
            try:
                while self.recording:
                    data = wavfile.readframes(self.chunk)
                    if data == b"":
                        break

                    audio_array = self.bytes_to_float_array(data)
                    self.send_packet_to_server(audio_array.tobytes())
                    self.stream.write(data)

                wavfile.close()

                assert self.last_response_recieved
                while time.time() - self.last_response_recieved < self.disconnect_if_no_response_for:
                    continue
                self.send_packet_to_server(Client.END_OF_AUDIO.encode('utf-8'))
                if self.server_backend == "faster_whisper":
                    self.write_srt_file(self.srt_file_path)
                self.stream.close()
                self.close_websocket()

            except KeyboardInterrupt:
                wavfile.close()
                self.stream.stop_stream()
                self.stream.close()
                self.p.terminate()
                self.close_websocket()
                if self.server_backend == "faster_whisper":
                    self.write_srt_file(self.srt_file_path)
                print("[INFO]: Keyboard interrupt.")

    def close_websocket(self):
        """
        Close the WebSocket connection and join the WebSocket thread.

        First attempts to close the WebSocket connection using `self.client_socket.close()`. After
        closing the connection, it joins the WebSocket thread to ensure proper termination.

        """
        try:
            self.client_socket.close()
        except Exception as e:
            print("[ERROR]: Error closing WebSocket:", e)

        try:
            self.ws_thread.join()
        except Exception as e:
            print("[ERROR:] Error joining WebSocket thread:", e)

    def get_client_socket(self):
        """
        Get the WebSocket client socket instance.

        Returns:
            WebSocketApp: The WebSocket client socket instance currently in use by the client.
        """
        return self.client_socket

    def write_audio_frames_to_file(self, frames, file_name):
        """
        Write audio frames to a WAV file.

        The WAV file is created or overwritten with the specified name. The audio frames should be
        in the correct format and match the specified channel, sample width, and sample rate.

        Args:
            frames (bytes): The audio frames to be written to the file.
            file_name (str): The name of the WAV file to which the frames will be written.

        """
        with wave.open(file_name, "wb") as wavfile:
            wavfile: wave.Wave_write
            wavfile.setnchannels(self.channels)
            wavfile.setsampwidth(2)
            wavfile.setframerate(self.rate)
            wavfile.writeframes(frames)

    def process_hls_stream(self, hls_url):
        """
        Connect to an HLS source, process the audio stream, and send it for transcription.

        Args:
            hls_url (str): The URL of the HLS stream source.
        """
        print("[INFO]: Connecting to HLS stream...")
        process = None  # Initialize process to None

        try:
            # Connecting to the HLS stream using ffmpeg-python
            process = (
                ffmpeg
                .input(hls_url, threads=0)
                .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=self.rate)
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )

            # Process the stream
            while True:
                in_bytes = process.stdout.read(self.chunk * 2)  # 2 bytes per sample
                if not in_bytes:
                    break
                audio_array = self.bytes_to_float_array(in_bytes)
                self.send_packet_to_server(audio_array.tobytes())

        except Exception as e:
            print(f"[ERROR]: Failed to connect to HLS stream: {e}")
        finally:
            if process:
                process.kill()

        print("[INFO]: HLS stream processing finished.")

    def record(self, out_file="output_recording.wav"):
        """
        Record audio data from the input stream and save it to a WAV file.

        Continuously records audio data from the input stream, sends it to the server via a WebSocket
        connection, and simultaneously saves it to multiple WAV files in chunks. It stops recording when
        the `RECORD_SECONDS` duration is reached or when the `RECORDING` flag is set to `False`.

        Audio data is saved in chunks to the "chunks" directory. Each chunk is saved as a separate WAV file.
        The recording will continue until the specified duration is reached or until the `RECORDING` flag is set to `False`.
        The recording process can be interrupted by sending a KeyboardInterrupt (e.g., pressing Ctrl+C). After recording,
        the method combines all the saved audio chunks into the specified `out_file`.

        Args:
            out_file (str, optional): The name of the output WAV file to save the entire recording.
                                      Default is "output_recording.wav".

        """
        n_audio_file = 0
        if not os.path.exists("chunks"):
            os.makedirs("chunks", exist_ok=True)
        try:
            for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                if not self.recording:
                    break
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames += data

                audio_array = Client.bytes_to_float_array(data)

                self.send_packet_to_server(audio_array.tobytes())

                # save frames if more than a minute
                if len(self.frames) > 60 * self.rate:
                    t = threading.Thread(
                        target=self.write_audio_frames_to_file,
                        args=(
                            self.frames[:],
                            f"chunks/{n_audio_file}.wav",
                        ),
                    )
                    t.start()
                    n_audio_file += 1
                    self.frames = b""
            if self.server_backend == "faster_whisper":
                self.write_srt_file(self.srt_file_path)

        except KeyboardInterrupt:
            if len(self.frames):
                self.write_audio_frames_to_file(
                    self.frames[:], f"chunks/{n_audio_file}.wav"
                )
                n_audio_file += 1
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            self.close_websocket()

            self.write_output_recording(n_audio_file, out_file)
            if self.server_backend == "faster_whisper":
                self.write_srt_file(self.srt_file_path)

    def write_output_recording(self, n_audio_file, out_file):
        """
        Combine and save recorded audio chunks into a single WAV file.

        The individual audio chunk files are expected to be located in the "chunks" directory. Reads each chunk
        file, appends its audio data to the final recording, and then deletes the chunk file. After combining
        and saving, the final recording is stored in the specified `out_file`.


        Args:
            n_audio_file (int): The number of audio chunk files to combine.
            out_file (str): The name of the output WAV file to save the final recording.

        """
        input_files = [
            f"chunks/{i}.wav"
            for i in range(n_audio_file)
            if os.path.exists(f"chunks/{i}.wav")
        ]
        with wave.open(out_file, "wb") as wavfile:
            wavfile: wave.Wave_write
            wavfile.setnchannels(self.channels)
            wavfile.setsampwidth(2)
            wavfile.setframerate(self.rate)
            for in_file in input_files:
                with wave.open(in_file, "rb") as wav_in:
                    while True:
                        data = wav_in.readframes(self.chunk)
                        if data == b"":
                            break
                        wavfile.writeframes(data)
                # remove this file
                os.remove(in_file)
        wavfile.close()

    def write_srt_file(self, output_path="output.srt"):
        self.transcript.append(self.last_segment)
        utils.create_srt_file(self.transcript, output_path)


class TranscriptionClient:
    """
    Client for handling audio transcription tasks via a WebSocket connection.

    Acts as a high-level client for audio transcription tasks using a WebSocket connection. It can be used
    to send audio data for transcription to a server and receive transcribed text segments.

    Args:
        host (str): The hostname or IP address of the server.
        port (int): The port number to connect to on the server.
        lang (str, optional): The primary language for transcription. Default is None, which defaults to English ('en').
        translate (bool, optional): Indicates whether translation tasks are required (default is False).

    Attributes:
        client (Client): An instance of the underlying Client class responsible for handling the WebSocket connection.

    Example:
        To create a TranscriptionClient and start transcription on microphone audio:
        ```python
        transcription_client = TranscriptionClient(host="localhost", port=9090)
        transcription_client()
        ```
    """
    def __init__(self, host, port, lang=None, translate=False, model="small", use_vad=True):
        self.client = Client(host, port, lang, translate, model, srt_file_path="output.srt", use_vad=use_vad)

    def __call__(self, audio=None, hls_url=None):
        """
        Start the transcription process.

        Initiates the transcription process by connecting to the server via a WebSocket. It waits for the server
        to be ready to receive audio data and then sends audio for transcription. If an audio file is provided, it
        will be played and streamed to the server; otherwise, it will perform live recording.

        Args:
            audio (str, optional): Path to an audio file for transcription. Default is None, which triggers live recording.

        """
        print("[INFO]: Waiting for server ready ...")
        while not self.client.recording:
            if self.client.waiting or self.client.server_error:
                self.client.close_websocket()
                return

        print("[INFO]: Server Ready!")
        if hls_url is not None:
            self.client.process_hls_stream(hls_url)
        elif audio is not None:
            resampled_file = utils.resample(audio)
            self.client.play_file(resampled_file)
        else:
            self.client.record()
