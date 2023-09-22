import os
import argparse
import wave

import numpy as np
import scipy
import ffmpeg
import pyaudio
import threading
import textwrap
import json
import websocket
import uuid
import time


def resample(file: str, sr: int = 16000):
    """
    # https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/audio.py#L22
    Open an audio file and read as mono waveform, resampling as necessary,
    save the resampled audio

    Args:
        file (str): The audio file to open
        sr (int): The sample rate to resample the audio if necessary
    
    Returns:
        resampled_file (str): The resampled audio file
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    np_buffer = np.frombuffer(out, dtype=np.int16)

    resampled_file = f"{file.split('.')[0]}_resampled.wav"
    scipy.io.wavfile.write(resampled_file, sr, np_buffer.astype(np.int16))
    return resampled_file


class Client:
    """
    Represents a client for audio recording and streaming to a server using WebSocket communication.

    This class allows audio recording from the microphone or playing audio from a file while streaming it to
    a server for transcription or translation. It uses PyAudio for audio recording and playback and WebSocket
    for communication with the server.

    Attributes:
        CHUNK (int): The size of audio chunks for recording and playback.
        FORMAT: The audio format used by PyAudio (paInt16 for 16-bit PCM).
        CHANNELS (int): The number of audio channels (1 for mono).
        RATE (int): The audio sampling rate in Hz (samples per second).
        RECORD_SECONDS (int): The maximum duration for audio recording in seconds.
        RECORDING (bool): Indicates whether recording is currently active.
        multilingual (bool): Indicates if multilingual transcription is enabled.
        language (str): The selected language for transcription.
        task (str): The transcription or translation task to be performed.
        uid (str): A unique identifier for the client.
        WAITING (bool): Indicates if the client is waiting for server availability.
        LAST_RESPONSE_RECIEVED (float): Timestamp of the last response received from the server.
        DISCONNECT_IF_NO_RESPONSE_FOR (int): Maximum time without server response before disconnection.

    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 60000
    RECORDING = False
    multilingual = False
    language = None
    task = "transcribe"
    uid = str(uuid.uuid4())
    WAITING = False
    LAST_RESPONSE_RECIEVED = None
    DISCONNECT_IF_NO_RESPONSE_FOR = 15
    
    def __init__(self, host=None, port=None, is_multilingual=False, lang=None, translate=False):
        """
        Initializes a Client instance for audio recording and streaming to a server.

        If host and port are not provided, the WebSocket connection will not be established.
        When translate is True, the task will be set to "translate" instead of "transcribe".
        he audio recording starts immediately upon initialization.

        Args:
            host (str): The hostname or IP address of the server.
            port (int): The port number for the WebSocket server.
            is_multilingual (bool, optional): Specifies if multilingual transcription is enabled. Default is False.
            lang (str, optional): The selected language for transcription when multilingual is disabled. Default is None.
            translate (bool, optional): Specifies if the task is translation. Default is False.

        Attributes:
            timestamp_offset (float): A timestamp offset for tracking audio timing.
            audio_bytes (bytes): A buffer for storing audio data.
            p (pyaudio.PyAudio): An instance of PyAudio for audio streaming.
            stream (pyaudio.Stream): The audio stream for recording.
            client_socket (websocket.WebSocketApp): The WebSocket client for server communication.
            ws_thread (threading.Thread): A thread for running the WebSocket client.
            frames (bytes): A buffer for accumulating audio frames.

        """
        Client.multilingual = is_multilingual
        Client.language = lang if is_multilingual else "en"
        if translate:
            Client.task = "translate"

        self.timestamp_offset = 0.0
        self.audio_bytes = None
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)

        if host is not None and port is not None:
            socket_url = f"ws://{host}:{port}"    
            self.client_socket = websocket.WebSocketApp(socket_url,
                                  on_open=Client.on_open,
                                  on_message=Client.on_message,
                                  on_error=Client.on_error,
                                  on_close=Client.on_close)
        else:
            print("[ERROR]: No host or port specified.")
            return

        self.ws_thread = threading.Thread(target=self.client_socket.run_forever)
        self.ws_thread.setDaemon(True)
        self.ws_thread.start()

        self.frames = b""
        print("[INFO]: * recording")
    
    @staticmethod
    def on_message(ws, message):
        """
        Callback function called when a message is received from the server.
        
        It updates various attributes of the client based on the received message, including
        recording status, language detection, and server messages. If a disconnect message
        is received, it sets the recording status to False.

        Args:
            ws (websocket.WebSocketApp): The WebSocket client instance.
            message (str): The received message from the server.

        """
        Client.LAST_RESPONSE_RECIEVED = time.time()
        message = json.loads(message)
        if message.get('uid')!=Client.uid:
            print("[ERROR]: invalid client uid")
            return
        
        if "status" in message.keys() and  message["status"] == "WAIT":
            Client.WAITING = True
            print(f"[INFO]:Server is full. Estimated wait time {round(message['message'])} minutes.")
        
        if "message" in message.keys() and message["message"] == "DISCONNECT":
            print("[INFO]: Server overtime disconnected.")
            Client.RECORDING = False

        if "message" in message.keys() and message["message"] == "SERVER_READY":
            Client.RECORDING = True
            return

        if "language" in message.keys():
            Client.language = message.get("language")
            lang_prob = message.get("language_prob")
            print(f"[INFO]: Server detected language {Client.language} with probability {lang_prob}")
            return

        if "segments" not in message.keys():
            return
    
        message = message["segments"]
        text = []
        if len(message):
            for seg in message:
                if len(text):
                    if text[-1] != seg["text"]:
                        text.append(seg["text"])
                else:
                    text.append(seg["text"])
        if len(text) > 3:
            text = text[-3:]
        wrapper = textwrap.TextWrapper(width=60)
        word_list = wrapper.wrap(text="".join(text))
        # Print each line.
        if os.name=='nt':
            os.system('cls')
        else:
            os.system('clear')
        for element in word_list:
            print(element)

    @staticmethod
    def on_error(ws, error):
        print(error)

    @staticmethod
    def on_close(ws, close_status_code, close_msg):
        print(f"[INFO]: Websocket connection closed.")

    @staticmethod
    def on_open(ws):
        """
        Callback function called when the WebSocket connection is successfully opened.
        
        Sends an initial configuration message to the server, including client UID, multilingual mode,
        language selection, and task type.

        Args:
            ws (websocket.WebSocketApp): The WebSocket client instance.

        """
        print(Client.multilingual, Client.language, Client.task)

        print("[INFO]: Opened connection")
        ws.send(json.dumps({
            'uid': Client.uid,
            'multilingual': Client.multilingual,
            'language': Client.language,
            'task': Client.task
        }))

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
        raw_data = np.frombuffer(
            buffer=audio_bytes, dtype=np.int16
        )
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
        self.wf = wave.open(filename, 'rb')
        self.stream = self.p.open(format=self.p.get_format_from_width(self.wf.getsampwidth()),
                channels=self.wf.getnchannels(),
                rate=self.wf.getframerate(),
                input=True,
                output=True,
                frames_per_buffer=self.CHUNK)
        try:
            while Client.RECORDING:
                data = self.wf.readframes(self.CHUNK)
                if data==b'': break

                audio_array = Client.bytes_to_float_array(data)
                self.send_packet_to_server(audio_array.tobytes())
                self.stream.write(data)

            self.wf.close()
            elapsed_time = time.time() - self.LAST_RESPONSE_RECIEVED
            while elapsed_time < self.DISCONNECT_IF_NO_RESPONSE_FOR:
                continue
            self.stream.close()
            self.close_websocket()

        except KeyboardInterrupt:
            self.wf.close()
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            self.close_websocket()
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
        wf = wave.open(file_name, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(self.RATE)
        wf.writeframes(frames)
        wf.close()

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
            out_file (str, optional): The name of the output WAV file to save the entire recording. Default is "output_recording.wav".

        """
        n_audio_file = 0
        if not os.path.exists("chunks"):
            os.makedirs("chunks", exist_ok=True)
        try:
            for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                if not Client.RECORDING: break
                data = self.stream.read(self.CHUNK)
                self.frames += data

                audio_array = Client.bytes_to_float_array(data)
                
                self.send_packet_to_server(audio_array.tobytes())

                # save frames if more than a minute
                if len(self.frames) > 60*self.RATE:
                    t = threading.Thread(
                        target=self.write_audio_frames_to_file,
                        args=(self.frames[:], f"chunks/{n_audio_file}.wav", )
                    )
                    t.start()
                    n_audio_file += 1
                    self.frames = b""

        except KeyboardInterrupt:
            if len(self.frames):
                self.write_audio_frames_to_file(
                    self.frames[:], f"chunks/{n_audio_file}.wav")
                n_audio_file += 1
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            self.close_websocket()

            self.write_output_recording(n_audio_file, out_file)
    
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
        input_files = [f"chunks/{i}.wav" for i in range(n_audio_file) if os.path.exists(f"chunks/{i}.wav")]
        wf = wave.open(out_file, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(self.RATE)
        for in_file in input_files:
            w = wave.open(in_file, 'rb')
            while True:
                data = w.readframes(self.CHUNK)
                if data==b'': break
                wf.writeframes(data)
            w.close()
            # remove this file
            os.remove(in_file)
        wf.close()


class TranscriptionClient:
    """
    Client for handling audio transcription tasks via a WebSocket connection.

    Acts as a high-level client for audio transcription tasks using a WebSocket connection. It can be used
    to send audio data for transcription to a server and receive transcribed text segments.

    Args:
        host (str): The hostname or IP address of the server.
        port (int): The port number to connect to on the server.
        is_multilingual (bool, optional): Indicates whether the transcription should support multiple languages (default is False).
        lang (str, optional): The primary language for transcription (used if `is_multilingual` is False). Default is None, which defaults to English ('en').
        translate (bool, optional): Indicates whether translation tasks are required (default is False).

    Attributes:
        client (Client): An instance of the underlying Client class responsible for handling the WebSocket connection.

    Example:
        To create a TranscriptionClient and start transcription on microphone audio:
        ```python
        transcription_client = TranscriptionClient(host="localhost", port=9090, is_multilingual=True)
        transcription_client()
        ```
    """
    def __init__(self, host, port, is_multilingual=False, lang=None, translate=False):
        self.client = Client(host, port, is_multilingual, lang, translate)
        
    def __call__(self, audio=None):
        """
        Start the transcription process.

        Initiates the transcription process by connecting to the server via a WebSocket. It waits for the server
        to be ready to receive audio data and then sends audio for transcription. If an audio file is provided, it 
        will be played and streamed to the server; otherwise, it will perform live recording.

        Args:
            audio (str, optional): Path to an audio file for transcription. Default is None, which triggers live recording.
                   
        """
        print("[INFO]: Waiting for server ready ...")
        while not Client.RECORDING:
            if Client.WAITING:
                self.client.close_websocket()
                return
            pass
        print("[INFO]: Server Ready!")
        if audio is not None:
            resampled_file = resample(audio)
            self.client.play_file(resampled_file)
        else:
            self.client.record()

