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


def resample(file: str, sr: int = 16000):
    """
    # https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/audio.py#L22
    Open an audio file and read as mono waveform, resampling as necessary,
    save the resampled audio
    Parameters
    ----------
    file: str
        The audio file to open
    sr: int
        The sample rate to resample the audio if necessary
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
    
    def __init__(self, host=None, port=None, is_multilingual=False, lang=None, translate=False):
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

        # create websocket connection
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

        # start websocket client in a thread
        self.ws_thread = threading.Thread(target=self.client_socket.run_forever)
        self.ws_thread.setDaemon(True)
        self.ws_thread.start()

        self.frames = b""
        print("[INFO]: * recording")
    
    @staticmethod
    def on_message(ws, message):
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
        raw_data = np.frombuffer(
            buffer=audio_bytes, dtype=np.int16
        )
        return raw_data.astype(np.float32) / 32768.0
    
    def send_packet_to_server(self, message):
        try:
            self.client_socket.send(message, websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            print(e)
    
    def play_file(self, filename):
        # read audio and create pyaudio stream
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
            self.stream.close()

        except KeyboardInterrupt:
            self.wf.close()
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            self.close_websocket()
            print("[INFO]: Keyboard interrupt.")

    def close_websocket(self):
        try:
            self.client_socket.close()  # Close the WebSocket connection
        except Exception as e:
            print("[ERROR]: Error closing WebSocket:", e)

        try:
            self.ws_thread.join()  # Wait for the WebSocket thread to finish
        except Exception as e:
            print("[ERROR:] Error joining WebSocket thread:", e)

    def get_client_socket(self):
        return self.client_socket
    
    def write_audio_frames_to_file(self, frames, file_name):
        wf = wave.open(file_name, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(self.RATE)
        wf.writeframes(frames)
        wf.close()

    def record(self, out_file="output_recording.wav"):
        n_audio_file = 0
        # create dir for saving audio chunks
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

            # combine all the audio files
            self.write_output_recording(n_audio_file, out_file)
    
    def write_output_recording(self, n_audio_file, out_file):
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
    def __init__(self, host, port, is_multilingual=False, lang=None, translate=False):
        self.client = Client(host, port, is_multilingual, lang, translate)
        
    def __call__(self, audio=None):
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

