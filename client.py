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


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 60000
START_RECORDING = False
multilingual = False
language = None



def on_message(ws, message):
    global START_RECORDING, language
    message = json.loads(message)
    if message == "SERVER_READY":
        START_RECORDING = True
        return

    if isinstance(message, dict):
        language = message.get("language")
        lang_prob = message.get("language_prob")
        print(f"Server detected language {language} with probability {lang_prob}")
        return

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

def on_error(ws, error):
    print(error)

def on_close(ws, close_status_code, close_msg):
    print("### websocket connection closed ###")

def on_open(ws):
    global multilingual, language, task
    print(multilingual, language, task)

    print("Opened connection")
    ws.send(json.dumps({
        'multilingual': multilingual[0],
        'language': language[0],
        'task': task
    }))


class Client:
    def __init__(self, host=None, port=None):
        self.timestamp_offset = 0.0
        self.audio_bytes = None
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        print(self.p.get_sample_size(FORMAT))

        # create websocket connection
        if host is not None and port is not None:
            socket_url = f"ws://{host}:{port}"    
            self.client_socket = websocket.WebSocketApp(socket_url,
                                  on_open=on_open,
                                  on_message=on_message,
                                  on_error=on_error,
                                  on_close=on_close)
        else:
            print("No host or port specified.")
            return

        # start websocket client in a thread
        self.ws_thread = threading.Thread(target=self.client_socket.run_forever)
        self.ws_thread.setDaemon(True)
        self.ws_thread.start()

        self.frames = b""
        print("* recording")

    def send_packet_to_server(self, message):
        try:
            self.client_socket.send(message, websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            print(e)

    @staticmethod
    def bytes_to_float_array(audio_bytes):
        raw_data = np.frombuffer(
            buffer=audio_bytes, dtype=np.int16
        )
        return raw_data.astype(np.float32) / 32768.0
    
    def play_file(self, filename):
        # read audio and create pyaudio stream
        self.wf = wave.open(filename, 'rb')
        self.stream = self.p.open(format=self.p.get_format_from_width(self.wf.getsampwidth()),
                channels=self.wf.getnchannels(),
                rate=self.wf.getframerate(),
                input=True,
                output=True,
                frames_per_buffer=CHUNK)
        try:
            while True:
                data = self.wf.readframes(CHUNK)
                if data==b'': break

                audio_array = Client.bytes_to_float_array(data)
                self.send_packet_to_server(audio_array.tobytes())
                self.stream.write(data)

            self.wf.close()
            self.stream.close()

        except KeyboardInterrupt:
            print("Keyboard interrupt.")


    def get_client_socket(self):
        return self.client_socket
    
    def write_audio_frames_to_file(self, frames, file_name):
        wf = wave.open(file_name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(frames)
        wf.close()

    def record(self, out_file="output_recording.wav"):
        n_audio_file = 0
        # create dir for saving audio chunks
        if not os.path.exists("chunks"):
            os.makedirs("chunks", exist_ok=True)
        try:
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = self.stream.read(CHUNK)
                self.frames += data

                audio_array = Client.bytes_to_float_array(data)
                
                self.send_packet_to_server(audio_array.tobytes())

                # save frames if more than a minute
                if len(self.frames) > 60*RATE:
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


            # combine all the audio files
            self.write_output_recording(n_audio_file, out_file)
    
    def write_output_recording(self, n_audio_file, out_file):
        input_files = [f"chunks/{i}.wav" for i in range(n_audio_file) if os.path.exists(f"chunks/{i}.wav")]
        wf = wave.open(out_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        for in_file in input_files:
            w = wave.open(in_file, 'rb')
            while True:
                data = w.readframes(CHUNK)
                if data==b'': break
                wf.writeframes(data)
            w.close()
            # remove this file
            os.remove(in_file)
        wf.close()


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


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, help='audio file to transcribe')
    parser.add_argument('--host', default=None, type=str, help='websocket server address to connect to')
    parser.add_argument('--port', default=None, type=str, help='websocket server port to connect to')
    parser.add_argument('--multilingual', action="store_true", help='use multilingual model')
    parser.add_argument('--language', default=None, type=str, help='languages to use')
    parser.add_argument(
        '--task', default="transcribe", type=str, help='task transcribe/translate (translates from any to english)')
    opt = parser.parse_args()
    print(opt)
    multilingual=opt.multilingual,
    language = opt.language if opt.multilingual else "en",
    task = opt.task
    c = Client(host=opt.host, port=opt.port)

    # while loop to wait for server to be ready
    print("Waiting for server ready ...")
    while not START_RECORDING:
        pass
    print("Server Ready!")

    if opt.audio is not None:
        resampled_file = resample(opt.audio)
        c.play_file(resampled_file)
    else:
        c.record()
