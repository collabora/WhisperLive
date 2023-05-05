import io
import os
import argparse
import wave
import uuid
import hashlib
import base64
import time

import numpy as np
import scipy
import ffmpeg
import torch
import socket, pickle, pyaudio, struct
import threading
import textwrap
import json
import torchaudio
from dataclasses import dataclass

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 60000
all_segments = []


@dataclass(frozen=True)
class Constants:
    ACK = b"acknowledged"
    RECORDING_OVER = b"audio_data_over"
    RECEIVED_AUDIO_FILE = b"audio_file_sent"
    RECEIVING_AUDIO_FILE = b"sending_audio_file"

class Client:
    def __init__(self, topic=None, host=None, port=None):
        self.timestamp_offset = 0.0
        self.audio_bytes = None
        self.p = pyaudio.PyAudio()
        self.payload_size = struct.calcsize("Q")
        self.stream = self.p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        print(self.p.get_sample_size(FORMAT))
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_ip = 'localhost' if host is None else host
        port = 5901 if port is None else port

        socket_address = (host_ip, port)
        self.client_socket.connect(socket_address)
        print("CLIENT CONNECTED TO", socket_address)

        # voice activity detection model
        self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=True)
        self.window_size = 1024
        self.vad_threshold = 0.4

        # subscribing to the correct topic
        if topic is not None:
            self.topic = topic
        else:
            self.topic = self.get_mac_address().decode()

        self.frames = b""
        data = b""
        while True:
            while len(data) < self.payload_size:
                packet = self.client_socket.recv(4*1024)  #4K
                if not packet: break
                data+=packet
            packed_msg_size = data[:self.payload_size]
            data = data[self.payload_size:]
            try:
                msg_size = struct.unpack("Q",packed_msg_size)[0]
            except struct.error:
                break
            while len(data) < msg_size:
                data += self.client_socket.recv(4*1024)
            frame_data = data[:msg_size]
            frame_data = pickle.loads(frame_data)
            if Constants.ACK in frame_data:
                print("Server is ready. Sending audio ...")
                break
        print("* recording")

    def send_packet_to_server(self, message):
        a = pickle.dumps(message)
        message = struct.pack("Q",len(a))+a
        self.client_socket.sendall(message)

    def get_mac_address(self):
        mac = hex(uuid.getnode())
        hasher = hashlib.sha1(mac.encode())
        return base64.urlsafe_b64encode(hasher.digest()[:5])

    @staticmethod
    def bytes_to_audio_tensor(audio_bytes):
        bytes_io = io.BytesIO()
        raw_data = np.frombuffer(
            buffer=audio_bytes, dtype=np.int16
        )
        scipy.io.wavfile.write(bytes_io, RATE, raw_data)
        audio, _ = torchaudio.load(bytes_io)
        return audio.squeeze(0)
    
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

                # voice activity detection
                chunk_tensor = Client.bytes_to_audio_tensor(data)
                try:
                    speech_prob = self.vad_model(chunk_tensor, RATE).item()
                except ValueError:
                    break   # input audio chunk is too short
                if speech_prob > self.vad_threshold:
                    data_dict = {
                        "topic": self.topic,
                        "audio": data
                    }
                    self.send_packet_to_server(data_dict)
                self.stream.write(data)

            self.wf.close()
            self.stream.close()

            # let the server know that we're done
            data = Constants.RECORDING_OVER
            self.send_packet_to_server(data)
            with open("results.json", "w") as f:
                json_dict = json.dumps(all_segments, indent=2)
                f.write(json_dict)

        except KeyboardInterrupt:
            # write all segments to a file
            with open("results.json", "w") as f:
                json_dict = json.dumps(all_segments, indent=2)
                f.write(json_dict)


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

                # voice activity detection
                chunk_tensor = Client.bytes_to_audio_tensor(data)
                
                speech_prob = self.vad_model(chunk_tensor, RATE).item()
                if speech_prob > self.vad_threshold:
                    data_dict = {
                        "topic": self.topic,
                        "audio": data
                    }
                    self.send_packet_to_server(data_dict)

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

            # let the server know that we're done
            data = Constants.RECORDING_OVER
            self.send_packet_to_server(data)

            # combine all the audio files
            self.write_output_recording(n_audio_file, out_file)
            # write all segments to a file
            with open("results.json", "w") as f:
                json_dict = json.dumps(all_segments, indent=2)
                f.write(json_dict)
    
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


def recieve_response(client_socket):
    data = b""
    payload_size = struct.calcsize("Q")
    
    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024) # 4K
            if not packet: break
            data+=packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        try:
            msg_size = struct.unpack("Q",packed_msg_size)[0]
        except struct.error:
            break
        while len(data) < msg_size:
            data += client_socket.recv(4*1024)
        frame_data = data[:msg_size]
        data  = data[msg_size:]
        response = pickle.loads(frame_data)

        if response is not None and isinstance(response, dict):
            os.system('clear')
            text = response['text']
            segments = response['segments']
            if len(segments):
                for seg in segments:
                    all_segments.append(seg)
            wrapper = textwrap.TextWrapper(width=50)
            word_list = wrapper.wrap(text=text)
            # Print each line.
            for element in word_list:
                print(element)


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
    parser.add_argument('--topic', default=None, type=str, help='topic to subscribe for results')
    parser.add_argument('--host', default=None, type=str, help='server address to connect to')
    parser.add_argument('--port', default=None, type=str, help='server port to connect to')
    opt = parser.parse_args()
    c = Client(topic=opt.topic, host=opt.host, port=opt.port)
    while True:
        if c.get_client_socket() is not None:
            break
    client_socket = c.get_client_socket()
    t2 = threading.Thread(target=recieve_response, args=(client_socket, ))
    t2.start()
    if opt.audio is not None:
        resampled_file = resample(opt.audio)
        c.play_file(resampled_file)
    else:
        c.record()
    t2.join()
