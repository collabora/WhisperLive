import socket, pickle, struct, time, pyaudio
import threading
import os
import wave
import textwrap
from collections import deque
from dataclasses import dataclass

import torch
import numpy as np
import paho.mqtt.client as mqtt

from transcriber import WhisperModel


def on_connect(mqttc, obj, flags, rc):
    pass

def on_message(mqttc, obj, msg):
    pass

def on_publish(mqttc, obj, mid):
    pass

def on_subscribe(mqttc, obj, mid, granted_qos):
    pass

def on_log(mqttc, obj, level, string):
    pass


@dataclass(frozen=True)
class Constants:
    AUDIO_OVER = b"audio_data_over"
    ACK = b"acknowledged"
    SENDING_FILE = b"sending_audio_file"
    FILE_SENT = b"audio_file_sent"


class ServeClient:
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    def __init__(self, client_socket, device=None, verbose=True):
        self.payload_size = struct.calcsize("Q")
        self.data = b""
        self.frames = b""
        self.frames_np = None
        self.transcriber = WhisperModel("medium.en", device="cuda", compute_type="float16")
        self.timestamp_offset = 0.0
        self.frames_offset = 0.0
        self.text = []
        self.current_out = ''
        self.prev_out = ''
        self.t_start=None
        self.client_socket = client_socket
        self.verbose = verbose
        self.exit = False
        self.same_output_threshold = 0
        self.show_prev_out_thresh = 5   # if pause(no output from whisper) show previous output for 5 seconds
        self.add_pause_thresh = 3       # add a blank to segment list as a pause(no speech) for 3 seconds

        # text formatting
        self.wrapper = textwrap.TextWrapper(width=50)
        self.pick_previous_segments = 2

        # setup mqtt
        self.topic = None
        self.mqttc = mqtt.Client()
        self.mqttc.on_message = on_message
        self.mqttc.on_connect = on_connect
        self.mqttc.on_publish = on_publish
        self.mqttc.on_subscribe = on_subscribe
        self.mqttc = mqtt.Client()
        self.mqttc.connect("mqtt.kurg.org", 1883, 60)
        self.mqttc.loop_start()

        # send response to client; server is ready
        self.send_response_to_client(Constants.ACK)

        # threading
        self.recv_thread = threading.Thread(target=self.recv_audio)
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.recv_thread.start()
        self.trans_thread.start()

    def recv_audio(self):
        """
        Receive audio chunks from client in an infinite loop.
        """
        if self.client_socket:
            try:
                while True:
                    while len(self.data) < self.payload_size:
                        packet = self.client_socket.recv(4*1024) # 4K
                        if not packet: break
                        self.data+=packet

                    packed_msg_size = self.data[:self.payload_size]
                    self.data = self.data[self.payload_size:]
                    msg_size = struct.unpack("Q",packed_msg_size)[0]
                    
                    while len(self.data) < msg_size:
                        self.data += self.client_socket.recv(4*1024)
                    frame_data = self.data[:msg_size]
                    self.data  = self.data[msg_size:]
                    frame_data = pickle.loads(frame_data)
                    if self.topic is None:
                        self.topic = frame_data["topic"]
                    
                    frame = frame_data["audio"]
                    
                    # client says audio over
                    if Constants.AUDIO_OVER in frame:
                        break
                    
                    frame_np = np.frombuffer(frame, dtype=np.int16)
                    if self.frames_np is not None and self.frames_np.shape[0] > 60*self.RATE:
                        self.frames_offset += 45.0
                        self.frames_np = self.frames_np[int(45*self.RATE):]
                    
                    if self.frames_np is None:
                        self.frames_np = frame_np.copy()
                    else:
                        self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)

                # set frames np to None so to stop translation for this client
                self.frames_np = None
                self.exit = True
            except Exception as e:
                if self.verbose: print(f"[ERROR]: {e}")
                self.exit = True
    
    def send_response_to_client(self, message):
        """
        Send serialized response to client.
        """
        a = pickle.dumps(message)
        message = struct.pack("Q",len(a))+a
        self.client_socket.sendall(message)
    
    def fill_output(self, output):
        """
        Format output with current and previous complete segments
        into two lines of 50 characters.

        Args:
            output(str): current incomplete segment
        
        Returns:
            transcription wrapped in two lines
        """
        text = ''
        pick_prev = min(len(self.text), self.pick_previous_segments)
        for seg in self.text[-pick_prev:]:
            # discard everything before a 3 second pause
            if seg == '':
                text = ''
            else:
                text += seg
        wrapped = self.wrapper.wrap(
            text="".join(text + output))[-2:]
        return " ".join(wrapped)

    def speech_to_text(self):
        """
        Process audio stream in an infinite loop.
        """
        while True:
            if self.exit: 
                self.mqttc.disconnect()
                self.client_socket.close()
                self.transcriber.destroy()
                break

            if self.frames_np is None: continue

            # clip audio if the current chunk exceeds 25 seconds, this basically implies that
            # no valid segment for the last 25 seconds from whisper
            if self.frames_np[int((self.timestamp_offset - self.frames_offset)*self.RATE):].shape[0] > 25 * self.RATE:
                duration = self.frames_np.shape[0] / self.RATE
                self.timestamp_offset = self.frames_offset + duration - 5
            
            # add 200 ms from the last chunk if available
            if len(self.text) and self.frames_np[:-int((self.timestamp_offset - self.frames_offset)*self.RATE)].shape[0]:
                samples_take = max(0, (self.timestamp_offset - self.frames_offset)*self.RATE - 0.2*self.RATE)
            else:
                samples_take = max(0, (self.timestamp_offset - self.frames_offset)*self.RATE)
            input_bytes = self.frames_np[int(samples_take):].copy()
            duration = input_bytes.shape[0] / self.RATE
            if duration<1.0: continue

            try:
                input_sample = input_bytes.astype(np.float32) / 32768.0
                # set previous complete segment as initial prompt
                if len(self.text) and self.text[-1] != '': 
                    initial_prompt = self.text[-1]
                else: 
                    initial_prompt = None

                # whisper transcribe with prompt
                result = self.transcriber.transcribe(input_sample, initial_prompt=initial_prompt)
                if len(result):
                    self.t_start = None
                    output, segments = self.update_segments(result, duration)
                    out_dict = {
                        'text': output,
                        'segments': segments
                    }
                    if self.topic is not None:
                        self.mqttc.publish(self.topic, payload=str(out_dict))
                    self.send_response_to_client(out_dict)
                else:
                    # show previous output if there is pause i.e. no output from whisper
                    output = ''
                    if self.t_start is None: self.t_start = time.time()

                    if time.time() - self.t_start < self.show_prev_out_thresh:
                        output = self.fill_output('')

                    # add a blank if there is no speech for 3 seconds
                    if len(self.text) and self.text[-1] != '':
                        if time.time() - self.t_start > self.add_pause_thresh:
                            self.text.append('')

                    # publish outputs
                    out_dict = {
                        'text': output,
                        'segments': []
                    }
                    if self.topic is not None:
                        self.mqttc.publish(self.topic, payload=str(out_dict))
                    self.send_response_to_client(out_dict)
            except Exception as e:
                if self.verbose: print(f"[ERROR]: {e}")
                time.sleep(0.01)
    
    def update_segments(self, segments, duration):
        """
        Processes the segments from whisper. Appends all the segments to the list
        except for the last segment assuming that it is incomplete.

        Args:
            segments(dict) : dictionary of segments as returned by whisper
            duration(float): duration of the current chunk
        
        Returns:
            transcription for the current chunk
        """
        offset = None
        transcript = []
        self.current_out = ''
        # process complete segments
        if len(segments) > 1:
            for i, s in enumerate(segments[:-1]):
                text_ = s.text
                self.text.append(text_)
                start, end = self.timestamp_offset + s.start, self.timestamp_offset + min(duration, s.end)
                transcript.append(
                    {
                        'start': start,
                        'end': end,
                        'text': text_
                    }
                )
                
                offset = min(duration, s.end)

        self.current_out += segments[-1].text
        
        # if same incomplete segment is seen multiple times then update the offset
        # and append the segment to the list
        if self.current_out.strip() == self.prev_out.strip() and self.current_out != '': 
            self.same_output_threshold += 1
        else: 
            self.same_output_threshold = 0
        
        if self.same_output_threshold > 5:
            if not len(self.text) or self.text[-1].strip().lower()!=self.current_out.strip().lower():          
                self.text.append(self.current_out)
                transcript.append(
                    {
                        'start': self.timestamp_offset,
                        'end': self.timestamp_offset + duration,
                        'text': self.current_out
                    }
                )
            self.current_out = ''
            offset = duration
            self.same_output_threshold = 0
        else:
            self.prev_out = self.current_out
        
        # update offset
        if offset is not None:
            self.timestamp_offset += offset

        # format and return output
        output = self.current_out
        return self.fill_output(output), transcript


if __name__=="__main__":
    # create socket
    server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host='127.0.0.1'
    port=5901
    backlog=5
    socket_address = (host, port)
    print('STARTING SERVER AT',socket_address,'...')
    server_socket.bind(socket_address)
    server_socket.listen(backlog)
    client_sockets = []
    device = 0
    try:
        while True:
            client_socket, addr = server_socket.accept()
            print('GOT CONNECTION FROM:', addr)
            client = ServeClient(client_socket, device=f'cuda:{device}')
            client_sockets.append(client_socket)
            print("waiting for new connection")
    except Exception as e:
        print(f"[ERROR main]: {e}")
        for sock in client_sockets:
            try:
                sock.close()
            except:
                pass

