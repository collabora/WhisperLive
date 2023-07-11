import websockets
import pickle, struct, time, pyaudio
import threading
import os, json
import wave
import textwrap

import logging
logging.basicConfig(level = logging.INFO)

from collections import deque
from dataclasses import dataclass

import torch
import numpy as np
from websockets.sync import server
from websockets.sync.server import serve
from transcriber import WhisperModel


clients = {}
SERVER_READY = "SERVER_READY"

def recv_audio(websocket):
    """
    Receive audio chunks from client in an infinite loop.
    """
    global clients
    client = ServeClient(websocket)
    clients[websocket] = client
    while True:
        try:
            frame_data = websocket.recv()
            if isinstance(frame_data, str):
                logging.info(frame_data)
                continue
            frame_np = np.frombuffer(frame_data, np.float32)
            clients[websocket].add_frames(frame_np)
            
        except Exception as e:
            clients[websocket].cleanup()
            clients.pop(websocket)
            logging.info("Connection Closed.")
            break


class ServeClient:
    RATE = 16000
    def __init__(self, websocket, topic=None, device=None):
        self.payload_size = struct.calcsize("Q")
        self.data = b""
        self.frames = b""
        self.transcriber = WhisperModel(
            "small.en", 
            device="cuda",
            compute_type="float16", 
            local_files_only=False
        )
        
        # voice activity detection model
        self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=True,
                                           onnx=True
                                           )
        self.vad_threshold = 0.4
        
        self.timestamp_offset = 0.0
        self.frames_np = None
        self.frames_offset = 0.0
        self.text = []
        self.current_out = ''
        self.prev_out = ''
        self.t_start=None
        self.exit = False
        self.same_output_threshold = 0
        self.show_prev_out_thresh = 5   # if pause(no output from whisper) show previous output for 5 seconds
        self.add_pause_thresh = 3       # add a blank to segment list as a pause(no speech) for 3 seconds
        self.transcript = []
        self.send_last_n_segments = 10

        # text formatting
        self.wrapper = textwrap.TextWrapper(width=50)
        self.pick_previous_segments = 2

        # setup mqtt
        self.topic = topic

        # threading
        self.websocket = websocket
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()
        self.websocket.send(json.dumps(SERVER_READY))
    
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
        wrapped = "".join(text + output)
        return wrapped
    
    def add_frames(self, frame_np):
        try:
            speech_prob = self.vad_model(torch.from_numpy(frame_np.copy()), self.RATE).item()
            if speech_prob < self.vad_threshold:
                return
            
        except Exception as e:
            logging.error(e)
            return
        
        if self.frames_np is not None and self.frames_np.shape[0] > 45*self.RATE:
            self.frames_offset += 45.0
            self.frames_np = self.frames_np[int(30*self.RATE):]
        if self.frames_np is None:
            self.frames_np = frame_np.copy()
        else:
            self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)

    def speech_to_text(self):
        """
        Process audio stream in an infinite loop.
        """
        while True:
            if self.exit:
                logging.info("Exiting speech to text thread")
                break
            
            if self.frames_np is None: 
                continue

            # clip audio if the current chunk exceeds 30 seconds, this basically implies that
            # no valid segment for the last 30 seconds from whisper
            if self.frames_np[int((self.timestamp_offset - self.frames_offset)*self.RATE):].shape[0] > 25 * self.RATE:
                duration = self.frames_np.shape[0] / self.RATE
                self.timestamp_offset = self.frames_offset + duration - 5
    
            samples_take = max(0, (self.timestamp_offset - self.frames_offset)*self.RATE)
            input_bytes = self.frames_np[int(samples_take):].copy()
            duration = input_bytes.shape[0] / self.RATE
            if duration<1.0: 
                continue
            try:
                input_sample = input_bytes.copy()
                # set previous complete segment as initial prompt
                if len(self.text) and self.text[-1] != '': 
                    initial_prompt = self.text[-1]
                else: 
                    initial_prompt = None
                
                # whisper transcribe with prompt
                result = self.transcriber.transcribe(input_sample, initial_prompt=initial_prompt)
                if len(result):
                    self.t_start = None
                    last_segment = self.update_segments(result, duration)
                    if len(self.transcript) < self.send_last_n_segments:
                        segments = self.transcript
                    else:
                        segments = self.transcript[-self.send_last_n_segments:]
                    if last_segment is not None:
                        segments = segments + [last_segment]
                    
                    try:
                        self.websocket.send(json.dumps(segments))
                    except Exception as e:
                        logging.info(f"[ERROR]: {e}")
                else:
                    # show previous output if there is pause i.e. no output from whisper
                    segments = []
                    if self.t_start is None: self.t_start = time.time()
                    if time.time() - self.t_start < self.show_prev_out_thresh:
                        if len(self.transcript) < self.send_last_n_segments:
                            segments = self.transcript
                        else:
                            segments = self.transcript[-self.send_last_n_segments:]
                    
                    # add a blank if there is no speech for 3 seconds
                    if len(self.text) and self.text[-1] != '':
                        if time.time() - self.t_start > self.add_pause_thresh:
                            self.text.append('')

                    try:
                        self.websocket.send(json.dumps(segments))
                    except Exception as e:
                        logging.info(f"[INFO]: {e}")
            except Exception as e:
                logging.info(f"[INFO]: {e}")
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
        self.current_out = ''
        last_segment = None
        # process complete segments
        if len(segments) > 1:
            for i, s in enumerate(segments[:-1]):
                text_ = s.text
                self.text.append(text_)
                start, end = self.timestamp_offset + s.start, self.timestamp_offset + min(duration, s.end)
                self.transcript.append(
                    {
                        'start': start,
                        'end': end,
                        'text': text_
                    }
                )
                
                offset = min(duration, s.end)

        self.current_out += segments[-1].text
        last_segment = {
            'start': self.timestamp_offset + segments[-1].start,
            'end': self.timestamp_offset + min(duration, segments[-1].end),
            'text': self.current_out
        }
        
        # if same incomplete segment is seen multiple times then update the offset
        # and append the segment to the list
        if self.current_out.strip() == self.prev_out.strip() and self.current_out != '': 
            self.same_output_threshold += 1
        else: 
            self.same_output_threshold = 0
        
        if self.same_output_threshold > 5:
            if not len(self.text) or self.text[-1].strip().lower()!=self.current_out.strip().lower():          
                self.text.append(self.current_out)
                self.transcript.append(
                    {
                        'start': self.timestamp_offset,
                        'end': self.timestamp_offset + duration,
                        'text': self.current_out
                    }
                )
            self.current_out = ''
            offset = duration
            self.same_output_threshold = 0
            last_segment = None
        else:
            self.prev_out = self.current_out
        
        # update offset
        if offset is not None:
            self.timestamp_offset += offset

        return last_segment
    
    def cleanup(self):
        logging.info("Cleaning up.")
        self.exit = True
        self.transcriber.destroy()
    

if __name__ == "__main__":
    with serve(recv_audio, "0.0.0.0", 9090) as server:
        server.serve_forever()