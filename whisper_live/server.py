import websockets
import pickle, struct, time, pyaudio
import threading
import os, json
import base64
import wave
import textwrap

import logging
# logging.basicConfig(level = logging.INFO)

from collections import deque
from dataclasses import dataclass
from websockets.sync.server import serve

import torch
import numpy as np
import time
from whisper_live.transcriber import WhisperModel


class TranscriptionServer:
    """
    Represents a transcription server that handles incoming audio from clients.

    Attributes:
        clients (dict): A dictionary to store connected clients.
    """
    RATE = 16000
    def __init__(self):
        # voice activity detection model
        self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=True,
                                           onnx=True
                                           )
        self.vad_threshold = 0.4
        self.clients = {}
        self.websockets = {}
        self.clients_start_time = {}
        self.max_clients = 1
        self.max_connection_time = 600 # in seconds
    
    def get_wait_time(self):
        wait_time = None
        for k,v in self.clients_start_time.items():
            current_client_time_remaining = self.max_connection_time - (time.time() - v)
            if wait_time is None:
                wait_time = current_client_time_remaining
            elif current_client_time_remaining < wait_time:
                wait_time = current_client_time_remaining
        return wait_time/60

    def recv_audio(self, websocket):
        """
        Receive audio chunks from a client in an infinite loop.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
        """
        logging.info("New client connected")
        options = websocket.recv()
        options = json.loads(options)

        if len(self.clients) >= self.max_clients:
            logging.warning("Client Queue Full. Asking client to wait ...")
            wait_time = self.get_wait_time()
            response = {
                "uid" : options["uid"],
                "status": "WAIT",
                "message": wait_time,
            }
            websocket.send(json.dumps(response))
            websocket.close()
            del websocket
            return
        
        client = ServeClient(
            websocket,
            multilingual=options["multilingual"],
            language=options["language"],
            task=options["task"],
            client_uid=options["uid"]
        )
        
        self.clients[websocket] = client
        self.clients_start_time[websocket] = time.time() 

        while True:
            try:
                frame_data = websocket.recv()
                data = json.loads(frame_data)
                base64_audio = data["audio"]
                binary_audio = base64.b64decode(base64_audio)
                frame_np = np.frombuffer(binary_audio, dtype=np.float32)

                try:
                    speech_prob = self.vad_model(torch.from_numpy(frame_np.copy()), self.RATE).item()
                    if speech_prob < self.vad_threshold:
                        continue
                    
                except Exception as e:
                    logging.error(e)
                    return
                self.clients[websocket].add_frames(frame_np)

                elapsed_time = time.time() - self.clients_start_time[websocket]
                if elapsed_time >= self.max_connection_time:
                    self.clients[websocket].disconnect()
                    logging.warning(f"{self.clients[websocket]} Client disconnected due to overtime.")
                    self.clients[websocket].cleanup()
                    self.clients.pop(websocket)
                    self.clients_start_time.pop(websocket)
                    websocket.close()
                    del websocket
                    break


            except Exception as e:
                self.clients[websocket].cleanup()
                self.clients.pop(websocket)
                self.clients_start_time.pop(websocket)
                logging.info("Connection Closed.")
                logging.info(self.clients)

                del websocket
                break

    def run(self, host, port=9090):
        """
        Run the transcription server.

        Args:
            host (str): The host address to bind the server.
            port (int): The port number to bind the server.
        """
        with serve(self.recv_audio, host, port) as server:
            server.serve_forever()


class ServeClient:
    RATE = 16000
    SERVER_READY = "SERVER_READY"
    DISCONNECT = "DISCONNECT"

    def __init__(self, websocket, task="transcribe", device=None, multilingual=False, language=None, client_uid=None):
        self.client_uid = client_uid
        self.data = b""
        self.frames = b""
        self.language = language if multilingual else "en"
        self.task = task
        self.transcriber = WhisperModel(
            "small" if multilingual else "small.en", 
            device=device if device else "cuda",
            compute_type="float16", 
            local_files_only=False,
        )
        
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

        # threading
        self.websocket = websocket
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()
        self.websocket.send(
            json.dumps(
                {
                    "uid": self.client_uid,
                    "message": self.SERVER_READY
                }
            )
        )
    
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
        if self.frames_np is not None and self.frames_np.shape[0] > 45*self.RATE:
            self.frames_offset += 30.0
            self.frames_np = self.frames_np[int(30*self.RATE):]
        if self.frames_np is None:
            self.frames_np = frame_np.copy()
        else:
            self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)

    def speech_to_text(self):
        """
        Process audio stream in an infinite loop.
        """
        # detect language
        if self.language is None:
            # wait for 30s of audio
            while self.frames_np is None or self.frames_np.shape[0] < 30*self.RATE:
                time.sleep(1)
            input_bytes = self.frames_np[-30*self.RATE:].copy()
            self.frames_np = None
            duration = input_bytes.shape[0] / self.RATE

            self.language, lang_prob = self.transcriber.transcribe(
                    input_bytes, 
                    initial_prompt=None,
                    language=self.language,
                    task=self.task
                )
            logging.info(f"Detected language {self.language} with probability {lang_prob}")
            self.websocket.send(json.dumps(
                {"uid": self.client_uid, "language": self.language, "language_prob": lang_prob}))

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
                result = self.transcriber.transcribe(
                    input_sample, 
                    initial_prompt=initial_prompt,
                    language=self.language,
                    task=self.task
                )

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
                        self.websocket.send(
                            json.dumps({
                                "uid": self.client_uid,
                                "segments": segments
                            })
                        )
                    except Exception as e:
                        logging.error(f"[ERROR]: {e}")
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
                        self.websocket.send(
                            json.dumps({
                                "uid": self.client_uid,
                                "segments": segments
                            })
                        )
                    except Exception as e:
                        logging.error(f"[ERROR]: {e}")
            except Exception as e:
                logging.error(f"[ERROR]: {e}")
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
    
    def disconnect(self):
        self.websocket.send(
            json.dumps(
                {
                    "uid": self.client_uid,
                    "message": self.DISCONNECT
                }
            )
        )
    
    def cleanup(self):
        logging.info("Cleaning up.")
        self.exit = True
        self.transcriber.destroy()
