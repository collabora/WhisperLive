import websockets
import time
import threading
import json
import textwrap

import logging
# logging.basicConfig(level = logging.INFO)

from websockets.sync.server import serve

import torch
import numpy as np
import time
from whisper_live.transcriber import WhisperModel


class TranscriptionServer:
    """
    Represents a transcription server that handles incoming audio from clients.

    Attributes:
        RATE (int): The audio sampling rate (constant) set to 16000.
        vad_model (torch.Module): The voice activity detection model.
        vad_threshold (float): The voice activity detection threshold.
        clients (dict): A dictionary to store connected clients.
        websockets (dict): A dictionary to store WebSocket connections.
        clients_start_time (dict): A dictionary to track client start times.
        max_clients (int): Maximum allowed connected clients.
        max_connection_time (int): Maximum allowed connection time in seconds.
    """

    RATE = 16000

    def __init__(self):
        # voice activity detection model

        self.clients = {}
        self.websockets = {}
        self.clients_start_time = {}
        self.max_clients = 4
        self.max_connection_time = 600

    def get_wait_time(self):
        """
        Calculate and return the estimated wait time for clients.

        Returns:
            float: The estimated wait time in minutes.
        """
        wait_time = None

        for k, v in self.clients_start_time.items():
            current_client_time_remaining = self.max_connection_time - (time.time() - v)

            if wait_time is None or current_client_time_remaining < wait_time:
                wait_time = current_client_time_remaining

        return wait_time / 60

    def recv_audio(self, websocket):
        """
        Receive audio chunks from a client in an infinite loop.
        
        Continuously receives audio frames from a connected client
        over a WebSocket connection. It processes the audio frames using a
        voice activity detection (VAD) model to determine if they contain speech
        or not. If the audio frame contains speech, it is added to the client's
        audio data for ASR.
        If the maximum number of clients is reached, the method sends a
        "WAIT" status to the client, indicating that they should wait
        until a slot is available.
        If a client's connection exceeds the maximum allowed time, it will
        be disconnected, and the client's resources will be cleaned up.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
        
        Raises:
            Exception: If there is an error during the audio frame processing.
        """
        logging.info("New client connected")
        options = websocket.recv()
        options = json.loads(options)

        if len(self.clients) >= self.max_clients:
            logging.warning("Client Queue Full. Asking client to wait ...")
            wait_time = self.get_wait_time()
            response = {
                "uid": options["uid"],
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
            client_uid=options["uid"],
            model_size=options["model_size"],
            initial_prompt=options.get("initial_prompt"),
            vad_parameters=options.get("vad_parameters")
        )
        
        self.clients[websocket] = client
        self.clients_start_time[websocket] = time.time()

        while True:
            try:
                frame_data = websocket.recv()
                frame_np = np.frombuffer(frame_data, dtype=np.float32)

                self.clients[websocket].add_frames(frame_np)

                elapsed_time = time.time() - self.clients_start_time[websocket]
                if elapsed_time >= self.max_connection_time:
                    self.clients[websocket].disconnect()
                    logging.warning(f"Client with uid '{self.clients[websocket].client_uid}' disconnected due to overtime.")
                    self.clients[websocket].cleanup()
                    self.clients.pop(websocket)
                    self.clients_start_time.pop(websocket)
                    websocket.close()
                    del websocket
                    break

            except Exception as e:
                logging.info(f"[ERROR]: Client with uid '{self.clients[websocket].client_uid}' Disconnected.")
                if self.clients[websocket].model_size is not None:
                    self.clients[websocket].cleanup()
                self.clients.pop(websocket)
                self.clients_start_time.pop(websocket)
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
    """
    Attributes:
        RATE (int): The audio sampling rate (constant) set to 16000.
        SERVER_READY (str): A constant message indicating that the server is ready.
        DISCONNECT (str): A constant message indicating that the client should disconnect.
        client_uid (str): A unique identifier for the client.
        data (bytes): Accumulated audio data.
        frames (bytes): Accumulated audio frames.
        language (str): The language for transcription.
        task (str): The task type, e.g., "transcribe."
        transcriber (WhisperModel): The Whisper model for speech-to-text.
        timestamp_offset (float): The offset in audio timestamps.
        frames_np (numpy.ndarray): NumPy array to store audio frames.
        frames_offset (float): The offset in audio frames.
        text (list): List of transcribed text segments.
        current_out (str): The current incomplete transcription.
        prev_out (str): The previous incomplete transcription.
        t_start (float): Timestamp for the start of transcription.
        exit (bool): A flag to exit the transcription thread.
        same_output_threshold (int): Threshold for consecutive same output segments.
        show_prev_out_thresh (int): Threshold for showing previous output segments.
        add_pause_thresh (int): Threshold for adding a pause (blank) segment.
        transcript (list): List of transcribed segments.
        send_last_n_segments (int): Number of last segments to send to the client.
        wrapper (textwrap.TextWrapper): Text wrapper for formatting text.
        pick_previous_segments (int): Number of previous segments to include in the output.
        websocket: The WebSocket connection for the client.
    """
    RATE = 16000
    SERVER_READY = "SERVER_READY"
    DISCONNECT = "DISCONNECT"

    def __init__(
        self,
        websocket,
        task="transcribe",
        device=None,
        multilingual=False,
        language=None,
        client_uid=None,
        model_size="small",
        initial_prompt=None,
        vad_parameters=None
        ):
        """
        Initialize a ServeClient instance.
        The Whisper model is initialized based on the client's language and device availability.
        The transcription thread is started upon initialization. A "SERVER_READY" message is sent
        to the client to indicate that the server is ready.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            task (str, optional): The task type, e.g., "transcribe." Defaults to "transcribe".
            device (str, optional): The device type for Whisper, "cuda" or "cpu". Defaults to None.
            multilingual (bool, optional): Whether the client supports multilingual transcription. Defaults to False.
            language (str, optional): The language for transcription. Defaults to None.
            client_uid (str, optional): A unique identifier for the client. Defaults to None.

        """
        self.client_uid = client_uid
        self.data = b""
        self.frames = b""
        self.model_sizes = [
            "tiny", "base", "small", "medium", "large-v2", "large-v3"
        ]
        self.multilingual = multilingual
        self.model_size = self.get_model_size(model_size)
        self.language = language if self.multilingual else "en"
        self.task = task
        self.websocket = websocket
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters or {"threshold": 0.5}
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.model_size == None:
            return
        
        self.transcriber = WhisperModel(
            self.model_size, 
            device=device,
            compute_type="int8" if device=="cpu" else "float16", 
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
    
    def get_model_size(self, model_size):
        """
        Returns the whisper model size based on multilingual.
        """
        if model_size not in self.model_sizes:
            self.websocket.send(
                json.dumps(
                    {
                        "uid": self.client_uid,
                        "status": "ERROR",
                        "message": f"Invalid model size {model_size}. Available choices: {self.model_sizes}"
                    }
                )
            )
            return None
        
        if model_size in ["large-v2", "large-v3"]:
            self.multilingual = True
            return model_size

        if not self.multilingual:
            model_size = model_size + ".en"

        return model_size
    
    def add_frames(self, frame_np):
        """
        Add audio frames to the ongoing audio stream buffer.

        This method is responsible for maintaining the audio stream buffer, allowing the continuous addition
        of audio frames as they are received. It also ensures that the buffer does not exceed a specified size
        to prevent excessive memory usage.

        If the buffer size exceeds a threshold (45 seconds of audio data), it discards the oldest 30 seconds
        of audio data to maintain a reasonable buffer size. If the buffer is empty, it initializes it with the provided
        audio frame. The audio stream buffer is used for real-time processing of audio data for transcription.

        Args:
            frame_np (numpy.ndarray): The audio frame data as a NumPy array.

        """
        if self.frames_np is not None and self.frames_np.shape[0] > 45*self.RATE:
            self.frames_offset += 30.0
            self.frames_np = self.frames_np[int(30*self.RATE):]
        if self.frames_np is None:
            self.frames_np = frame_np.copy()
        else:
            self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)

    def speech_to_text(self):
        """
        Process an audio stream in an infinite loop, continuously transcribing the speech.

        This method continuously receives audio frames, performs real-time transcription, and sends
        transcribed segments to the client via a WebSocket connection.

        If the client's language is not detected, it waits for 30 seconds of audio input to make a language prediction.
        It utilizes the Whisper ASR model to transcribe the audio, continuously processing and streaming results. Segments
        are sent to the client in real-time, and a history of segments is maintained to provide context.Pauses in speech 
        (no output from Whisper) are handled by showing the previous output for a set duration. A blank segment is added if 
        there is no speech for a specified duration to indicate a pause.

        Raises:
            Exception: If there is an issue with audio processing or WebSocket communication.

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
                
                # whisper transcribe with prompt
                result, info = self.transcriber.transcribe(
                    input_sample, 
                    initial_prompt=self.initial_prompt,
                    language=self.language,
                    task=self.task,
                    vad_filter=True,
                    vad_parameters=self.vad_parameters
                )

                if self.language is None:
                    if info.language_probability > 0.5:
                        self.language = info.language
                        logging.info(f"Detected language {self.language} with probability {info.language_probability}")
                        self.websocket.send(json.dumps(
                            {"uid": self.client_uid, "language": self.language, "language_prob": info.language_probability}))
                    else:
                        # detect language again
                        continue

                if len(result):
                    self.t_start = None
                    last_segment = self.update_segments(result, duration)
                    if len(self.transcript) < self.send_last_n_segments:
                        segments = self.transcript
                    else:
                        segments = self.transcript[-self.send_last_n_segments:]
                    if last_segment is not None:
                        segments = segments + [last_segment]                    
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
                    logging.error(f"[ERROR]: Failed to send message to client: {e}")

            except Exception as e:
                logging.error(f"[ERROR]: Failed to transcribe audio chunk: {e}")
                time.sleep(0.01)
    
    def format_segment(self, start, end, text):
        """Helper function to format a segment with string timestamps."""
        return {
            'start': "{:.3f}".format(start),
            'end': "{:.3f}".format(end),
            'text': text
        }

    def update_segments(self, segments, duration):
        """
        Processes the segments from whisper. Appends all the segments to the list
        except for the last segment assuming that it is incomplete.

        Updates the ongoing transcript with transcribed segments, including their start and end times.
        Complete segments are appended to the transcript in chronological order. Incomplete segments 
        (assumed to be the last one) are processed to identify repeated content. If the same incomplete 
        segment is seen multiple times, it updates the offset and appends the segment to the transcript.
        A threshold is used to detect repeated content and ensure it is only included once in the transcript.
        The timestamp offset is updated based on the duration of processed segments. The method returns the 
        last processed segment, allowing it to be sent to the client for real-time updates.

        Args:
            segments(dict) : dictionary of segments as returned by whisper
            duration(float): duration of the current chunk
        
        Returns:
            dict or None: The last processed segment with its start time, end time, and transcribed text.
                     Returns None if there are no valid segments to process.
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
                self.transcript.append(self.format_segment(start, end, text_))
                
                offset = min(duration, s.end)

        self.current_out += segments[-1].text
        last_segment = self.format_segment(
            self.timestamp_offset + segments[-1].start,
            self.timestamp_offset + min(duration, segments[-1].end),
            self.current_out
        )
        
        # if same incomplete segment is seen multiple times then update the offset
        # and append the segment to the list
        if self.current_out.strip() == self.prev_out.strip() and self.current_out != '': 
            self.same_output_threshold += 1
        else: 
            self.same_output_threshold = 0
        
        if self.same_output_threshold > 5:
            if not len(self.text) or self.text[-1].strip().lower()!=self.current_out.strip().lower():          
                self.text.append(self.current_out)
                self.transcript.append(self.format_segment(
                    self.timestamp_offset,
                    self.timestamp_offset + duration,
                    self.current_out
                ))
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
        """
        Notify the client of disconnection and send a disconnect message.

        This method sends a disconnect message to the client via the WebSocket connection to notify them
        that the transcription service is disconnecting gracefully.

        """
        self.websocket.send(
            json.dumps(
                {
                    "uid": self.client_uid,
                    "message": self.DISCONNECT
                }
            )
        )
    
    def cleanup(self):
        """
        Perform cleanup tasks before exiting the transcription service.

        This method performs necessary cleanup tasks, including stopping the transcription thread, marking
        the exit flag to indicate the transcription thread should exit gracefully, and destroying resources
        associated with the transcription process.

        """
        logging.info("Cleaning up.")
        self.exit = True
