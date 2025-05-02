import json
import logging
import threading
import time

from whisper_live.backend.base import ServeClientBase
from whisper_live.transcriber.transcriber_tensorrt import WhisperTRTLLM


class ServeClientTensorRT(ServeClientBase):
    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()

    def __init__(
        self,
        websocket,
        task="transcribe",
        multilingual=False,
        language=None,
        client_uid=None,
        model=None,
        single_model=False,
        use_py_session=False,
        max_new_tokens=225,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=10,
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
            single_model (bool, optional): Whether to instantiate a new model for each client connection. Defaults to False.
            use_py_session (bool, optional): Use python session or cpp session. Defaults to Cpp Session.
            max_new_tokens (int, optional): Max number of tokens to generate.
            send_last_n_segments (int, optional): Number of most recent segments to send to the client. Defaults to 10.
            no_speech_thresh (float, optional): Segments with no speech probability above this threshold will be discarded. Defaults to 0.45.
            clip_audio (bool, optional): Whether to clip audio with no valid segments. Defaults to False.
            same_output_threshold (int, optional): Number of repeated outputs before considering it as a valid segment. Defaults to 10.
        """
        super().__init__(
            client_uid,
            websocket,
            send_last_n_segments,
            no_speech_thresh,
            clip_audio,
            same_output_threshold,
        )

        self.language = language if multilingual else "en"
        self.task = task
        self.eos = False
        self.max_new_tokens = max_new_tokens

        if single_model:
            if ServeClientTensorRT.SINGLE_MODEL is None:
                self.create_model(model, multilingual, use_py_session=use_py_session)
                ServeClientTensorRT.SINGLE_MODEL = self.transcriber
            else:
                self.transcriber = ServeClientTensorRT.SINGLE_MODEL
        else:
            self.create_model(model, multilingual, use_py_session=use_py_session)

        # threading
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()

        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.SERVER_READY,
            "backend": "tensorrt"
        }))

    def create_model(self, model, multilingual, warmup=True, use_py_session=False):
        """
        Instantiates a new model, sets it as the transcriber and does warmup if desired.
        """
        self.transcriber = WhisperTRTLLM(
            model,
            assets_dir="assets",
            device="cuda",
            is_multilingual=multilingual,
            language=self.language,
            task=self.task,
            use_py_session=use_py_session,
            max_output_len=self.max_new_tokens,
        )
        if warmup:
            self.warmup()

    def warmup(self, warmup_steps=10):
        """
        Warmup TensorRT since first few inferences are slow.

        Args:
            warmup_steps (int): Number of steps to warm up the model for.
        """
        logging.info("[INFO:] Warming up TensorRT engine..")
        mel, _ = self.transcriber.log_mel_spectrogram("assets/jfk.flac")
        for i in range(warmup_steps):
            self.transcriber.transcribe(mel)

    def set_eos(self, eos):
        """
        Sets the End of Speech (EOS) flag.

        Args:
            eos (bool): The value to set for the EOS flag.
        """
        self.lock.acquire()
        self.eos = eos
        self.lock.release()

    def handle_transcription_output(self, last_segment, duration):
        """
        Handle the transcription output, updating the transcript and sending data to the client.

        Args:
            last_segment (str): The last segment from the whisper output which is considered to be incomplete because
                                of the possibility of word being truncated.
            duration (float): Duration of the transcribed audio chunk.
        """
        segments = self.prepare_segments({"text": last_segment})
        self.send_transcription_to_client(segments)
        if self.eos:
            self.update_timestamp_offset(last_segment, duration)

    def transcribe_audio(self, input_bytes):
        """
        Transcribe the audio chunk and send the results to the client.

        Args:
            input_bytes (np.array): The audio chunk to transcribe.
        """
        if ServeClientTensorRT.SINGLE_MODEL:
            ServeClientTensorRT.SINGLE_MODEL_LOCK.acquire()
        logging.info(f"[WhisperTensorRT:] Processing audio with duration: {input_bytes.shape[0] / self.RATE}")
        mel, duration = self.transcriber.log_mel_spectrogram(input_bytes)
        last_segment = self.transcriber.transcribe(
            mel,
            text_prefix=f"<|startoftranscript|><|{self.language}|><|{self.task}|><|notimestamps|>",
        )
        if ServeClientTensorRT.SINGLE_MODEL:
            ServeClientTensorRT.SINGLE_MODEL_LOCK.release()
        if last_segment:
            self.handle_transcription_output(last_segment, duration)

    def update_timestamp_offset(self, last_segment, duration):
        """
        Update timestamp offset and transcript.

        Args:
            last_segment (str): Last transcribed audio from the whisper model.
            duration (float): Duration of the last audio chunk.
        """
        if not len(self.transcript):
            self.transcript.append({"text": last_segment + " "})
        elif self.transcript[-1]["text"].strip() != last_segment:
            self.transcript.append({"text": last_segment + " "})
        
        with self.lock:
            self.timestamp_offset += duration

    def speech_to_text(self):
        """
        Process an audio stream in an infinite loop, continuously transcribing the speech.

        This method continuously receives audio frames, performs real-time transcription, and sends
        transcribed segments to the client via a WebSocket connection.

        If the client's language is not detected, it waits for 30 seconds of audio input to make a language prediction.
        It utilizes the Whisper ASR model to transcribe the audio, continuously processing and streaming results. Segments
        are sent to the client in real-time, and a history of segments is maintained to provide context.

        Raises:
            Exception: If there is an issue with audio processing or WebSocket communication.

        """
        while True:
            if self.exit:
                logging.info("Exiting speech to text thread")
                break

            if self.frames_np is None:
                time.sleep(0.02)    # wait for any audio to arrive
                continue

            self.clip_audio_if_no_valid_segment()

            input_bytes, duration = self.get_audio_chunk_for_processing()
            if duration < 0.4:
                continue

            try:
                input_sample = input_bytes.copy()
                logging.info(f"[WhisperTensorRT:] Processing audio with duration: {duration}")
                self.transcribe_audio(input_sample)

            except Exception as e:
                logging.error(f"[ERROR]: {e}")
