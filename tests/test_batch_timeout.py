import threading
import unittest
from unittest import mock

import numpy as np

from whisper_live.backend.base import ServeClientBase
from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper


class NeverAnsweringWorker:
    def __init__(self):
        self.submitted = []

    def submit(self, request):
        self.submitted.append(request)


class TestBatchWaitTimeout(unittest.TestCase):
    def setUp(self):
        self.client = ServeClientFasterWhisper.__new__(ServeClientFasterWhisper)
        self.client.language = "en"
        self.client.task = "transcribe"
        self.client.initial_prompt = None
        self.client.use_vad = False
        self.client.vad_parameters = None
        self.client.word_timestamps = False
        self.client.hotwords = None
        self.client.client_uid = "uid"
        self.worker = NeverAnsweringWorker()

    def test_timeout_raises_and_marks_request_abandoned(self):
        with (
            mock.patch.object(ServeClientFasterWhisper, "BATCH_WORKER", self.worker),
            mock.patch.object(ServeClientFasterWhisper, "BATCH_WAIT_TIMEOUT_SECONDS", 0.01),
        ):
            with self.assertRaises(TimeoutError):
                self.client.transcribe_audio(np.zeros(16000, dtype=np.float32))

        self.assertEqual(len(self.worker.submitted), 1)
        self.assertTrue(self.worker.submitted[0].abandoned)


class TimingOutClient(ServeClientBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.language = "en"
        self.calls = 0

    def transcribe_audio(self, input_sample):
        self.calls += 1
        self.exit = True
        raise TimeoutError("late")

    def handle_transcription_output(self, result, duration):
        raise AssertionError("no output expected")


class TestTimeoutDropsChunk(unittest.TestCase):
    def test_speech_to_text_advances_offset_past_timed_out_audio(self):
        client = TimingOutClient(client_uid="uid", websocket=mock.MagicMock())
        client.frames_np = np.zeros(5 * ServeClientBase.RATE, dtype=np.float32)

        client.speech_to_text()

        self.assertEqual(client.calls, 1)
        self.assertAlmostEqual(client.timestamp_offset, 5.0)


class TestBatchChunkCap(unittest.TestCase):
    def setUp(self):
        self.client = ServeClientFasterWhisper.__new__(ServeClientFasterWhisper)
        self.client.lock = threading.Lock()
        self.client.frames_offset = 0.0
        self.client.timestamp_offset = 0.0
        self.client.frames_np = np.zeros(40 * ServeClientFasterWhisper.RATE, dtype=np.float32)

    def test_batch_mode_caps_chunk_at_30s(self):
        with mock.patch.object(ServeClientFasterWhisper, "BATCH_WORKER", object()):
            audio, duration = self.client.get_audio_chunk_for_processing()
        self.assertEqual(duration, 30.0)
        self.assertEqual(audio.shape[0], 30 * ServeClientFasterWhisper.RATE)

    def test_non_batch_mode_returns_whole_chunk(self):
        with mock.patch.object(ServeClientFasterWhisper, "BATCH_WORKER", None):
            audio, duration = self.client.get_audio_chunk_for_processing()
        self.assertEqual(duration, 40.0)
        self.assertEqual(audio.shape[0], 40 * ServeClientFasterWhisper.RATE)


if __name__ == "__main__":
    unittest.main()
