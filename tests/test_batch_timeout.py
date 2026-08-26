import unittest
from unittest import mock

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
