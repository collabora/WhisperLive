import time
import unittest
from unittest import mock
from unittest.mock import MagicMock

import numpy as np

from whisper_live.batch_inference import BatchInferenceWorker, BatchRequest


class TestBatchInferenceWorker(unittest.TestCase):
    def setUp(self):
        self.mock_transcriber = MagicMock()
        self.worker = BatchInferenceWorker(
            transcriber=self.mock_transcriber,
            max_batch_size=8,
            batch_window_ms=200,
        )
        self.worker.start()

    def tearDown(self):
        self.worker.stop()

    def _make_audio(self, duration_s=1.0):
        return np.random.randn(int(16000 * duration_s)).astype(np.float32)

    def test_single_request_uses_transcribe(self):
        """Single request should fall back to transcriber.transcribe()."""
        fake_segment = MagicMock()
        fake_info = MagicMock()
        self.mock_transcriber.transcribe.return_value = ([fake_segment], fake_info)

        req = BatchRequest(audio=self._make_audio(), language="en", use_vad=False)
        self.worker.submit(req)
        req.future.wait(timeout=5)

        self.assertTrue(req.future.is_set())
        self.assertIsNone(req.error)
        self.assertEqual(req.result, [fake_segment])
        self.assertEqual(req.info, fake_info)
        self.mock_transcriber.transcribe.assert_called_once()

    @mock.patch('whisper_live.batch_inference.get_suppressed_tokens', return_value=[-1])
    @mock.patch('whisper_live.batch_inference.Tokenizer')
    def test_multiple_requests_batched(self, mock_tokenizer_cls, mock_suppress):
        """Multiple concurrent requests should go through the batched GPU path."""
        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.decode.return_value = "hello world"
        mock_tokenizer_cls.return_value = mock_tok

        # Mock feature extractor
        self.mock_transcriber.feature_extractor.return_value = np.zeros(
            (80, 3000), dtype=np.float32
        )
        self.mock_transcriber.feature_extractor.sampling_rate = 16000

        # Mock encode
        self.mock_transcriber.encode.return_value = np.zeros(
            (3, 1500, 512), dtype=np.float32
        )

        # Mock model.generate — one result per item
        gen_result = MagicMock()
        gen_result.sequences_ids = [[50257, 50362, 1234, 50256]]
        gen_result.scores = [np.float32(-1.0)]
        gen_result.no_speech_prob = 0.1
        self.mock_transcriber.model.generate.return_value = [gen_result] * 3

        # Mock remaining model attributes
        self.mock_transcriber.model.is_multilingual = False
        self.mock_transcriber.max_length = 448
        self.mock_transcriber.frames_per_second = 50
        self.mock_transcriber.get_prompt.return_value = [50258]
        self.mock_transcriber._split_segments_by_timestamps.return_value = (
            [{"start": 0.0, "end": 1.0, "tokens": [1234], "seek": 0}],
            None,
            None,
        )

        requests = [
            BatchRequest(audio=self._make_audio(), language="en", use_vad=False)
            for _ in range(3)
        ]
        for req in requests:
            self.worker.submit(req)
        for req in requests:
            req.future.wait(timeout=5)

        for req in requests:
            self.assertTrue(req.future.is_set())
            self.assertIsNone(req.error)
            self.assertIsNotNone(req.result)

        # Verify the batched encode path was used (not transcribe)
        self.mock_transcriber.encode.assert_called()
        self.mock_transcriber.transcribe.assert_not_called()

    def test_error_propagation(self):
        """Transcriber errors should propagate to the request without crashing the worker."""
        self.mock_transcriber.transcribe.side_effect = RuntimeError("GPU OOM")

        req = BatchRequest(audio=self._make_audio(), language="en", use_vad=False)
        self.worker.submit(req)
        req.future.wait(timeout=5)

        self.assertTrue(req.future.is_set())
        self.assertIsInstance(req.error, RuntimeError)
        self.assertIn("GPU OOM", str(req.error))

        # Worker should still be alive — submit another request
        self.mock_transcriber.transcribe.side_effect = None
        self.mock_transcriber.transcribe.return_value = ([MagicMock()], MagicMock())

        req2 = BatchRequest(audio=self._make_audio(), language="en", use_vad=False)
        self.worker.submit(req2)
        req2.future.wait(timeout=5)

        self.assertIsNone(req2.error)
        self.assertIsNotNone(req2.result)

    def test_worker_stop(self):
        """Worker thread should exit cleanly when stop() is called."""
        self.assertTrue(self.worker._thread.is_alive())
        self.worker.stop()
        self.assertFalse(self.worker._thread.is_alive())

    def test_batch_respects_max_size(self):
        """Batches should not exceed max_batch_size."""
        self.worker.stop()  # Stop the default worker

        observed_batch_sizes = []
        original_process = BatchInferenceWorker._process_batch

        def tracking_process(self_inner, batch):
            observed_batch_sizes.append(len(batch))
            original_process(self_inner, batch)

        self.worker = BatchInferenceWorker(
            transcriber=self.mock_transcriber,
            max_batch_size=2,
            batch_window_ms=100,
        )

        self.mock_transcriber.transcribe.return_value = ([MagicMock()], MagicMock())

        with mock.patch.object(
            BatchInferenceWorker, '_process_batch', tracking_process
        ):
            self.worker.start()

            requests = [
                BatchRequest(audio=self._make_audio(), language="en", use_vad=False)
                for _ in range(4)
            ]
            for req in requests:
                self.worker.submit(req)
            for req in requests:
                req.future.wait(timeout=5)

        for size in observed_batch_sizes:
            self.assertLessEqual(size, 2)
        self.assertTrue(all(req.future.is_set() for req in requests))
