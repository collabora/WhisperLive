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

    def _mock_multi_path(self, mock_tokenizer_cls, n_items, no_speech_prob=0.1, score=-1.0):
        mock_tok = MagicMock()
        mock_tok.decode.return_value = "hello world"
        mock_tokenizer_cls.return_value = mock_tok
        self.mock_transcriber.feature_extractor.return_value = np.zeros((80, 3000), dtype=np.float32)
        self.mock_transcriber.feature_extractor.sampling_rate = 16000
        self.mock_transcriber.encode.return_value = np.zeros((n_items, 1500, 512), dtype=np.float32)
        gen_result = MagicMock()
        gen_result.sequences_ids = [[50257, 50362, 1234, 50256]]
        gen_result.scores = [np.float32(score)]
        gen_result.no_speech_prob = no_speech_prob
        self.mock_transcriber.model.generate.return_value = [gen_result] * n_items
        self.mock_transcriber.model.is_multilingual = False
        self.mock_transcriber.max_length = 448
        self.mock_transcriber.frames_per_second = 50
        self.mock_transcriber.get_prompt.return_value = [50258]
        self.mock_transcriber._split_segments_by_timestamps.return_value = (
            [{"start": 0.0, "end": 1.0, "tokens": [1234], "seek": 0}],
            None,
            None,
        )

    def _run_batch(self, requests):
        for req in requests:
            self.worker.submit(req)
        for req in requests:
            req.future.wait(timeout=5)
            self.assertIsNone(req.error)

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

    def test_long_audio_routed_to_single_path(self):
        """Audio longer than 30s must not be truncated by the batched encode."""
        self.mock_transcriber.feature_extractor.sampling_rate = 16000
        self.mock_transcriber.transcribe.return_value = ([MagicMock()], MagicMock())

        long1 = BatchRequest(audio=self._make_audio(31.0), language="en", use_vad=False)
        long2 = BatchRequest(audio=self._make_audio(45.0), language="en", use_vad=False)
        self.worker._process_batch([long1, long2])

        # both long items go through the windowed transcribe() path, not encode()
        self.assertEqual(self.mock_transcriber.transcribe.call_count, 2)
        self.mock_transcriber.encode.assert_not_called()
        self.assertTrue(long1.future.is_set())
        self.assertTrue(long2.future.is_set())

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

    @mock.patch('whisper_live.batch_inference.collect_chunks')
    @mock.patch('whisper_live.batch_inference.get_speech_timestamps')
    @mock.patch('whisper_live.batch_inference.get_suppressed_tokens', return_value=[-1])
    @mock.patch('whisper_live.batch_inference.Tokenizer')
    def test_vad_timestamps_restored(self, mock_tokenizer_cls, mock_suppress, mock_vad, mock_collect):
        """Segment times must refer to the original audio, not the VAD-collapsed audio."""
        self._mock_multi_path(mock_tokenizer_cls, n_items=2)
        mock_vad.return_value = [{"start": 16000, "end": 32000}]
        mock_collect.side_effect = lambda audio, chunks: ([audio[16000:32000]], None)

        requests = [
            BatchRequest(audio=self._make_audio(2.0), language="en", use_vad=True)
            for _ in range(2)
        ]
        self._run_batch(requests)

        for req in requests:
            self.assertEqual(req.result[0].start, 1.0)
            self.assertEqual(req.result[0].end, 2.0)
            self.assertEqual(req.info.duration, 2.0)
            self.assertEqual(req.info.duration_after_vad, 1.0)

    @mock.patch('whisper_live.batch_inference.get_suppressed_tokens', return_value=[-1])
    @mock.patch('whisper_live.batch_inference.Tokenizer')
    def test_silence_yields_no_segments(self, mock_tokenizer_cls, mock_suppress):
        """High no_speech_prob with low logprob is silence, so no text is emitted."""
        self._mock_multi_path(mock_tokenizer_cls, n_items=2, no_speech_prob=0.9, score=-2.0)
        self.mock_transcriber._split_segments_by_timestamps.return_value = ([], None, None)

        requests = [
            BatchRequest(audio=self._make_audio(), language="en", use_vad=False)
            for _ in range(2)
        ]
        self._run_batch(requests)

        for req in requests:
            self.assertEqual(req.result, [])
        split_calls = self.mock_transcriber._split_segments_by_timestamps.call_args_list
        self.assertTrue(all(call.kwargs["tokens"] == [] for call in split_calls))

    @mock.patch('whisper_live.batch_inference.get_suppressed_tokens', return_value=[-1])
    @mock.patch('whisper_live.batch_inference.Tokenizer')
    def test_hotwords_reach_both_paths(self, mock_tokenizer_cls, mock_suppress):
        self._mock_multi_path(mock_tokenizer_cls, n_items=2)
        self.mock_transcriber.transcribe.return_value = ([MagicMock()], MagicMock())

        batched = [
            BatchRequest(audio=self._make_audio(), language="en", use_vad=False, hotwords="aavaaz")
            for _ in range(2)
        ]
        single = BatchRequest(audio=self._make_audio(31.0), language="en", hotwords="aavaaz")
        self._run_batch(batched + [single])

        for call in self.mock_transcriber.get_prompt.call_args_list:
            self.assertEqual(call.kwargs["hotwords"], "aavaaz")
        self.assertEqual(self.mock_transcriber.transcribe.call_args.kwargs["hotwords"], "aavaaz")

    @mock.patch('whisper_live.batch_inference.get_suppressed_tokens', return_value=[-1])
    @mock.patch('whisper_live.batch_inference.Tokenizer')
    def test_word_timestamps_take_single_path(self, mock_tokenizer_cls, mock_suppress):
        self._mock_multi_path(mock_tokenizer_cls, n_items=2)
        self.mock_transcriber.transcribe.return_value = ([MagicMock()], MagicMock())

        batched = [
            BatchRequest(audio=self._make_audio(), language="en", use_vad=False)
            for _ in range(2)
        ]
        with_words = BatchRequest(audio=self._make_audio(), language="en", word_timestamps=True)
        self._run_batch(batched + [with_words])

        self.mock_transcriber.transcribe.assert_called_once()
        self.assertTrue(self.mock_transcriber.transcribe.call_args.kwargs["word_timestamps"])
        self.assertEqual(self.mock_transcriber.model.generate.call_args.args[0].shape[0], 2)

    def test_abandoned_request_skipped(self):
        """A request whose session stopped waiting must not reach the model."""
        self.mock_transcriber.transcribe.return_value = ([MagicMock()], MagicMock())
        abandoned = BatchRequest(audio=self._make_audio(), language="en", abandoned=True)
        live = BatchRequest(audio=self._make_audio(), language="en")
        self.worker.submit(abandoned)
        self.worker.submit(live)
        live.future.wait(timeout=5)

        self.assertTrue(live.future.is_set())
        self.assertFalse(abandoned.future.is_set())
        self.mock_transcriber.transcribe.assert_called_once()

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
