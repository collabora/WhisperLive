import json
import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from whisper_live.backend.base import ServeClientBase


class ConcreteServeClient(ServeClientBase):
    """Concrete subclass for testing the abstract base class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.language = "en"

    def transcribe_audio(self, input_sample):
        return None

    def handle_transcription_output(self, result, duration):
        pass


class TestServeClientBaseInit(unittest.TestCase):
    def test_default_values(self):
        ws = MagicMock()
        client = ConcreteServeClient(client_uid="test-uid", websocket=ws)
        self.assertEqual(client.client_uid, "test-uid")
        self.assertEqual(client.send_last_n_segments, 10)
        self.assertAlmostEqual(client.no_speech_thresh, 0.45)
        self.assertFalse(client.clip_audio)
        self.assertEqual(client.same_output_threshold, 10)
        self.assertIsNone(client.frames_np)
        self.assertAlmostEqual(client.timestamp_offset, 0.0)
        self.assertFalse(client.exit)
        self.assertEqual(client.transcript, [])

    def test_custom_values(self):
        ws = MagicMock()
        q = queue.Queue()
        client = ConcreteServeClient(
            client_uid="uid2",
            websocket=ws,
            send_last_n_segments=5,
            no_speech_thresh=0.6,
            clip_audio=True,
            same_output_threshold=20,
            translation_queue=q,
        )
        self.assertEqual(client.send_last_n_segments, 5)
        self.assertAlmostEqual(client.no_speech_thresh, 0.6)
        self.assertTrue(client.clip_audio)
        self.assertEqual(client.same_output_threshold, 20)
        self.assertIs(client.translation_queue, q)


class TestAddFrames(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(client_uid="test", websocket=self.ws)

    def test_first_frame_initializes_buffer(self):
        frame = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.client.add_frames(frame)
        np.testing.assert_array_equal(self.client.frames_np, frame)

    def test_subsequent_frames_concatenated(self):
        frame1 = np.array([0.1, 0.2], dtype=np.float32)
        frame2 = np.array([0.3, 0.4], dtype=np.float32)
        self.client.add_frames(frame1)
        self.client.add_frames(frame2)
        expected = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        np.testing.assert_array_equal(self.client.frames_np, expected)

    def test_buffer_trimmed_at_45_seconds(self):
        # 45 seconds + 1 sample at 16kHz = 720001 samples
        self.client.frames_np = np.zeros(45 * 16000 + 1, dtype=np.float32)
        self.client.add_frames(np.array([1.0], dtype=np.float32))
        # after trimming 30s, buffer should be ~15s + 1 original + 1 new
        expected_len = (45 * 16000 + 1) - (30 * 16000) + 1
        self.assertEqual(self.client.frames_np.shape[0], expected_len)
        self.assertAlmostEqual(self.client.frames_offset, 30.0)

    def test_timestamp_offset_updated_on_trim(self):
        self.client.frames_np = np.zeros(45 * 16000 + 1, dtype=np.float32)
        self.client.timestamp_offset = 5.0  # behind frames_offset after trim
        self.client.add_frames(np.array([1.0], dtype=np.float32))
        # timestamp_offset should be bumped to at least frames_offset
        self.assertGreaterEqual(self.client.timestamp_offset, self.client.frames_offset)


class TestAddFramesThreadSafety(unittest.TestCase):
    def test_concurrent_add_frames(self):
        ws = MagicMock()
        client = ConcreteServeClient(client_uid="test", websocket=ws)
        errors = []

        def add_many():
            try:
                for _ in range(100):
                    client.add_frames(np.random.randn(160).astype(np.float32))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertIsNotNone(client.frames_np)


class TestGetAudioChunkForProcessing(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(client_uid="test", websocket=self.ws)

    def test_empty_buffer_returns_empty(self):
        self.client.frames_np = np.array([], dtype=np.float32)
        chunk, duration = self.client.get_audio_chunk_for_processing()
        self.assertEqual(duration, 0.0)
        self.assertEqual(chunk.shape[0], 0)

    def test_full_buffer_no_offset(self):
        audio = np.random.randn(16000).astype(np.float32)  # 1 second
        self.client.frames_np = audio
        chunk, duration = self.client.get_audio_chunk_for_processing()
        self.assertAlmostEqual(duration, 1.0)
        np.testing.assert_array_equal(chunk, audio)

    def test_with_offset(self):
        audio = np.random.randn(32000).astype(np.float32)  # 2 seconds
        self.client.frames_np = audio
        self.client.timestamp_offset = 1.0  # skip first second
        chunk, duration = self.client.get_audio_chunk_for_processing()
        self.assertAlmostEqual(duration, 1.0)
        self.assertEqual(chunk.shape[0], 16000)


class TestClipAudioIfNoValidSegment(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(
            client_uid="test", websocket=self.ws, clip_audio=True
        )

    def test_clips_when_chunk_exceeds_25s(self):
        # 30 seconds of audio with no valid segments
        self.client.frames_np = np.zeros(30 * 16000, dtype=np.float32)
        self.client.timestamp_offset = 0.0
        self.client.frames_offset = 0.0
        self.client.clip_audio_if_no_valid_segment()
        # offset should have advanced to leave ~5s of remaining audio
        expected_offset = (30 * 16000 / 16000) - 5
        self.assertAlmostEqual(self.client.timestamp_offset, expected_offset, places=1)

    def test_no_clip_when_short(self):
        self.client.frames_np = np.zeros(10 * 16000, dtype=np.float32)
        self.client.timestamp_offset = 0.0
        self.client.frames_offset = 0.0
        self.client.clip_audio_if_no_valid_segment()
        self.assertAlmostEqual(self.client.timestamp_offset, 0.0)


class TestPrepareSegments(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(
            client_uid="test", websocket=self.ws, send_last_n_segments=3
        )

    def test_empty_transcript_no_last(self):
        segments = self.client.prepare_segments()
        self.assertEqual(segments, [])

    def test_empty_transcript_with_last(self):
        last = {"start": "0.000", "end": "1.000", "text": "hello", "completed": False}
        segments = self.client.prepare_segments(last_segment=last)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "hello")

    def test_fewer_than_n_segments(self):
        self.client.transcript = [
            {"start": "0.000", "end": "1.000", "text": "a", "completed": True},
            {"start": "1.000", "end": "2.000", "text": "b", "completed": True},
        ]
        segments = self.client.prepare_segments()
        self.assertEqual(len(segments), 2)

    def test_more_than_n_segments_truncated(self):
        self.client.transcript = [
            {"start": f"{i}.000", "end": f"{i+1}.000", "text": f"seg{i}", "completed": True}
            for i in range(10)
        ]
        segments = self.client.prepare_segments()
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0]["text"], "seg7")

    def test_last_segment_appended(self):
        self.client.transcript = [
            {"start": "0.000", "end": "1.000", "text": "a", "completed": True},
        ]
        last = {"start": "1.000", "end": "2.000", "text": "in progress", "completed": False}
        segments = self.client.prepare_segments(last_segment=last)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[-1]["text"], "in progress")


class TestFormatSegment(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(client_uid="test", websocket=self.ws)

    def test_format(self):
        seg = self.client.format_segment(1.234, 5.678, "hello world", completed=True)
        self.assertEqual(seg["start"], "1.234")
        self.assertEqual(seg["end"], "5.678")
        self.assertEqual(seg["text"], "hello world")
        self.assertTrue(seg["completed"])

    def test_format_not_completed(self):
        seg = self.client.format_segment(0.0, 1.0, "text")
        self.assertFalse(seg["completed"])


class TestSendTranscriptionToClient(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(client_uid="test-uid", websocket=self.ws)

    def test_sends_json(self):
        segments = [{"start": "0.000", "end": "1.000", "text": "hi", "completed": True}]
        self.client.send_transcription_to_client(segments)
        self.ws.send.assert_called_once()
        sent = json.loads(self.ws.send.call_args[0][0])
        self.assertEqual(sent["uid"], "test-uid")
        self.assertEqual(len(sent["segments"]), 1)

    def test_send_failure_logged_not_raised(self):
        self.ws.send.side_effect = ConnectionError("broken pipe")
        # should not raise
        self.client.send_transcription_to_client([])


class TestDisconnect(unittest.TestCase):
    def test_sends_disconnect_message(self):
        ws = MagicMock()
        client = ConcreteServeClient(client_uid="uid1", websocket=ws)
        client.disconnect()
        sent = json.loads(ws.send.call_args[0][0])
        self.assertEqual(sent["uid"], "uid1")
        self.assertEqual(sent["message"], "DISCONNECT")


class TestCleanup(unittest.TestCase):
    def test_sets_exit_flag(self):
        ws = MagicMock()
        client = ConcreteServeClient(client_uid="uid1", websocket=ws)
        self.assertFalse(client.exit)
        client.cleanup()
        self.assertTrue(client.exit)


class TestTrimTranscript(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(client_uid="test", websocket=self.ws)

    def test_transcript_trimmed_when_over_max(self):
        self.client.transcript = [
            {"start": f"{i}.000", "end": f"{i+1}.000", "text": f"seg{i}", "completed": True}
            for i in range(self.client.MAX_TRANSCRIPT_LENGTH + 100)
        ]
        self.client._trim_transcript()
        self.assertEqual(len(self.client.transcript), self.client.MAX_TRANSCRIPT_LENGTH)
        self.assertEqual(self.client.transcript[0]["text"], "seg100")

    def test_transcript_not_trimmed_when_under_max(self):
        self.client.transcript = [
            {"start": "0.000", "end": "1.000", "text": "a", "completed": True}
        ]
        self.client._trim_transcript()
        self.assertEqual(len(self.client.transcript), 1)

    def test_text_list_trimmed(self):
        self.client.text = ["word"] * (self.client.MAX_TRANSCRIPT_LENGTH + 50)
        self.client._trim_transcript()
        self.assertEqual(len(self.client.text), self.client.MAX_TRANSCRIPT_LENGTH)


class TestUpdateSegments(unittest.TestCase):
    """Tests for the core update_segments() logic."""

    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(
            client_uid="test",
            websocket=self.ws,
            no_speech_thresh=0.45,
            same_output_threshold=3,
        )
        self.client.frames_np = np.zeros(16000 * 5, dtype=np.float32)

    def _make_segment(self, start, end, text, no_speech_prob=0.0):
        seg = MagicMock()
        seg.start = start
        seg.end = end
        seg.text = text
        seg.no_speech_prob = no_speech_prob
        return seg

    def test_single_segment_becomes_last(self):
        segs = [self._make_segment(0.0, 1.0, " hello")]
        last = self.client.update_segments(segs, duration=2.0)
        self.assertIsNotNone(last)
        self.assertIn("hello", last["text"])
        self.assertFalse(last["completed"])
        self.assertEqual(len(self.client.transcript), 0)

    def test_multiple_segments_completes_all_but_last(self):
        segs = [
            self._make_segment(0.0, 1.0, " first"),
            self._make_segment(1.0, 2.0, " second"),
        ]
        last = self.client.update_segments(segs, duration=3.0)
        self.assertEqual(len(self.client.transcript), 1)
        self.assertTrue(self.client.transcript[0]["completed"])
        self.assertIn("first", self.client.transcript[0]["text"])
        self.assertIsNotNone(last)
        self.assertIn("second", last["text"])

    def test_high_no_speech_prob_skipped(self):
        segs = [
            self._make_segment(0.0, 1.0, " noise", no_speech_prob=0.9),
            self._make_segment(1.0, 2.0, " also noise", no_speech_prob=0.9),
        ]
        last = self.client.update_segments(segs, duration=3.0)
        self.assertEqual(len(self.client.transcript), 0)
        self.assertIsNone(last)

    def test_segment_with_start_gte_end_skipped(self):
        segs = [
            self._make_segment(1.0, 0.5, " backwards"),
            self._make_segment(1.5, 2.0, " normal"),
        ]
        last = self.client.update_segments(segs, duration=3.0)
        self.assertEqual(len(self.client.transcript), 0)
        self.assertIsNotNone(last)

    def test_repeated_output_triggers_completion(self):
        seg = self._make_segment(0.0, 1.0, " repeated")
        for _ in range(self.client.same_output_threshold + 2):
            last = self.client.update_segments([seg], duration=2.0)
        # after enough repeats, should be added to transcript
        self.assertTrue(len(self.client.transcript) >= 1)

    def test_translation_queue_receives_completed(self):
        q = queue.Queue()
        self.client.translation_queue = q
        segs = [
            self._make_segment(0.0, 1.0, " first"),
            self._make_segment(1.0, 2.0, " second"),
        ]
        self.client.update_segments(segs, duration=3.0)
        self.assertFalse(q.empty())
        item = q.get_nowait()
        self.assertIn("first", item["text"])

    def test_timestamp_offset_advances(self):
        segs = [
            self._make_segment(0.0, 1.0, " first"),
            self._make_segment(1.0, 2.0, " second"),
        ]
        self.client.update_segments(segs, duration=3.0)
        self.assertGreater(self.client.timestamp_offset, 0.0)


class TestGetSegmentHelpers(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(client_uid="test", websocket=self.ws)

    def test_get_segment_no_speech_prob_attr(self):
        seg = MagicMock()
        seg.no_speech_prob = 0.3
        self.assertAlmostEqual(self.client.get_segment_no_speech_prob(seg), 0.3)

    def test_get_segment_no_speech_prob_fallback(self):
        seg = MagicMock(spec=[])  # no attributes
        self.assertEqual(self.client.get_segment_no_speech_prob(seg), 0)

    def test_get_segment_start_uses_start(self):
        seg = MagicMock()
        seg.start = 1.5
        self.assertAlmostEqual(self.client.get_segment_start(seg), 1.5)

    def test_get_segment_end_uses_end(self):
        seg = MagicMock()
        seg.end = 3.0
        self.assertAlmostEqual(self.client.get_segment_end(seg), 3.0)

    def test_get_segment_start_fallback_to_start_ts(self):
        seg = MagicMock(spec=["start_ts"])
        seg.start_ts = 2.0
        self.assertAlmostEqual(self.client.get_segment_start(seg), 2.0)


if __name__ == "__main__":
    unittest.main()
