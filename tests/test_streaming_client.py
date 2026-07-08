import json
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from whisper_live.client import Client, StreamingTranscriptionClient


class StreamingClientTestCase(unittest.TestCase):
    @patch('whisper_live.client.websocket.WebSocketApp')
    def setUp(self, mock_websocket):
        self.mock_websocket = mock_websocket
        self.mock_ws_app = mock_websocket.return_value
        self.mock_ws_app.send = MagicMock()

        self.committed = []
        self.partials = []
        self.session_started = []

        self.client = StreamingTranscriptionClient(
            host='localhost',
            port=9090,
            lang="en",
            on_session_started=lambda: self.session_started.append(True),
            on_committed_transcript=lambda text, segs: self.committed.append((text, segs)),
            on_partial_transcript=lambda text, segs: self.partials.append((text, segs)),
        )
        self._inner = self.client._client

    def tearDown(self):
        self._inner.close_websocket()
        self.mock_websocket.stop()

    def _server_ready(self, backend="faster_whisper"):
        self._inner.on_message(self.mock_ws_app, json.dumps({
            "uid": self._inner.uid,
            "message": "SERVER_READY",
            "backend": backend,
        }))

    def _send_segments(self, segments):
        self._inner.on_message(self.mock_ws_app, json.dumps({
            "uid": self._inner.uid,
            "segments": segments,
        }))


class TestPcmFormatConversion(StreamingClientTestCase):
    def test_int16_is_normalized_to_float32(self):
        self._server_ready()
        raw = np.array([0, 16384, -32768], dtype=np.int16).tobytes()
        with patch.object(self._inner, 'send_packet_to_server') as mock_send:
            self.client.send(raw, pcm_format="int16")
        sent = np.frombuffer(mock_send.call_args[0][0], dtype=np.float32)
        np.testing.assert_allclose(sent, [0.0, 0.5, -1.0], atol=1e-4)

    def test_float32_passes_through(self):
        self._server_ready()
        raw = np.array([0.1, -0.2], dtype=np.float32).tobytes()
        with patch.object(self._inner, 'send_packet_to_server') as mock_send:
            self.client.send(raw, pcm_format="float32")
        self.assertEqual(mock_send.call_args[0][0], raw)

    def test_default_format_is_int16(self):
        self._server_ready()
        raw = np.array([32767], dtype=np.int16).tobytes()
        with patch.object(self._inner, 'send_packet_to_server') as mock_send:
            self.client.send(raw)
        sent = np.frombuffer(mock_send.call_args[0][0], dtype=np.float32)
        self.assertAlmostEqual(float(sent[0]), 32767 / 32768.0, places=4)

    def test_unsupported_format_raises(self):
        self._server_ready()
        with self.assertRaises(ValueError):
            self.client.send(b"\x00\x00", pcm_format="int8")

    def test_send_after_close_raises(self):
        self.client._closed = True
        with self.assertRaises(RuntimeError):
            self.client.send(b"\x00\x00", pcm_format="int16")

    def test_send_array_normalizes_integers(self):
        with patch.object(self._inner, 'send_packet_to_server') as mock_send:
            self.client.send_array(np.array([0, 16384, -32768], dtype=np.int16))
        sent = np.frombuffer(mock_send.call_args[0][0], dtype=np.float32)
        np.testing.assert_allclose(sent, [0.0, 0.5, -1.0], atol=1e-4)


class TestTranscriptDispatch(StreamingClientTestCase):
    def test_partial_then_committed(self):
        self._server_ready()
        self._send_segments([{"start": 0, "end": 1, "text": "hello", "completed": False}])
        self.assertEqual(len(self.partials), 1)
        self.assertEqual(self.partials[0][0], "hello")
        self.assertEqual(len(self.committed), 0)

        self._send_segments([{"start": 0, "end": 1, "text": "hello world", "completed": True}])
        self.assertEqual(len(self.committed), 1)
        self.assertEqual(self.committed[0][0], "hello world")
        self.assertEqual(len(self.client.transcript), 1)

    def test_committed_deduplicated(self):
        self._server_ready()
        seg = {"start": 0, "end": 1, "text": "hi", "completed": True}
        self._send_segments([seg])
        self._send_segments([seg])
        self.assertEqual(len(self.committed), 1)
        self.assertEqual(len(self.client.transcript), 1)

    def test_committed_backend_agnostic(self):
        """Committed dispatch must work for non-faster_whisper backends."""
        self._server_ready(backend="tensorrt")
        self._send_segments([{"start": 0, "end": 1, "text": "trt seg", "completed": True}])
        self.assertEqual(len(self.committed), 1)
        self.assertEqual(len(self.client.transcript), 1)

    def test_last_partial_alias(self):
        self._server_ready()
        self._send_segments([{"start": 0, "end": 1, "text": "pending", "completed": False}])
        self.assertIsNotNone(self.client.last_partial)
        self.assertIs(self.client.last_partial, self.client.last_segment)


class TestConnectLifecycle(StreamingClientTestCase):
    def test_connect_returns_after_ready(self):
        self._server_ready()
        self.assertIs(self.client.connect(), self.client)
        self.assertEqual(len(self.session_started), 1)

    def test_connect_times_out(self):
        self.client._ready_timeout = 0.1
        with self.assertRaises(TimeoutError):
            self.client.connect()

    def test_connect_raises_on_server_error(self):
        self._inner.on_message(self.mock_ws_app, json.dumps({
            "uid": self._inner.uid,
            "status": "ERROR",
            "message": "boom",
        }))
        with self.assertRaises(RuntimeError):
            self.client.connect()

    def test_connect_raises_when_server_full(self):
        self._inner.on_message(self.mock_ws_app, json.dumps({
            "uid": self._inner.uid,
            "status": "WAIT",
            "message": 5,
        }))
        with self.assertRaises(RuntimeError):
            self.client.connect()

    def test_close_sends_end_of_audio(self):
        self._server_ready()
        with patch.object(self._inner, 'send_packet_to_server') as mock_send, \
                patch.object(self._inner, 'close_websocket') as mock_close:
            self.client.close(drain_seconds=0)
        mock_send.assert_called_once_with(Client.END_OF_AUDIO.encode("utf-8"))
        mock_close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
