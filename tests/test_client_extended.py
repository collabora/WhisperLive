import json
import time
import unittest
from unittest.mock import patch, MagicMock, PropertyMock

from whisper_live.client import Client, TranscriptionTeeClient


class TestClientStatusMessages(unittest.TestCase):
    """Tests for Client.handle_status_messages() and on_message() branches."""

    @patch("whisper_live.client.websocket.WebSocketApp")
    @patch("whisper_live.client.pyaudio.PyAudio")
    def setUp(self, mock_pyaudio, mock_websocket):
        mock_pyaudio.return_value.open.return_value = MagicMock()
        self.client = Client(host="localhost", port=9090, lang="en")

    def tearDown(self):
        self.client.close_websocket()

    def test_wait_status(self):
        msg = {"uid": self.client.uid, "status": "WAIT", "message": 5.0}
        self.client.handle_status_messages(msg)
        self.assertTrue(self.client.waiting)

    def test_error_status(self):
        msg = {"uid": self.client.uid, "status": "ERROR", "message": "model not found"}
        self.client.handle_status_messages(msg)
        self.assertTrue(self.client.server_error)

    def test_warning_status_no_side_effects(self):
        msg = {"uid": self.client.uid, "status": "WARNING", "message": "fallback backend"}
        self.client.handle_status_messages(msg)
        self.assertFalse(self.client.server_error)
        self.assertFalse(self.client.waiting)

    def test_on_message_wrong_uid_ignored(self):
        msg = json.dumps({"uid": "wrong-uid", "segments": [{"start": 0, "end": 1, "text": "hi", "completed": True}]})
        self.client.on_message(MagicMock(), msg)
        self.assertEqual(len(self.client.transcript), 0)

    def test_on_message_disconnect(self):
        self.client.recording = True
        msg = json.dumps({"uid": self.client.uid, "message": "DISCONNECT"})
        self.client.on_message(MagicMock(), msg)
        self.assertFalse(self.client.recording)

    def test_on_message_server_ready(self):
        msg = json.dumps({
            "uid": self.client.uid,
            "message": "SERVER_READY",
            "backend": "faster_whisper",
        })
        self.client.on_message(MagicMock(), msg)
        self.assertTrue(self.client.recording)
        self.assertEqual(self.client.server_backend, "faster_whisper")

    def test_on_message_language_detection(self):
        msg = json.dumps({
            "uid": self.client.uid,
            "language": "fr",
            "language_prob": 0.95,
        })
        self.client.on_message(MagicMock(), msg)
        self.assertEqual(self.client.language, "fr")


class TestClientTranslationFlow(unittest.TestCase):
    """Tests for the translation-related client functionality."""

    @patch("whisper_live.client.websocket.WebSocketApp")
    @patch("whisper_live.client.pyaudio.PyAudio")
    def setUp(self, mock_pyaudio, mock_websocket):
        mock_pyaudio.return_value.open.return_value = MagicMock()
        self.client = Client(
            host="localhost",
            port=9090,
            lang="en",
            enable_translation=True,
            target_language="es",
        )
        # simulate SERVER_READY so server_backend is set
        ready_msg = json.dumps({
            "uid": self.client.uid,
            "message": "SERVER_READY",
            "backend": "faster_whisper",
        })
        self.client.on_message(MagicMock(), ready_msg)

    def tearDown(self):
        self.client.close_websocket()

    def test_on_open_includes_translation_fields(self):
        mock_ws = MagicMock()
        self.client.on_open(mock_ws)
        sent = json.loads(mock_ws.send.call_args[0][0])
        self.assertTrue(sent["enable_translation"])
        self.assertEqual(sent["target_language"], "es")

    def test_translated_segments_processed(self):
        msg = json.dumps({
            "uid": self.client.uid,
            "translated_segments": [
                {"start": "0.000", "end": "1.000", "text": "Hola mundo", "completed": True},
            ],
        })
        self.client.on_message(MagicMock(), msg)
        self.assertEqual(len(self.client.translated_transcript), 1)
        self.assertEqual(self.client.translated_transcript[0]["text"], "Hola mundo")

    def test_translation_callback_invoked(self):
        callback = MagicMock()
        self.client.translation_callback = callback
        msg = json.dumps({
            "uid": self.client.uid,
            "translated_segments": [
                {"start": "0.000", "end": "1.000", "text": "Hola", "completed": True},
            ],
        })
        self.client.on_message(MagicMock(), msg)
        callback.assert_called_once()

    def test_translation_callback_exception_handled(self):
        callback = MagicMock(side_effect=RuntimeError("callback broke"))
        self.client.translation_callback = callback
        msg = json.dumps({
            "uid": self.client.uid,
            "translated_segments": [
                {"start": "0.000", "end": "1.000", "text": "Hola", "completed": True},
            ],
        })
        # should not raise
        self.client.on_message(MagicMock(), msg)


class TestClientTranscriptionCallback(unittest.TestCase):
    """Tests for the transcription callback feature."""

    @patch("whisper_live.client.websocket.WebSocketApp")
    @patch("whisper_live.client.pyaudio.PyAudio")
    def setUp(self, mock_pyaudio, mock_websocket):
        mock_pyaudio.return_value.open.return_value = MagicMock()
        self.callback = MagicMock()
        self.client = Client(
            host="localhost",
            port=9090,
            lang="en",
            transcription_callback=self.callback,
        )
        ready_msg = json.dumps({
            "uid": self.client.uid,
            "message": "SERVER_READY",
            "backend": "faster_whisper",
        })
        self.client.on_message(MagicMock(), ready_msg)

    def tearDown(self):
        self.client.close_websocket()

    def test_callback_receives_text_and_segments(self):
        msg = json.dumps({
            "uid": self.client.uid,
            "segments": [
                {"start": "0.000", "end": "1.000", "text": "Hello", "completed": True},
            ],
        })
        self.client.on_message(MagicMock(), msg)
        self.callback.assert_called_once()
        text_arg, segments_arg = self.callback.call_args[0]
        self.assertIn("Hello", text_arg)
        self.assertIsInstance(segments_arg, list)

    def test_callback_exception_does_not_crash(self):
        self.callback.side_effect = ValueError("boom")
        msg = json.dumps({
            "uid": self.client.uid,
            "segments": [
                {"start": "0.000", "end": "1.000", "text": "Test", "completed": True},
            ],
        })
        # should not raise
        self.client.on_message(MagicMock(), msg)


class TestClientSrtWriting(unittest.TestCase):
    """Tests for Client.write_srt_file() edge cases."""

    @patch("whisper_live.client.websocket.WebSocketApp")
    @patch("whisper_live.client.pyaudio.PyAudio")
    def setUp(self, mock_pyaudio, mock_websocket):
        mock_pyaudio.return_value.open.return_value = MagicMock()
        self.client = Client(host="localhost", port=9090, lang="en")
        self.client.server_backend = "faster_whisper"

    def tearDown(self):
        self.client.close_websocket()
        import os
        for f in ["test_out.srt"]:
            if os.path.exists(f):
                os.remove(f)

    def test_write_srt_empty_transcript_with_last_segment(self):
        self.client.transcript = []
        self.client.last_segment = {"start": "0.000", "end": "1.000", "text": "final"}
        self.client.write_srt_file("test_out.srt")
        self.assertEqual(len(self.client.transcript), 1)
        self.assertEqual(self.client.transcript[0]["text"], "final")

    def test_write_srt_appends_last_segment_if_different(self):
        self.client.transcript = [{"start": "0.000", "end": "1.000", "text": "first"}]
        self.client.last_segment = {"start": "1.000", "end": "2.000", "text": "second"}
        self.client.write_srt_file("test_out.srt")
        self.assertEqual(len(self.client.transcript), 2)

    def test_write_srt_no_duplicate_last_segment(self):
        self.client.transcript = [{"start": "0.000", "end": "1.000", "text": "same"}]
        self.client.last_segment = {"start": "0.000", "end": "1.000", "text": "same"}
        self.client.write_srt_file("test_out.srt")
        self.assertEqual(len(self.client.transcript), 1)


class TestWaitBeforeDisconnect(unittest.TestCase):
    """Tests for Client.wait_before_disconnect()."""

    @patch("whisper_live.client.websocket.WebSocketApp")
    @patch("whisper_live.client.pyaudio.PyAudio")
    def setUp(self, mock_pyaudio, mock_websocket):
        mock_pyaudio.return_value.open.return_value = MagicMock()
        self.client = Client(host="localhost", port=9090, lang="en")

    def tearDown(self):
        self.client.close_websocket()

    def test_raises_if_no_response(self):
        self.client.last_response_received = None
        with self.assertRaises(AssertionError):
            self.client.wait_before_disconnect()

    def test_returns_immediately_if_timeout_elapsed(self):
        self.client.last_response_received = time.time() - 100
        self.client.disconnect_if_no_response_for = 15
        start = time.time()
        self.client.wait_before_disconnect()
        elapsed = time.time() - start
        self.assertLess(elapsed, 1.0)


class TestTeeClientEdgeCases(unittest.TestCase):
    """Edge cases for TranscriptionTeeClient."""

    def test_empty_clients_raises(self):
        with self.assertRaises(Exception):
            TranscriptionTeeClient([])


class TestClientReconnect(unittest.TestCase):
    """Tests for reconnection logic."""

    @patch("whisper_live.client.websocket.WebSocketApp")
    @patch("whisper_live.client.pyaudio.PyAudio")
    def test_reconnect_on_close(self, mock_pyaudio, mock_websocket):
        mock_pyaudio.return_value.open.return_value = MagicMock()
        client = Client(host="localhost", port=9090, lang="en", max_retries=2, retry_delay=0)
        initial_socket = client.client_socket
        client.on_close(MagicMock(), 1006, "abnormal closure")
        self.assertEqual(client._retry_count, 1)
        # A new websocket should have been created
        self.assertIsNotNone(client.client_socket)
        client.close_websocket()

    @patch("whisper_live.client.websocket.WebSocketApp")
    @patch("whisper_live.client.pyaudio.PyAudio")
    def test_no_reconnect_on_server_error(self, mock_pyaudio, mock_websocket):
        mock_pyaudio.return_value.open.return_value = MagicMock()
        client = Client(host="localhost", port=9090, lang="en", max_retries=2, retry_delay=0)
        client.server_error = True
        client.on_close(MagicMock(), 1000, "normal")
        self.assertEqual(client._retry_count, 0)
        client.close_websocket()

    @patch("whisper_live.client.websocket.WebSocketApp")
    @patch("whisper_live.client.pyaudio.PyAudio")
    def test_no_reconnect_when_max_retries_zero(self, mock_pyaudio, mock_websocket):
        mock_pyaudio.return_value.open.return_value = MagicMock()
        client = Client(host="localhost", port=9090, lang="en", max_retries=0, retry_delay=0)
        client.on_close(MagicMock(), 1006, "abnormal closure")
        self.assertEqual(client._retry_count, 0)
        client.close_websocket()

    @patch("whisper_live.client.websocket.WebSocketApp")
    @patch("whisper_live.client.pyaudio.PyAudio")
    def test_stops_after_max_retries(self, mock_pyaudio, mock_websocket):
        mock_pyaudio.return_value.open.return_value = MagicMock()
        client = Client(host="localhost", port=9090, lang="en", max_retries=2, retry_delay=0)
        client.on_close(MagicMock(), 1006, "closed")
        client.on_close(MagicMock(), 1006, "closed")
        self.assertEqual(client._retry_count, 2)
        # third close should NOT retry
        client.on_close(MagicMock(), 1006, "closed")
        self.assertEqual(client._retry_count, 2)
        client.close_websocket()


if __name__ == "__main__":
    unittest.main()
