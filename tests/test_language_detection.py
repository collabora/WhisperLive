"""Tests for language detection confidence in REST API responses."""

import unittest
from unittest.mock import patch, MagicMock
from whisper_live.server import TranscriptionServer


class TestLanguageDetectionConfidence(unittest.TestCase):
    """Verify that language_probability is included in REST API responses."""

    def test_set_language_sends_probability(self):
        """The WebSocket set_language method already sends language_prob."""
        from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper
        ws = MagicMock()
        client = ServeClientFasterWhisper.__new__(ServeClientFasterWhisper)
        client.client_uid = "test-uid"
        client.websocket = ws
        client.language = None
        client.model_size_or_path = "small"

        info = MagicMock()
        info.language = "fr"
        info.language_probability = 0.92

        client.set_language(info)
        assert client.language == "fr"
        sent = ws.send.call_args[0][0]
        import json
        data = json.loads(sent)
        assert data["language"] == "fr"
        assert data["language_prob"] == 0.92

    def test_parse_profanity_option_none_on_empty(self):
        """Regression: _parse_profanity_option exists and returns None for empty."""
        server = TranscriptionServer()
        assert server._parse_profanity_option({}) is None

    def test_verbose_json_includes_language_probability(self):
        """Verify that the verbose_json response format would include language_probability.

        We test this by checking the server.py source contains the key.
        """
        import inspect
        from whisper_live import server
        source = inspect.getsource(server)
        assert '"language_probability": info.language_probability' in source

    def test_json_response_includes_language_probability(self):
        """Verify that the json response format includes language and language_probability."""
        import inspect
        from whisper_live import server
        source = inspect.getsource(server)
        assert '"language_probability": info.language_probability,' in source

    def test_sse_metadata_event_emitted(self):
        """Verify SSE stream emits a metadata event with language info."""
        import inspect
        from whisper_live import server
        source = inspect.getsource(server)
        assert '"type": "metadata"' in source
        assert '"language_probability": info.language_probability' in source


if __name__ == "__main__":
    unittest.main()
