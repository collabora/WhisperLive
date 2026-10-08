"""Tests for diarization/translation support on the OpenVINO backend.

The openvino/openvino_genai packages are optional (commented out in
requirements/server.txt) and are not installed in CI's default test
environment, so they're stubbed via sys.modules before importing the
modules under test - mirroring how this suite already treats other
optional, hardware-gated dependencies.
"""
import queue
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np


def _install_openvino_stubs():
    """Install minimal stand-ins for openvino/openvino_genai.

    huggingface_hub is deliberately NOT stubbed here: it's a real,
    already-installed dependency (pulled in transitively via transformers),
    and replacing it in sys.modules would break transformers' own imports
    for the rest of the test process. WhisperOpenVINO (the only caller of
    hf_hub.snapshot_download) is itself mocked out wherever it's
    constructed in this file, so the real huggingface_hub is never invoked.
    """
    if "openvino" not in sys.modules:
        openvino_mod = types.ModuleType("openvino")
        openvino_mod.Core = MagicMock()
        sys.modules["openvino"] = openvino_mod

    if "openvino_genai" not in sys.modules:
        openvino_genai_mod = types.ModuleType("openvino_genai")
        openvino_genai_mod.WhisperPipeline = MagicMock()
        sys.modules["openvino_genai"] = openvino_genai_mod


_install_openvino_stubs()

from whisper_live.backend.openvino_backend import ServeClientOpenVINO  # noqa: E402


class TestServeClientOpenVINODiarizationTranslation(unittest.TestCase):
    """ServeClientOpenVINO must forward diarization/translation_queue to the
    shared ServeClientBase, which already implements both features."""

    def _make_client(self, **kwargs):
        ws = MagicMock()
        with patch(
            "whisper_live.backend.openvino_backend.WhisperOpenVINO"
        ) as mock_whisper_openvino, patch(
            "whisper_live.backend.openvino_backend.threading.Thread"
        ):
            # Real transcription thread loops forever (speech_to_text) until
            # .exit is set; stub Thread so __init__ doesn't spawn one.
            mock_whisper_openvino.return_value = MagicMock()
            client = ServeClientOpenVINO(
                ws,
                client_uid="test-uid",
                model="OpenVINO/whisper-tiny-fp16-ov",
                **kwargs,
            )
            return client

    def test_defaults_have_no_diarization_or_translation(self):
        client = self._make_client()
        self.assertIsNone(client.diarization)
        self.assertIsNone(client.translation_queue)

    def test_diarization_and_translation_queue_are_stored(self):
        diarizer = MagicMock()
        tq = queue.Queue()
        client = self._make_client(diarization=diarizer, translation_queue=tq)
        self.assertIs(client.diarization, diarizer)
        self.assertIs(client.translation_queue, tq)

    def test_diarization_and_translation_actually_fire_via_update_segments(self):
        """End-to-end proof: once diarization/translation_queue are wired in,
        the OpenVINO backend's own handle_transcription_output (which calls
        the shared update_segments()) already triggers both - no other code
        change needed, confirming this is a wiring gap, not a missing
        capability."""
        diarizer = MagicMock()
        diarizer.identify_speaker.return_value = "SPEAKER_00"
        tq = queue.Queue()
        client = self._make_client(diarization=diarizer, translation_queue=tq)

        # Simulate 45s of buffered audio so _identify_speaker's slice check passes.
        client.frames_np = np.zeros(client.RATE * 45, dtype=np.float32)
        client.frames_offset = 0.0
        client.timestamp_offset = 0.0

        # Two OpenVINO-shaped chunks (start_ts/end_ts, per WhisperDecodedResultChunk);
        # only the first is "completed" by update_segments (all but the last segment).
        segments = [
            SimpleNamespace(text="Hello there", start_ts=0.0, end_ts=2.0),
            SimpleNamespace(text=" incomplete tail", start_ts=2.0, end_ts=3.0),
        ]

        client.handle_transcription_output(segments, duration=3.0)

        diarizer.identify_speaker.assert_called_once()
        self.assertEqual(client.transcript[-1]["speaker"], "SPEAKER_00")

        queued_segment = tq.get_nowait()
        self.assertEqual(queued_segment["text"], "Hello there")
        self.assertEqual(queued_segment["speaker"], "SPEAKER_00")


class TestServerForwardsDiarizationTranslationToOpenVINO(unittest.TestCase):
    """TranscriptionServer.initialize_client must forward a diarizer and the
    translation queue through to ServeClientOpenVINO."""

    def test_diarization_and_translation_queue_are_forwarded(self):
        from whisper_live.server import TranscriptionServer, BackendType, ClientManager

        server = TranscriptionServer()
        server.backend = BackendType.OPENVINO
        server.single_model = False
        server.client_manager = ClientManager(max_clients=2, max_connection_time=60)

        ws = MagicMock()
        options = {
            "uid": "abc",
            "language": "en",
            "task": "transcribe",
            "model": "OpenVINO/whisper-tiny-fp16-ov",
            "enable_diarization": True,
        }

        fake_diarizer = MagicMock()
        with patch(
            "whisper_live.backend.openvino_backend.ServeClientOpenVINO"
        ) as mock_backend, patch.object(
            TranscriptionServer, "_create_diarizer", return_value=fake_diarizer
        ):
            server.initialize_client(ws, options, None, None, False)

        _, kwargs = mock_backend.call_args
        self.assertIs(kwargs["diarization"], fake_diarizer)
        # No enable_translation in options -> queue stays None, but the kwarg
        # must still be forwarded (proves the wiring exists, not just a default).
        self.assertIsNone(kwargs["translation_queue"])

    def test_translation_queue_is_forwarded_when_enabled(self):
        import whisper_live.backend.translation_backend  # noqa: F401 (registers submodule for patch())
        from whisper_live.server import TranscriptionServer, BackendType, ClientManager

        server = TranscriptionServer()
        server.backend = BackendType.OPENVINO
        server.single_model = False
        server.client_manager = ClientManager(max_clients=2, max_connection_time=60)

        ws = MagicMock()
        options = {
            "uid": "abc",
            "language": "en",
            "task": "transcribe",
            "model": "OpenVINO/whisper-tiny-fp16-ov",
            "enable_translation": True,
            "target_language": "fr",
        }

        with patch(
            "whisper_live.backend.openvino_backend.ServeClientOpenVINO"
        ) as mock_backend, patch(
            "whisper_live.backend.translation_backend.ServeClientTranslation"
        ), patch(
            "threading.Thread"
        ):
            server.initialize_client(ws, options, None, None, False)

        _, kwargs = mock_backend.call_args
        self.assertIsInstance(kwargs["translation_queue"], queue.Queue)


if __name__ == "__main__":
    unittest.main()
