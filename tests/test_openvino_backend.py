"""Tests for the OpenVINO backend's initial_prompt/hotwords wiring.

The openvino/openvino_genai packages are optional (commented out in
requirements/server.txt) and are not installed in CI's default test
environment, so they're stubbed via sys.modules before importing the
modules under test - mirroring how this suite already treats other
optional, hardware-gated dependencies.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _install_openvino_stubs():
    """Install minimal stand-ins for openvino/openvino_genai/huggingface_hub."""
    openvino_mod = types.ModuleType("openvino")
    openvino_mod.Core = MagicMock()

    openvino_genai_mod = types.ModuleType("openvino_genai")
    openvino_genai_mod.WhisperPipeline = MagicMock()

    hf_hub_mod = types.ModuleType("huggingface_hub")
    hf_hub_mod.snapshot_download = MagicMock()

    sys.modules["openvino"] = openvino_mod
    sys.modules["openvino_genai"] = openvino_genai_mod
    sys.modules["huggingface_hub"] = hf_hub_mod


_install_openvino_stubs()

from whisper_live.backend.openvino_backend import ServeClientOpenVINO  # noqa: E402
from whisper_live.transcriber.transcriber_openvino import WhisperOpenVINO  # noqa: E402


class TestWhisperOpenVINOInitialPromptHotwords(unittest.TestCase):
    """WhisperOpenVINO must forward initial_prompt/hotwords to generate()."""

    def _make_transcriber(self, **kwargs):
        with patch("whisper_live.transcriber.transcriber_openvino.hf_hub"), \
             patch("os.path.exists", return_value=True):
            return WhisperOpenVINO(**kwargs)

    def test_defaults_pass_none_through_to_generate(self):
        transcriber = self._make_transcriber()
        transcriber.model = MagicMock()
        transcriber.model.generate.return_value.chunks = []

        transcriber.transcribe([0.0, 0.0])

        _, kwargs = transcriber.model.generate.call_args
        self.assertIsNone(kwargs["initial_prompt"])
        self.assertIsNone(kwargs["hotwords"])

    def test_initial_prompt_and_hotwords_reach_generate(self):
        transcriber = self._make_transcriber(
            initial_prompt="Jane Doe context",
            hotwords="WhisperLive,TensorRT",
        )
        transcriber.model = MagicMock()
        transcriber.model.generate.return_value.chunks = []

        transcriber.transcribe([0.0, 0.0])

        _, kwargs = transcriber.model.generate.call_args
        self.assertEqual(kwargs["initial_prompt"], "Jane Doe context")
        self.assertEqual(kwargs["hotwords"], "WhisperLive,TensorRT")
        # Existing behavior (return_timestamps/language/task) must be unaffected.
        self.assertTrue(kwargs["return_timestamps"])


class TestServeClientOpenVINOInitialPromptHotwords(unittest.TestCase):
    """ServeClientOpenVINO must store and forward initial_prompt/hotwords."""

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
            return client, mock_whisper_openvino

    def test_default_has_no_prompt_or_hotwords(self):
        client, mock_whisper_openvino = self._make_client()
        self.assertIsNone(client.initial_prompt)
        self.assertIsNone(client.hotwords)
        _, kwargs = mock_whisper_openvino.call_args
        self.assertIsNone(kwargs["initial_prompt"])
        self.assertIsNone(kwargs["hotwords"])

    def test_initial_prompt_and_hotwords_stored_and_forwarded(self):
        client, mock_whisper_openvino = self._make_client(
            initial_prompt="steer the model",
            hotwords="acme,foobar",
        )
        self.assertEqual(client.initial_prompt, "steer the model")
        self.assertEqual(client.hotwords, "acme,foobar")
        _, kwargs = mock_whisper_openvino.call_args
        self.assertEqual(kwargs["initial_prompt"], "steer the model")
        self.assertEqual(kwargs["hotwords"], "acme,foobar")


class TestServerForwardsInitialPromptHotwordsToOpenVINO(unittest.TestCase):
    """TranscriptionServer.initialize_client must forward the client's
    initial_prompt/hotwords options through to ServeClientOpenVINO."""

    def test_options_are_forwarded(self):
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
            "initial_prompt": "Jane Doe context",
            "hotwords": "WhisperLive,TensorRT",
        }

        with patch(
            "whisper_live.backend.openvino_backend.ServeClientOpenVINO"
        ) as mock_backend:
            server.initialize_client(ws, options, None, None, False)

        _, kwargs = mock_backend.call_args
        self.assertEqual(kwargs["initial_prompt"], "Jane Doe context")
        self.assertEqual(kwargs["hotwords"], "WhisperLive,TensorRT")


if __name__ == "__main__":
    unittest.main()
