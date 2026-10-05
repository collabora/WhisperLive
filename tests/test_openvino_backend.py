"""Tests for the OpenVINO backend's initial_prompt/hotwords wiring.

openvino/openvino_genai are listed in requirements/server.txt, but a
developer running this suite without the optional OpenVINO extras installed
should still be able to run it, so each missing package is stubbed via
sys.modules before the modules under test are imported. Packages that are
installed are left untouched.
"""
import importlib
import inspect
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _install_openvino_stubs():
    """Install minimal stand-ins for whichever of these packages is missing."""
    stubs = {
        "openvino": {"Core": MagicMock()},
        "openvino_genai": {"WhisperPipeline": MagicMock()},
        "huggingface_hub": {"snapshot_download": MagicMock()},
    }
    for name, attrs in stubs.items():
        try:
            importlib.import_module(name)
        except ImportError:
            module = types.ModuleType(name)
            for attr, value in attrs.items():
                setattr(module, attr, value)
            sys.modules[name] = module


_install_openvino_stubs()

from whisper_live.backend.openvino_backend import ServeClientOpenVINO  # noqa: E402
from whisper_live.transcriber.transcriber_openvino import WhisperOpenVINO  # noqa: E402


class TestWhisperOpenVINOInitialPromptHotwords(unittest.TestCase):
    """WhisperOpenVINO.transcribe must forward per-call initial_prompt/hotwords
    to generate()."""

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
        transcriber = self._make_transcriber()
        transcriber.model = MagicMock()
        transcriber.model.generate.return_value.chunks = []

        transcriber.transcribe(
            [0.0, 0.0],
            initial_prompt="Jane Doe context",
            hotwords="WhisperLive,TensorRT",
        )

        _, kwargs = transcriber.model.generate.call_args
        self.assertEqual(kwargs["initial_prompt"], "Jane Doe context")
        self.assertEqual(kwargs["hotwords"], "WhisperLive,TensorRT")
        # Existing behavior (return_timestamps/language/task) must be unaffected.
        self.assertTrue(kwargs["return_timestamps"])


class TestServeClientOpenVINOInitialPromptHotwords(unittest.TestCase):
    """ServeClientOpenVINO must store each client's initial_prompt/hotwords and
    pass them on every transcribe call."""

    def setUp(self):
        # SINGLE_MODEL is class-level state; keep tests independent.
        self._saved_single_model = ServeClientOpenVINO.SINGLE_MODEL
        ServeClientOpenVINO.SINGLE_MODEL = None
        self.addCleanup(self._restore_single_model)

    def _restore_single_model(self):
        ServeClientOpenVINO.SINGLE_MODEL = self._saved_single_model

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
        client, _ = self._make_client()
        self.assertIsNone(client.initial_prompt)
        self.assertIsNone(client.hotwords)

        client.transcribe_audio([0.0, 0.0])

        _, kwargs = client.transcriber.transcribe.call_args
        self.assertIsNone(kwargs["initial_prompt"])
        self.assertIsNone(kwargs["hotwords"])

    def test_initial_prompt_and_hotwords_stored_and_passed_per_call(self):
        client, _ = self._make_client(
            initial_prompt="steer the model",
            hotwords="acme,foobar",
        )
        self.assertEqual(client.initial_prompt, "steer the model")
        self.assertEqual(client.hotwords, "acme,foobar")

        client.transcribe_audio([0.0, 0.0])

        _, kwargs = client.transcriber.transcribe.call_args
        self.assertEqual(kwargs["initial_prompt"], "steer the model")
        self.assertEqual(kwargs["hotwords"], "acme,foobar")

    def test_hotwords_is_last_positional_parameter(self):
        # Adding hotwords must not shift the position of existing parameters
        # for any caller that passes them positionally.
        params = list(inspect.signature(ServeClientOpenVINO.__init__).parameters)
        self.assertEqual(params[-1], "hotwords")

    def test_single_model_clients_keep_their_own_options(self):
        # With single_model=True the second client reuses the first client's
        # transcriber, so its own prompt/hotwords must still reach generate().
        first, _ = self._make_client(
            single_model=True, initial_prompt="first prompt", hotwords="first"
        )
        second, _ = self._make_client(
            single_model=True, initial_prompt="second prompt", hotwords="second"
        )
        self.assertIs(second.transcriber, first.transcriber)

        first.transcribe_audio([0.0, 0.0])
        _, kwargs = first.transcriber.transcribe.call_args
        self.assertEqual(kwargs["initial_prompt"], "first prompt")
        self.assertEqual(kwargs["hotwords"], "first")

        second.transcribe_audio([0.0, 0.0])
        _, kwargs = second.transcriber.transcribe.call_args
        self.assertEqual(kwargs["initial_prompt"], "second prompt")
        self.assertEqual(kwargs["hotwords"], "second")


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
