"""Tests for the OpenVINO backend: initial_prompt/hotwords wiring and
diarization/translation support.

openvino/openvino_genai are listed in requirements/server.txt, but a
developer running this suite without the optional OpenVINO extras installed
should still be able to run it, so each missing package is stubbed via
sys.modules before the modules under test are imported. Packages that are
installed are left untouched.
"""
import importlib
import inspect
import queue
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np


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
        # Patch the pipeline too: where openvino_genai is really installed (CI),
        # constructing a real WhisperPipeline from a fake path would fail.
        with patch("whisper_live.transcriber.transcriber_openvino.hf_hub"), \
             patch("whisper_live.transcriber.transcriber_openvino.ov_genai"), \
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

    def test_new_parameters_are_appended_after_existing_ones(self):
        # Adding hotwords (and later diarization/translation_queue) must not
        # shift the position of existing parameters for any caller that
        # passes them positionally: each new one is appended in order.
        params = list(inspect.signature(ServeClientOpenVINO.__init__).parameters)
        self.assertEqual(
            params[-4:],
            ["same_output_threshold", "hotwords", "diarization", "translation_queue"],
        )

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


class TestServeClientOpenVINODiarizationTranslation(unittest.TestCase):
    """ServeClientOpenVINO must forward diarization/translation_queue to the
    shared ServeClientBase, which already implements both features."""

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
