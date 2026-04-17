import unittest
from unittest.mock import MagicMock, patch
import numpy as np


class TestSpeakerDiarizer(unittest.TestCase):
    """Tests for SpeakerDiarizer with mocked embedding model."""

    def _make_diarizer(self, **kwargs):
        from whisper_live.diarization import SpeakerDiarizer
        d = SpeakerDiarizer(**kwargs)
        # Mock the embedding model to return deterministic embeddings
        d._model = MagicMock()
        return d

    def _set_embedding(self, diarizer, embedding):
        """Configure mock model to return a specific embedding."""
        emb = np.array(embedding, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        diarizer._model.return_value = emb

    def test_first_speaker_creates_new(self):
        d = self._make_diarizer()
        self._set_embedding(d, [1.0, 0.0, 0.0])
        audio = np.zeros(16000, dtype=np.float32)  # 1 second of audio
        speaker = d.identify_speaker(audio)
        self.assertEqual(speaker, "SPEAKER_00")
        self.assertEqual(len(d.speakers), 1)

    def test_same_speaker_matches(self):
        d = self._make_diarizer(similarity_threshold=0.8)
        self._set_embedding(d, [1.0, 0.0, 0.0])
        audio = np.zeros(16000, dtype=np.float32)
        d.identify_speaker(audio)  # SPEAKER_00
        # Same embedding should match
        self._set_embedding(d, [0.99, 0.01, 0.0])
        speaker = d.identify_speaker(audio)
        self.assertEqual(speaker, "SPEAKER_00")
        self.assertEqual(len(d.speakers), 1)

    def test_different_speaker_creates_new(self):
        d = self._make_diarizer(similarity_threshold=0.8)
        self._set_embedding(d, [1.0, 0.0, 0.0])
        audio = np.zeros(16000, dtype=np.float32)
        d.identify_speaker(audio)  # SPEAKER_00

        # Very different embedding
        self._set_embedding(d, [0.0, 1.0, 0.0])
        speaker = d.identify_speaker(audio)
        self.assertEqual(speaker, "SPEAKER_01")
        self.assertEqual(len(d.speakers), 2)

    def test_max_speakers_limit(self):
        d = self._make_diarizer(similarity_threshold=0.95, max_speakers=2)
        audio = np.zeros(16000, dtype=np.float32)

        self._set_embedding(d, [1.0, 0.0, 0.0])
        d.identify_speaker(audio)  # SPEAKER_00
        self._set_embedding(d, [0.0, 1.0, 0.0])
        d.identify_speaker(audio)  # SPEAKER_01

        # Third distinct speaker should be assigned to closest existing
        self._set_embedding(d, [0.0, 0.0, 1.0])
        speaker = d.identify_speaker(audio)
        self.assertIn(speaker, ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual(len(d.speakers), 2)

    def test_short_audio_returns_none(self):
        d = self._make_diarizer()
        # Less than 0.3 seconds
        audio = np.zeros(3000, dtype=np.float32)
        speaker = d.identify_speaker(audio)
        self.assertIsNone(speaker)

    def test_reset_clears_state(self):
        d = self._make_diarizer()
        self._set_embedding(d, [1.0, 0.0, 0.0])
        audio = np.zeros(16000, dtype=np.float32)
        d.identify_speaker(audio)
        self.assertEqual(len(d.speakers), 1)
        d.reset()
        self.assertEqual(len(d.speakers), 0)
        self.assertEqual(d._speaker_count, 0)

    def test_import_error_without_pyannote(self):
        from whisper_live.diarization import SpeakerDiarizer
        d = SpeakerDiarizer()
        with patch.dict("sys.modules", {"pyannote": None, "pyannote.audio": None}):
            with self.assertRaises(ImportError):
                d._load_model()


class TestDiarizationInBase(unittest.TestCase):
    """Test diarization integration in ServeClientBase."""

    def _make_client(self, diarization=None):
        from whisper_live.backend.base import ServeClientBase

        class ConcreteClient(ServeClientBase):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.language = "en"
            def transcribe_audio(self, input_sample):
                return None
            def handle_transcription_output(self, result, duration):
                pass

        ws = MagicMock()
        return ConcreteClient(
            client_uid="test-uid", websocket=ws, diarization=diarization
        )

    def test_no_diarization_by_default(self):
        client = self._make_client()
        self.assertIsNone(client.diarization)

    def test_format_segment_with_speaker(self):
        client = self._make_client()
        seg = client.format_segment(0.0, 1.0, "hello", speaker="SPEAKER_00")
        self.assertEqual(seg["speaker"], "SPEAKER_00")

    def test_format_segment_without_speaker(self):
        client = self._make_client()
        seg = client.format_segment(0.0, 1.0, "hello")
        self.assertNotIn("speaker", seg)

    def test_identify_speaker_disabled(self):
        client = self._make_client(diarization=None)
        seg = MagicMock()
        seg.start = 0.0
        seg.end = 1.0
        result = client._identify_speaker(seg)
        self.assertIsNone(result)

    def test_identify_speaker_calls_diarizer(self):
        mock_diarizer = MagicMock()
        mock_diarizer.identify_speaker.return_value = "SPEAKER_01"
        client = self._make_client(diarization=mock_diarizer)
        # Set up audio buffer
        client.frames_np = np.zeros(48000, dtype=np.float32)
        client.frames_offset = 0.0
        client.timestamp_offset = 0.0
        seg = MagicMock()
        seg.start = 0.5
        seg.end = 1.5
        result = client._identify_speaker(seg)
        self.assertEqual(result, "SPEAKER_01")
        mock_diarizer.identify_speaker.assert_called_once()


if __name__ == "__main__":
    unittest.main()
