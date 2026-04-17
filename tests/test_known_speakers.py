"""Tests for known speaker matching / enrollment in diarization."""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock


class FakeSpeakerDiarizer:
    """Minimal diarizer for testing enrollment without pyannote."""

    def __init__(self):
        self.speakers = {}
        self._speaker_count = 0
        self.similarity_threshold = 0.55
        self.max_speakers = 10
        self._model = None
        self._embedding_model_name = "test"
        self._hf_token = None

    def _compute_embedding(self, audio_np, sample_rate=16000):
        if len(audio_np) < sample_rate * 0.3:
            return None
        # Return a simple normalized embedding based on audio stats
        emb = np.array([np.mean(audio_np), np.std(audio_np), float(len(audio_np))], dtype=np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb


class TestEnrollSpeaker(unittest.TestCase):
    def _make_diarizer(self):
        from whisper_live.diarization import SpeakerDiarizer
        d = SpeakerDiarizer.__new__(SpeakerDiarizer)
        d.speakers = {}
        d._speaker_count = 0
        d.similarity_threshold = 0.55
        d.max_speakers = 10
        d._model = None
        d._embedding_model_name = "test"
        d._hf_token = None
        # Patch _compute_embedding to avoid needing pyannote
        d._compute_embedding = FakeSpeakerDiarizer()._compute_embedding
        return d

    def test_enroll_success(self):
        d = self._make_diarizer()
        audio = np.random.randn(16000).astype(np.float32)
        result = d.enroll_speaker("Alice", audio)
        assert result is True
        assert "Alice" in d.speakers

    def test_enroll_too_short(self):
        d = self._make_diarizer()
        audio = np.random.randn(100).astype(np.float32)  # too short
        result = d.enroll_speaker("Bob", audio)
        assert result is False
        assert "Bob" not in d.speakers

    def test_enroll_replaces_existing(self):
        d = self._make_diarizer()
        audio1 = np.random.randn(16000).astype(np.float32)
        audio2 = np.random.randn(16000).astype(np.float32) * 2
        d.enroll_speaker("Alice", audio1)
        d.enroll_speaker("Alice", audio2)
        assert "Alice" in d.speakers

    def test_get_enrolled_speakers(self):
        d = self._make_diarizer()
        audio = np.random.randn(16000).astype(np.float32)
        d.enroll_speaker("Alice", audio)
        d.enroll_speaker("Bob", np.random.randn(16000).astype(np.float32))
        enrolled = d.get_enrolled_speakers()
        assert "Alice" in enrolled
        assert "Bob" in enrolled

    def test_enroll_speakers_from_files_with_arrays(self):
        d = self._make_diarizer()
        refs = {
            "Alice": np.random.randn(16000).astype(np.float32),
            "Bob": np.random.randn(16000).astype(np.float32),
        }
        results = d.enroll_speakers_from_files(refs)
        assert results["Alice"] is True
        assert results["Bob"] is True

    def test_enroll_speakers_from_files_with_short_audio(self):
        d = self._make_diarizer()
        refs = {
            "Alice": np.random.randn(100).astype(np.float32),
        }
        results = d.enroll_speakers_from_files(refs)
        assert results["Alice"] is False

    def test_enroll_speakers_from_files_invalid_type(self):
        d = self._make_diarizer()
        refs = {
            "Alice": 12345,  # invalid type
        }
        results = d.enroll_speakers_from_files(refs)
        assert results["Alice"] is False

    def test_reset_clears_enrolled(self):
        d = self._make_diarizer()
        audio = np.random.randn(16000).astype(np.float32)
        d.enroll_speaker("Alice", audio)
        d.reset()
        assert len(d.speakers) == 0


class TestServerKnownSpeakerParsing(unittest.TestCase):
    def test_create_diarizer_without_refs(self):
        """Server _create_diarizer works when no known_speaker_refs provided."""
        from whisper_live.server import TranscriptionServer
        server = TranscriptionServer()
        # This will return None because enable_diarization is False
        result = server._create_diarizer({"enable_diarization": False})
        assert result is None

    def test_source_has_known_speaker_refs(self):
        import inspect
        from whisper_live import server
        source = inspect.getsource(server)
        assert "known_speaker_refs" in source
        assert "enroll_speakers_from_files" in source


if __name__ == "__main__":
    unittest.main()
