"""Tests for noise reduction module."""

import unittest
import numpy as np
from whisper_live.noise_reduction import NoiseReducer, is_available

_skip_no_noisereduce = unittest.skipUnless(
    is_available(), "noisereduce not installed"
)


class TestIsAvailable(unittest.TestCase):
    def test_noisereduce_returns_bool(self):
        self.assertIsInstance(is_available(), bool)


@_skip_no_noisereduce
class TestNoiseReducer(unittest.TestCase):
    def test_near_field_mode(self):
        nr = NoiseReducer(mode="near_field")
        assert nr.mode == "near_field"
        assert nr._stationary is True

    def test_far_field_mode(self):
        nr = NoiseReducer(mode="far_field")
        assert nr.mode == "far_field"
        assert nr._stationary is False

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            NoiseReducer(mode="invalid")

    def test_prop_decrease_clamped(self):
        nr = NoiseReducer(prop_decrease=1.5)
        assert nr.prop_decrease == 1.0
        nr2 = NoiseReducer(prop_decrease=-0.5)
        assert nr2.prop_decrease == 0.0

    def test_stationary_override(self):
        nr = NoiseReducer(mode="far_field", stationary=True)
        assert nr._stationary is True

    def test_reduce_empty_array(self):
        nr = NoiseReducer()
        result = nr.reduce(np.array([], dtype=np.float32))
        assert result.size == 0

    def test_reduce_preserves_shape(self):
        nr = NoiseReducer(sample_rate=16000)
        # Generate 1 second of noisy audio
        np.random.seed(42)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = nr.reduce(audio)
        assert result.shape == audio.shape
        assert result.dtype == np.float32

    def test_reduce_actually_reduces_noise(self):
        nr = NoiseReducer(sample_rate=16000, prop_decrease=1.0)
        np.random.seed(42)
        # Pure noise signal
        noise = np.random.randn(16000).astype(np.float32) * 0.1
        reduced = nr.reduce(noise)
        # Reduced should have lower RMS than original
        rms_original = np.sqrt(np.mean(noise**2))
        rms_reduced = np.sqrt(np.mean(reduced**2))
        assert rms_reduced < rms_original

    def test_reduce_file_mono(self):
        nr = NoiseReducer(sample_rate=16000)
        np.random.seed(42)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = nr.reduce_file(audio, 16000)
        assert result.shape == audio.shape

    def test_reduce_file_stereo(self):
        nr = NoiseReducer(sample_rate=16000)
        np.random.seed(42)
        audio = np.random.randn(16000, 2).astype(np.float32) * 0.1
        result = nr.reduce_file(audio, 16000)
        assert result.shape == audio.shape

    def test_reduce_file_empty(self):
        nr = NoiseReducer()
        result = nr.reduce_file(np.array([], dtype=np.float32), 16000)
        assert result.size == 0

    def test_int_input_converted(self):
        nr = NoiseReducer(sample_rate=16000)
        audio = np.zeros(16000, dtype=np.int16)
        result = nr.reduce(audio)
        assert result.dtype == np.float32


class TestServerIntegration(unittest.TestCase):
    def test_server_accepts_noise_reduction(self):
        from whisper_live.server import TranscriptionServer
        server = TranscriptionServer()
        assert server._noise_reducer is None

    def test_noise_reduction_in_source(self):
        import inspect
        from whisper_live import server
        source = inspect.getsource(server)
        assert "self._noise_reducer" in source
        assert "noise_reduction" in source


if __name__ == "__main__":
    unittest.main()
