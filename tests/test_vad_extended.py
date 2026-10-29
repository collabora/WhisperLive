import os
import shutil
import tempfile
import unittest
import urllib.error
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from whisper_live.vad import VoiceActivityDetection, VoiceActivityDetector


class TestVoiceActivityDetectionValidation(unittest.TestCase):
    """Tests for VoiceActivityDetection input validation without requiring the ONNX model."""

    @patch.object(VoiceActivityDetection, "__init__", lambda self, **kw: None)
    def setUp(self):
        self.vad = VoiceActivityDetection()
        self.vad.sample_rates = [8000, 16000]

    def test_1d_input_unsqueezed(self):
        x = torch.randn(512)
        x_out, sr_out = self.vad._validate_input(x, 16000)
        self.assertEqual(x_out.dim(), 2)
        self.assertEqual(sr_out, 16000)

    def test_3d_input_raises(self):
        x = torch.randn(1, 1, 512)
        with self.assertRaises(ValueError):
            self.vad._validate_input(x, 16000)

    def test_unsupported_sample_rate_raises(self):
        x = torch.randn(1, 512)
        with self.assertRaises(ValueError):
            self.vad._validate_input(x, 44100)

    def test_too_short_audio_raises(self):
        x = torch.randn(1, 1)
        with self.assertRaises(ValueError):
            self.vad._validate_input(x, 16000)

    def test_downsample_multiple_of_16k(self):
        x = torch.randn(1, 512 * 3)
        x_out, sr_out = self.vad._validate_input(x, 48000)
        self.assertEqual(sr_out, 16000)
        self.assertEqual(x_out.shape[1], 512)


class TestVoiceActivityDetectionStateReset(unittest.TestCase):
    """Tests for VoiceActivityDetection.reset_states()."""

    @patch.object(VoiceActivityDetection, "__init__", lambda self, **kw: None)
    def setUp(self):
        self.vad = VoiceActivityDetection()

    def test_reset_creates_correct_shapes(self):
        self.vad.reset_states(batch_size=4)
        self.assertEqual(self.vad._state.shape, (2, 4, 128))
        self.assertEqual(self.vad._context.shape[0], 0)
        self.assertEqual(self.vad._last_sr, 0)
        self.assertEqual(self.vad._last_batch_size, 0)

    def test_reset_default_batch_size(self):
        self.vad.reset_states()
        self.assertEqual(self.vad._state.shape, (2, 1, 128))


def _write_model(url, filename):
    with open(filename, "wb") as model_file:
        model_file.write(b"onnx model")


class TestVoiceActivityDetectionDownload(unittest.TestCase):
    """Tests for the model download function."""

    def setUp(self):
        home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, home)
        patch.dict(os.environ, {"HOME": home}).start()
        self.addCleanup(patch.stopall)
        self.model_path = os.path.join(home, ".cache", "whisper-live", "silero_vad.onnx")

    @patch("urllib.request.urlretrieve")
    def test_skips_download_if_exists(self, mock_urlretrieve):
        os.makedirs(os.path.dirname(self.model_path))
        _write_model(None, self.model_path)
        self.assertEqual(VoiceActivityDetection.download(), self.model_path)
        mock_urlretrieve.assert_not_called()

    @patch("urllib.request.urlretrieve", side_effect=_write_model)
    def test_downloads_if_missing(self, mock_urlretrieve):
        path = VoiceActivityDetection.download()
        self.assertEqual(path, self.model_path)
        with open(path, "rb") as model_file:
            self.assertEqual(model_file.read(), b"onnx model")

    @patch("urllib.request.urlretrieve")
    def test_failed_download_raises_and_leaves_no_model(self, mock_urlretrieve):
        def interrupted_download(url, filename):
            with open(filename, "wb") as model_file:
                model_file.write(b"onn")
            raise urllib.error.URLError("connection reset")

        mock_urlretrieve.side_effect = interrupted_download
        with self.assertRaises(urllib.error.URLError):
            VoiceActivityDetection.download()
        self.assertFalse(os.path.exists(self.model_path))


class TestVoiceActivityDetectorThreshold(unittest.TestCase):
    """Tests for VoiceActivityDetector threshold behavior."""

    @patch.object(VoiceActivityDetection, "__init__", lambda self, **kw: None)
    def test_above_threshold_returns_true(self):
        detector = VoiceActivityDetector.__new__(VoiceActivityDetector)
        detector.model = VoiceActivityDetection()
        detector.threshold = 0.5
        detector.frame_rate = 16000

        mock_probs = torch.tensor([[0.9, 0.8, 0.7]])
        with patch.object(detector.model, "audio_forward", return_value=mock_probs):
            result = detector(np.random.randn(16000).astype(np.float32))
        self.assertTrue(result)

    @patch.object(VoiceActivityDetection, "__init__", lambda self, **kw: None)
    def test_below_threshold_returns_false(self):
        detector = VoiceActivityDetector.__new__(VoiceActivityDetector)
        detector.model = VoiceActivityDetection()
        detector.threshold = 0.5
        detector.frame_rate = 16000

        mock_probs = torch.tensor([[0.1, 0.2, 0.3]])
        with patch.object(detector.model, "audio_forward", return_value=mock_probs):
            result = detector(np.random.randn(16000).astype(np.float32))
        self.assertFalse(result)

    @patch.object(VoiceActivityDetection, "__init__", lambda self, **kw: None)
    def test_custom_threshold(self):
        detector = VoiceActivityDetector.__new__(VoiceActivityDetector)
        detector.model = VoiceActivityDetection()
        detector.threshold = 0.95
        detector.frame_rate = 16000

        mock_probs = torch.tensor([[0.9]])
        with patch.object(detector.model, "audio_forward", return_value=mock_probs):
            result = detector(np.random.randn(16000).astype(np.float32))
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
