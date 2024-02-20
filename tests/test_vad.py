import unittest
import numpy as np
from whisper_live.tensorrt_utils import load_audio
from whisper_live.vad import VoiceActivityDetector


class TestVoiceActivityDetection(unittest.TestCase):
    def setUp(self):
        self.vad = VoiceActivityDetector()
        self.sample_rate = 16000

    def generate_silence(self, duration_seconds):
        return np.zeros(int(self.sample_rate * duration_seconds), dtype=np.float32)

    def load_speech_segment(self, filepath):
        return load_audio(filepath)

    def test_vad_silence_detection(self):
        silence = self.generate_silence(3)
        is_speech_present = self.vad(silence.copy())
        self.assertFalse(is_speech_present, "VAD incorrectly identified silence as speech.")

    def test_vad_speech_detection(self):
        audio_tensor = load_audio("assets/jfk.flac")
        is_speech_present = self.vad(audio_tensor)
        self.assertTrue(is_speech_present, "VAD failed to identify speech segment.")
