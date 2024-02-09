import unittest
import numpy as np
import torch
from whisper_live.tensorrt_utils import load_audio
from whisper_live.vad import VoiceActivityDetection


class TestVoiceActivityDetection(unittest.TestCase):
    def setUp(self):
        self.vad = VoiceActivityDetection()
        self.sample_rate = 16000

    def generate_silence(self, duration_seconds):
        return np.zeros(int(self.sample_rate * duration_seconds), dtype=np.float32)

    def load_speech_segment(self, filepath):
        return load_audio(filepath)

    def test_vad_silence_detection(self):
        silence = self.generate_silence(3)
        speech_prob = self.vad(torch.from_numpy(silence.copy()), self.sample_rate).item()
        self.assertLess(speech_prob, 0.5, "VAD incorrectly identified silence as speech.")

    def test_vad_speech_detection(self):
        audio_tensor = torch.from_numpy(load_audio("assets/jfk.flac"))
        speech_prob = self.vad(audio_tensor, self.sample_rate).item()
        self.assertGreater(speech_prob, 0.5, "VAD failed to identify speech segment.")
