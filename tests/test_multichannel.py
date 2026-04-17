import unittest
import numpy as np

from whisper_live.multichannel import (
    split_channels,
    merge_channel_segments,
    detect_channels_from_wav,
)


class TestSplitChannels(unittest.TestCase):
    def test_mono_passthrough(self):
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = split_channels(audio, channels=1)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], audio)

    def test_stereo_split(self):
        # Interleaved: L0, R0, L1, R1, L2, R2
        audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        result = split_channels(audio, channels=2)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], [1.0, 3.0, 5.0])  # left
        np.testing.assert_array_equal(result[1], [2.0, 4.0, 6.0])  # right

    def test_three_channels(self):
        audio = np.arange(9, dtype=np.float32)  # 0..8
        result = split_channels(audio, channels=3)
        self.assertEqual(len(result), 3)
        np.testing.assert_array_equal(result[0], [0, 3, 6])
        np.testing.assert_array_equal(result[1], [1, 4, 7])
        np.testing.assert_array_equal(result[2], [2, 5, 8])

    def test_truncates_incomplete_frame(self):
        audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = split_channels(audio, channels=2)
        # 5 samples -> 4 usable (2 frames)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(len(result[1]), 2)

    def test_empty_audio(self):
        audio = np.array([], dtype=np.float32)
        result = split_channels(audio, channels=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 0)
        self.assertEqual(len(result[1]), 0)

    def test_preserves_dtype(self):
        audio = np.array([1, 2, 3, 4], dtype=np.int16)
        result = split_channels(audio, channels=2)
        self.assertEqual(result[0].dtype, np.int16)


class TestMergeChannelSegments(unittest.TestCase):
    def test_merge_two_channels(self):
        ch0 = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        ch1 = [{"start": 0.5, "end": 1.5, "text": "world"}]
        merged = merge_channel_segments([ch0, ch1])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["channel"], "ch0")
        self.assertEqual(merged[0]["text"], "hello")
        self.assertEqual(merged[1]["channel"], "ch1")
        self.assertEqual(merged[1]["text"], "world")

    def test_custom_labels(self):
        ch0 = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        ch1 = [{"start": 0.5, "end": 1.5, "text": "world"}]
        merged = merge_channel_segments([ch0, ch1], channel_labels=["agent", "customer"])
        self.assertEqual(merged[0]["channel"], "agent")
        self.assertEqual(merged[1]["channel"], "customer")

    def test_sorted_by_start(self):
        ch0 = [{"start": 2.0, "end": 3.0, "text": "second"}]
        ch1 = [{"start": 0.0, "end": 1.0, "text": "first"}]
        merged = merge_channel_segments([ch0, ch1])
        self.assertEqual(merged[0]["text"], "first")
        self.assertEqual(merged[1]["text"], "second")

    def test_empty_channels(self):
        merged = merge_channel_segments([[], []])
        self.assertEqual(merged, [])

    def test_does_not_mutate_originals(self):
        ch0 = [{"start": 0.0, "end": 1.0, "text": "hi"}]
        merge_channel_segments([ch0])
        self.assertNotIn("channel", ch0[0])

    def test_string_start_times(self):
        ch0 = [{"start": "2.000", "end": "3.000", "text": "b"}]
        ch1 = [{"start": "0.500", "end": "1.500", "text": "a"}]
        merged = merge_channel_segments([ch0, ch1])
        self.assertEqual(merged[0]["text"], "a")


class TestDetectChannelsFromWav(unittest.TestCase):
    def test_mono_wav(self):
        import tempfile
        import wave
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        wf = wave.open(path, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 320)
        wf.close()
        self.assertEqual(detect_channels_from_wav(path), 1)
        import os
        os.unlink(path)

    def test_stereo_wav(self):
        import tempfile
        import wave
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        wf = wave.open(path, "wb")
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 640)
        wf.close()
        self.assertEqual(detect_channels_from_wav(path), 2)
        import os
        os.unlink(path)

    def test_invalid_file_returns_1(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not a wav")
            path = f.name
        self.assertEqual(detect_channels_from_wav(path), 1)
        import os
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
