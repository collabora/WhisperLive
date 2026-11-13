import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from whisper_live.transcriber import tensorrt_utils


class TestLoadAudioCleanup(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        tempdir = patch.object(tempfile, "tempdir", str(self.directory))
        tempdir.start()
        self.addCleanup(tempdir.stop)

        self.source = self.directory / "meeting.wav"
        with wave.open(str(self.source), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8000)
            output.writeframes(np.arange(-400, 400, dtype=np.int16).tobytes())
        self.existing = self.directory / "meeting_resampled.wav"
        self.existing.write_bytes(b"keep this existing file")
        self.before = {path: path.read_bytes() for path in self.directory.iterdir()}

    def assert_files_preserved(self):
        self.assertEqual(set(self.directory.iterdir()), set(self.before))
        for path, content in self.before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_successful_reads_remove_each_temporary_file(self):
        for _ in range(2):
            audio = tensorrt_utils.load_audio(str(self.source))
            self.assertEqual(audio.dtype, np.float32)
            self.assertEqual(audio.ndim, 1)
            self.assertGreater(audio.size, 0)
            self.assertTrue(np.isfinite(audio).all())
            self.assert_files_preserved()

    def test_failed_open_removes_temporary_file(self):
        with patch.object(tensorrt_utils.wave, "open", side_effect=wave.Error("open failed")):
            with self.assertRaisesRegex(wave.Error, "open failed"):
                tensorrt_utils.load_audio(str(self.source))
        self.assert_files_preserved()

    def check_failed_read(self, error):
        with patch.object(tensorrt_utils.wave, "open") as open_wave:
            wav_file = open_wave.return_value.__enter__.return_value
            wav_file.getnframes.return_value = 100
            wav_file.readframes.side_effect = error
            with self.assertRaises(type(error)):
                tensorrt_utils.load_audio(str(self.source))
        self.assert_files_preserved()

    def test_failed_read_removes_temporary_file(self):
        self.check_failed_read(RuntimeError("read failed"))

    def test_interrupted_read_removes_temporary_file(self):
        self.check_failed_read(KeyboardInterrupt())


if __name__ == "__main__":
    unittest.main()
