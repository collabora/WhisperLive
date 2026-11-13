import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from whisper_live import utils
from whisper_live.client import TranscriptionTeeClient


class TestResampleFiles(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        original_directory = Path.cwd()
        os.chdir(self.directory)
        self.addCleanup(os.chdir, original_directory)
        tempdir = patch.object(tempfile, "tempdir", str(self.directory))
        tempdir.start()
        self.addCleanup(tempdir.stop)

        self.audio = self.directory / "meeting.wav"
        with wave.open(str(self.audio), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8000)
            output.writeframes(b"\x00\x00" * 800)

    def test_resampling_preserves_existing_files_and_uses_distinct_outputs(self):
        existing = self.audio.with_name("meeting_resampled.wav")
        existing.write_bytes(b"keep this existing file")
        converted = []
        try:
            for _ in range(2):
                converted.append(Path(utils.resample(str(self.audio))))
            self.assertEqual(existing.read_bytes(), b"keep this existing file")
            self.assertNotEqual(converted[0], converted[1])
            for path in converted:
                with wave.open(str(path), "rb") as output:
                    self.assertEqual(output.getframerate(), 16000)
                    self.assertEqual(output.getnchannels(), 1)
        finally:
            for path in converted:
                if path.resolve() != existing.resolve():
                    path.unlink(missing_ok=True)

    def check_failed_conversion(self, error_type):
        before = set(self.directory.iterdir())

        class FailingResampler:
            def resample(self, frame):
                raise error_type("conversion failed")

        with patch.object(utils.av, "AudioResampler", return_value=FailingResampler()):
            with self.assertRaisesRegex(error_type, "conversion failed"):
                utils.resample(str(self.audio))
        self.assertEqual(set(self.directory.iterdir()), before)

    def test_failed_conversion_removes_only_its_partial_output(self):
        self.check_failed_conversion(RuntimeError)

    def test_interrupted_conversion_removes_only_its_partial_output(self):
        self.check_failed_conversion(KeyboardInterrupt)

    def check_file_transcription(self, fail):
        client = TranscriptionTeeClient.__new__(TranscriptionTeeClient)
        client.clients = []
        before = set(self.directory.iterdir())

        def play(path):
            self.assertTrue(Path(path).is_file())
            if fail:
                raise RuntimeError("playback failed")

        client.play_file = play
        if fail:
            with self.assertRaisesRegex(RuntimeError, "playback failed"):
                client(audio=str(self.audio))
        else:
            client(audio=str(self.audio))
        self.assertEqual(set(self.directory.iterdir()), before)

    def test_successful_file_transcription_cleans_up_its_resampled_file(self):
        self.check_file_transcription(False)

    def test_failed_file_transcription_cleans_up_its_resampled_file(self):
        self.check_file_transcription(True)


if __name__ == "__main__":
    unittest.main()
