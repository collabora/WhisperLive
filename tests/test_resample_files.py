import tempfile
import wave
from pathlib import Path

import pytest

from whisper_live import utils
from whisper_live.client import TranscriptionTeeClient


@pytest.fixture
def audio(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    source = tmp_path / "meeting.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b"\x00\x00" * 800)
    return source


def test_resampling_preserves_existing_files_and_uses_distinct_outputs(audio):
    existing = audio.with_name("meeting_resampled.wav")
    existing.write_bytes(b"keep this existing file")
    converted = []
    try:
        converted = [Path(utils.resample(str(audio))) for _ in range(2)]
        assert existing.read_bytes() == b"keep this existing file"
        assert converted[0] != converted[1]
        for path in converted:
            with wave.open(str(path), "rb") as output:
                assert output.getframerate() == 16000
                assert output.getnchannels() == 1
    finally:
        for path in converted:
            if path.resolve() != existing.resolve():
                path.unlink(missing_ok=True)


def test_failed_conversion_removes_only_its_partial_output(audio, monkeypatch):
    before = set(audio.parent.iterdir())

    class FailingResampler:
        def resample(self, frame):
            raise RuntimeError("conversion failed")

    monkeypatch.setattr(utils.av, "AudioResampler", lambda **kwargs: FailingResampler())
    with pytest.raises(RuntimeError, match="conversion failed"):
        utils.resample(str(audio))
    assert set(audio.parent.iterdir()) == before


@pytest.mark.parametrize("fail", [False, True])
def test_file_transcription_cleans_up_its_resampled_file(audio, fail):
    client = TranscriptionTeeClient.__new__(TranscriptionTeeClient)
    client.clients = []
    before = set(audio.parent.iterdir())

    def play(path):
        assert Path(path).is_file()
        if fail:
            raise RuntimeError("playback failed")

    client.play_file = play
    if fail:
        with pytest.raises(RuntimeError, match="playback failed"):
            client(audio=str(audio))
    else:
        client(audio=str(audio))
    assert set(audio.parent.iterdir()) == before
