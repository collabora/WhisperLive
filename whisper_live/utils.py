import os
import shutil
import tempfile
import textwrap
import scipy
import numpy as np
import av
from pathlib import Path


def clear_screen():
    """Clears the console screen and its scrollback buffer."""
    print("\033[H\033[2J\033[3J", end="", flush=True)


def print_transcript(text, translated=False, timestamps=False, max_lines=3):
    """Prints the last `max_lines` wrapped lines of transcript text in a subtitle-like block."""
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    wrap_width = max(10, min(80, terminal_width - 8))

    if timestamps:
        lines = []
        for t in text:
            prefix = f'[{t["start"]} -> {t["end"]}] '
            wrapper = textwrap.TextWrapper(
                width=wrap_width,
                subsequent_indent=" " * len(prefix),
            )
            lines.extend(wrapper.wrap(f'{prefix}{t["text"]}'))
    else:
        wrapper = textwrap.TextWrapper(width=wrap_width)
        transcript = " ".join(text) if translated else "".join(text)
        lines = wrapper.wrap(text=transcript)

    for line in lines[-max_lines:]:
        print(line.center(terminal_width))


def format_time(s):
    """Convert seconds (float) to SRT time format."""
    hours = int(s // 3600)
    minutes = int((s % 3600) // 60)
    seconds = int(s % 60)
    milliseconds = int((s - int(s)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def create_srt_file(segments, resampled_file):
    with open(resampled_file, 'w', encoding='utf-8') as srt_file:
        segment_number = 1
        for segment in segments:
            start_time = format_time(float(segment['start']))
            end_time = format_time(float(segment['end']))
            text = segment['text']

            srt_file.write(f"{segment_number}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text}\n\n")

            segment_number += 1


def resample(file: str, sr: int = 16000):
    """
    Resample the audio file to 16kHz.

    Args:
        file (str): The audio file to open
        sr (int): The sample rate to resample the audio if necessary

    Returns:
        resampled_file (str): A temporary WAV file. The caller must remove it after use.
    """
    with tempfile.NamedTemporaryFile(prefix='whisperlive_', suffix='_resampled.wav', delete=False) as temp:
        resampled_file = temp.name

    completed = False
    try:
        with av.open(file) as container, av.open(resampled_file, mode='w') as output_container:
            resampler = av.AudioResampler(format='s16', layout='mono', rate=sr)
            output_stream = output_container.add_stream('pcm_s16le', rate=sr)
            output_stream.layout = 'mono'

            for frame in container.decode(audio=0):
                frame.pts = None
                resampled_frames = resampler.resample(frame)
                if resampled_frames is not None:
                    for resampled_frame in resampled_frames:
                        for packet in output_stream.encode(resampled_frame):
                            output_container.mux(packet)

            for packet in output_stream.encode(None):
                output_container.mux(packet)

        completed = True
        return resampled_file
    finally:
        if not completed:
            Path(resampled_file).unlink(missing_ok=True)
