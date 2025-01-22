import os
import textwrap
import scipy
import numpy as np
import av
from pathlib import Path


def clear_screen():
    """Clears the console screen."""
    os.system("cls" if os.name == "nt" else "clear")


def print_transcript(text):
    """Prints formatted transcript text."""
    wrapper = textwrap.TextWrapper(width=60)
    for line in wrapper.wrap(text="".join(text)):
        print(line)


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
        resampled_file (str): The resampled audio file
    """
    container = av.open(file)
    stream = next(s for s in container.streams if s.type == 'audio')

    resampler = av.AudioResampler(
        format='s16',
        layout='mono',
        rate=sr,
    )

    resampled_file = Path(file).stem + "_resampled.wav"
    output_container = av.open(resampled_file, mode='w')
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

    output_container.close()
    return resampled_file
