"""
Manual audio chunking example for WhisperLive.

Streams an audio file to a running WhisperLive server in real-time sized chunks,
printing partial transcripts when speech is detected and committed transcripts
when each segment is finalized.

Usage:
    python examples/manual_audio_chunking.py --file assets/jfk.flac
"""

import argparse
import os
import sys
import time
import wave

try:
    from whisper_live.client import StreamingTranscriptionClient
    from whisper_live.utils import resample
except ImportError: # just in case whisper_live isn't installed.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    print("[INFO] whisper_live not installed or the current version does not have StreamingTranscriptionClient. Will attempt to import from local source.")
    from whisper_live.client import StreamingTranscriptionClient
    from whisper_live.utils import resample

SAMPLE_RATE = 16000


def stream_audio_file(path: str, client: StreamingTranscriptionClient, chunk_ms: int = 50) -> None:
    """Read an audio file, resample to 16 kHz mono if needed, and pace chunks in real time."""
    resampled_path = resample(path)
    try:
        with wave.open(resampled_path, "rb") as wf:
            frames_per_chunk = SAMPLE_RATE * chunk_ms // 1000
            chunk_duration = frames_per_chunk / SAMPLE_RATE
            while chunk := wf.readframes(frames_per_chunk):
                client.send(chunk, pcm_format="int16")
                time.sleep(chunk_duration)
    finally:
        os.remove(resampled_path)


def main():
    parser = argparse.ArgumentParser(description="Stream an audio file to WhisperLive.")
    parser.add_argument("--file", "-f", required=True, help="Audio file to transcribe (any format supported by ffmpeg).")
    parser.add_argument("--server", "-s", default="localhost")
    parser.add_argument("--port", "-p", type=int, default=9090)
    parser.add_argument("--model", "-m", default="small")
    parser.add_argument("--lang", "-l", default="en")
    parser.add_argument("--chunk_ms", type=int, default=50, help="Chunk size in ms.")
    args = parser.parse_args()

    client = StreamingTranscriptionClient(
        args.server, args.port,
        lang=args.lang,
        model=args.model,
        on_session_started=lambda: print("[INFO] Server ready.\n"),
        on_partial_transcript=lambda text, _: print(f"\r… {text:<80}", end="", flush=True),
        on_committed_transcript=lambda text, _: print(f"\r✓ {text:<80}"),
        on_error=lambda e: print(f"\n[ERROR] {e}"),
        on_close=lambda: print("\n[INFO] Connection closed."),
    )

    with client:
        print(f"[INFO] Streaming {args.file} in {args.chunk_ms} ms chunks.")
        stream_audio_file(args.file, client, chunk_ms=args.chunk_ms)

    print("\n[INFO] Final transcript:")
    for seg in client.transcript:
        print(f"  [{float(seg['start']):.2f}s → {float(seg['end']):.2f}s] {seg['text'].strip()}")
    if client.last_partial:
        seg = client.last_segment
        print(f"  [{float(seg['start']):.2f}s → {float(seg['end']):.2f}s] {seg['text'].strip()} (partial)")


if __name__ == "__main__":
    main()
