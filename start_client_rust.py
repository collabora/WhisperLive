#!/usr/bin/env python3
"""
WhisperLive Client for Rust Integration
Outputs JSON transcriptions to stdout (one JSON per line) for easy parsing
"""
import json
import sys
import time
from whisper_live.client import TranscriptionClient


def on_transcription(text, segments):
    """Callback to output JSON to stdout (one line per transcription)."""
    # Build the JSON payload
    entry = {
        "timestamp": time.time(),
        "text": text,
        "segments": [
            {
                "text": seg["text"],
                "completed": seg["completed"],
            } for seg in segments
        ]
    }

    # Print as single-line JSON to stdout
    print(json.dumps(entry, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    MODEL = "kml93/whisper-large-v3-turbo-int8-asym-ov"

    # Print startup info to stderr (so stdout only contains JSON)
    print("=" * 60, file=sys.stderr)
    print("WhisperLive Client - Rust Integration Mode", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Model: {MODEL}", file=sys.stderr)
    print("Language: French (fr)", file=sys.stderr)
    print("Server: localhost:9090", file=sys.stderr)
    print("Starting transcription...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    sys.stderr.flush()

    # Create client
    client = TranscriptionClient(
        host="localhost",
        port=9090,
        lang="fr",
        translate=False,
        model=MODEL,
        use_vad=True,
        save_output_recording=False,
        log_transcription=False,  # Don't log to console
        transcription_callback=on_transcription,
    )

    # Start microphone transcription (runs until killed)
    try:
        client()
    except KeyboardInterrupt:
        print("Stopped by user", file=sys.stderr)
