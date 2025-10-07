#!/usr/bin/env python3
"""
WhisperLive Client - Microphone real-time transcription
Captures microphone audio and displays French transcription in real-time
"""
import json
from datetime import datetime
import time
from whisper_live.client import TranscriptionClient

# Global list to accumulate all received transcriptions
transcription_history = []


def on_transcription(text, segments):
    """Callback to handle transcription display."""
    entry = {
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "segments": [
            {
                "text": seg["text"],
                "completed": seg["completed"],
            } for seg in segments
        ]
    }
    print(json.dumps(entry, indent=2, ensure_ascii=False) + ",")

if __name__ == "__main__":
    # Whisper model: "tiny", "base", "small", "medium", "large-v3-turbo"
    # MODEL = "tiny"
    # MODEL = "base"
    # MODEL = "small"
    # MODEL = "medium"
    # MODEL = "large-v3-turbo"
    # MODEL = "Systran/faster-whisper-tiny"
    # MODEL = "Systran/faster-whisper-base"
    # MODEL = "Systran/faster-whisper-small"
    # MODEL = "Systran/faster-whisper-medium"
    # MODEL = "Systran/faster-whisper-large-v3"
    # MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
    # MODEL = "Zoont/faster-whisper-large-v3-turbo-int8-ct2"

    # For OpenVINO backend (format OpenVINO IR):
    # MODEL = "OpenVINO/whisper-tiny-int8-ov"
    # MODEL = "OpenVINO/whisper-base-int8-ov"
    # MODEL = "OpenVINO/whisper-small-int8-ov"
    # MODEL = "OpenVINO/whisper-medium-int8-ov"
    MODEL = "OpenVINO/whisper-large-v3-int8-ov"

    print("=" * 60)
    print("WhisperLive Client - Transcription Microphone en Temps Réel")
    print("=" * 60)
    print("Configuration:")
    print("  - Langue: Français (fr)")
    print(f"  - Modèle: {MODEL}")
    print("  - VAD: Activé (Voice Activity Detection)")
    print("  - Serveur: localhost:9090")
    print()
    print("Démarrage de la transcription...")
    print("Parlez dans votre microphone - la transcription s'affichera ici")
    print("Appuyez sur Ctrl+C pour arrêter")
    print("=" * 60)
    print()

    # Create client optimized for real-time French transcription
    client = TranscriptionClient(
        host="localhost",
        port=9090,
        lang="fr",
        translate=False,
        model=MODEL,
        use_vad=True,
        save_output_recording=False,
        log_transcription=False,
        transcription_callback=on_transcription,
    )

    # Start microphone transcription (runs until Ctrl+C)
    client()
