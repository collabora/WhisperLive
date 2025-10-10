#!/usr/bin/env python3
"""
WhisperLive Server - CPU Only with 20 threads
Starts the transcription server with optimal settings for CPU-based real-time transcription
"""
import os

if __name__ == "__main__":
  # Configuration - Modify these values as needed
  THREADS = "20"  # Number of CPU threads (e.g., "4", "8", "20")
  SINGLE_MODEL = False
  BACKEND = "openvino"

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
  # MODEL = "Systran/faster-whisper-large-v3-turbo"
  # MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
  # MODEL = "Zoont/faster-whisper-large-v3-turbo-int8-ct2"

  # For OpenVINO backend (format OpenVINO IR):
  # MODEL = "OpenVINO/whisper-tiny-int8-ov"
  # MODEL = "OpenVINO/whisper-base-int8-ov"
  # MODEL = "OpenVINO/whisper-small-int8-ov"
  # MODEL = "OpenVINO/whisper-medium-int8-ov"
  # MODEL = "OpenVINO/whisper-large-v3-int8-ov"
  # MODEL = "OpenVINO/distil-whisper-large-v3-int8-ov"

  print("=" * 60)
  print("WhisperLive Server - Starting...")
  print("=" * 60)
  print("Configuration:")
  print(f"  - Backend: {BACKEND} (CPU)")
  if 'MODEL' in locals(): print(f"  - Model: {MODEL} (forced for all clients)") # pyright: ignore[reportUndefinedVariable]
  print(f"  - Threads: {THREADS} ")
  print("  - Port: 9090")
  print("  - Max clients: 4")
  print(f"  - Single model mode: {SINGLE_MODEL}")
  print("=" * 60)
  print()

  from whisper_live.server import TranscriptionServer

  # Convert model name to Hugging Face cache format and get the snapshot path
  # model_cache_name = f"models--{MODEL.replace('/', '--')}"
  # base_path = f"/home/kml93/.config/local/share/huggingface/hub/{model_cache_name}"

  # # Find the snapshot hash (there should be only one)
  # snapshots_dir = os.path.join(base_path, "snapshots")
  # if os.path.exists(snapshots_dir):
  #   snapshots = os.listdir(snapshots_dir)
  #   if snapshots:
  #     model_path = os.path.join(snapshots_dir, snapshots[0])
  #   else:
  #     raise FileNotFoundError(f"No snapshot found in {snapshots_dir}")
  # else:
  #   raise FileNotFoundError(f"Snapshots directory not found: {snapshots_dir}")

  server = TranscriptionServer()
  server.run(
    host="0.0.0.0",
    # port=9090,
    # backend="faster_whisper",
    backend=BACKEND,
    # faster_whisper_custom_model_path=model_path,
    single_model=SINGLE_MODEL,
    cpu_threads=int(THREADS),
    cache_path="/home/kml93/.config/cache/whisper-live/"
  )
