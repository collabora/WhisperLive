import os

from whisper_live.server import TranscriptionServer

if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"

if __name__ == "__main__":
    server = TranscriptionServer()
    server.run("0.0.0.0", port=9090, backend="faster_whisper",
               faster_whisper_custom_model_path="/Users/aliuspetraska/Documents/Git/faster-whisper-large-v2")
