
from whisper_live.server import TranscriptionServer

if __name__ == "__main__":
    server = TranscriptionServer()
    server.run_server("0.0.0.0", 9090)