from whisper_live.server import TranscriptionServer

if __name__ == "__main__":
    server = TranscriptionServer()
    server.run("0.0.0.0")
