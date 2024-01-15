import argparse
from whisper_live.server import TranscriptionServer

if __name__ == "__main__":
    server = TranscriptionServer()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help="Custom Faster Whisper Model")
    args = parser.parse_args()
    server.run(
        "0.0.0.0",
        9090,
        custom_model_path=args.model_path
    )
