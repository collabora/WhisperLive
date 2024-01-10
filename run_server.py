import argparse
from whisper_live.server import TranscriptionServer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='tensorrt', help='Backends from ["tensorrt", "faster_whisper"]')
    parser.add_argument('--whisper_tensorrt_path',
                        type=str,
                        default="/root/TensorRT-LLM/examples/whisper/whisper_small_en",
                        help='Whisper TensorRT model path')
    args = parser.parse_args()

    if args.backend == "tensorrt":
        if args.whisper_tensorrt_path is None:
            raise ValueError("Please Provide a valid tensorrt model path")

    server = TranscriptionServer()
    server.run("0.0.0.0", port=6006, backend=args.backend, whisper_tensorrt_path=args.whisper_tensorrt_path)
