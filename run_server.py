import argparse
from whisper_live.server import TranscriptionServer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9090, help="Websocket port to run the server on.")
    parser.add_argument('--backend', type=str, default='faster_whisper', help='Backends from ["tensorrt", "faster_whisper"]')
    parser.add_argument('--whisper_tensorrt_path',
                        type=str,
                        default=None,
                        help='Whisper TensorRT model path')
    parser.add_argument('--trt_multilingual',
                        action="store_true",
                        help='Boolean only for TensorRT model. True if multilingual.')
    args = parser.parse_args()

    if args.backend == "tensorrt":
        if args.whisper_tensorrt_path is None:
            raise ValueError("Please Provide a valid tensorrt model path")

    server = TranscriptionServer()
    server.run(
        "0.0.0.0",
        port=6006, 
        backend=args.backend, 
        whisper_tensorrt_path=args.whisper_tensorrt_path,
        multilingual=args.trt_multilingual
    )
