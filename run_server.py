import argparse
import os
import threading
import logging
from fastapi import FastAPI
from fastapi import UploadFile, Form
import uvicorn
import tempfile
import shutil
import json
from starlette.responses import PlainTextResponse, JSONResponse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p',
                        type=int,
                        default=9090,
                        help="Websocket port to run the server on.")
    parser.add_argument('--backend', '-b',
                        type=str,
                        default='faster_whisper',
                        help='Backends from ["tensorrt", "faster_whisper", "openvino"]')
    parser.add_argument('--faster_whisper_custom_model_path', '-fw',
                        type=str, default=None,
                        help="Custom Faster Whisper Model")
    parser.add_argument('--trt_model_path', '-trt',
                        type=str,
                        default=None,
                        help='Whisper TensorRT model path')
    parser.add_argument('--trt_multilingual', '-m',
                        action="store_true",
                        help='Boolean only for TensorRT model. True if multilingual.')
    parser.add_argument('--trt_py_session',
                        action="store_true",
                        help='Boolean only for TensorRT model. Use python session or cpp session, By default uses Cpp.')
    parser.add_argument('--omp_num_threads', '-omp',
                        type=int,
                        default=1,
                        help="Number of threads to use for OpenMP")
    parser.add_argument('--no_single_model', '-nsm',
                        action='store_true',
                        help='Set this if every connection should instantiate its own model. Only relevant for custom model, passed using -trt or -fw.')
    parser.add_argument('--max_clients',
                        type=int,
                        default=4,
                        help='Maximum clients supported by the server.')
    parser.add_argument('--max_connection_time',
                        type=int,
                        default=300,
                        help='The maximum duration (in seconds) a client can stay connected. Defaults to 300 seconds (5 minutes)')
    parser.add_argument('--cache_path', '-c',
                        type=str,
                        default="~/.cache/whisper-live/",
                        help='Path to cache the converted ctranslate2 models.')
    parser.add_argument(
        "--rest_port", type=int, default=8000, help="Port for the REST API server."
    )
    parser.add_argument(
        "--enable_rest",
        action="store_true",
        help="Enable the OpenAI-compatible REST API endpoint.",
    )
    parser.add_argument(
        '--cors-origins',
        type=str,
        default=None,
        help="Comma-separated list of allowed CORS origins (e.g., 'http://localhost:3000,http://example.com'). Defaults to localhost/127.0.0.1 on the WebSocket port."
    )
    args = parser.parse_args()

    if args.backend == "tensorrt":
        if args.trt_model_path is None:
            raise ValueError("Please Provide a valid tensorrt model path")

    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)

    from whisper_live.server import TranscriptionServer
    server = TranscriptionServer()
    server.run(
        "0.0.0.0",
        port=args.port,
        backend=args.backend,
        faster_whisper_custom_model_path=args.faster_whisper_custom_model_path,
        whisper_tensorrt_path=args.trt_model_path,
        trt_multilingual=args.trt_multilingual,
        trt_py_session=args.trt_py_session,
        single_model=not args.no_single_model,
        max_clients=args.max_clients,
        max_connection_time=args.max_connection_time,
        cache_path=args.cache_path,
        rest_port=args.rest_port,
        enable_rest=args.enable_rest,
        cors_origins=args.cors_origins,
    )