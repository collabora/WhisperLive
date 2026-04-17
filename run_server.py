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
    parser.add_argument(
        '--batch_inference',
        action='store_true',
        help='Enable batched GPU inference for concurrent sessions. '
             'Batches multiple sessions into a single GPU call for higher throughput.'
    )
    parser.add_argument(
        '--batch_max_size',
        type=int,
        default=8,
        help='Maximum batch size for batched inference (default: 8).'
    )
    parser.add_argument(
        '--batch_window_ms',
        type=int,
        default=50,
        help='Maximum time in ms to wait for batch to fill (default: 50).'
    )
    parser.add_argument(
        '--raw_pcm_input',
        action='store_true',
        help='Expect raw PCM int16 audio from clients instead of float32. '
             'Audio will be normalized to float32 range [-1.0, 1.0].'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='Optional API key for authenticating REST API requests. '
             'Clients must send "Authorization: Bearer <key>" header.'
    )
    parser.add_argument(
        '--rate_limit_rpm',
        type=int,
        default=0,
        help='Maximum REST API requests per minute per client IP. 0 = unlimited (default).'
    )
    parser.add_argument(
        '--metrics_port',
        type=int,
        default=0,
        help='Port for Prometheus /metrics endpoint. 0 = disabled (default). Requires prometheus_client.'
    )
    parser.add_argument(
        '--noise_reduction',
        type=str,
        default=None,
        choices=['near_field', 'far_field'],
        help='Enable audio noise reduction. "near_field" for close-mic, "far_field" for distant audio. Requires noisereduce.'
    )
    parser.add_argument(
        '--json_logs',
        action='store_true',
        help='Emit structured JSON logs (for CloudWatch, ELK, or similar). Default: human-readable.'
    )
    parser.add_argument(
        '--storage_backend',
        type=str,
        default='local',
        choices=['local', 's3'],
        help='Storage backend for audio files and results. "local" (default) or "s3".'
    )
    parser.add_argument(
        '--storage_bucket',
        type=str,
        default=None,
        help='S3 bucket name (required when --storage_backend=s3).'
    )
    parser.add_argument(
        '--storage_prefix',
        type=str,
        default='whisperlive/',
        help='S3 key prefix for stored files. Default: "whisperlive/".'
    )
    parser.add_argument(
        '--data_retention_days',
        type=int,
        default=0,
        help='Auto-delete stored data older than N days. 0 = disabled (default).'
    )
    parser.add_argument(
        '--user_store',
        type=str,
        default=None,
        help='Path to JSON file for user management. Enables multi-user API keys with roles/quotas.'
    )
    parser.add_argument(
        '--jwt_jwks_url',
        type=str,
        default=None,
        help='JWKS URL for JWT validation (Cognito, Auth0, Keycloak). Requires PyJWT[crypto].'
    )
    parser.add_argument(
        '--jwt_secret',
        type=str,
        default=None,
        help='Shared secret for HS256 JWT validation.'
    )
    parser.add_argument(
        '--jwt_audience',
        type=str,
        default=None,
        help='Expected JWT audience claim.'
    )
    parser.add_argument(
        '--jwt_issuer',
        type=str,
        default=None,
        help='Expected JWT issuer claim.'
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
        batch_enabled=args.batch_inference,
        batch_max_size=args.batch_max_size,
        batch_window_ms=args.batch_window_ms,
        raw_pcm_input=args.raw_pcm_input,
        api_key=args.api_key,
        rate_limit_rpm=args.rate_limit_rpm,
        metrics_port=args.metrics_port,
        noise_reduction=args.noise_reduction,
        json_logs=args.json_logs,
        storage_backend=args.storage_backend,
        storage_bucket=args.storage_bucket,
        storage_prefix=args.storage_prefix,
        data_retention_days=args.data_retention_days,
        user_store_path=args.user_store,
        jwt_jwks_url=args.jwt_jwks_url,
        jwt_secret=args.jwt_secret,
        jwt_audience=args.jwt_audience,
        jwt_issuer=args.jwt_issuer,
    )