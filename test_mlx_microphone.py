#!/usr/bin/env python3
"""
Test script for MLX Whisper backend with microphone input.

This script connects to a WhisperLive server running the MLX backend
and streams audio from your microphone for real-time transcription.

Usage:
    python test_mlx_microphone.py --host localhost --port 9090 --model small.en

Before running:
    1. Start the server: python run_server.py --backend mlx_whisper --port 9090
    2. Install client dependencies: pip install -e .
"""

import argparse
import logging
from whisper_live.client import Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(
        description='Test MLX Whisper backend with microphone input'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Server host (default: localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=9090,
        help='Server port (default: 9090)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='small.en',
        help='Model size (e.g., tiny, base, small.en, medium, large-v3, turbo)'
    )
    parser.add_argument(
        '--lang',
        type=str,
        default=None,
        help='Language code (e.g., en, es, fr). Auto-detect if not specified.'
    )
    parser.add_argument(
        '--translate',
        action='store_true',
        help='Translate to English instead of transcribing'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_mlx.srt',
        help='Output SRT file path (default: output_mlx.srt)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MLX Whisper Live Transcription Test")
    print("=" * 70)
    print(f"Server: {args.host}:{args.port}")
    print(f"Model: {args.model}")
    print(f"Language: {args.lang if args.lang else 'Auto-detect'}")
    print(f"Task: {'Translate to English' if args.translate else 'Transcribe'}")
    print(f"Output: {args.output}")
    print("=" * 70)
    print("\nStarting microphone recording...")
    print("Speak into your microphone. Press Ctrl+C to stop.\n")

    try:
        # Initialize client with MLX backend
        client = Client(
            host=args.host,
            port=args.port,
            lang=args.lang,
            translate=args.translate,
            model=args.model,
            srt_file_path=args.output,
            use_vad=True,
            log_transcription=True,
        )

        # The client automatically starts recording when initialized
        # Keep the script running until user interrupts
        print("Recording... (Press Ctrl+C to stop)")

        # Wait indefinitely (client runs in background threads)
        import time
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopping recording...")
        if 'client' in locals():
            client.close_websocket()
        print(f"Transcription saved to: {args.output}")
        print("\nThank you for using MLX Whisper!")

    except ConnectionRefusedError:
        print(f"\n ERROR: Could not connect to server at {args.host}:{args.port}")
        print("\nMake sure the server is running:")
        print(f"  python run_server.py --backend mlx_whisper --port {args.port}")

    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
