import csv
import time
from pathlib import Path
import sys
import argparse
import sounddevice as sd
from whisper_live.client import TranscriptionClient

# --- Benchmark result logger ---
CSV_FILE = "latency_results.csv"

def log_latency_to_csv(model, avg, minv, maxv):
    file_exists = Path(CSV_FILE).exists()
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:  # write header once
            writer.writerow(["timestamp", "model", "avg_latency", "min_latency", "max_latency"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), model, avg, minv, maxv])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=int, default=9090)
    parser.add_argument('--server', '-s', type=str, default='localhost')
    parser.add_argument('--files', '-f', type=str, nargs='+')
    parser.add_argument('--output_file', '-o', type=str, default='./output_recording.wav')
    parser.add_argument('--model', '-m', type=str, default='small')
    parser.add_argument('--lang', '-l', type=str, default='en')
    parser.add_argument('--translate', '-t', action='store_true')
    parser.add_argument('--mute_audio_playback', '-a', action='store_true')
    parser.add_argument('--save_output_recording', '-r', action='store_true')
    parser.add_argument('--enable_translation', action='store_true')
    parser.add_argument('--target_language', '-tl', type=str, default='fr')
    parser.add_argument('--disable-vad', action='store_true')
    parser.add_argument('--input-device', '-i', type=int)
    parser.add_argument('-b', '--benchmark', nargs='+',
                        help="Benchmark mode: run multiple models sequentially (e.g. tiny small medium large-v3)")
    args = parser.parse_args()

    # --- Normalize server string ---
    if args.server.startswith("ws://"):
        server_str = args.server.replace("ws://", "")
    elif args.server.startswith("wss://"):
        server_str = args.server.replace("wss://", "")
    else:
        server_str = args.server

    if ":" in server_str:
        host, port_str = server_str.split(":")
        port = int(port_str)
    else:
        host, port = server_str, args.port

    if args.input_device is not None:
        print(f"[INFO]: Using input device index {args.input_device}")
        sd.default.device = (args.input_device, None)

    # --- Benchmark mode ---
    if args.benchmark:
        for model in args.benchmark:
            print("="*60)
            print(f"[BENCHMARK] Starting run with model: {model}")
            print("="*60)

            client = TranscriptionClient(
                host, port,
                lang=args.lang,
                translate=args.translate,
                model=model,
                use_vad=not args.disable_vad,
                save_output_recording=args.save_output_recording,
                output_recording_filename=args.output_file,
                mute_audio_playback=args.mute_audio_playback,
                enable_translation=args.enable_translation,
                target_language=args.target_language,
            )

            client()  # mic or file run

            # after client ends, assume we store latency stats in client
            if hasattr(client.client, "latency_stats"):
                avg, minv, maxv = client.client.latency_stats
                log_latency_to_csv(model, avg, minv, maxv)
                print(f"[CSV] Logged results for {model} → {CSV_FILE}")

        sys.exit(0)

    # --- Normal single run ---
    client = TranscriptionClient(
        host, port,
        lang=args.lang,
        translate=args.translate,
        model=args.model,
        use_vad=not args.disable_vad,
        save_output_recording=args.save_output_recording,
        output_recording_filename=args.output_file,
        mute_audio_playback=args.mute_audio_playback,
        enable_translation=args.enable_translation,
        target_language=args.target_language,
    )

    if args.files:
        valid_files = []
        for file_path in args.files:
            path = Path(file_path)
            if path.exists() and path.is_file():
                valid_files.append(str(path))
            else:
                print(f"Warning: File not found: {file_path}")

        if not valid_files:
            print("Error: No valid audio files found!")
            sys.exit(1)

        for f in valid_files:
            client(f)
    else:
        print("[INFO]: No files provided → using microphone input")
        print("[INFO]: Speak into your mic, transcription will appear below...")
        client()
