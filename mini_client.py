#!/usr/bin/env python3
"""
mini_client.py — simple WhisperLive client

Usage:
  python mini_client.py --file ./audio.wav \
                        --server ws://127.0.0.1:9090 \
                        --language nl \
                        --model small \
                        --realtime \
                        --save

Notes:
- Requires: websocket-client, soundfile, numpy, librosa
  pip install websocket-client soundfile numpy librosa
- Make sure your WhisperLive server is already running, e.g.:
  python -u -m whisper_live.server --host 127.0.0.1 --port 9090 --backend faster_whisper
"""

import argparse
import json
import threading
import time
from typing import List, Dict

import numpy as np
import soundfile as sf
import websocket  # from websocket-client

# --- Argument Parsing ---

def parse_args():
    ap = argparse.ArgumentParser(description="Minimal WhisperLive audio streaming client")
    ap.add_argument("--file", required=True, help="Path to audio file (wav/mp3/flac/etc.)")
    ap.addargument("--server", default="ws://127.0.0.1:9090", help="WebSocket server URL")
    ap.add_argument("--language", default="en", help="Language code, e.g. en, nl, fr")
    ap.add_argument("--model", default="tiny", help="Whisper model name, e.g. tiny, base, small, medium")
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Task to perform")
    ap.add_argument("--use_vad", action="store_true", help="Enable server-side VAD")
    ap.add_argument("--no_vad", action="store_true", help="Disable server-side VAD (overrides --use_vad)")
    ap.add_argument("--send_last_n_segments", type=int, default=5, help="How many recent segments to echo back")
    ap.add_argument("--same_output_threshold", type=int, default=20, help="Debounce repeated partials")
    ap.add_argument("--target_language", default="en", help="Target language for translate task")
    ap.add_argument("--chunk", type=int, default=4096, help="Samples per audio frame (at 16 kHz)")
    ap.add_argument("--realtime", action="store_true", help="Pace sending chunks in real time")
    ap.add_argument("--save", action="store_true", help="Save transcript.txt and transcript.srt")
    ap.add_argument("--uid", default="mini-1", help="Client UID to send to server")
    return ap.parse_args()


# --- SRT Formatting Utilities ---

def fmt_srt_time(sec_float: float) -> str:
    """Convert seconds -> SRT timestamp string HH:MM:SS,mmm"""
    s = float(sec_float)
    h = int(s // 3600)
    s -= h * 3600
    m = int(s // 60)
    s -= m * 60
    sec = int(s)
    ms = int(round((s - sec) * 1000))
    # This is the line that was cut off.
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def make_srt_block(idx: int, start_sec: float, end_sec: float, text: str) -> str:
    """Format a single SRT block"""
    start_ts = fmt_srt_time(start_sec)
    end_ts = fmt_srt_time(end_sec)
    return f"{idx}\n{start_ts} --> {end_ts}\n{text}\n\n"


# --- Client Class ---

class MiniClient:
    """
    Manages WebSocket connection, audio streaming, and transcription handling.
    """
    def __init__(self, args):
        self.args = args
        self.ws = None
        self.full_transcript: List[Dict] = []
        self.last_segment = None
        self.is_ready = False
        self.audio_thread = None
        self.ws_thread = None
        self.start_time = time.time()
        self.END_OF_AUDIO_SIGNAL = b"END_OF_AUDIO"

        # Setup WebSocket client
        self.ws = websocket.WebSocketApp(
            args.server,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

    def log(self, *msg):
        """Simple logger with timestamp."""
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:8.3f}s] {' '.join(map(str, msg))}")

    def on_open(self, ws):
        """Called when WebSocket connection is established."""
        self.log("INFO: WebSocket connection opened.")
        options = {
            "uid": self.args.uid,
            "language": self.args.language,
            "model": self.args.model,
            "task": self.args.task,
            "use_vad": not self.args.no_vad if self.args.no_vad else self.args.use_vad,
            "send_last_n_segments": self.args.send_last_n_segments,
            "same_output_threshold": self.args.same_output_threshold,
            "target_language": self.args.target_language,
        }
        self.log("INFO: Sending client options:", json.dumps(options))
        ws.send(json.dumps(options))

    def on_message(self, ws, message):
        """Called on every message from the server."""
        data = json.loads(message)

        if data.get("uid") != self.args.uid:
            self.log(f"WARN: Ignoring message for other UID: {data.get('uid')}")
            return

        if data.get("message") == "SERVER_READY":
            self.log("INFO: Server is ready.")
            self.is_ready = True
            return

        if "segments" in data:
            self.handle_segments(data["segments"])

    def handle_segments(self, segments: List[Dict]):
        """Process transcription segments from the server."""
        if not segments:
            return

        # Print real-time transcript
        texts = [s["text"].strip() for s in segments]
        self.log("TRANSCRIPT:", " ".join(texts))

        # Store finalized segments for SRT/TXT saving
        for seg in segments:
            if seg.get("completed", False):
                # Add to full transcript if it's new
                if not self.full_transcript or float(seg['start']) >= float(self.full_transcript[-1]['end']):
                    self.log(f"  -> FINALIZED: [{seg['start']}s -> {seg['end']}s] {seg['text']}")
                    self.full_transcript.append(seg)
        
        # Keep track of the *very* last segment for partials
        self.last_segment = segments[-1]

    def on_error(self, ws, error):
        """Called on WebSocket error."""
        self.log(f"ERROR: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Called when WebSocket connection closes."""
        self.log(f"INFO: WebSocket closed: {close_status_code} {close_msg}")
        self.is_ready = False
        self.save_results()  # Save transcript when done

    def send_audio_chunks(self):
        """Load audio file and stream it to the server in chunks."""
        try:
            # Wait for server to be ready
            self.log("INFO: Waiting for server to be ready...")
            while not self.is_ready:
                time.sleep(0.1)
                if not self.ws_thread.is_alive():
                    self.log("ERROR: WebSocket connection failed before ready.")
                    return

            self.log(f"INFO: Loading audio file: {self.args.file}")
            
            # Use librosa to load *any* format and resample to 16kHz
            # This is much more robust than using soundfile alone.
            try:
                import librosa
                audio_data, _ = librosa.load(self.args.file, sr=16000, mono=True)
            except ImportError:
                self.log("WARN: librosa not found. Falling back to soundfile. "
                         "Audio MUST be 16kHz mono WAV for this to work.")
                self.log("      (pip install librosa)")
                audio_data, sr = sf.read(self.args.file, dtype='float32')
                if sr != 16000:
                    self.log(f"ERROR: Audio sample rate is {sr}Hz, but server expects 16000Hz.")
                    self.log("       Please resample or install librosa for auto-resampling.")
                    return
                if audio_data.ndim > 1:
                    self.log("ERROR: Audio is not mono. Please convert or install librosa.")
                    return

            self.log(f"INFO: Audio loaded ({len(audio_data) / 16000:.2f} seconds).")
            
            chunk_size = self.args.chunk
            total_samples = len(audio_data)
            chunk_duration_s = chunk_size / 16000.0

            self.log(f"INFO: Streaming audio in {chunk_size}-sample chunks...")

            for i in range(0, total_samples, chunk_size):
                if not self.ws.sock or not self.ws.sock.connected:
                    self.log("WARN: WebSocket disconnected. Stopping audio stream.")
                    break
                
                chunk = audio_data[i : i + chunk_size]
                
                # Send audio chunk as binary data
                self.ws.send(chunk.tobytes(), websocket.ABNF.OPCODE_BINARY)

                if self.args.realtime:
                    # Wait for chunk duration to simulate real-time
                    time.sleep(chunk_duration_s)

            self.log("INFO: Audio streaming finished.")
            # Send end-of-audio signal
            self.ws.send(self.END_OF_AUDIO_SIGNAL)
            self.log("INFO: Sent END_OF_AUDIO.")

        except Exception as e:
            self.log(f"ERROR in audio thread: {e}")
        finally:
            # Give server a moment to process final segment
            time.sleep(2) 
            self.ws.close()

    def run(self):
        """Start the WebSocket and audio streaming threads."""
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.audio_thread = threading.Thread(target=self.send_audio_chunks)

        self.ws_thread.daemon = True
        self.audio_thread.daemon = True

        self.ws_thread.start()
        self.audio_thread.start()

        # Wait for threads to finish
        try:
            self.audio_thread.join()
            self.ws_thread.join()
        except KeyboardInterrupt:
            self.log("INFO: Keyboard interrupt. Closing.")
            self.ws.close()
            # Wait for threads to clean up
            self.audio_thread.join()
            self.ws_thread.join()

    def save_results(self):
        """Save full transcript to TXT and SRT files if --save is used."""
        if not self.args.save:
            return
        
        # Ensure we have the very last segment
        if self.last_segment and (not self.full_transcript or self.full_transcript[-1]["text"] != self.last_segment["text"]):
             self.full_transcript.append(self.last_segment)

        if not self.full_transcript:
            self.log("WARN: No finalized transcript to save.")
            return

        # Save TXT
        txt_path = "transcript.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in self.full_transcript:
                f.write(f"{seg['text'].strip()}\n")
        self.log(f"INFO: Plain text transcript saved to {txt_path}")

        # Save SRT
        srt_path = "transcript.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(self.full_transcript):
                try:
                    start = float(seg["start"])
                    end = float(seg["end"])
                    text = seg["text"].strip()
                    f.write(make_srt_block(i + 1, start, end, text))
                except KeyError:
                    self.log(f"WARN: Skipping malformed segment for SRT: {seg}")
                except Exception as e:
                    self.log(f"WARN: Error writing SRT block for segment {seg}: {e}")
                    
        self.log(f"INFO: SRT transcript saved to {srt_path}")


# --- Main Execution ---

def main():
    args = parse_args()
    
    # --- Dependency Check ---
    try:
        import librosa
    except ImportError:
        print("="*50)
        print("WARNING: `librosa` is not installed.")
        print("         Audio file MUST be a 16kHz MONO WAV file.")
        print("         Install with `pip install librosa` for automatic resampling.")
        print("="*50)
        time.sleep(2) # Give user time to see warning

    client = MiniClient(args)
    client.run()


if __name__ == "__main__":
    main()