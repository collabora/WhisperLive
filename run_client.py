from pathlib import Path
import sys
from whisper_live.client import TranscriptionClient
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', '-p',
                        type=int,
                        default=9090,
                        help="Websocket port to run the server on.")
  parser.add_argument('--server', '-s',
                        type=str,
                        default='localhost',
                        help='hostname or ip address of server')
  parser.add_argument('--files', '-f',
                        type=str,
                        nargs='+',
                        help='hostname or ip address of server')
  parser.add_argument('--output_file', '-o',
                        type=str,
                        default='./output_recording.wav',
                        help='hostname or ip address of server')
  args = parser.parse_args()

  # Validate audio files
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

  print(f"Found {len(valid_files)} audio file(s) to stream:")
  for file_path in valid_files:
      print(f"  - {file_path}")

  for f in valid_files:
    client = TranscriptionClient(
      args.server,
      args.port,
      lang="en",
      translate=False,
      model="large-v3",                                   # also support hf_model => `Systran/faster-whisper-small`
      use_vad=False,
      save_output_recording=False,                         # Only used for microphone input, False by Default
      output_recording_filename=args.output_file, # Only used for microphone input
      max_clients=4,
      max_connection_time=600,
      mute_audio_playback=True,                          # Only used for file input, False by Default
    )
    client(f)
