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
                          help='Files to transcribe, separated by spaces. '
                              'If not provided, will use microphone input.')
    parser.add_argument('--output_file', '-o',
                          type=str,
                          default='./output_recording.wav',
                          help='output recording filename, only used for microphone input.')
    parser.add_argument('--model', '-m',
                          type=str,
                          default='small',
                          help='Model to use for transcription, e.g., "tiny, small.en, large-v3".')
    parser.add_argument('--lang', '-l',
                          type=str,
                          default='en',
                          help='Language code for transcription, e.g., "en" for English.')
    parser.add_argument('--translate', '-t',
                          action='store_true',
                          help='Enable translation of the transcription output.')
    parser.add_argument('--mute_audio_playback', '-a',
                          action='store_true',
                          help='Mute audio playback during transcription.') 
    parser.add_argument('--save_output_recording', '-r',
                          action='store_true',
                          help='Save the output recording, only used for microphone input.')
    parser.add_argument('--enable_translation',
                          action='store_true',
                          help='Enable translation of the transcription output.')
    parser.add_argument('--target_language', '-tl',
                          type=str,
                          default='fr',
                          help='Target language for translation, e.g., "fr" for French.')

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
            lang=args.lang,
            translate=args.translate,
            model=args.model,                                  # also support hf_model => `Systran/faster-whisper-small`
            use_vad=True,
            save_output_recording=args.save_output_recording,  # Only used for microphone input, False by Default
            output_recording_filename=args.output_file,        # Only used for microphone input
            mute_audio_playback=args.mute_audio_playback,      # Only used for file input, False by Default
            enable_translation=args.enable_translation,        # Enable translation of the transcription output
            target_language=args.target_language,              # Target language for translation, e.g., "fr
        )
        client(f)
