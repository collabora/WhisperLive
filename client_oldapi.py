import sys
from whisper_live.client import TranscriptionClient

if len(sys.argv) < 2:
    print("Usage: python transcribe_file.py <path_to_audio_file>")
    sys.exit(1)

audio_file = sys.argv[1]

client = TranscriptionClient(
    "localhost",
    9090,
    lang="en",
    translate=False,
    model="small",  # also support hf_model => `Systran/faster-whisper-small`
    use_vad=False,
    save_output_recording=True,  # Only used for microphone input, False by Default
    output_recording_filename="./output_recording.wav",  # Only used for microphone input
    mute_audio_playback=False,  # Only used for file input, False by Default
    enable_translation=True,
    target_language="hi",
)

# Transcribe the offline audio file
client(audio_file)
