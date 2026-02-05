import sys
import requests

if len(sys.argv) < 2:
    print("Usage: python transcribe_file.py <path_to_audio_file>")
    sys.exit(1)

audio_file = sys.argv[1]

# Configuration
host = "localhost"
port = 8000  # Default REST port; change if you used --rest_port
url = f"http://{host}:{port}/v1/audio/transcriptions"
model = "small"  # Or "whisper-1" (mapped to small internally)
language = "en"  # Or "hi" for Hindi
response_format = "json"  # Options: "json", "text", "verbose_json", "srt", "vtt"

# Prepare the request
files = {"file": open(audio_file, "rb")}
data = {
    "model": model,
    "language": language,
    "response_format": response_format,
    # Optional: Add "prompt" for style guidance, "temperature" (0-1), etc.
}

# Send the request
response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    if response_format == "json" or response_format == "verbose_json":
        result = response.json()
        print("Transcript:", result.get("text", "No text found"))
        # If you need translation, post-process here (e.g., using another API like Google Translate)
    else:
        print("Transcript:", response.text)
else:
    print("Error:", response.status_code, response.json().get("error", "Unknown error"))
