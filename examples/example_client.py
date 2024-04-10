from whisper_live.client import TranscriptionClient
import time

last_text = ""

def sample_callback(text, is_final):
  global last_text
  if is_final and text != last_text:
    print(text[-1])
    last_text = text

client = TranscriptionClient(
  "localhost",
  9090,
  lang="en",
  translate=False,
  model="tiny.en",
  use_vad=True,
  callback=sample_callback
)

client()
