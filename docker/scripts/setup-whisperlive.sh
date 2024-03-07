#!/bin/bash -e

## Clone this repo and install requirements
[ -d "WhisperLive" ] || git clone https://github.com/collabora/WhisperLive.git

cd WhisperLive
apt update
apt-get install portaudio19-dev ffmpeg wget -y

## Install all the other dependencies normally
pip install -r requirements/server.txt

mkdir -p /root/.cache/whisper-live/
curl -L -o /root/.cache/whisper-live/silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx

# the sound filter definitions
wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
