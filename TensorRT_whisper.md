# WhisperLive-TensorRT
We have only tested the TensorRT backend in docker so, we recommend docker for a smooth TensorRT backend setup.
**Note**: We use `tensorrt_llm==0.9.0`

## Installation
- Install [docker](https://docs.docker.com/engine/install/)
- Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

- Clone this repo.
```bash
git clone https://github.com/collabora/WhisperLive.git
cd WhisperLive
```

- Run WhisperLive TensorRT in docker
```bash
docker run -p 9090:9090 --runtime=nvidia --gpus all --entrypoint /bin/bash -it ghcr.io/collabora/whisperlive-tensorrt:latest
```

## Whisper TensorRT Engine
- We build `small.en` and `small` multilingual TensorRT engine as examples below. The script logs the path of the directory with Whisper TensorRT engine. We need that model_path to run the server.
```bash
# convert small.en
bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en

# convert small multilingual model
bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small
```

## Run WhisperLive Server with TensorRT Backend
```bash
# Run English only model
python3 run_server.py --port 9090 \
                      --backend tensorrt \
                      --trt_model_path "/app/TensorRT-LLM-examples/whisper/whisper_small_en"

# Run Multilingual model
python3 run_server.py --port 9090 \
                      --backend tensorrt \
                      --trt_model_path "/app/TensorRT-LLM-examples/whisper/whisper_small" \
                      --trt_multilingual
```
