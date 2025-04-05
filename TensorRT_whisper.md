# WhisperLive-TensorRT
We have only tested the TensorRT backend in docker so, we recommend docker for a smooth TensorRT backend setup.
**Note**: We use `tensorrt_llm==0.15.0.dev2024111200`

## Installation
- Install [docker](https://docs.docker.com/engine/install/)
- Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

- Run WhisperLive TensorRT in docker
```bash
docker run -p 9090:9090 --runtime=nvidia --gpus all --entrypoint /bin/bash -it ghcr.io/collabora/whisperlive-tensorrt:latest
```

## Whisper TensorRT Engine
- We build `small.en` and `small` multilingual TensorRT engine as examples below. The script logs the path of the directory with Whisper TensorRT engine. We need that model_path to run the server.
```bash
# convert small.en
bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en        # float16
bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en int8   # int8 weight only quantization
bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en int4   # int4 weight only quantization
bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples medium

# convert small multilingual model
bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small
```

we have committed a docker image for medium, reuse this one!
```
REPOSITORY                         TAG                IMAGE ID       CREATED          SIZE
whisperlive-trt-medium-ready       latest             8596e0157dbf   2 seconds ago    19.1GB
```

## Run WhisperLive Server with TensorRT Backend
```bash
# Run English only model
python3 run_server.py --port 9090 \
                      --backend tensorrt \
                      --trt_model_path "/app/TensorRT-LLM-examples/whisper/whisper_small_en_float16"

# Run Multilingual model
python3 run_server.py --port 9090 \
                      --backend tensorrt \
                      --trt_model_path "/app/TensorRT-LLM-examples/whisper/whisper_small_float16" \
                      --trt_multilingual
```


python3 run_server.py --port 9090 \
                      --backend tensorrt \
                      --trt_model_path "/app/TensorRT-LLM-examples/whisper/whisper_medium_float16" \
                      --trt_multilingual
