# WhisperLive-TensorRT
We have only tested the TensorRT backend in docker so, we recommend docker for a smooth TensorRT backend setup.
**Note**: We use `tensorrt_llm==0.18.2`

## Installation
- Install [docker](https://docs.docker.com/engine/install/)
- Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

- Run WhisperLive TensorRT in docker
```bash
docker build . -f docker/Dockerfile.tensorrt -t whisperlive-tensorrt
docker run -p 9090:9090 --runtime=nvidia --gpus all --entrypoint /bin/bash -it whisperlive-tensorrt
```

## Whisper TensorRT Engine
- We build `small.en` and `small` multilingual TensorRT engine as examples below. The script logs the path of the directory with Whisper TensorRT engine. We need that model_path to run the server.
```bash
# convert small.en
bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en        # float16
bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en int8   # int8 weight only quantization
bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en int4   # int4 weight only quantization

# convert small multilingual model
bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small
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

By default trt_backend uses cpp_session, to use python session pass `--trt_py_session` to run_server.py
```bash
python3 run_server.py --port 9090 \
                      --backend tensorrt \
                      --trt_model_path "/app/TensorRT-LLM-examples/whisper/whisper_small_float16" \
                      --trt_py_session
```