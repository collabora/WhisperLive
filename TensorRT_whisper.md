# Whisper-TensorRT
We have only tested the TensorRT backend in docker so, we recommend docker for a smooth TensorRT backend setup.
**Note**: We use [our fork to setup TensorRT](https://github.com/makaveli10/TensorRT-LLM)

## Installation
- Install [docker](https://docs.docker.com/engine/install/)
- Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

- Clone this repo.
```bash
git clone https://github.com/collabora/WhisperLive.git
cd WhisperLive
```

- Build the TensorRT-LLM docker image
```bash
docker build --file docker/Dockerfile.tensorrt --tag tensorrt_llm:latest .
```
**NOTE**: This could take some time.

- Next, we run the docker image and mount WhisperLive repo to the containers `/home` directory.
```bash
docker run -it --gpus all --shm-size=8g \
       --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
       -v /path/to/WhisperLive:/home/WhisperLive \
       tensorrt_llm:latest

cd /home/TensorRT-LLM
python3 scripts/build_wheel.py --cuda_architectures "89-real" --clean --trt_root /usr/local/tensorrt
pip install build/tensorrt_llm*.whl
```
**NOTE**: `--cuda_architectures "89-real"` builds for 4090, change according to your device.

- Make sure to test the installation. 
```bash
# export ENV=${ENV:-/etc/shinit_v2} 
# source $ENV
python -c "import torch; import tensorrt; import tensorrt_llm"
```
**NOTE**: Uncomment and update library paths if imports fail.

## Whisper TensorRT Engine
- We build `small.en` and `small` multilingual TensorRT engine. The script logs the path of the directory with Whisper TensorRT engine. We need the model_path to run the server.
```bash
# convert small.en
bash build_whisper_tensorrt /path/to/TensorRT-LLM/examples small.en

# convert small multilingual model
bash build_whisper_tensorrt /path/to/TensorRT-LLM/examples small
```

## Run WhisperLive Server with TensorRT Backend
```bash
cd /home/WhisperLive

# Required to create mel spectogram
wget --directory-prefix=assets assets/mel_filters.npz https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz

# Run English only model
python3 run_server.py --port 9090 \
                      --backend tensorrt \
                      --trt_model_path "path/from/build/step"

# Run Multilingual model
python3 run_server.py --port 9090 \
                      --backend tensorrt \
                      --trt_model_path "path/from/build/step" \
                      --trt_multilingual
```
