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

- Build docker image for the gpu architecture. By default the image is built for 4090 i.e. `CUDA_ARCH=89-real;90-real`
```
mkdir docker/scratch-space
cp docker/scripts/build-whisper-tensorrt.sh docker/scratch-space
cp docker/scripts/run-whisperlive.sh docker/scratch-space

# For e.g. 3090 RTX cuda architecture is 86-real
CUDA_ARCH=86-real docker compose build
```

## Run WhisperLive Server with TensorRT Backend
We run the container with docker compose which builds the tensorrt engine for specified model
if it doesnt exist already in the mounted volume `docker/scratch-space`. Optionally, if you want to run `faster_whisper` backend use `BACKEND=faster_whisper`
```bash
MODEL_SIZE=small.en BACKEND=tensorrt docker compose up
```
