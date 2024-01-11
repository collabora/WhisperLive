# Whisper-TensorRT
We have only tested the TensorRT backend in docker so, we recommend docker for a smooth TensorRT backend setup.

## Installation
- Install [docker](https://docs.docker.com/engine/install/)
- Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Pull the pytorch docker image.
```bash
docker pull nvcr.io/nvidia/pytorch_23.10-py3
```
- Clone this repo.
```bash
git clone https://github.com/collabora/WhisperLive.git
```
- Next, we run the docker image and mount WhisperLive repo to the containers `/home` directory.
```bash
docker run -it --gpus all --shm-size=64g /path/to/WhisperLive:/home/WhisperLive nvcr.io/nvidia/pytorch_23.10-py3
```
- Build `tensorrt-llm`.
```bash
cd /home/
cp WhisperLive/scripts/install_tensorrt_llm.sh .
bash install_tensorrt_llm.sh
```
This should clone the [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and build it as well.

- Test the installation.
```bash
export ENV=${ENV:-/etc/shinit_v2}
source $ENV
python -c "import torch; import tensorrt; import tensorrt_llm"
```

## Whisper TensorRT Engine
- Change working dir to the [whisper example dir](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/whisper) in TensorRT-LLM.
```bash
cd /home/TensorRT-LLM/examples/whisper
``` 
- Currently, by default TensorRT-LLM only supports `large-v2` and `large-v3`. In this repo, we use `small.en`.

- Edit `build.py` to support all the model sizes i.e. `["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en"]`. In order to do that, update the list [`choices`](https://github.com/NVIDIA/TensorRT-LLM/blob/a75618df24e97ecf92b8899ca3c229c4b8097dda/examples/whisper/build.py#L58) with the model size you prefer for your WhisperLive server.

- Download the models from [here](https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/__init__.py#L17C1-L30C2)
```bash
# small.en model
wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt

# small multilingual model
wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt
```

- For this demo we build `small.en` and `small` multilingual TensorRT engine.
```bash
pip install -r requirements.txt

# convert small.en
python3 build.py --output_dir whisper_small_en --use_gpt_attention_plugin --use_gemm_plugin  --use_bert_attention_plugin --model_name small.en

# convert small multilingual model
python3 build.py --output_dir whisper_small --use_gpt_attention_plugin --use_gemm_plugin  --use_bert_attention_plugin --model_name small
```

- Whisper/small.en tensorrt model engine is saved in `/home/TensorRT-LLM/examples/whisper/whisper_small_en` dir and if you converted the `small` multilingual model it should be saved in `/home/TensorRT-LLM/examples/whisper/whisper_small` dir.

## Run WhisperLive Server with TensorRT Backend
```bash

cd /home/WhisperLive
bash scripts/setup.sh
pip install -r requirements.txt

# Required to create mel spectogram
wget --directory-prefix=assets assets/mel_filters.npz https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz

# Run English only model
python3 run_server.py --port 9090 \
                      --backend tensorrt \
                      --whisper_tensorrt_path /home/TensorRT-LLM/examples/whisper/whisper_small_en

# Run Multilingual model
python3 run_server.py --port 9090 \
                      --backend tensorrt \
                      --whisper_tensorrt_path /home/TensorRT-LLM/examples/whisper/whisper_small_en \
                      --trt_multilingual
```

