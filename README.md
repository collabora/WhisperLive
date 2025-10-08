# WhisperLive

<!-- markdownlint-disable MD033 -->
<h2 align="center">
  <a href="https://www.youtube.com/watch?v=0PHWCApIcCI"><img
src="https://img.youtube.com/vi/0PHWCApIcCI/0.jpg" style="background-color:rgba(0,0,0,0);" height=300 alt="WhisperLive"></a>
  <a href="https://www.youtube.com/watch?v=0f5oiG4oPWQ"><img
  src="https://img.youtube.com/vi/0f5oiG4oPWQ/0.jpg" style="background-color:rgba(0,0,0,0);" height=300 alt="WhisperLive"></a>
  <br><br>A nearly-live implementation of OpenAI's Whisper.
<br><br>
</h2>
<!-- markdownlint-enable MD033 -->

This project is a real-time transcription application that uses the OpenAI Whisper model
to convert speech input into text output. It can be used to transcribe both live audio
input from microphone and pre-recorded audio files.

- [WhisperLive](#whisperlive)
  - [Installation](#installation)
    - [Using uv (Recommended)](#using-uv-recommended)
    - [Using pip](#using-pip)
    - [Setting up NVIDIA/TensorRT-LLM for TensorRT backend](#setting-up-nvidiatensorrt-llm-for-tensorrt-backend)
  - [Getting Started](#getting-started)
    - [Fork Enhancements](#fork-enhancements)
    - [Running with uv](#running-with-uv)
    - [Running the Server](#running-the-server)
      - [Controlling OpenMP Threads](#controlling-openmp-threads)
      - [Single model mode](#single-model-mode)
    - [Running the Client](#running-the-client)
  - [Browser Extensions](#browser-extensions)
  - [iOS Client](#ios-client)
  - [Whisper Live Server in Docker](#whisper-live-server-in-docker)
  - [Future Work](#future-work)
  - [Blog Posts](#blog-posts)
  - [Contact](#contact)
  - [Citations](#citations)

## Installation

### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Choose the installation option based on your usage:

**Client only** (for running transcription clients):

```bash
uv sync --group client
```

**Server with faster-whisper backend**:

```bash
uv sync --group server
```

**Server with OpenVINO backend** (includes client):

```bash
uv sync --group openvino
```

**Full installation** (all backends and tools):

```bash
uv sync --all-groups
```

**Dependency groups:**

- `client`: Core client dependencies (PyAudio, WebSocket)
- `server`: Server with faster-whisper backend
- `openvino`: OpenVINO backend (includes client dependencies)

### Using pip

```bash
# Install PyAudio
bash scripts/setup.sh

# Install whisper-live
pip install whisper-live
```

### Setting up NVIDIA/TensorRT-LLM for TensorRT backend

- Please follow [TensorRT_whisper readme](https://github.com/collabora/WhisperLive/blob/main/TensorRT_whisper.md) for setup of [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and for building Whisper-TensorRT engine.

## Getting Started

The server supports 3 backends `faster_whisper`, `tensorrt` and `openvino`. If running `tensorrt` backend follow [TensorRT_whisper readme](https://github.com/collabora/WhisperLive/blob/main/TensorRT_whisper.md). For `openvino` backend setup and advanced configuration (VAD, CPU threads), see [OpenVINO readme](https://github.com/kml93/WhisperLive/blob/main/OpenVINO.md).

### Fork Enhancements

This fork adds the following enhancements to the original WhisperLive:

- **OpenVINO Backend Improvements**
  - Voice Activity Detection (VAD) support for OpenVINO backend
  - Configurable CPU threads (`--cpu_threads`) for performance optimization (supports OpenVINO and faster_whisper backends)
  - Cache path propagation for VAD models
- **Project Management**: [uv](https://docs.astral.sh/uv/) package manager support with dependency groups
- **Custom Scripts**: `start_client.py` and `start_server.py` for streamlined usage

See [OpenVINO.md](https://github.com/kml93/WhisperLive/blob/main/OpenVINO.md) for detailed OpenVINO backend documentation.

### Running with uv

If you installed with `uv`, use `uv run` to execute scripts with the project's virtual environment:

**Run server:**

```bash
uv run python run_server.py --port 9090 --backend openvino
```

**Run client:**

```bash
uv run python run_client.py
```

**Run custom scripts:**

```bash
uv run python start_server.py  # Pre-configured server
uv run python start_client.py  # Pre-configured client
```

**Benefits of `uv run`:**

- Automatically uses the correct virtual environment (`.venv`)
- No need to manually activate the environment
- Ensures all dependencies are available
- Works with all installed dependency groups

**Alternatively**, activate the virtual environment manually:

```bash
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
python3 run_server.py --port 9090 --backend openvino
```

### Running the Server

- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) backend

```bash
python3 run_server.py --port 9090 \
                      --backend faster_whisper \
                      --max_clients 4 \
                      --max_connection_time 600

# running with custom model and cache_dir to save auto-converted ctranslate2 models
python3 run_server.py --port 9090 \
                      --backend faster_whisper \
                      --max_clients 4 \
                      --max_connection_time 600 \
                      -fw "/path/to/custom/faster/whisper/model" \
                      -c ~/.cache/whisper-live/
```

- TensorRT backend. Currently, we recommend to only use the docker setup for TensorRT. Follow [TensorRT_whisper readme](https://github.com/collabora/WhisperLive/blob/main/TensorRT_whisper.md) which works as expected. Make sure to build your TensorRT Engines before running the server with TensorRT backend.

```bash
# Run English only model
python3 run_server.py -p 9090 \
                      -b tensorrt \
                      -trt /home/TensorRT-LLM/examples/whisper/whisper_small_en \
                      --max_clients 4 \
                      --max_connection_time 600

# Run Multilingual model
python3 run_server.py -p 9090 \
                      -b tensorrt \
                      -trt /home/TensorRT-LLM/examples/whisper/whisper_small \
                      -m \
                      --max_clients 4 \
                      --max_connection_time 600
```

- Use `--max_clients` option to restrict the number of clients the server should allow. Defaults to 4.
- Use `--max_connection_time` options to limit connection time for a client in seconds. Defaults to 600.
- [OpenVINO](https://github.com/openvinotoolkit/openvino) backend. For efficient inference on Intel CPUs, iGPU and dGPUs. See [OpenVINO.md](https://github.com/kml93/WhisperLive/blob/main/OpenVINO.md) for detailed setup and configuration.
  - > **Docker Recommended:** Running WhisperLive with OpenVINO inside Docker automatically enables GPU support (iGPU/dGPU) without requiring additional host setup.
  - > **Native (non-Docker) Use:** If you prefer running outside Docker, ensure the Intel drivers and OpenVINO runtime are installed and properly configured on your system. Refer to the documentation for [installing OpenVINO](https://docs.openvino.ai/2025/get-started/install-openvino.html?PACKAGE=OPENVINO_BASE&VERSION=v_2025_0_0&OP_SYSTEM=LINUX&DISTRIBUTION=PIP#).

```bash
# Basic usage
python3 run_server.py -p 9090 -b openvino

# With CPU thread optimization and caching
python3 run_server.py -p 9090 -b openvino \
                      --cpu_threads 20 \
                      --cache_path ~/.cache/whisper-live/
```

**Note**: Model selection and VAD are configured on the client side. See [OpenVINO.md](https://github.com/kml93/WhisperLive/blob/main/OpenVINO.md) for client configuration examples.

#### Controlling CPU Threads

To control the number of CPU threads used for inference, use the `--cpu_threads` argument. This parameter works for both OpenVINO and faster_whisper backends, and will also set the `OMP_NUM_THREADS` environment variable if needed. Setting it to `0` (default) enables auto-detection:

```bash
python3 run_server.py --port 9090 \
                      --backend faster_whisper \
                      --cpu_threads 4
```

#### Single model mode

By default, when running the server without specifying a model, the server will instantiate a new whisper model for every client connection. This has the advantage, that the server can use different model sizes, based on the client's requested model size. On the other hand, it also means you have to wait for the model to be loaded upon client connection and you will have increased (V)RAM usage.

When serving a custom TensorRT model using the `-trt` or a custom faster_whisper model using the `-fw` option, the server will instead only instantiate the custom model once and then reuse it for all client connections.

If you don't want this, set `--no_single_model`.

### Running the Client

- Initializing the client with below parameters:
  - `lang`: Language of the input audio, applicable only if using a multilingual model.
  - `translate`: If set to `True` then translate from any language to `en`.
  - `model`: Whisper model size.
  - `use_vad`: Whether to use `Voice Activity Detection` on the server.
  - `save_output_recording`: Set to True to save the microphone input as a `.wav` file during live transcription. This option is helpful for recording sessions for later playback or analysis. Defaults to `False`.
  - `output_recording_filename`: Specifies the `.wav` file path where the microphone input will be saved if `save_output_recording` is set to `True`.
  - `mute_audio_playback`: Whether to mute audio playback when transcribing an audio file. Defaults to False.
  - `enable_translation`: Start translation thread on the server (from any to any).
  - `target_language`: Server translation thread's target translation language.

```python
from whisper_live.client import TranscriptionClient
client = TranscriptionClient(
  "localhost",
  9090,
  lang="en",
  translate=False,
  model="small",                                      # also support hf_model => `Systran/faster-whisper-small`
  use_vad=False,
  save_output_recording=True,                         # Only used for microphone input, False by Default
  output_recording_filename="./output_recording.wav", # Only used for microphone input
  mute_audio_playback=False,                          # Only used for file input, False by Default
  enable_translation=True,
  target_language="hi",
)
```

It connects to the server running on localhost at port 9090. Using a multilingual model, language for the transcription will be automatically detected. You can also use the language option to specify the target language for the transcription, in this case, English ("en"). The translate option should be set to `True` if we want to translate from the source language to English and `False` if we want to transcribe in the source language.

- Transcribe an audio file:

```python
client("tests/jfk.wav")
```

- To transcribe from microphone:

```python
client()
```

- To transcribe from a RTSP stream:

```python
client(rtsp_url="rtsp://admin:admin@192.168.0.1/rtsp")
```

- To transcribe from a HLS stream:

```python
client(hls_url="http://as-hls-ww-live.akamaized.net/pool_904/live/ww/bbc_1xtra/bbc_1xtra.isml/bbc_1xtra-audio%3d96000.norewind.m3u8")
```

## Browser Extensions

- Run the server with your desired backend as shown in [Running the Server](#running-the-server).
- Transcribe audio directly from your browser using our Chrome or Firefox extensions. Refer to [Audio-Transcription-Chrome](https://github.com/collabora/whisper-live/tree/main/Audio-Transcription-Chrome#readme).

## iOS Client

Use WhisperLive on iOS with our native iOS client.
Refer to [`ios-client`](https://github.com/collabora/WhisperLive/tree/main/Audio-Transcription-iOS) and [`ios-client/README.md`](https://github.com/collabora/WhisperLive/blob/main/Audio-Transcription-iOS/README.md) for setup and usage instructions.

## Whisper Live Server in Docker

- GPU
  - Faster-Whisper

  ```bash
  docker run -it --gpus all -p 9090:9090 ghcr.io/collabora/whisperlive-gpu:latest
  ```

  - TensorRT. Refer to [TensorRT_whisper readme](https://github.com/collabora/WhisperLive/blob/main/TensorRT_whisper.md) for setup and more tensorrt backend configurations.

  ```bash
  docker build . -f docker/Dockerfile.tensorrt -t whisperlive-tensorrt
  docker run -p 9090:9090 --runtime=nvidia --entrypoint /bin/bash -it whisperlive-tensorrt

  # Build small.en engine
  bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en        # float16
  bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en int8   # int8 weight only quantization
  bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en int4   # int4 weight only quantization

  # Run server with small.en
  python3 run_server.py --port 9090 \
                        --backend tensorrt \
                        --trt_model_path "/app/TensorRT-LLM-examples/whisper/whisper_small_en_float16"
                        --trt_model_path "/app/TensorRT-LLM-examples/whisper/whisper_small_en_int8"
                        --trt_model_path "/app/TensorRT-LLM-examples/whisper/whisper_small_en_int4"
  ```

  - OpenVINO

  ```bash
  docker run -it --device=/dev/dri -p 9090:9090 ghcr.io/collabora/whisperlive-openvino
  ```

- CPU
  - Faster-whisper

  ```bash
  docker run -it -p 9090:9090 ghcr.io/collabora/whisperlive-cpu:latest
  ```

## Future Work

- [x] Add translation to other languages on top of transcription.
- [x] Add VAD support for OpenVINO backend.
- [x] Add CPU thread configuration for OpenVINO backend.

## Blog Posts

- [Transforming speech technology with WhisperLive](https://www.collabora.com/news-and-blog/blog/2024/05/28/transforming-speech-technology-with-whisperlive/)
- [WhisperFusion: Ultra-low latency conversations with an AI chatbot](https://www.collabora.com/news-and-blog/news-and-events/whisperfusion-ultra-low-latency-conversations-with-an-ai-chatbot.html) powered by WhisperLive
- [Breaking language barriers 2.0: Moving closer towards fully reliable, production-ready Hindi ASR](https://www.collabora.com/news-and-blog/news-and-events/breaking-language-barriers-20-moving-closer-production-ready-hindi-asr.html) which is used in WhisperLive for hindi.

## Contact

We are available to help you with both Open Source and proprietary AI projects. You can reach us via the Collabora website or [vineet.suryan@collabora.com](mailto:vineet.suryan@collabora.com) and [marcus.edel@collabora.com](mailto:marcus.edel@collabora.com).

## Citations

```bibtex
@article{Whisper
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  publisher = {arXiv},
  year = {2022},
}
```

```bibtex
@misc{Silero VAD,
  author = {Silero Team},
  title = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snakers4/silero-vad}},
  email = {hello@silero.ai}
}
```
