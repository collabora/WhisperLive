# WhisperLive-OpenVINO

WhisperLive supports the OpenVINO backend for efficient inference on Intel CPUs, iGPUs and dGPUs. This guide covers setup and usage.

## Installation

### Using pip

Ensure Intel drivers and OpenVINO runtime are installed and configured.
Refer to [OpenVINO installation guide](https://docs.openvino.ai/2025/get-started/install-openvino.html?PACKAGE=OPENVINO_BASE&VERSION=v_2025_0_0&OP_SYSTEM=LINUX&DISTRIBUTION=PIP#).

```bash
# Install dependencies
bash scripts/setup.sh
pip install whisper-live

# Install OpenVINO dependencies
pip install openvino openvino-tokenizers onnx
```

### Using uv (Recommended)

Install [uv](https://docs.astral.sh/uv/) package manager for faster dependency resolution:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Choose the installation option that matches your usage:

**Client only** (for running transcription clients):

```bash
uv sync --group client
```

**Server with OpenVINO backend** (includes client dependencies):

```bash
uv sync --group openvino
```

**Full installation** (client, server, and OpenVINO):

```bash
uv sync --all-groups
```

**Dependency groups:**

- `client`: Core client dependencies (PyAudio, WebSocket)
- `server`: Server dependencies with faster-whisper backend
- `openvino`: OpenVINO backend dependencies (includes client group)

## Server Configuration

### Basic Usage

```bash
python3 run_server.py --port 9090 --backend openvino
```

### Recommended Configuration

With CPU thread optimization and caching:

```bash
python3 run_server.py --port 9090 \
                      --backend openvino \
                      --openvino_cpu_threads 20 \
                      --cache_path ~/.cache/whisper-live/ \
                      --max_clients 4 \
                      --max_connection_time 600
```

**Server Parameters:**

- `--openvino_cpu_threads`: Number of CPU threads for inference (adjust based on your CPU)
- `--cache_path`: Directory to cache models and VAD components
- `--max_clients`: Maximum simultaneous connections (default: 4)
- `--max_connection_time`: Maximum connection duration in seconds (default: 300)

**Note**: Model selection and VAD are configured client-side, not server-side.

## Client Configuration

The client specifies the model and VAD settings:

```python
from whisper_live.client import TranscriptionClient

client = TranscriptionClient(
    host="localhost",
    port=9090,
    lang="en",
    model="OpenVINO/whisper-large-v3",  # OpenVINO model from HuggingFace
    use_vad=True,                        # Enable Voice Activity Detection
    translate=False,
)

# Transcribe from microphone
client()
```

**Client Parameters:**

- `model`: OpenVINO model path (HuggingFace format)
- `use_vad`: Enable Voice Activity Detection to filter silence
- `lang`: Language code (e.g., "en", "fr", "es")
- `translate`: Translate to English (True/False)

## Available Models

WhisperLive has been tested with models from [OpenVINO's HuggingFace collection](https://huggingface.co/OpenVINO?search_models=whisper):

- `OpenVINO/whisper-tiny`
- `OpenVINO/whisper-base`
- `OpenVINO/whisper-small`
- `OpenVINO/whisper-medium`
- `OpenVINO/whisper-large-v3`

## Performance Optimization

- **CPU Threads**: Adjust `--openvino_cpu_threads` based on your CPU core count
- **VAD**: Reduces processing time by skipping silent audio frames (client-side)
- **Cache Path**: Reuse compiled models with `--cache_path` for faster startup
- **GPU Acceleration**: Ensure Intel GPU drivers are installed for iGPU/dGPU support

## Troubleshooting

- Ensure Intel GPU drivers are installed for GPU acceleration
- For CPU-only inference, no special drivers are required
- Check OpenVINO installation: `python -c "import openvino; print(openvino.__version__)"`
