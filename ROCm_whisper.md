# WhisperLive-ROCm
Run WhisperLive's `faster_whisper` backend on AMD GPUs using the official [CTranslate2 ROCm wheel](https://github.com/OpenNMT/CTranslate2/releases). Tested on Radeon AI PRO R9700 (gfx1201/RDNA4) and Ryzen AI Max+ 395 / Radeon 8060S (gfx1151/Strix Halo).

## Docker Installation (recommended)
- Install [docker](https://docs.docker.com/engine/install/)

- Build and run the WhisperLive ROCm image:
```bash
docker build -f docker/Dockerfile.rocm -t whisperlive-rocm .
docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri \
    --group-add "$(getent group video | cut -d: -f3)" \
    --group-add "$(getent group render | cut -d: -f3)" \
    -p 9090:9090 whisperlive-rocm
```

## Native Installation

### Prerequisites
- AMD GPU with ROCm support (see [supported GPUs](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html))
- ROCm 7.2+ installed ([installation guide](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html))
- User in `video` and `render` groups (`sudo usermod -aG video,render $USER`, re-login)
- Python 3.12

### Verify ROCm is working
```bash
rocminfo | grep -E 'Name:|gfx'
# Should show your GPU, e.g. "Name: gfx1151" or "Name: gfx1201"
```

### Install CTranslate2 ROCm wheel
The default `pip install ctranslate2` installs a CUDA-only wheel. Replace it with the official ROCm wheel from the [CTranslate2 releases page](https://github.com/OpenNMT/CTranslate2/releases):

```bash
# Download the ROCm wheels archive (v4.8.0)
curl -LO https://github.com/OpenNMT/CTranslate2/releases/download/v4.8.0/rocm-python-wheels-Linux.zip

# Extract the Python 3.12 wheel
unzip -j rocm-python-wheels-Linux.zip 'temp-linux/ctranslate2-*-cp312-*manylinux*x86_64.whl'

# Install (replaces any existing ctranslate2)
pip install ctranslate2-*-cp312-*.whl
```

### Install WhisperLive server requirements
```bash
pip install -r requirements/server.txt
```

### Verify GPU is visible to CTranslate2
```bash
python -c "import ctranslate2; print('devices:', ctranslate2.get_cuda_device_count())"
```
Expected output: `devices: 1` (CTranslate2 uses the name "cuda" even on ROCm).

If you see `devices: 0`, check:
- Your user is in `video` and `render` groups (re-login after adding)
- `/dev/kfd` exists and is accessible
- The ROCm wheel was installed (not the default PyPI CUDA-only one)

## Run WhisperLive Server with ROCm
```bash
python3 run_server.py --port 9090 --backend faster_whisper
```

The server automatically uses the AMD GPU when the CTranslate2 ROCm wheel is installed. For multi-GPU systems, use `HIP_VISIBLE_DEVICES=N` to select a specific GPU.
