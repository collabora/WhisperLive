# Edge & Embedded Deployment Guide

WhisperLive can run on edge devices including Raspberry Pi 4/5, NVIDIA Jetson Nano/Xavier/Orin, and other ARM64 platforms.

## Quick Start

### Raspberry Pi / Generic ARM64

```bash
# Build the edge-optimized image
docker build -f docker/Dockerfile.edge -t whisperlive-edge .

# Run with defaults (tiny model, REST enabled, 2 max clients)
docker run -p 9090:9090 -p 8000:8000 whisperlive-edge

# Run with noise reduction
docker run -p 9090:9090 -p 8000:8000 whisperlive-edge \
  --enable_rest --noise_reduction near_field
```

### NVIDIA Jetson (with CUDA)

```bash
# Build the Jetson-specific image (uses L4T base with CUDA)
docker build -f docker/Dockerfile.jetson -t whisperlive-jetson .

# Run with NVIDIA runtime
docker run --runtime nvidia -p 9090:9090 -p 8000:8000 whisperlive-jetson
```

## Model Selection for Edge

| Device | RAM | Recommended Model | Compute Type | Notes |
|--------|-----|-------------------|-------------|-------|
| RPi 4 (4GB) | 4GB | `tiny` | int8 | ~1GB RAM usage |
| RPi 5 (8GB) | 8GB | `base` | int8 | ~2GB RAM usage |
| Jetson Nano | 4GB | `tiny` | float16 | CUDA accelerated |
| Jetson Xavier NX | 8GB | `small` | float16 | CUDA accelerated |
| Jetson Orin | 16-64GB | `medium` or `large-v3` | float16 | Full models |

## Performance Tips

### Reduce Memory Usage
- Use `tiny` or `base` models with `int8` compute type
- Set `--max_clients 1` or `2` for limited RAM devices
- Disable features you don't need (metrics, diarization)

### Reduce Latency
- Enable `--noise_reduction near_field` for cleaner input
- Use `--raw_pcm_input` to avoid float32 conversion overhead
- Set lower `--max_connection_time` to free resources faster

### Docker Compose for Edge

```yaml
version: '3.8'
services:
  whisperlive:
    build:
      context: .
      dockerfile: docker/Dockerfile.edge
    ports:
      - "9090:9090"
      - "8000:8000"
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'
    command: >
      --enable_rest
      --max_clients 2
      --noise_reduction near_field
      --metrics_port 9091
```

## Without Docker

```bash
# Install on the device directly
pip install faster-whisper websockets fastapi uvicorn noisereduce soundfile

# Clone and run
git clone https://github.com/collabora/WhisperLive.git
cd WhisperLive
python run_server.py \
  --backend faster_whisper \
  --faster_whisper_custom_model_path tiny \
  --enable_rest \
  --max_clients 2 \
  --noise_reduction near_field
```

## Monitoring on Edge

Enable Prometheus metrics for lightweight monitoring:

```bash
docker run -p 9090:9090 -p 8000:8000 -p 9091:9091 whisperlive-edge \
  --enable_rest --metrics_port 9091
```

Access metrics at `http://device-ip:9091/metrics`.

## Tested Platforms

- Raspberry Pi 4 Model B (4GB/8GB) with Raspberry Pi OS 64-bit
- Raspberry Pi 5 (8GB) with Raspberry Pi OS 64-bit
- NVIDIA Jetson Nano (4GB) with JetPack 5.x
- NVIDIA Jetson Xavier NX (8GB) with JetPack 5.x
- NVIDIA Jetson AGX Orin with JetPack 6.x
- Generic x86_64 and ARM64 Linux with Docker
