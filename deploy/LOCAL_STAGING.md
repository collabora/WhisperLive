# Local Staging with Docker Compose

Run the full WhisperLive production stack on your local hardware — no cloud costs.

## Stack

| Service | Port | Purpose |
|---------|------|---------|
| WhisperLive | (behind Traefik) | GPU transcription server |
| Traefik | 80 (HTTP), 8080 (dashboard) | Reverse proxy / load balancer |
| MinIO | 9000 (API), 9001 (console) | S3-compatible object storage |
| Prometheus | 9091 | Metrics collection |
| Grafana | 3000 | Dashboards |

## Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- `docker compose` v2
- NVIDIA GPU(s)

Verify GPU support:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

```bash
cd deploy

# Start everything
docker compose -f docker-compose.local.yml up -d

# Check status
docker compose -f docker-compose.local.yml ps

# View logs
docker compose -f docker-compose.local.yml logs -f whisperlive
```

## Endpoints

- **Web UI**: http://localhost/
- **REST API**: http://localhost/v1/audio/transcriptions
- **API Docs**: http://localhost/docs
- **WebSocket**: ws://localhost/ws/
- **Health Check**: http://localhost/health
- **MinIO Console**: http://localhost:9001 (user: `minioadmin`, pass: `minioadmin`)
- **Traefik Dashboard**: http://localhost:8080
- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3000 (user: `admin`, pass: `admin`)

## Test It

```bash
# Health check
curl http://localhost/health

# Transcribe a file
curl -X POST http://localhost/v1/audio/transcriptions \
  -H "Authorization: Bearer test-api-key" \
  -F file=@test.wav \
  -F model=whisper-1

# Check MinIO for stored files
open http://localhost:9001
```

## Configuration

Edit `docker-compose.local.yml` or set environment variables:

```bash
# Custom API key
API_KEY=my-secret-key docker compose -f docker-compose.local.yml up -d
```

## Multi-GPU

To use multiple GPUs, scale the WhisperLive service:

```bash
docker compose -f docker-compose.local.yml up -d --scale whisperlive=4
```

Traefik will automatically load-balance across instances with sticky sessions for WebSocket connections.

## Cleanup

```bash
docker compose -f docker-compose.local.yml down -v  # -v removes volumes
```
