# Local Staging with Docker Compose

Run the full WhisperLive production stack on your local hardware — no cloud costs.

Uses a custom MinIO image built from source on Alpine with self-signed TLS certs
and virtual-host-style bucket access (adapted from
[GrokImageCompression/grok](https://github.com/GrokImageCompression/grok) network testing).

## Stack

| Service | Port | Purpose |
|---------|------|---------|
| WhisperLive | (behind Traefik) | GPU transcription server |
| Traefik | 80 (HTTP), 8080 (dashboard) | Reverse proxy / load balancer |
| MinIO | 9000 (S3 API, TLS), 9001 (console) | S3-compatible object storage (built from source) |
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

## MinIO Details

The local stack includes a custom MinIO image (`deploy/minio/Dockerfile`) built from source
on Alpine Linux with:

- **Self-signed TLS certificates** — generated at build time for HTTPS on port 9000
- **Virtual-host-style access** — `entrypoint.sh` configures `/etc/hosts` for S3 path-style
  and virtual-host-style bucket resolution
- **mc CLI included** — MinIO client baked into the image for bucket management

### Building MinIO Standalone

```bash
cd deploy/minio
docker build -t minio-alpine .
docker volume create minio-data
docker run -d --rm --cap-add=NET_ADMIN -v minio-data:/data \
  --name minio-container -p 9000:9000 -p 9001:9001 minio-alpine
```

### MinIO Client (mc)

```bash
# Install mc
curl -O https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc && sudo mv mc /usr/local/bin/

# Set alias
mc alias set local https://localhost:9000 minioadmin minioadmin --insecure

# Create bucket
mc mb local/whisperlive --insecure

# List contents
mc ls local/whisperlive --insecure
```

### AWS CLI with MinIO

```bash
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin"
export AWS_ENDPOINT_URL="https://localhost:9000"

# Use --no-verify-ssl for self-signed certs
aws s3 ls s3://whisperlive/ --no-verify-ssl
```

## Troubleshooting

### "docker: Error response from daemon: could not select device driver"
NVIDIA Container Toolkit isn't installed. Follow the [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### MinIO "connection refused" or TLS errors
The MinIO container needs ~30s to build from source on first run. Check status:
```bash
docker compose -f docker-compose.local.yml logs minio
docker compose -f docker-compose.local.yml logs minio-init
```

### WhisperLive can't connect to MinIO
The `minio-init` service must complete before WhisperLive starts (enforced by `depends_on`). If it fails:
```bash
docker compose -f docker-compose.local.yml restart minio-init
```

### Out of GPU memory
Reduce `--max_clients` in `docker-compose.local.yml` or use a smaller model (default is `small`).

### Port conflicts
If ports 80, 8080, 9000, 9001, 9091, or 3000 are in use, edit `docker-compose.local.yml` to change the host-side port mappings.
