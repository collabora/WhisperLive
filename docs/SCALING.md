# Horizontal Scaling & Deployment Guide

This document covers strategies for deploying WhisperLive across multiple GPU nodes behind a load balancer.

## Architecture Overview

```
                    ┌──────────────┐
                    │   Clients    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Load Balancer│
                    │ (nginx/HAProxy)
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──┐  ┌──────▼──┐  ┌──────▼──┐
       │ Node 1  │  │ Node 2  │  │ Node N  │
       │ GPU + WS│  │ GPU + WS│  │ GPU + WS│
       │ :9090   │  │ :9090   │  │ :9090   │
       └─────────┘  └─────────┘  └─────────┘
```

Each node runs an independent WhisperLive server with its own GPU.

## Load Balancer Configuration

### WebSocket Sticky Sessions

WebSocket connections are stateful — a client must stay connected to the same backend node for the duration of the session. Use **sticky sessions** (session affinity) in your load balancer.

#### nginx

```nginx
upstream whisperlive {
    ip_hash;  # sticky sessions based on client IP
    server node1:9090;
    server node2:9090;
    server node3:9090;
}

server {
    listen 443 ssl;
    server_name transcribe.example.com;

    ssl_certificate     /etc/ssl/cert.pem;
    ssl_certificate_key /etc/ssl/key.pem;

    location / {
        proxy_pass http://whisperlive;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
```

#### HAProxy

```haml
frontend ws_front
    bind *:9090
    default_backend ws_back

backend ws_back
    balance source          # sticky sessions by source IP
    timeout server 3600s
    timeout tunnel 3600s
    server node1 10.0.0.1:9090 check
    server node2 10.0.0.2:9090 check
    server node3 10.0.0.3:9090 check
```

### REST API (Stateless)

The REST `/v1/audio/transcriptions` endpoint is stateless and can use round-robin:

```nginx
upstream whisperlive_rest {
    least_conn;
    server node1:8080;
    server node2:8080;
    server node3:8080;
}

location /v1/ {
    proxy_pass http://whisperlive_rest;
}
```

## Docker Compose (Multi-Node)

```yaml
version: "3.8"

services:
  whisperlive-1:
    build:
      context: .
      dockerfile: docker/Dockerfile.gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
    ports:
      - "9091:9090"
    command: >
      python run_server.py
        --port 9090
        --backend faster_whisper
        --faster_whisper_custom_model_path large-v3
        --metrics_port 9100

  whisperlive-2:
    build:
      context: .
      dockerfile: docker/Dockerfile.gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
    ports:
      - "9092:9090"
    command: >
      python run_server.py
        --port 9090
        --backend faster_whisper
        --faster_whisper_custom_model_path large-v3
        --metrics_port 9100

  nginx:
    image: nginx:alpine
    ports:
      - "9090:9090"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - whisperlive-1
      - whisperlive-2
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: whisperlive
spec:
  replicas: 3
  selector:
    matchLabels:
      app: whisperlive
  template:
    metadata:
      labels:
        app: whisperlive
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9100"
    spec:
      containers:
        - name: whisperlive
          image: your-registry/whisperlive:latest
          ports:
            - containerPort: 9090
              name: websocket
            - containerPort: 8080
              name: rest
            - containerPort: 9100
              name: metrics
          args:
            - "python"
            - "run_server.py"
            - "--port"
            - "9090"
            - "--backend"
            - "faster_whisper"
            - "--metrics_port"
            - "9100"
          resources:
            limits:
              nvidia.com/gpu: "1"
            requests:
              nvidia.com/gpu: "1"
              memory: "8Gi"
              cpu: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: whisperlive
spec:
  type: ClusterIP
  sessionAffinity: ClientIP  # sticky sessions for WebSocket
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600
  ports:
    - name: websocket
      port: 9090
      targetPort: 9090
    - name: rest
      port: 8080
      targetPort: 8080
  selector:
    app: whisperlive
```

## Prometheus Monitoring

With `--metrics_port 9100`, each node exposes metrics at `http://<node>:9100/metrics`.

### Prometheus scrape config

```yaml
scrape_configs:
  - job_name: whisperlive
    static_configs:
      - targets:
          - node1:9100
          - node2:9100
          - node3:9100
```

### Key metrics to monitor

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `whisperlive_connections_active` | Current WebSocket sessions per node | > 80% of `--max_clients` |
| `whisperlive_transcription_latency_seconds` | Per-chunk transcription time | p99 > 2s |
| `whisperlive_audio_processed_seconds_total` | Total audio throughput | Rate drop > 50% |
| `whisperlive_errors_total` | Error count by type | Rate > 5/min |
| `whisperlive_connections_rejected_total` | Rejected connections | Rate > 0 |

### Grafana Dashboard

Import the Prometheus metrics above into Grafana with panels for:
- Active connections per node (gauge)
- Transcription latency heatmap (histogram)
- Audio throughput rate (counter rate)
- Error rate by type (counter rate)
- Connection rejection rate by reason (counter rate)

## Capacity Planning

### Per-Node Estimates (faster-whisper, large-v3)

| GPU | Concurrent Streams | Real-Time Factor | VRAM Used |
|-----|--------------------|-------------------|-----------|
| RTX 3090 (24GB) | 5–8 | ~0.1x | ~6GB |
| A100 (40GB) | 15–25 | ~0.05x | ~8GB |
| A100 (80GB) | 30–50 | ~0.05x | ~8GB |
| H100 (80GB) | 40–60 | ~0.03x | ~8GB |

*Estimates vary based on audio complexity, VAD filtering, and model size.*

### Scaling Formula

```
nodes_needed = ceil(peak_concurrent_streams / streams_per_node)
```

Add 20–30% headroom for burst capacity.

## Health Checks

Use the REST API for health probing:

```bash
# Simple liveness check
curl http://node1:8080/v1/audio/transcriptions \
  -F file=@/dev/null \
  -F model=whisper-1 \
  --max-time 5

# Or add a dedicated /health endpoint (recommended)
```

## Best Practices

1. **Use `--single_model`** — Shares one model instance across all connections per node. Reduces VRAM usage significantly.

2. **Enable batch inference** — Use `--batch_inference` with `--batch_max_size 8` to batch requests and improve GPU utilization.

3. **Set `--max_clients`** — Limit concurrent connections per node to prevent OOM. Match to your GPU capacity.

4. **Monitor metrics** — Use Prometheus + Grafana to track saturation and latency.

5. **Use WSS (TLS)** — Terminate TLS at the load balancer, not at each node.

6. **API key auth** — Use `--api_key` to protect both REST and WebSocket endpoints.

7. **Rate limiting** — Use `--rate_limit_rpm` to prevent REST API abuse.
