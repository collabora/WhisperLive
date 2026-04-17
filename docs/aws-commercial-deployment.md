# WhisperLive: Commercial AWS Deployment Guide

## Overview

This document outlines what is needed to deploy WhisperLive as a commercial transcription service hosted on AWS. It covers infrastructure, security, reliability, observability, business logic, compliance, and cost optimization.

## What's Already Built

The following features have been implemented across our feature branches:

| Feature | Status |
|---------|--------|
| REST API (OpenAI-compatible) + WebSocket server | ✅ |
| API key authentication & rate limiting | ✅ |
| Prometheus metrics instrumentation | ✅ |
| SSE streaming for real-time transcription | ✅ |
| Web transcription UI (drag-and-drop) | ✅ |
| OpenAPI/Swagger auto-docs | ✅ |
| GPU Docker image | ✅ |
| Horizontal scaling documentation | ✅ |
| Model hot-swap with LRU cache | ✅ |
| Audio noise reduction | ✅ |
| PII redaction | ✅ |
| Speaker diarization | ✅ |
| Multi-channel audio support | ✅ |
| Webhook callbacks for async jobs | ✅ |
| Plugin architecture for post-processing | ✅ |

---

## 1. Infrastructure (Critical)

### Load Balancer
- Deploy an **Application Load Balancer (ALB)** in front of GPU instances.
- Enable **WebSocket sticky sessions** (required for persistent WebSocket connections).
- Configure health checks against the `/health` endpoint.

### Compute
- Use **ECS on EC2** or **EKS** with GPU-enabled instances:
  - `g4dn.xlarge` (1× T4 GPU, 16 GB VRAM) — suitable for `small` and `medium` models.
  - `g5.xlarge` (1× A10G GPU, 24 GB VRAM) — suitable for `large-v3` model.
  - `g5.2xlarge` for concurrent model loading or large batch jobs.
- Configure **auto-scaling** based on GPU utilization (target: 70%) or SQS queue depth.
- Use **launch templates** with the NVIDIA GPU AMI or install drivers via user data.

### Storage
- **S3** for uploaded audio files. Currently files are stored in `/tmp` and deleted after processing — for a commercial service, persist originals and results in S3.
- **EFS** (optional) for shared model weights across instances, avoiding redundant downloads.

### Database
- **DynamoDB** or **PostgreSQL (RDS)** for:
  - Job state and results for async transcription jobs.
  - User accounts, API keys, and organization management.
  - Usage tracking (audio minutes processed per customer).

### Queue
- **SQS** or **Redis (ElastiCache)** for async transcription jobs (webhook-based flow).
- Dead-letter queue for failed jobs with retry logic.

### Networking
- **TLS termination** at the ALB using ACM certificates.
- **CloudFront** CDN for the web UI static files and API caching (for docs endpoints).
- **Route 53** for DNS management.

### Architecture Diagram

```
                    ┌──────────────┐
                    │  CloudFront  │
                    │  (Web UI +   │
                    │   API CDN)   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │     ALB      │
                    │ (TLS + WSS)  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼────┐ ┌────▼─────┐ ┌────▼─────┐
        │  ECS/EKS │ │  ECS/EKS │ │  ECS/EKS │
        │  GPU #1  │ │  GPU #2  │ │  GPU #N  │
        └─────┬────┘ └────┬─────┘ └────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
       ┌────▼────┐  ┌─────▼─────┐  ┌─────▼─────┐
       │   S3    │  │    SQS    │  │ DynamoDB / │
       │ (audio) │  │ (async    │  │  Postgres  │
       └─────────┘  │  jobs)    │  │  (state)   │
                    └───────────┘  └────────────┘
```

---

## 2. Security (Critical)

### Transport Security
- **HTTPS/WSS only** — redirect all HTTP to HTTPS at the ALB level.
- Use TLS 1.2+ with strong cipher suites.
- HSTS headers on all responses.

### Authentication & Authorization
- **AWS Cognito** or **Auth0** for user signup, login, and API key management.
- Per-user API keys with configurable rate limits and usage quotas.
- Support for organization-level API keys with multiple team members.

### Input Validation
- **File size limits**: Enforce maximum upload size (e.g., 500 MB) at the ALB and application level.
- **Audio format validation**: Verify file headers match declared content type.
- **Rate limiting**: Per-user, per-IP, and global rate limits (already implemented via our rate-limiting-auth branch).

### Network Security
- **WAF** (Web Application Firewall) + **AWS Shield** for DDoS protection.
- **VPC** with private subnets for GPU workers; only ALB in public subnets.
- Security groups restricting GPU instance access to ALB only.

### Secrets Management
- **AWS Secrets Manager** or **SSM Parameter Store** for API keys, database credentials, and configuration.
- Never store secrets in environment variables, Docker images, or code.

### Audit Logging
- **CloudTrail** for AWS API activity.
- Application-level audit logs for all API key usage, transcription requests, and admin actions.

---

## 3. Reliability (Critical)

### Health Checks
- The `/health` endpoint (already implemented) should be wired to the ALB target group health check.
- Include GPU availability and model readiness in the health response.

### Graceful Shutdown
- Handle `SIGTERM` signal: stop accepting new connections, drain in-flight WebSocket sessions and REST requests, then exit.
- ECS/EKS deregistration delay should match the drain timeout.

### Error Handling
- **Retry logic** for transient GPU OOM errors (reload model with smaller batch size).
- **Circuit breaker** pattern for downstream dependencies (S3, DynamoDB).
- **Dead-letter queue** for permanently failed async jobs with alerting.

### Data Durability
- Persist transcription results to S3/database, not just return in the HTTP response.
- Enable S3 versioning and cross-region replication for critical data.

### Multi-AZ Deployment
- Spread GPU instances across at least 2 Availability Zones.
- RDS Multi-AZ for database high availability.

---

## 4. Observability (Important)

### Logging
- **CloudWatch Logs** or **ELK stack** (Elasticsearch, Logstash, Kibana) for centralized logging.
- Structured JSON logs with request IDs, user IDs, and timing information.

### Metrics
- Prometheus metrics (already instrumented) scraped by a Prometheus server or **CloudWatch Agent**.
- Key metrics to monitor:
  - Request rate, error rate, and latency (p50, p95, p99).
  - GPU utilization, memory usage, and temperature.
  - Active WebSocket connections.
  - Transcription queue depth and processing time.
  - Model load time and cache hit rate.

### Tracing
- **AWS X-Ray** or **OpenTelemetry** for distributed tracing across the request lifecycle.
- Trace from ALB → API handler → model inference → response.

### Alerting
- **CloudWatch Alarms** or **PagerDuty** integration:
  - Error rate > 1% → warning.
  - Error rate > 5% → critical.
  - p99 latency > 30s → warning.
  - GPU utilization > 90% sustained for 5 min → auto-scale.
  - Queue depth > 100 → scale up workers.

### Dashboard
- **Grafana** dashboard consuming Prometheus metrics:
  - Real-time request volume, error rates, latency distributions.
  - GPU utilization heatmap across instances.
  - Per-customer usage breakdown.

---

## 5. Business Layer (Important)

### Billing & Metering
- Track **audio minutes processed** per customer (the standard billing unit for transcription services).
- Record start time, duration, model used, and features enabled for each request.
- Integrate with **Stripe** or **AWS Marketplace** for billing.

### Usage Tiers

| Tier | Rate Limit | Models | Features | Price |
|------|-----------|--------|----------|-------|
| Free | 60 min/month, 5 req/min | small | Basic transcription | $0 |
| Pro | 1,000 min/month, 50 req/min | small, medium, large-v3 | All features, SSE streaming, webhooks | $0.006/min |
| Enterprise | Unlimited, custom limits | All + custom fine-tuned | All features + SLA, dedicated capacity | Custom |

### Multi-Tenancy
- Isolate customer data: separate S3 prefixes per organization.
- API keys scoped to organizations with role-based access (admin, member, read-only).
- Per-tenant rate limits and usage quotas.

### API Versioning
- Version the REST API (e.g., `/v1/audio/transcriptions`) — already implemented.
- Maintain backward compatibility; deprecate with 6-month notice.

---

## 6. Compliance

### GDPR (If serving EU customers)
- **Data retention policies**: Auto-delete audio files after configurable period (default: 30 days).
- **Right to deletion**: API endpoint to delete all data for a user/organization.
- **Data processing agreements**: Document what data is processed and where.
- **Data residency**: Option to deploy in `eu-west-1` or `eu-central-1` for EU data.

### SOC 2 (If serving enterprise customers)
- **Audit logging**: All access and changes logged and retained for 1 year.
- **Access controls**: Least-privilege IAM roles, no shared credentials.
- **Encryption**: At rest (S3 SSE-KMS, RDS encryption) and in transit (TLS).
- **Incident response**: Documented runbook for security incidents.

### HIPAA (If serving healthcare customers)
- **BAA with AWS**: Required before processing PHI.
- **Encryption at rest**: KMS-managed keys for all storage.
- **Access logging**: CloudTrail + application audit logs.
- **Dedicated tenancy**: EC2 dedicated instances if required.

### Licensing
- **WhisperLive**: MIT License ✅ — fully permissive for commercial use.
- **faster-whisper**: MIT License ✅
- **openai-whisper**: MIT License ✅
- **PyTorch**: BSD License ✅
- **ONNX Runtime**: MIT License ✅
- No copyleft (GPL) dependencies in the critical path. All clear for commercial deployment.

---

## 7. Cost Optimization

### Spot Instances
- Use **EC2 Spot Instances** for async/batch transcription jobs — up to 70% savings on GPU instances.
- On-Demand or Reserved for real-time WebSocket connections (Spot interruptions would break sessions).

### Model Caching
- The **model hot-swap with LRU cache** (already implemented) avoids redundant model loading.
- Pre-load the most common model (e.g., `small`) on instance startup.
- Cache models on **EFS** shared across instances to avoid repeated downloads.

### Right-Sizing

| Model | Min GPU VRAM | Recommended Instance | On-Demand Cost |
|-------|-------------|---------------------|---------------|
| tiny | 1 GB | g4dn.xlarge | ~$0.526/hr |
| small | 2 GB | g4dn.xlarge | ~$0.526/hr |
| medium | 5 GB | g4dn.xlarge | ~$0.526/hr |
| large-v3 | 10 GB | g5.xlarge | ~$1.006/hr |
| large-v3 (batch) | 20+ GB | g5.2xlarge | ~$1.212/hr |

### Reserved Capacity
- **Savings Plans** or **Reserved Instances** for baseline load (1-year commitment: ~40% savings).
- Spot for burst capacity above baseline.

### Estimated Monthly Cost (Moderate Load)

| Component | Count | Cost/month |
|-----------|-------|-----------|
| g5.xlarge (On-Demand, 2 instances, 24/7) | 2 | ~$1,450 |
| g4dn.xlarge (Spot, async workers, ~12hr/day) | 2 | ~$230 |
| ALB | 1 | ~$25 |
| CloudFront | - | ~$10 |
| S3 (1 TB audio storage) | - | ~$23 |
| DynamoDB (on-demand) | - | ~$25 |
| SQS | - | ~$5 |
| CloudWatch | - | ~$30 |
| **Total** | | **~$1,800/month** |

---

## Quick-Start Checklist

- [ ] Set up VPC with public/private subnets across 2+ AZs
- [ ] Build and push GPU Docker image to ECR
- [ ] Deploy ECS/EKS cluster with GPU task definitions
- [ ] Configure ALB with TLS, WebSocket support, and health checks
- [ ] Set up S3 bucket for audio storage with lifecycle policies
- [ ] Deploy DynamoDB or RDS for state management
- [ ] Configure SQS for async job queue
- [ ] Set up Cognito/Auth0 for user management
- [ ] Wire Prometheus metrics to CloudWatch or Grafana
- [ ] Configure CloudWatch Alarms for error rates and latency
- [ ] Set up WAF rules and Shield protection
- [ ] Implement billing metering and Stripe integration
- [ ] Configure CloudFront for web UI and API caching
- [ ] Set up CI/CD pipeline (GitHub Actions → ECR → ECS deploy)
- [ ] Create runbooks for common operational tasks
- [ ] Load test with realistic traffic patterns
- [ ] Security audit and penetration testing
