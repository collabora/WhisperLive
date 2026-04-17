# WhisperLive AWS Deployment (Terraform)

One-command deployment of WhisperLive to AWS with GPU support.

## What Gets Created

| Resource | Purpose |
|----------|---------|
| VPC + Subnets | Networking across 2 AZs (public + private) |
| ALB | TLS termination, WebSocket sticky sessions, health checks |
| ECS Cluster | GPU container orchestration |
| EC2 Auto Scaling (GPU) | `g4dn.xlarge` instances with managed scaling |
| ECR Repository | Docker image registry |
| S3 Bucket | Audio file and result storage (KMS encrypted, versioned) |
| Secrets Manager | API key storage |
| CloudWatch Logs | Centralized logging (90-day retention) |
| CloudWatch Alarms | CPU and 5xx error monitoring |
| IAM Roles | Least-privilege task and execution roles |

## Prerequisites

- [Terraform >= 1.5](https://developer.hashicorp.com/terraform/install)
- [AWS CLI](https://aws.amazon.com/cli/) configured with credentials
- Docker (for building and pushing the image)

## Quick Start

```bash
# 1. Build and push Docker image
cd /path/to/WhisperLive
./deploy/ecr-push.sh us-east-1

# 2. Deploy infrastructure
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your settings

terraform init
terraform plan -var="api_key=your-secret-key"
terraform apply -var="api_key=your-secret-key"

# 3. Get the ALB URL
terraform output alb_dns_name
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `region` | `us-east-1` | AWS region |
| `gpu_instance_type` | `g4dn.xlarge` | GPU instance type |
| `desired_count` | `2` | Number of GPU tasks |
| `max_clients_per_instance` | `4` | WebSocket client limit per instance |
| `data_retention_days` | `30` | Auto-delete data after N days |
| `api_key` | (required) | API key for authentication |
| `certificate_arn` | (optional) | ACM cert for HTTPS |
| `domain_name` | (optional) | Custom domain |

## Architecture

```
Internet → ALB (TLS) → ECS Tasks (GPU) → S3 (storage)
                              ↓
                        Secrets Manager
                        CloudWatch Logs
```

## Cost Estimate

| Component | Monthly Cost |
|-----------|-------------|
| 2× g4dn.xlarge (On-Demand) | ~$760 |
| ALB | ~$25 |
| S3 (1 TB) | ~$23 |
| NAT Gateway | ~$35 |
| CloudWatch | ~$30 |
| **Total** | **~$875/mo** |

Use Spot Instances or Savings Plans to reduce GPU costs by 40-70%.

## Cleanup

```bash
terraform destroy -var="api_key=unused"
```
