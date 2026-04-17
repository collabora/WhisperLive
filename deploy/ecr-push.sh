#!/usr/bin/env bash
# Build and push WhisperLive GPU image to Amazon ECR.
#
# Usage:
#   ./deploy/ecr-push.sh [REGION] [ACCOUNT_ID] [REPO_NAME]
#
# Defaults can be overridden via environment variables or arguments.

set -euo pipefail

REGION="${1:-${AWS_DEFAULT_REGION:-us-east-1}}"
ACCOUNT_ID="${2:-$(aws sts get-caller-identity --query Account --output text)}"
REPO_NAME="${3:-whisperlive-gpu}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}"

echo "==> Logging in to ECR: ${ECR_URI}"
aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# Create repository if it doesn't exist
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" 2>/dev/null || \
  aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}" \
    --image-scanning-configuration scanOnPush=true

echo "==> Building Docker image"
docker build -f docker/Dockerfile.gpu -t "${REPO_NAME}:${IMAGE_TAG}" .

echo "==> Tagging and pushing to ECR"
docker tag "${REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:${IMAGE_TAG}"

echo "==> Done: ${ECR_URI}:${IMAGE_TAG}"
