#!/usr/bin/env bash
# Deploy WhisperLive to ECS.
#
# Prerequisites:
#   - ECS cluster already created
#   - ALB and target groups configured
#   - Task definition registered (or will be registered by this script)
#
# Usage:
#   ./deploy/ecs-deploy.sh [CLUSTER] [SERVICE] [REGION]

set -euo pipefail

CLUSTER="${1:-whisperlive}"
SERVICE="${2:-whisperlive-gpu}"
REGION="${3:-${AWS_DEFAULT_REGION:-us-east-1}}"
TASK_DEF_FILE="deploy/ecs-task-definition.json"

echo "==> Registering task definition"
TASK_DEF_ARN=$(aws ecs register-task-definition \
  --cli-input-json "file://${TASK_DEF_FILE}" \
  --region "${REGION}" \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)

echo "    Task definition: ${TASK_DEF_ARN}"

# Check if the service exists
if aws ecs describe-services --cluster "${CLUSTER}" --services "${SERVICE}" --region "${REGION}" \
   --query 'services[0].status' --output text 2>/dev/null | grep -q ACTIVE; then
  echo "==> Updating existing service: ${SERVICE}"
  aws ecs update-service \
    --cluster "${CLUSTER}" \
    --service "${SERVICE}" \
    --task-definition "${TASK_DEF_ARN}" \
    --region "${REGION}" \
    --force-new-deployment \
    --no-cli-pager
else
  echo "==> Creating new service: ${SERVICE}"
  aws ecs create-service \
    --cluster "${CLUSTER}" \
    --service-name "${SERVICE}" \
    --task-definition "${TASK_DEF_ARN}" \
    --desired-count 2 \
    --launch-type EC2 \
    --region "${REGION}" \
    --deployment-configuration "maximumPercent=200,minimumHealthyPercent=100" \
    --no-cli-pager
fi

echo "==> Waiting for service stability..."
aws ecs wait services-stable \
  --cluster "${CLUSTER}" \
  --services "${SERVICE}" \
  --region "${REGION}"

echo "==> Deployment complete!"
