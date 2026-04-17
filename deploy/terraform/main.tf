terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "whisperlive"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "gpu_instance_type" {
  description = "EC2 instance type for GPU workers"
  type        = string
  default     = "g4dn.xlarge"
}

variable "desired_count" {
  description = "Desired number of ECS tasks"
  type        = number
  default     = 2
}

variable "max_clients_per_instance" {
  description = "Max concurrent WebSocket clients per instance"
  type        = number
  default     = 4
}

variable "max_connection_time" {
  description = "Max WebSocket connection duration (seconds)"
  type        = number
  default     = 600
}

variable "data_retention_days" {
  description = "Auto-delete stored data after N days (0=disabled)"
  type        = number
  default     = 30
}

variable "domain_name" {
  description = "Custom domain name (leave empty to skip DNS/TLS)"
  type        = string
  default     = ""
}

variable "certificate_arn" {
  description = "ACM certificate ARN for HTTPS (leave empty for HTTP-only)"
  type        = string
  default     = ""
}

variable "api_key" {
  description = "API key for authentication"
  type        = string
  sensitive   = true
}

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ssm_parameter" "ecs_gpu_ami" {
  name = "/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id"
}

# ---------------------------------------------------------------------------
# Networking: VPC, Subnets, IGW, NAT, Routes
# ---------------------------------------------------------------------------
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = { Name = "${var.project_name}-vpc" }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.project_name}-igw" }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags                    = { Name = "${var.project_name}-public-${count.index}" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags              = { Name = "${var.project_name}-private-${count.index}" }
}

resource "aws_eip" "nat" {
  domain = "vpc"
  tags   = { Name = "${var.project_name}-nat-eip" }
}

resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
  tags          = { Name = "${var.project_name}-nat" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
  tags = { Name = "${var.project_name}-public-rt" }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat.id
  }
  tags = { Name = "${var.project_name}-private-rt" }
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# ---------------------------------------------------------------------------
# Security Groups
# ---------------------------------------------------------------------------
resource "aws_security_group" "alb" {
  name_prefix = "${var.project_name}-alb-"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-alb-sg" }
}

resource "aws_security_group" "ecs" {
  name_prefix = "${var.project_name}-ecs-"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "REST API from ALB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    description     = "WebSocket from ALB"
    from_port       = 9090
    to_port         = 9090
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-ecs-sg" }
}

# ---------------------------------------------------------------------------
# S3 Bucket for audio storage
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "data" {
  bucket = "${var.project_name}-data-${data.aws_caller_identity.current.account_id}"
  tags   = { Name = "${var.project_name}-data" }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "auto-expire"
    status = var.data_retention_days > 0 ? "Enabled" : "Disabled"

    expiration {
      days = var.data_retention_days > 0 ? var.data_retention_days : 365
    }

    filter {
      prefix = "whisperlive/"
    }
  }
}

data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------------------
# Secrets Manager — API Key
# ---------------------------------------------------------------------------
resource "aws_secretsmanager_secret" "api_key" {
  name = "${var.project_name}/api-key"
}

resource "aws_secretsmanager_secret_version" "api_key" {
  secret_id     = aws_secretsmanager_secret.api_key.id
  secret_string = var.api_key
}

# ---------------------------------------------------------------------------
# ECR Repository
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "app" {
  name                 = "${var.project_name}-gpu"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = { Name = "${var.project_name}-gpu" }
}

# ---------------------------------------------------------------------------
# IAM Roles
# ---------------------------------------------------------------------------
resource "aws_iam_role" "ecs_execution" {
  name = "${var.project_name}-ecs-execution"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_execution_secrets" {
  name = "secrets-access"
  role = aws_iam_role.ecs_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [aws_secretsmanager_secret.api_key.arn]
    }]
  })
}

resource "aws_iam_role" "ecs_task" {
  name = "${var.project_name}-ecs-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "ecs_task_s3" {
  name = "s3-access"
  role = aws_iam_role.ecs_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ]
      Resource = [
        aws_s3_bucket.data.arn,
        "${aws_s3_bucket.data.arn}/*"
      ]
    }]
  })
}

# ---------------------------------------------------------------------------
# CloudWatch Log Group
# ---------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.project_name}"
  retention_in_days = 90
  tags              = { Name = "${var.project_name}-logs" }
}

# ---------------------------------------------------------------------------
# ECS Cluster
# ---------------------------------------------------------------------------
resource "aws_ecs_cluster" "main" {
  name = var.project_name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = { Name = var.project_name }
}

# ---------------------------------------------------------------------------
# ECS EC2 Launch Template (GPU)
# ---------------------------------------------------------------------------
resource "aws_launch_template" "gpu" {
  name_prefix   = "${var.project_name}-gpu-"
  image_id      = data.aws_ssm_parameter.ecs_gpu_ami.value
  instance_type = var.gpu_instance_type

  iam_instance_profile {
    arn = aws_iam_instance_profile.ecs_instance.arn
  }

  network_interfaces {
    security_groups             = [aws_security_group.ecs.id]
    associate_public_ip_address = false
  }

  user_data = base64encode(<<-EOF
    #!/bin/bash
    echo "ECS_CLUSTER=${aws_ecs_cluster.main.name}" >> /etc/ecs/ecs.config
    echo "ECS_ENABLE_GPU_SUPPORT=true" >> /etc/ecs/ecs.config
  EOF
  )

  tag_specifications {
    resource_type = "instance"
    tags          = { Name = "${var.project_name}-gpu-worker" }
  }
}

resource "aws_iam_role" "ecs_instance" {
  name = "${var.project_name}-ecs-instance"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_instance" {
  role       = aws_iam_role.ecs_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "ecs_instance" {
  name = "${var.project_name}-ecs-instance"
  role = aws_iam_role.ecs_instance.name
}

# ---------------------------------------------------------------------------
# Auto Scaling Group for GPU instances
# ---------------------------------------------------------------------------
resource "aws_autoscaling_group" "gpu" {
  name                = "${var.project_name}-gpu-asg"
  min_size            = var.desired_count
  max_size            = var.desired_count * 3
  desired_capacity    = var.desired_count
  vpc_zone_identifier = aws_subnet.private[*].id

  launch_template {
    id      = aws_launch_template.gpu.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "${var.project_name}-gpu-worker"
    propagate_at_launch = true
  }

  tag {
    key                 = "AmazonECSManaged"
    value               = ""
    propagate_at_launch = true
  }
}

# ---------------------------------------------------------------------------
# ECS Capacity Provider
# ---------------------------------------------------------------------------
resource "aws_ecs_capacity_provider" "gpu" {
  name = "${var.project_name}-gpu"

  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.gpu.arn
    managed_termination_protection = "ENABLED"

    managed_scaling {
      status                    = "ENABLED"
      target_capacity           = 70
      minimum_scaling_step_size = 1
      maximum_scaling_step_size = 3
    }
  }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = [aws_ecs_capacity_provider.gpu.name]

  default_capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.gpu.name
    weight            = 1
    base              = var.desired_count
  }
}

# ---------------------------------------------------------------------------
# Application Load Balancer
# ---------------------------------------------------------------------------
resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  tags = { Name = "${var.project_name}-alb" }
}

resource "aws_lb_target_group" "rest" {
  name        = "${var.project_name}-rest"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path                = "/health"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
    matcher             = "200"
  }

  stickiness {
    type    = "lb_cookie"
    enabled = false
  }

  tags = { Name = "${var.project_name}-rest-tg" }
}

resource "aws_lb_target_group" "websocket" {
  name        = "${var.project_name}-ws"
  port        = 9090
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path                = "/health"
    port                = "8000"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
    matcher             = "200"
  }

  stickiness {
    type            = "lb_cookie"
    enabled         = true
    cookie_duration = 86400
  }

  tags = { Name = "${var.project_name}-ws-tg" }
}

# HTTP listener — redirect to HTTPS if cert provided, else serve directly
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = var.certificate_arn != "" ? "redirect" : "forward"

    dynamic "redirect" {
      for_each = var.certificate_arn != "" ? [1] : []
      content {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }

    dynamic "forward" {
      for_each = var.certificate_arn == "" ? [1] : []
      content {
        target_group {
          arn = aws_lb_target_group.rest.arn
        }
      }
    }
  }
}

# HTTPS listener (only if certificate ARN is provided)
resource "aws_lb_listener" "https" {
  count             = var.certificate_arn != "" ? 1 : 0
  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.rest.arn
  }
}

# WebSocket routing rule (path /ws/*)
resource "aws_lb_listener_rule" "websocket" {
  listener_arn = var.certificate_arn != "" ? aws_lb_listener.https[0].arn : aws_lb_listener.http.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.websocket.arn
  }

  condition {
    path_pattern { values = ["/ws/*", "/ws"] }
  }
}

# ---------------------------------------------------------------------------
# ECS Task Definition
# ---------------------------------------------------------------------------
resource "aws_ecs_task_definition" "app" {
  family                   = var.project_name
  requires_compatibilities = ["EC2"]
  network_mode             = "awsvpc"
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name      = var.project_name
    image     = "${aws_ecr_repository.app.repository_url}:latest"
    essential = true

    portMappings = [
      { containerPort = 8000, protocol = "tcp" },
      { containerPort = 9090, protocol = "tcp" }
    ]

    command = [
      "python", "run_server.py",
      "--enable_rest",
      "--backend", "faster_whisper",
      "--max_clients", tostring(var.max_clients_per_instance),
      "--max_connection_time", tostring(var.max_connection_time),
      "--rest_port", "8000",
      "--json_logs",
      "--storage_backend", "s3",
      "--storage_bucket", aws_s3_bucket.data.id,
      "--data_retention_days", tostring(var.data_retention_days),
    ]

    environment = [
      { name = "WHISPER_MAX_UPLOAD_BYTES", value = "524288000" }
    ]

    secrets = [
      {
        name      = "WHISPERLIVE_API_KEY"
        valueFrom = aws_secretsmanager_secret.api_key.arn
      }
    ]

    healthCheck = {
      command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\" || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 15
    }

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "ecs"
      }
    }

    resourceRequirements = [{ type = "GPU", value = "1" }]
    stopTimeout          = 30
    memory               = 8192
    cpu                  = 2048
  }])

  tags = { Name = var.project_name }
}

# ---------------------------------------------------------------------------
# ECS Service
# ---------------------------------------------------------------------------
resource "aws_ecs_service" "app" {
  name            = var.project_name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count

  capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.gpu.name
    weight            = 1
    base              = var.desired_count
  }

  network_configuration {
    subnets         = aws_subnet.private[*].id
    security_groups = [aws_security_group.ecs.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.rest.arn
    container_name   = var.project_name
    container_port   = 8000
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.websocket.arn
    container_name   = var.project_name
    container_port   = 9090
  }

  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 100
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  depends_on = [aws_lb_listener.http]

  tags = { Name = var.project_name }
}

# ---------------------------------------------------------------------------
# CloudWatch Alarms
# ---------------------------------------------------------------------------
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "${var.project_name}-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  period              = 60
  statistic           = "Average"
  threshold           = 80

  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.app.name
  }

  alarm_description = "ECS CPU utilization > 80% for 3 minutes"
  tags              = { Name = "${var.project_name}-high-cpu" }
}

resource "aws_cloudwatch_metric_alarm" "alb_5xx" {
  alarm_name          = "${var.project_name}-5xx-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "HTTPCode_Target_5XX_Count"
  namespace           = "AWS/ApplicationELB"
  period              = 300
  statistic           = "Sum"
  threshold           = 10

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
  }

  alarm_description = "ALB 5xx errors > 10 in 5 minutes"
  tags              = { Name = "${var.project_name}-5xx" }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
output "alb_dns_name" {
  description = "ALB DNS name (use this to access the service)"
  value       = aws_lb.main.dns_name
}

output "ecr_repository_url" {
  description = "ECR repository URL for pushing Docker images"
  value       = aws_ecr_repository.app.repository_url
}

output "s3_bucket_name" {
  description = "S3 bucket for audio and result storage"
  value       = aws_s3_bucket.data.id
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for ECS containers"
  value       = aws_cloudwatch_log_group.ecs.name
}
