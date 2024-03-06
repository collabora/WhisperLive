import base64

import pulumi
import pulumi_aws as aws

config = pulumi.Config()

vpc_id = config.require("vpc_id")

ecr_repository_name = "collabora/whisperlive-gpu"

public_subnet_id_a = config.require("public_subnet_id_a")
public_subnet_id_b = config.require("public_subnet_id_b")

# Assuming you have the ECR repository URL
ecr_repository_url = f"ghcr.io/{ecr_repository_name}"
image_tag = "latest"
container_name = "whisper-live-gpu-container"

# Define the ECR repository
ecr_repo = aws.ecr.Repository(ecr_repository_name)

# Define the ECS cluster
ecs_cluster = aws.ecs.Cluster("gpu_cluster")

# Define the IAM role for the EC2 instances
instance_role = aws.iam.Role(
    "instance-role",
    assume_role_policy="""{
        "Version": "2012-10-17",
        "Statement": [{
            "Action": "sts:AssumeRole",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Effect": "Allow",
            "Sid": ""
        }]
    }"""
)

# Attach the AmazonEC2ContainerServiceforEC2Role policy to the role
policy_attachment = aws.iam.RolePolicyAttachment(
    "ecs-instance-role-attachment",
    role=instance_role.name,
    policy_arn="arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
)

# Create an instance profile for the EC2 instances
instance_profile = aws.iam.InstanceProfile("instance-profile", role=instance_role.name)

# Create a new security group for allowing traffic on port 9090
security_group = aws.ec2.SecurityGroup(
    'whisper-live-security-group',
    description='Allow inbound traffic on port 9090',
    ingress=[
        {
            'protocol': 'tcp',
            'from_port': 9090,
            'to_port': 9090,
            'cidr_blocks': ['0.0.0.0/0'],
        }
    ],
    egress=[
        {
            'protocol': '-1',  # '-1' indicates all protocols
            'from_port': 0,    # '0' indicates all ports.
            'to_port': 0,
            'cidr_blocks': ['0.0.0.0/0'],
        }
    ]
)

private_subnet_a = aws.ec2.Subnet(
    "private-subnet-a",
    vpc_id=vpc_id,
    cidr_block="172.31.64.0/24",
    availability_zone="us-west-1a",
    map_public_ip_on_launch=False
)

private_subnet_b = aws.ec2.Subnet(
    "private-subnet-b",
    vpc_id=vpc_id,
    cidr_block="172.31.65.0/24",
    availability_zone="us-west-1c",
    map_public_ip_on_launch=False
)

private_route_table_a = aws.ec2.RouteTable(
    "private-route-table-a",
    vpc_id=vpc_id,
    tags={"Name": "Private Route Table AZ1"}
)

private_route_table_b = aws.ec2.RouteTable(
    "private-route-table-b",
    vpc_id=vpc_id,
    tags={"Name": "Private Route Table AZ2"}
)

route_table_association_a = aws.ec2.RouteTableAssociation(
    "route-table-association-a",
    subnet_id=private_subnet_a.id,
    route_table_id=private_route_table_a.id
)

route_table_association_b = aws.ec2.RouteTableAssociation(
    "route-table-association-b",
    subnet_id=private_subnet_b.id,
    route_table_id=private_route_table_b.id
)

# Allocate an Elastic IP for each NAT Gateway
eip_nat_gw_a = aws.ec2.Eip("eip-nat-gw-a", domain='vpc')
eip_nat_gw_b = aws.ec2.Eip("eip-nat-gw-b", domain='vpc')

# Create NAT Gateways in the public subnets and associate with the EIPs
nat_gateway_a = aws.ec2.NatGateway(
    "nat-gateway-a",
    allocation_id=eip_nat_gw_a.id,
    subnet_id=public_subnet_id_a,
)

nat_gateway_b = aws.ec2.NatGateway(
    "nat-gateway-b",
    allocation_id=eip_nat_gw_b.id,
    subnet_id=public_subnet_id_b,
)

# Add a routes to the internet through the NAT Gateways
nat_route_a = aws.ec2.Route(
    "nat-route-a",
    route_table_id=private_route_table_a.id,
    destination_cidr_block="0.0.0.0/0",
    nat_gateway_id=nat_gateway_a.id,
)

nat_route_b = aws.ec2.Route(
    "nat-route-b",
    route_table_id=private_route_table_b.id,
    destination_cidr_block="0.0.0.0/0",
    nat_gateway_id=nat_gateway_b.id,
)

# Define the ALB
alb = aws.lb.LoadBalancer(
    "whisper-live-lb",
    internal=False,
    load_balancer_type="network",
    security_groups=[security_group.id],
    subnets=[public_subnet_id_a, public_subnet_id_b],
    enable_deletion_protection=False,
    enable_cross_zone_load_balancing=True,
)

# Define a Target Group for the ECS service
tg = aws.lb.TargetGroup(
    "whisper-live-tg",
    port=9090,
    protocol="TCP",
    target_type="ip",
    vpc_id=vpc_id,
    health_check=aws.lb.TargetGroupHealthCheckArgs(
        protocol="TCP",
        healthy_threshold=2,
        unhealthy_threshold=2,
        timeout=10,
        interval=30,
        port="9090",
    ),
)

# Define a Listener for the ALB to forward TCP requests to the Target Group
listener = aws.lb.Listener(
    "listener",
    load_balancer_arn=alb.arn,
    port=9090,
    protocol="TCP",
    default_actions=[aws.lb.ListenerDefaultActionArgs(
        type="forward",
        target_group_arn=tg.arn,  # Forward to the Target Group
    )],
)

# Specify the ECS cluster in the user data for the EC2 instances
user_data_encoded = ecs_cluster.name.apply(lambda name:
    base64.b64encode(
        f"#!/bin/bash\necho ECS_CLUSTER={name} >> /etc/ecs/ecs.config".encode("ascii")
    ).decode("ascii")
)

launch_template = aws.ec2.LaunchTemplate(
    "gpu-launch-template",
    image_id="ami-0a7b82bd04a728ae5",
    instance_type="g4dn.xlarge",
    iam_instance_profile=aws.ec2.LaunchTemplateIamInstanceProfileArgs(
        arn=instance_profile.arn
    ),
    key_name="kevin-local",
    user_data=user_data_encoded,
    vpc_security_group_ids=[security_group.id],
    block_device_mappings=[aws.ec2.LaunchTemplateBlockDeviceMappingArgs(
        device_name="/dev/xvda",  # or the device name for your AMI
        ebs=aws.ec2.LaunchTemplateBlockDeviceMappingEbsArgs(
            delete_on_termination="true",
            volume_size=60,  # Specify your desired volume size in GiB
            volume_type="gp3"
        ),
    )],
)

# Define an Auto Scaling Group that uses the defined launch template
auto_scaling_group = aws.autoscaling.Group(
    "ecs-autoscaling-group",
    desired_capacity=1,
    max_size=2,
    min_size=1,
    launch_template=aws.autoscaling.GroupLaunchTemplateArgs(
        id=launch_template.id,
        version="$Latest"
    ),
    vpc_zone_identifiers=[private_subnet_a.id, private_subnet_b.id],
    tags=[{
        'key': 'Name',
        'value': 'ECS Instance - GPU',
        'propagate_at_launch': True,
    }]
)

ecs_task_execution_role = aws.iam.Role(
    "ecs-task-execution-role",
    assume_role_policy="""{
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Principal": {"Service": "ecs-tasks.amazonaws.com"},
          "Action": "sts:AssumeRole"
        }
      ]
    }"""
)

ecs_task_execution_policy_attachment = aws.iam.RolePolicyAttachment(
    "ecs-task-execution-policy-attachment",
    role=ecs_task_execution_role.name,
    policy_arn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
)

log_group = aws.cloudwatch.LogGroup(
    "whisper-live-gpu-log-group",
    retention_in_days=7,
)

# Task definition with GPU support
task_definition = aws.ecs.TaskDefinition(
    "whisper-live-gpu-task-definition",
    family="whisper-live-gpu-task",
    cpu="1024",
    memory="4096",
    requires_compatibilities=["EC2"],
    execution_role_arn=ecs_task_execution_role.arn,
    network_mode="awsvpc",
    container_definitions=pulumi.Output.all(ecr_repository_url, image_tag, log_group.name).apply(lambda args: f"""
    [
        {{
            "name": "{container_name}",
            "image": "{args[0]}:{args[1]}",
            "cpu": 1024,
            "memory": 4096,
            "essential": true,
            "resourceRequirements": [
                {{
                    "value": "1",
                    "type": "GPU"
                }}
            ],
            "portMappings": [
                {{
                    "containerPort": 9090,
                    "hostPort": 9090,
                    "protocol": "tcp"
                }}
            ],
            "logConfiguration": {{
                "logDriver": "awslogs",
                "options": {{
                    "awslogs-group": "{args[2]}",
                    "awslogs-region": "us-west-1",
                    "awslogs-stream-prefix": "whisper-live"
                }}
            }}
        }}
    ]
    """)
)

capacity_provider = aws.ecs.CapacityProvider(
    "gpu-cluster-capacity-provider",
    auto_scaling_group_provider=aws.ecs.CapacityProviderAutoScalingGroupProviderArgs(
        auto_scaling_group_arn=auto_scaling_group.arn,
        managed_scaling=aws.ecs.CapacityProviderAutoScalingGroupProviderManagedScalingArgs(
            status="ENABLED",
            target_capacity=100,  # Adjust based on your needs
            minimum_scaling_step_size=1,
            maximum_scaling_step_size=1,
        )
    ),
    tags={"Name": "gpuClusterCapacityProvider"}  # Optional tags
)

capacity_provider_association = aws.ecs.ClusterCapacityProviders(
    "gpu-cluster-capacity-provider-association",
    cluster_name=ecs_cluster.name,
    capacity_providers=[capacity_provider.name],
    default_capacity_provider_strategies=[aws.ecs.ClusterCapacityProvidersDefaultCapacityProviderStrategyArgs(
        capacity_provider=capacity_provider.name,
        weight=1
    )]
)

# Create an ECS Service
ecs_service = aws.ecs.Service(
    "gpu-ecs-service",
    cluster=ecs_cluster.arn,
    desired_count=1,  # Number of tasks to run
    deployment_minimum_healthy_percent=0,  # Be sure to adjust this in production!
    task_definition=task_definition.arn,
    network_configuration=aws.ecs.ServiceNetworkConfigurationArgs(
        subnets=[private_subnet_a.id, private_subnet_b.id],
        security_groups=[security_group.id],
    ),
    load_balancers=[aws.ecs.ServiceLoadBalancerArgs(
        target_group_arn=tg.arn,
        container_name=container_name,
        container_port=9090,
    )],
    opts=pulumi.ResourceOptions(depends_on=[task_definition]),
    capacity_provider_strategies=[
        aws.ecs.ServiceCapacityProviderStrategyArgs(
            capacity_provider=capacity_provider.name,
            weight=1,
        )
    ],
)
