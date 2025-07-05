# Pulumi Whisper Live GPU Setup

This project uses Pulumi to provision AWS resources necessary to deploy an ECS
service that runs the WhisperLive service with GPU. It sets up a VPC, subnets,
security groups, NAT gateways, an Auto Scaling Group, an ECS cluster, and
related components required to deploy a containerized application with GPU
support.

Note: This is provided for reference only! Please be sure you understand the
costs of running GPU instances in AWS before deploying this stack. This stack
may not be set up to ensure security best practices, so use it at your own risk!

### Prerequisites

- An AWS account and AWS CLI installed and configured
- Pulumi CLI installed
- A configured Pulumi stack or access to create one

### Setup Steps

1. **Install AWS CLI**

   First, ensure you have the AWS CLI installed. If you don't, install it
   following [these instructions](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html).

2. **Configure AWS CLI**

   Once the AWS CLI is installed, configure it by running `aws configure`.
   You'll need to input your AWS Access Key ID, Secret Access Key, region, and
   output format (e.g., json).

    ```sh
    aws configure
    ```

3. **Install Pulumi**

   If Pulumi is not installed on your system, follow
   the [Get Started with Pulumi](https://www.pulumi.com/docs/get-started/aws/)
   guide to install it.

4. **Set up Pulumi Stack**

   Initialize a new Pulumi stack if you haven't already. A stack represents an
   isolated deployment (e.g., development, staging, production).

    ```sh
    pulumi stack init dev
    ```

5. **Configure Pulumi for AWS**

   Configure Pulumi to use your AWS credentials. You can specify the AWS region
   in which to deploy resources.

    ```sh
    pulumi config set aws:region <region> # e.g., us-west-1
    ```

6. **Define Required Configuration Variables**

   The Pulumi program requires certain configuration variables. Set them using
   the `pulumi config set` command.

    ```sh
    pulumi config set vpc_id <your-vpc-id>
    pulumi config set public_subnet_id_a <your-public-subnet-a-id>
    pulumi config set public_subnet_id_b <your-public-subnet-b-id>
    pulumi config set --optional ecr_repository_url <your-ecr-repo-url>  # Default: ghcr.io/collabora/whisperlive-gpu
    pulumi config set --optional image_tag <your-image-tag>  # Default: latest
    pulumi config set --optional ami_id <your-ami-id> # Find the AMI ID for the desired ECS compatible GPU instance type available in your region
    ```

7. **Deploy with Pulumi**

   Run the following command to provision the AWS resources as per the Pulumi
   program.

    ```sh
    pulumi up
    ```

    Review the proposed changes and type `yes` to proceed with the deployment.

### Teardown

1. **Destroy Resources**

    To tear down the resources and remove the stack, run:

    ```sh
    pulumi destroy
    pulumi stack rm
    ```

    Confirm the destruction of resources when prompted.

### Important Notes

- Be mindful of the AWS region where you deploy this project. The AMI ID and
  resource availability may vary by region.
- Adjust the `desired_capacity`, `max_size`, and `min_size` for the Auto Scaling
  Group based on your application's requirements.
- Always review AWS resource costs and Pulumi's resource management to ensure
  that you're operating within your budget and requirements.
