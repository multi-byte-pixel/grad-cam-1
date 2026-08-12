"""
aws/config.py — Centralized AWS configuration for all boto3 scripts.

ALL account-specific values come from environment variables or a local
aws/.env file.  Nothing sensitive is ever hardcoded here.

Quick-start:
    cp aws/.env.example aws/.env
    # edit aws/.env with your values
    python aws/preflight.py          # verify everything looks good
"""

import os
from pathlib import Path

# -- Load aws/.env if it exists (never committed — see .gitignore) ------------─
# python-dotenv lets us keep credentials in a file rather than setting shell
# variables every time.  If the package isn't installed, we fall back to
# whatever is already in the real environment.
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)   # real env vars take priority
except ImportError:
    pass  # pip install python-dotenv  if you want .env file support


# -- AWS region ----------------------------------------------------------------
# All resources (instances, SGs, key pairs) must live in the same region.
REGION: str = os.getenv("AWS_REGION", "us-east-1")


# -- EC2 instance --------------------------------------------------------------
# g4dn.xlarge  → 1× NVIDIA T4 GPU, 4 vCPU, 16 GB RAM  (~$0.53 / hr on-demand)
# g4dn.2xlarge → same GPU, 8 vCPU, 32 GB RAM           (~$0.75 / hr)
INSTANCE_TYPE: str = os.getenv("EC2_INSTANCE_TYPE", "g4dn.xlarge")

# Deep Learning AMI (Ubuntu 22.04, us-east-1) — ships with CUDA 12.4,
# cuDNN, and a ready-to-use conda env containing PyTorch.
# To find the latest AMI ID in your region:
#   aws ec2 describe-images --owners amazon \
#     --filters 'Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*' \
#     --query 'sort_by(Images,&CreationDate)[-1].[ImageId,Name]' --output text
AMI_ID: str = os.getenv("EC2_AMI_ID", "ami-05ac9f889ac42dd39")

# Name of the EC2 key pair that is registered in your AWS account.
# Create one with:  aws ec2 create-key-pair --key-name MyKey --query KeyMaterial \
#                      --output text > ~/.ssh/MyKey.pem && chmod 400 ~/.ssh/MyKey.pem
KEY_NAME: str = os.getenv("EC2_KEY_NAME", "my-key-pair")

# Full local path to the .pem private key.  Used by deploy/monitor for SSH.
KEY_PATH: str = os.getenv("EC2_KEY_PATH", str(Path.home() / ".ssh" / "id_rsa"))

# EBS root volume size in GB.
# 45 GB is enough for CIFAR-100 + ImageNet subsets + model checkpoints.
VOLUME_SIZE_GB: int = int(os.getenv("EC2_VOLUME_SIZE_GB", "45"))

# SSH username depends on the AMI:
#   Deep Learning AMI (Ubuntu 22.04) → "ubuntu"
#   Amazon Linux 2 / Amazon Linux 2023 → "ec2-user"
# The configured AMI is a Deep Learning Base AMI on Amazon Linux 2023,
# so the default is "ec2-user".  Override via EC2_SSH_USER if you swap AMIs.
SSH_USER: str = os.getenv("EC2_SSH_USER", "ec2-user")

# SSH port (usually 22; some hardened configs move it)
SSH_PORT: int = int(os.getenv("EC2_SSH_PORT", "22"))


# -- Security group ------------------------------------------------------------
# If EC2_SG_ID is set, instances are placed in that existing security group.
# If empty, standup.py creates a new one named EC2_SG_NAME automatically
# and restricts SSH ingress to your current public IP (not 0.0.0.0/0).
SG_ID: str   = os.getenv("EC2_SG_ID", "")
SG_NAME: str = os.getenv("EC2_SG_NAME", "ml-training-sg")


# -- IAM instance profile ------------------------------------------------------
# Grants the EC2 instance AWS permissions (e.g. read S3 buckets, write SSM
# parameters) without embedding long-term credentials on the machine.
# Leave empty if the training job doesn't need to call any AWS APIs.
# Example value: "EC2InstanceProfileForTraining"
IAM_PROFILE: str = os.getenv("EC2_IAM_PROFILE", "")


# -- Tagging ------------------------------------------------------------------─
# Every launched resource gets a "Project" tag — useful for cost allocation.
PROJECT_TAG: str = os.getenv("EC2_PROJECT_TAG", "MLTraining")


# -- Runtime state file --------------------------------------------------------
# standup.py writes instance IDs and public IPs here after launch.
# All other scripts (deploy, monitor, teardown) read it so you don't have
# to pass IDs manually.  This file is git-ignored (contains live IPs).
#
# Overridable via EC2_STATE_FILE so you can run a second, isolated batch of
# instances without clobbering an existing run's state (e.g. a re-run of one
# task while another task's instance is still live).  Relative paths are
# resolved against this directory.
_state_file_env = os.getenv("EC2_STATE_FILE")
if _state_file_env:
    _sf = Path(_state_file_env)
    STATE_FILE: Path = _sf if _sf.is_absolute() else (Path(__file__).parent / _sf)
else:
    STATE_FILE: Path = Path(__file__).parent / "instances.json"


# -- AWS service-quota codes --------------------------------------------------─
# Used by preflight.py to check that you won't hit quota limits mid-launch.
# "Running On-Demand G and VT instances" — covers g4dn, g5, g6 families.
G_VCPU_QUOTA_CODE: str = "L-DB2E81BA"

# vCPUs per instance type — used to calculate total vCPUs needed.
VCPUS_BY_TYPE: dict = {
    "g4dn.xlarge":  4,
    "g4dn.2xlarge": 8,
    "g4dn.4xlarge": 16,
    "g4dn.12xlarge": 48,
    "g5.xlarge":    4,
    "g5.2xlarge":   8,
}
