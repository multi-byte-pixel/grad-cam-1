"""
aws/standup.py — Launch EC2 training instances and write instances.json.

This script:
  1. Gets or creates a security group that allows SSH from your public IP only.
  2. Launches N instances with the configured AMI, instance type, and key pair.
  3. Tags each instance with a name and project tag for easy identification.
  4. Waits (with progress logging) until all instances reach 'running' state.
  5. Fetches public IPs and writes everything to aws/instances.json.

Usage:
    python aws/standup.py --count 3 --tags mlp dcnn tl
    python aws/standup.py --count 1 --tags single-run
    python aws/standup.py --count 4                    # tags auto-assigned 0..3

Prerequisites:
    - Run python aws/preflight.py first.
    - aws/.env filled in (or equivalent env vars set).
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# -- Bootstrap: make sure aws/ is on the path when run as a script ------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import aws.config as cfg
from botocore.exceptions import ClientError

# -- Logging ------------------------------------------------------------------─
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("standup")
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _my_public_ip() -> str:
    """
    Return the caller's current public IPv4 address by querying a public
    echo service.  Used to restrict SSH ingress to this machine only —
    far more secure than allowing 0.0.0.0/0.
    """
    import urllib.request
    try:
        with urllib.request.urlopen("https://checkip.amazonaws.com", timeout=5) as r:
            ip = r.read().decode().strip()
        log.debug("Detected public IP: %s", ip)
        return ip
    except Exception as exc:
        log.warning("Could not auto-detect public IP (%s). Falling back to 0.0.0.0/0.", exc)
        return "0.0.0.0"


def get_or_create_security_group(ec2_client) -> str:
    """
    Return an SG ID.

    Priority:
      1. EC2_SG_ID in config (user pre-created SG)
      2. An existing SG named EC2_SG_NAME
      3. Create a new SG named EC2_SG_NAME with SSH-only ingress
         restricted to the current public IP.
    """
    # -- Option 1: SG ID explicitly provided ----------------------------------
    if cfg.SG_ID:
        log.info("Using pre-configured security group: %s", cfg.SG_ID)
        return cfg.SG_ID

    # -- Option 2 / 3: look up or create by name ------------------------------─
    log.info("Looking for security group named '%s' …", cfg.SG_NAME)
    resp = ec2_client.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [cfg.SG_NAME]}]
    )
    existing = resp["SecurityGroups"]

    if existing:
        sg_id = existing[0]["GroupId"]
        log.info("Reusing existing security group %s (%s)", sg_id, cfg.SG_NAME)
        return sg_id

    # -- Create a new SG ------------------------------------------------------─
    my_ip = _my_public_ip()
    cidr  = f"{my_ip}/32" if my_ip != "0.0.0.0" else "0.0.0.0/0"
    log.info("Creating new security group '%s' with SSH ingress from %s …",
             cfg.SG_NAME, cidr)

    create_resp = ec2_client.create_security_group(
        GroupName=cfg.SG_NAME,
        Description=(
            f"ML training SG - SSH ingress restricted to {cidr}. "
            "Created by aws/standup.py."
        ),
        TagSpecifications=[{
            "ResourceType": "security-group",
            "Tags": [
                {"Key": "Name",    "Value": cfg.SG_NAME},
                {"Key": "Project", "Value": cfg.PROJECT_TAG},
            ],
        }],
    )
    sg_id = create_resp["GroupId"]
    log.info("Created security group %s", sg_id)

    # Add SSH inbound rule
    ec2_client.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{
            "IpProtocol": "tcp",
            "FromPort":    cfg.SSH_PORT,
            "ToPort":      cfg.SSH_PORT,
            "IpRanges": [{
                "CidrIp": cidr,
                "Description": f"SSH from {my_ip} (set by aws/standup.py)",
            }],
        }],
    )
    log.info("Authorized SSH ingress from %s on port %d", cidr, cfg.SSH_PORT)
    return sg_id


def launch_instances(ec2_client, sg_id: str, tags: list[str]) -> list[dict]:
    """
    Launch len(tags) EC2 instances, one per tag, and return a list of
    dicts with instance_id and tag.  Each instance gets a unique Name tag
    so you can identify it in the AWS console.

    We launch one at a time (instead of using the Count parameter) so that
    each instance can be given a distinct Name tag at launch time.
    """
    launched = []
    for i, tag in enumerate(tags):
        name = f"{cfg.PROJECT_TAG}-{tag}"
        log.info("[%s] Launching %s instance %d/%d (AMI=%s) …",
                 tag, cfg.INSTANCE_TYPE, i + 1, len(tags), cfg.AMI_ID)

        # Build the RunInstances request
        launch_kwargs: dict = dict(
            ImageId=cfg.AMI_ID,
            InstanceType=cfg.INSTANCE_TYPE,
            KeyName=cfg.KEY_NAME,
            MinCount=1,
            MaxCount=1,
            # Enforce IMDSv2 (prevents SSRF-based metadata attacks)
            MetadataOptions={
                "HttpTokens":   "required",
                "HttpEndpoint": "enabled",
            },
            BlockDeviceMappings=[{
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize":          cfg.VOLUME_SIZE_GB,
                    "VolumeType":          "gp3",
                    "Iops":                3000,
                    "DeleteOnTermination": True,
                },
            }],
            SecurityGroupIds=[sg_id],
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name",    "Value": name},
                    {"Key": "Project", "Value": cfg.PROJECT_TAG},
                    {"Key": "Task",    "Value": tag},
                ],
            }],
        )

        # Attach IAM instance profile if configured
        if cfg.IAM_PROFILE:
            launch_kwargs["IamInstanceProfile"] = {"Name": cfg.IAM_PROFILE}
            log.debug("[%s] Attaching IAM profile: %s", tag, cfg.IAM_PROFILE)

        resp        = ec2_client.run_instances(**launch_kwargs)
        instance_id = resp["Instances"][0]["InstanceId"]
        log.info("[%s] Instance launched: %s", tag, instance_id)
        launched.append({"tag": tag, "instance_id": instance_id, "public_ip": None})

    return launched


def wait_for_running(ec2_client, instances: list[dict], timeout_s: int = 300) -> list[dict]:
    """
    Poll EC2 until every instance in the list reaches state 'running'.
    Logs progress every 15 seconds.  Raises RuntimeError on timeout.

    After 'running', instances need another ~60–120 s for SSH to become
    available (cloud-init finishes, sshd starts).  deploy.py handles that.
    """
    log.info("Waiting for %d instance(s) to reach 'running' state (timeout=%ds) …",
             len(instances), timeout_s)

    ids       = [i["instance_id"] for i in instances]
    deadline  = time.time() + timeout_s
    poll_interval = 10  # seconds between API calls

    while time.time() < deadline:
        # RunInstances returns instance IDs before they are guaranteed visible
        # to DescribeInstances (eventual consistency).  A describe call issued
        # immediately after launch can raise InvalidInstanceID.NotFound even
        # though the instances are launching fine.  Treat that as "not yet
        # visible" and retry rather than aborting the whole stand-up.
        try:
            resp = ec2_client.describe_instances(InstanceIds=ids)
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "InvalidInstanceID.NotFound":
                log.info("  instances not yet visible to DescribeInstances "
                         "(eventual consistency) — retrying in %ds …", poll_interval)
                time.sleep(poll_interval)
                continue
            raise

        states = {}
        for reservation in resp["Reservations"]:
            for inst in reservation["Instances"]:
                states[inst["InstanceId"]] = inst["State"]["Name"]

        running = [iid for iid, s in states.items() if s == "running"]
        pending = [iid for iid, s in states.items() if s == "pending"]
        other   = {iid: s for iid, s in states.items()
                   if s not in ("running", "pending")}

        log.info("  running=%d  pending=%d  other=%s",
                 len(running), len(pending),
                 ", ".join(f"{iid}:{s}" for iid, s in other.items()) or "—")

        if other:
            raise RuntimeError(
                f"Unexpected instance state(s): {other}.  "
                "Check the EC2 console for details."
            )

        if len(running) == len(ids):
            log.info("All instances are running.")
            break

        time.sleep(poll_interval)
    else:
        raise RuntimeError(
            f"Timed out after {timeout_s}s waiting for instances to reach 'running'."
        )

    # Fetch public IPs now that instances are running
    resp = ec2_client.describe_instances(InstanceIds=ids)
    ip_map = {}
    for reservation in resp["Reservations"]:
        for inst in reservation["Instances"]:
            ip_map[inst["InstanceId"]] = inst.get("PublicIpAddress")

    for rec in instances:
        rec["public_ip"] = ip_map.get(rec["instance_id"])
        log.info("  %-30s  id=%-22s  ip=%s",
                 rec["tag"], rec["instance_id"], rec["public_ip"])

    return instances


def write_state(sg_id: str, sg_created: bool, instances: list[dict]) -> None:
    """
    Persist instance metadata to aws/instances.json.
    This file is read by deploy.py, monitor.py, and teardown.py so you
    don't have to copy/paste IDs between scripts.
    """
    state = {
        "project":     cfg.PROJECT_TAG,
        "region":      cfg.REGION,
        "instance_type": cfg.INSTANCE_TYPE,
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "sg_id":       sg_id,
        "sg_created":  sg_created,     # True → teardown.py will delete it
        "instances":   instances,
    }
    cfg.STATE_FILE.write_text(json.dumps(state, indent=2))
    log.info("State written to %s", cfg.STATE_FILE)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def standup(tags: list[str]) -> list[dict]:
    """
    Full stand-up flow: SG → launch → wait → save state.
    Returns the list of instance dicts (with IDs and IPs).
    """
    import boto3
    ec2 = boto3.client("ec2", region_name=cfg.REGION)

    log.info("=" * 60)
    log.info("STAND-UP  |  project=%s  region=%s  type=%s  count=%d",
             cfg.PROJECT_TAG, cfg.REGION, cfg.INSTANCE_TYPE, len(tags))
    log.info("          |  tags: %s", tags)
    log.info("=" * 60)

    # -- Security group --------------------------------------------------------
    sg_existed_before = bool(cfg.SG_ID) or bool(
        ec2.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [cfg.SG_NAME]}]
        )["SecurityGroups"]
    )
    sg_id      = get_or_create_security_group(ec2)
    sg_created = not sg_existed_before

    # -- Launch ----------------------------------------------------------------
    t_launch = time.time()
    instances = launch_instances(ec2, sg_id, tags)

    # -- Wait for running ------------------------------------------------------
    instances = wait_for_running(ec2, instances)
    log.info("Launch + running: %.1f s", time.time() - t_launch)

    # -- Save state ------------------------------------------------------------
    write_state(sg_id, sg_created, instances)

    log.info("=" * 60)
    log.info("STAND-UP COMPLETE — %d instance(s) running.", len(instances))
    log.info("SSH example:  ssh -i %s %s@%s",
             cfg.KEY_PATH, cfg.SSH_USER, instances[0]["public_ip"])
    log.info("Next step:    python aws/deploy.py --files <file1> [<file2>…] --cmd '<command>'")
    log.info("=" * 60)

    return instances


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch EC2 training instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--count", type=int, default=1,
        help="Number of instances to launch (ignored if --tags is given).",
    )
    parser.add_argument(
        "--tags", nargs="+", default=None,
        help=(
            "Logical names for each instance (e.g. mlp dcnn tl). "
            "One instance per tag.  Overrides --count."
        ),
    )
    args = parser.parse_args()

    tags = args.tags if args.tags else [str(i) for i in range(args.count)]

    try:
        standup(tags)
    except Exception as exc:
        log.exception("Stand-up failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
