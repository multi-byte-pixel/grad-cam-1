"""
aws/preflight.py — Pre-flight checks for AWS EC2 training launches.

Run this before standup.py to catch configuration problems that would otherwise
waste money (e.g. wrong key name, insufficient vCPU quota) or leave instances
running without being able to connect to them.

Usage:
    python aws/preflight.py                        # check for 1 instance
    python aws/preflight.py --count 4              # check quota for 4 instances
    python aws/preflight.py --instance-type g5.xlarge --count 2

Exit codes:
    0  all checks passed
    1  one or more checks failed (details printed to stderr / log)
"""

import argparse
import importlib.util
import logging
import socket
import sys
import time
from pathlib import Path

# -- Bootstrap: make sure aws/ is on the path when run as a script ------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import aws.config as cfg

# -- Logging setup ------------------------------------------------------------─
# Format: timestamp  [LEVEL   ]  message
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("preflight")

# Suppress noisy boto3 / botocore debug chatter so our messages stand out.
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# Individual check functions
# Each returns True on success, False on failure.
# They log the result themselves so the caller just collects a pass/fail flag.
# ═══════════════════════════════════════════════════════════════════════════════

def check_dependencies() -> bool:
    """Verify that boto3, paramiko, and python-dotenv are importable."""
    log.info("-- Check 1 / 7: Python dependencies --------------------------")
    ok = True
    for pkg, install_name in [
        ("boto3",   "boto3"),
        ("paramiko","paramiko"),
        ("dotenv",  "python-dotenv"),
    ]:
        if importlib.util.find_spec(pkg) is not None:
            mod = importlib.import_module(pkg)
            # dotenv exposes version via importlib.metadata, not __version__
            ver = getattr(mod, "__version__", None)
            if ver is None:
                try:
                    from importlib.metadata import version as pkg_version
                    ver = pkg_version(install_name)
                except Exception:
                    ver = "installed"
            log.info("  [OK]  %-12s  version %s", pkg, ver)
        else:
            log.error("  [FAIL]  %s not found.  Fix: pip install %s", pkg, install_name)
            ok = False
    return ok


def check_aws_credentials() -> bool:
    """
    Call STS.GetCallerIdentity — the cheapest possible AWS API call.
    It returns the account ID and IAM identity without doing anything.
    If this fails, no other AWS call will work.
    """
    log.info("-- Check 2 / 7: AWS credentials (STS GetCallerIdentity) ------")
    try:
        import boto3
        sts = boto3.client("sts", region_name=cfg.REGION)
        identity = sts.get_caller_identity()
        log.info("  [OK]  Account : %s", identity["Account"])
        log.info("  [OK]  ARN     : %s", identity["Arn"])
        log.info("  [OK]  Region  : %s", cfg.REGION)
        return True
    except Exception as exc:
        log.error("  [FAIL]  AWS credential check failed: %s", exc)
        log.error("     Fix: run 'aws configure' or set AWS_ACCESS_KEY_ID /"
                  " AWS_SECRET_ACCESS_KEY / AWS_PROFILE environment variables.")
        return False


def check_ami(ami_id: str) -> bool:
    """
    Verify the AMI exists and is in 'available' state in the target region.
    A missing or deregistered AMI would cause RunInstances to fail immediately.
    """
    log.info("-- Check 3 / 7: AMI availability (%s) ----------------------", ami_id)
    try:
        import boto3
        ec2 = boto3.client("ec2", region_name=cfg.REGION)
        resp = ec2.describe_images(ImageIds=[ami_id])
        images = resp.get("Images", [])
        if not images:
            log.error("  [FAIL]  AMI %s not found in region %s.", ami_id, cfg.REGION)
            log.error("     Fix: update EC2_AMI_ID in aws/.env with a valid AMI for %s.", cfg.REGION)
            return False
        img = images[0]
        state = img["State"]
        name  = img.get("Name", "(no name)")
        if state != "available":
            log.error("  [FAIL]  AMI %s found but state is '%s' (need 'available').", ami_id, state)
            return False
        log.info("  [OK]  AMI state : %s", state)
        log.info("  [OK]  AMI name  : %s", name)
        return True
    except Exception as exc:
        log.error("  [FAIL]  AMI check failed: %s", exc)
        return False


def check_key_pair(key_name: str) -> bool:
    """
    Verify the named key pair is registered in the AWS account AND that the
    corresponding private key file exists locally (needed for SSH / SCP).
    """
    log.info("-- Check 4 / 7: Key pair (%s) ------------------------------", key_name)
    ok = True

    # Check the key pair exists in AWS
    try:
        import boto3
        ec2 = boto3.client("ec2", region_name=cfg.REGION)
        resp = ec2.describe_key_pairs(KeyNames=[key_name])
        kp = resp["KeyPairs"][0]
        log.info("  [OK]  Key pair '%s' found in AWS (fingerprint: %s…)",
                 key_name, kp.get("KeyFingerprint", "")[:20])
    except Exception as exc:
        log.error("  [FAIL]  Key pair '%s' not found in AWS: %s", key_name, exc)
        log.error("     Fix: create it with 'aws ec2 create-key-pair --key-name %s …'", key_name)
        ok = False

    # Check the local .pem file exists
    key_path = Path(cfg.KEY_PATH)
    if key_path.exists():
        # On POSIX: warn if permissions are too open (SSH will refuse the key)
        import stat
        mode = key_path.stat().st_mode
        if sys.platform != "win32" and (mode & 0o077):
            log.warning("  [WARN]  %s has loose permissions (%s). SSH may reject it.",
                        key_path, oct(mode))
            log.warning("     Fix: chmod 400 %s", key_path)
        else:
            log.info("  [OK]  Local key file : %s", key_path)
    else:
        log.error("  [FAIL]  Local key file not found at %s.", key_path)
        log.error("     Fix: set EC2_KEY_PATH in aws/.env to the correct path.")
        ok = False

    return ok


def check_vcpu_quota(instance_type: str, count: int) -> bool:
    """
    Query the AWS Service Quotas API to confirm enough G-instance vCPU quota
    is available.  AWS limits are account-wide per region, not per launch.
    Hitting the quota causes RunInstances to fail with an InsufficientCapacity
    or RequestLimitExceeded error after the instances are partially started.
    """
    log.info("-- Check 5 / 7: G-instance vCPU quota ----------------------")
    vcpus_per = cfg.VCPUS_BY_TYPE.get(instance_type, 4)
    needed    = vcpus_per * count
    log.info("  Instance type : %s  (%d vCPUs each × %d = %d needed)",
             instance_type, vcpus_per, count, needed)

    try:
        import boto3
        sq = boto3.client("service-quotas", region_name=cfg.REGION)
        resp  = sq.get_service_quota(
            ServiceCode="ec2",
            QuotaCode=cfg.G_VCPU_QUOTA_CODE,
        )
        quota_value = int(resp["Quota"]["Value"])
        log.info("  Quota value   : %d vCPUs", quota_value)

        if quota_value < needed:
            log.error("  [FAIL]  Quota too low: have %d, need %d.", quota_value, needed)
            log.error("     Fix: request an increase at "
                      "https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas")
            return False
        log.info("  [OK]  Quota sufficient (%d >= %d)", quota_value, needed)
        return True
    except Exception as exc:
        log.warning("  [WARN]  Could not query service quotas: %s", exc)
        log.warning("     Proceeding without quota confirmation.")
        return True   # non-fatal — quota API may not be accessible in all accounts


def check_security_group() -> bool:
    """
    If EC2_SG_ID is set in config, verify it exists.
    If not set, verify that a new SG named EC2_SG_NAME can be created
    (i.e. the name is not already taken by an incompatible SG).
    """
    log.info("-- Check 6 / 7: Security group ------------------------------")
    try:
        import boto3
        ec2 = boto3.client("ec2", region_name=cfg.REGION)

        if cfg.SG_ID:
            # User provided an existing SG ID — verify it exists
            resp = ec2.describe_security_groups(GroupIds=[cfg.SG_ID])
            sg = resp["SecurityGroups"][0]
            log.info("  [OK]  SG '%s' (%s) exists — will reuse.", cfg.SG_ID, sg["GroupName"])
        else:
            # No SG ID set — standup.py will create one.
            # Check if the name is already taken.
            resp = ec2.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": [cfg.SG_NAME]}]
            )
            existing = resp["SecurityGroups"]
            if existing:
                sg = existing[0]
                log.info("  [OK]  SG '%s' (%s) already exists — standup.py will reuse it.",
                         sg["GroupId"], cfg.SG_NAME)
            else:
                log.info("  [OK]  SG '%s' does not yet exist — standup.py will create it.",
                         cfg.SG_NAME)
        return True
    except Exception as exc:
        log.error("  [FAIL]  Security group check failed: %s", exc)
        return False


def check_internet_connectivity() -> bool:
    """
    Verify we can reach the internet (needed for fetching our public IP
    when restricting SSH ingress in standup.py).
    """
    log.info("-- Check 7 / 7: Internet connectivity ----------------------─")
    try:
        # Attempt a TCP connection to the AWS EC2 endpoint
        sock = socket.create_connection(("ec2.amazonaws.com", 443), timeout=5)
        sock.close()
        log.info("  [OK]  Can reach ec2.amazonaws.com:443")
        return True
    except OSError as exc:
        log.error("  [FAIL]  No internet connectivity: %s", exc)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_checks(instance_type: str, count: int, ami_id: str, key_name: str) -> bool:
    """Run all checks and return True only if every check passes."""
    log.info("=" * 60)
    log.info("AWS Pre-flight Checks")
    log.info("  Region        : %s", cfg.REGION)
    log.info("  Instance type : %s × %d", instance_type, count)
    log.info("  AMI           : %s", ami_id)
    log.info("  Key pair      : %s", key_name)
    log.info("=" * 60)

    t0     = time.time()
    checks = [
        check_dependencies(),
        check_aws_credentials(),
        check_ami(ami_id),
        check_key_pair(key_name),
        check_vcpu_quota(instance_type, count),
        check_security_group(),
        check_internet_connectivity(),
    ]

    passed = sum(checks)
    total  = len(checks)
    elapsed = time.time() - t0

    log.info("=" * 60)
    if all(checks):
        log.info("RESULT: ALL %d / %d CHECKS PASSED  (%.1fs)", passed, total, elapsed)
        log.info("You are ready to run: python aws/standup.py")
    else:
        log.error("RESULT: %d / %d CHECKS FAILED  (%.1fs)", total - passed, total, elapsed)
        log.error("Fix the issues above before running standup.py.")
    log.info("=" * 60)

    return all(checks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify AWS config before launching EC2 training instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count",         type=int, default=1,
                        help="Number of instances to check quota for.")
    parser.add_argument("--instance-type", default=cfg.INSTANCE_TYPE,
                        help="EC2 instance type (overrides EC2_INSTANCE_TYPE in .env).")
    parser.add_argument("--ami-id",        default=cfg.AMI_ID,
                        help="AMI ID to check (overrides EC2_AMI_ID in .env).")
    parser.add_argument("--key-name",      default=cfg.KEY_NAME,
                        help="Key pair name to check (overrides EC2_KEY_NAME in .env).")
    args = parser.parse_args()

    ok = run_checks(
        instance_type=args.instance_type,
        count=args.count,
        ami_id=args.ami_id,
        key_name=args.key_name,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
