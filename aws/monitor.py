"""
aws/monitor.py — Poll instance health and stream training logs.

For each instance in aws/instances.json this script reports:
  • EC2 instance state (running / stopped / terminated …)
  • System status checks (passed / failed)
  • Last N lines of the remote training log
  • GPU utilization via nvidia-smi (if a GPU instance)
  • Elapsed wall-clock time since launch

Usage:
    # One-shot status report for all instances
    python aws/monitor.py

    # Watch mode: refresh every 60 s until Ctrl-C
    python aws/monitor.py --watch

    # Tail a specific instance's log continuously (like `tail -f`)
    python aws/monitor.py --tail --tags dcnn --lines 50

    # Only check certain tags
    python aws/monitor.py --tags mlp dcnn

Prerequisites:
    - aws/instances.json exists (written by standup.py).
    - paramiko installed:  pip install paramiko
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# -- Bootstrap ----------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import aws.config as cfg

# -- Logging ------------------------------------------------------------------─
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("monitor")
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("paramiko").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_state(tags_filter: list[str] | None) -> tuple[dict, list[dict]]:
    """Load instances.json.  Returns (full_state_dict, filtered_instance_list)."""
    if not cfg.STATE_FILE.exists():
        raise FileNotFoundError(
            f"{cfg.STATE_FILE} not found.  Run python aws/standup.py first."
        )
    state     = json.loads(cfg.STATE_FILE.read_text())
    instances = state.get("instances", [])
    if tags_filter:
        instances = [i for i in instances if i["tag"] in tags_filter]
    return state, instances


def _elapsed(launched_at_iso: str) -> str:
    """Return a human-readable elapsed time string since the given ISO timestamp."""
    try:
        start   = datetime.fromisoformat(launched_at_iso)
        elapsed = datetime.now(timezone.utc) - start
        h, rem  = divmod(int(elapsed.total_seconds()), 3600)
        m, s    = divmod(rem, 60)
        return f"{h}h {m}m {s}s"
    except Exception:
        return "unknown"


def check_ec2_status(instance_ids: list[str]) -> dict[str, dict]:
    """
    Query EC2 for instance state + system/instance status checks.
    Returns a dict keyed by instance_id.

    Status check states: "ok" | "impaired" | "insufficient-data" | "not-applicable"
    An "impaired" system check means the underlying AWS hardware has a problem.
    An "impaired" instance check means the OS is unresponsive.
    """
    import boto3
    ec2    = boto3.client("ec2", region_name=cfg.REGION)
    result = {}

    # describe_instances gives us the lifecycle state (running / stopped / …)
    inst_resp = ec2.describe_instances(InstanceIds=instance_ids)
    for reservation in inst_resp["Reservations"]:
        for inst in reservation["Instances"]:
            iid = inst["InstanceId"]
            result[iid] = {
                "state":         inst["State"]["Name"],
                "public_ip":     inst.get("PublicIpAddress", "—"),
                "system_check":  "—",
                "instance_check":"—",
            }

    # describe_instance_status gives us the health checks
    # Note: stopped instances don't appear here, so we only call this if needed.
    running_ids = [iid for iid, d in result.items() if d["state"] == "running"]
    if running_ids:
        chk_resp = ec2.describe_instance_status(InstanceIds=running_ids)
        for chk in chk_resp["InstanceStatuses"]:
            iid = chk["InstanceId"]
            result[iid]["system_check"]   = chk["SystemStatus"]["Status"]
            result[iid]["instance_check"] = chk["InstanceStatus"]["Status"]

    return result


def _ssh_run_silent(ip: str, cmd: str) -> str:
    """
    Run a command on the remote instance and return stdout as a string.
    Returns empty string on any error (monitor should never crash).
    """
    try:
        import paramiko
        key_path = Path(cfg.KEY_PATH)
        try:
            pkey = paramiko.RSAKey.from_private_key_file(str(key_path))
        except paramiko.ssh_exception.SSHException:
            pkey = paramiko.Ed25519Key.from_private_key_file(str(key_path))

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=ip, port=cfg.SSH_PORT, username=cfg.SSH_USER,
            pkey=pkey, timeout=10, banner_timeout=20, auth_timeout=10,
        )
        _, stdout, _ = client.exec_command(cmd, timeout=15)
        out = stdout.read().decode(errors="replace").strip()
        client.close()
        return out
    except Exception as exc:
        log.debug("SSH command failed on %s: %s", ip, exc)
        return ""


def fetch_log_tail(ip: str, tag: str, log_file: str, lines: int) -> str:
    """Fetch the last N lines of the remote training log via SSH."""
    # log_file may contain {tag} placeholder
    remote_log = log_file.replace("{tag}", tag)
    out = _ssh_run_silent(ip, f"tail -n {lines} ~/{remote_log} 2>/dev/null")
    return out


def fetch_gpu_stats(ip: str) -> str:
    """
    Run nvidia-smi on the instance and return a one-line summary:
    GPU utilization %, memory used / total, temperature.
    Returns empty string if nvidia-smi is unavailable (CPU instance).
    """
    cmd = (
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu "
        "--format=csv,noheader,nounits 2>/dev/null | head -1"
    )
    raw = _ssh_run_silent(ip, cmd)
    if not raw:
        return ""
    # raw = "42, 3800, 15360, 67"  (util%, mem_used_MB, mem_total_MB, temp_C)
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) == 4:
        util, used, total, temp = parts
        return f"GPU: {util}% util  {used}/{total} MB  {temp}°C"
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════

def report(instances: list[dict], state: dict, log_file: str, lines: int) -> None:
    """Print a status report for all instances."""
    launched_at = state.get("launched_at", "")
    ids         = [i["instance_id"] for i in instances]

    log.info("=" * 60)
    log.info("MONITOR REPORT  —  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Elapsed since launch: %s", _elapsed(launched_at))
    log.info("=" * 60)

    # Batch EC2 API call for all instances at once (one API call, not N)
    ec2_status = {}
    try:
        ec2_status = check_ec2_status(ids)
    except Exception as exc:
        log.warning("Could not fetch EC2 status: %s", exc)

    for inst in instances:
        tag = inst["tag"]
        iid = inst["instance_id"]
        ip  = inst.get("public_ip", "—")

        log.info("")
        log.info("-- Instance: %-10s  id=%-22s  ip=%s", tag, iid, ip)

        # EC2 health
        if iid in ec2_status:
            s = ec2_status[iid]
            log.info("   State          : %s", s["state"])
            log.info("   System check   : %s", s["system_check"])
            log.info("   Instance check : %s", s["instance_check"])
        else:
            log.info("   State          : (unavailable)")

        # GPU stats (skip if no IP or instance not running)
        if ip and ip != "—":
            gpu = fetch_gpu_stats(ip)
            if gpu:
                log.info("   %s", gpu)
            else:
                log.debug("   No GPU stats (CPU instance or nvidia-smi unavailable)")

        # Training log tail
        if ip and ip != "—":
            tail = fetch_log_tail(ip, tag, log_file, lines)
            if tail:
                log.info("   -- Last %d log lines (%s) --", lines, log_file.replace("{tag}", tag))
                for line in tail.splitlines():
                    log.info("   %s", line)
            else:
                log.info("   -- Log not yet available (training may not have started) --")

    log.info("")
    log.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check EC2 instance health and tail training logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tags",    nargs="+", default=None,
                        help="Only show instances with these tags.")
    parser.add_argument("--lines",   type=int, default=20,
                        help="Number of log tail lines to show.")
    parser.add_argument("--log",     default="{tag}.log",
                        help="Remote log filename ({tag} replaced per instance).")
    parser.add_argument("--watch",   action="store_true",
                        help="Continuously refresh the report every --interval seconds.")
    parser.add_argument("--tail",    action="store_true",
                        help="Stream log continuously (like tail -f). Use with --tags.")
    parser.add_argument("--interval", type=int, default=60,
                        help="Refresh interval for --watch mode (seconds).")
    args = parser.parse_args()

    try:
        state, instances = _load_state(args.tags)
    except Exception as exc:
        log.error("Failed to load state: %s", exc)
        sys.exit(1)

    if args.tail:
        # -- Continuous tail mode ----------------------------------------------
        if not instances:
            log.error("No instances to tail.")
            sys.exit(1)
        inst = instances[0]
        ip   = inst["public_ip"]
        tag  = inst["tag"]
        rlog = args.log.replace("{tag}", tag)
        log.info("Tailing ~/%s on %s (%s) … (Ctrl-C to stop)", rlog, ip, tag)
        try:
            # Use `tail -f` and stream output line by line
            import paramiko
            key_path = Path(cfg.KEY_PATH)
            try:
                pkey = paramiko.RSAKey.from_private_key_file(str(key_path))
            except paramiko.ssh_exception.SSHException:
                pkey = paramiko.Ed25519Key.from_private_key_file(str(key_path))
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=ip, port=cfg.SSH_PORT, username=cfg.SSH_USER,
                           pkey=pkey, timeout=15, banner_timeout=30)
            _, stdout, _ = client.exec_command(f"tail -f ~/{rlog}")
            for line in iter(stdout.readline, ""):
                print(f"[{tag}] {line}", end="", flush=True)
        except KeyboardInterrupt:
            log.info("Tail stopped.")
        return

    if args.watch:
        # -- Watch mode: repeat every --interval seconds ----------------------─
        log.info("Watch mode — refreshing every %ds (Ctrl-C to stop)", args.interval)
        try:
            while True:
                report(instances, state, args.log, args.lines)
                log.info("Next refresh in %ds …", args.interval)
                time.sleep(args.interval)
                # Reload state in case IPs were updated
                state, instances = _load_state(args.tags)
        except KeyboardInterrupt:
            log.info("Watch mode stopped.")
    else:
        # -- One-shot report --------------------------------------------------─
        report(instances, state, args.log, args.lines)


if __name__ == "__main__":
    main()
