"""
aws/teardown.py — Terminate EC2 instances and clean up resources.

This script reads aws/instances.json (written by standup.py) and:
  1. Optionally downloads any result files from each instance before terminating.
  2. Terminates all (or selected) instances.
  3. Waits until they reach 'terminated' state.
  4. If standup.py created the security group, deletes it.
  5. Removes (or archives) aws/instances.json.

Always run teardown.py after training is complete to avoid unexpected AWS charges.
A g4dn.xlarge costs ~$0.53/hr — an instance left running overnight = ~$4.

Usage:
    # Normal teardown (no file collection)
    python aws/teardown.py

    # Collect result files before terminating
    python aws/teardown.py --collect --collect-files "*.pt" "*.csv" "*.png" "*.log"

    # Dry run — show what would be done without actually terminating
    python aws/teardown.py --dry-run

    # Only terminate specific instances
    python aws/teardown.py --tags tinycnn resnet

    # Skip SG deletion even if standup.py created it
    python aws/teardown.py --keep-sg
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
log = logging.getLogger("teardown")
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("paramiko").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_state(tags_filter: list[str] | None) -> tuple[dict, list[dict]]:
    """Load instances.json.  Raises FileNotFoundError if it doesn't exist."""
    if not cfg.STATE_FILE.exists():
        raise FileNotFoundError(
            f"{cfg.STATE_FILE} not found.  Nothing to tear down."
        )
    state     = json.loads(cfg.STATE_FILE.read_text())
    instances = state.get("instances", [])
    if tags_filter:
        instances = [i for i in instances if i["tag"] in tags_filter]
    return state, instances


# Directories searched (in order) when collecting artifacts.
# ~/training/ is where the run scripts write their output (they run from that
# directory via `cd ~/training && ...` in deploy.py).
# ~/ is kept as a fallback so log files (written there by deploy.py) are
# still collected.
COLLECT_DIRS = ["~/training", "~"]

# ── Required artifacts per task ────────────────────────────────────────────────
# Used by the pre-termination gate to verify collection succeeded locally.
REQUIRED_ARTIFACTS = {
    "tinycnn": ["cifar10_model.pt",    "stl10_model.pt",    "metrics.csv"],
    "resnet":  ["cifar10_resnet18.pt", "stl10_resnet18.pt", "metrics.csv"],
}


def collect_results(instances: list[dict], patterns: list[str], local_dir: Path) -> None:
    """
    Download files matching glob patterns from each instance before terminating.
    Searches COLLECT_DIRS (~/training first, then ~/) on the remote side so that
    training artifacts (in ~/training/) and log files (in ~/) are both captured.
    Files are saved to local_dir/<tag>/<filename>.
    Missing files are logged as warnings, not errors.
    """
    try:
        import paramiko
    except ImportError:
        log.warning("paramiko not installed — skipping file collection.  "
                    "pip install paramiko  to enable this.")
        return

    key_path = Path(cfg.KEY_PATH)
    if not key_path.exists():
        log.warning("Key file %s not found — skipping collection.", key_path)
        return

    try:
        pkey = paramiko.RSAKey.from_private_key_file(str(key_path))
    except paramiko.ssh_exception.SSHException:
        pkey = paramiko.Ed25519Key.from_private_key_file(str(key_path))

    for inst in instances:
        tag = inst["tag"]
        ip  = inst.get("public_ip")
        if not ip:
            log.warning("[%s] No public IP — cannot collect files.", tag)
            continue

        dest_dir = local_dir / tag
        dest_dir.mkdir(parents=True, exist_ok=True)

        log.info("[%s] Collecting results from %s → %s", tag, ip, dest_dir)

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=ip, port=cfg.SSH_PORT, username=cfg.SSH_USER,
                pkey=pkey, timeout=15, banner_timeout=30,
            )
            sftp = client.open_sftp()

            for pattern in patterns:
                # Search every COLLECT_DIR so training artifacts (~/training/)
                # and log files (~/) are both found.  We deduplicate by filename
                # so a file present in both locations is only downloaded once
                # (the ~/training/ copy wins as it is listed first).
                seen_names: set[str] = set()
                remote_files: list[str] = []
                for rdir in COLLECT_DIRS:
                    _, stdout, _ = client.exec_command(
                        f"ls {rdir}/{pattern} 2>/dev/null || true"
                    )
                    for line in stdout.read().decode().strip().splitlines():
                        line = line.strip()
                        if line and Path(line).name not in seen_names:
                            seen_names.add(Path(line).name)
                            remote_files.append(line)

                if not remote_files:
                    log.debug("[%s] No files matching %s in %s", tag, pattern, COLLECT_DIRS)
                    continue

                for remote_path in remote_files:
                    fname = Path(remote_path).name
                    local_path = dest_dir / fname
                    try:
                        sftp.get(remote_path, str(local_path))
                        size = local_path.stat().st_size
                        log.info("[%s] Downloaded %s (%d bytes)", tag, fname, size)
                    except Exception as exc:
                        log.warning("[%s] Could not download %s: %s", tag, remote_path, exc)

            sftp.close()
        except Exception as exc:
            log.warning("[%s] Connection failed during collection: %s", tag, exc)
        finally:
            client.close()


def verify_collected(instances: list[dict], collect_dir: Path) -> dict[str, list[str]]:
    """
    Pre-termination gate: verify that every required artifact for each instance
    was actually downloaded to collect_dir/<tag>/.

    Returns a dict mapping tag -> list of missing filenames.
    An empty list means all artifacts are present.
    Instances whose tag is not in REQUIRED_ARTIFACTS (e.g. 'spare') are skipped.
    """
    missing_by_tag: dict[str, list[str]] = {}
    for inst in instances:
        tag = inst["tag"]
        if tag not in REQUIRED_ARTIFACTS:
            continue
        dest_dir = collect_dir / tag
        missing = []
        for fname in REQUIRED_ARTIFACTS[tag]:
            local_path = dest_dir / fname
            if not local_path.exists() or local_path.stat().st_size == 0:
                missing.append(fname)
        if missing:
            missing_by_tag[tag] = missing
            log.warning(
                "[%s] GATE FAIL — %d artifact(s) not collected: %s",
                tag, len(missing), missing,
            )
        else:
            log.info("[%s] GATE PASS — all %d required artifacts present locally.",
                     tag, len(REQUIRED_ARTIFACTS[tag]))
    return missing_by_tag


def terminate_instances(ec2_client, instance_ids: list[str], dry_run: bool) -> None:
    """
    Send TerminateInstances API call.  In dry_run mode, only log what would happen.
    """
    if dry_run:
        log.info("[DRY RUN] Would terminate: %s", instance_ids)
        return

    log.info("Terminating %d instance(s): %s", len(instance_ids), instance_ids)
    resp = ec2_client.terminate_instances(InstanceIds=instance_ids)
    for change in resp["TerminatingInstances"]:
        log.info("  %s : %s → %s",
                 change["InstanceId"],
                 change["PreviousState"]["Name"],
                 change["CurrentState"]["Name"])


def wait_for_terminated(ec2_client, instance_ids: list[str], timeout_s: int = 180) -> None:
    """
    Poll until all instances reach 'terminated' state or timeout elapses.
    We do NOT use boto3.Waiter here because its error messages are cryptic —
    we prefer our own loop with explicit logging.
    """
    log.info("Waiting for %d instance(s) to terminate (timeout=%ds) …",
             len(instance_ids), timeout_s)
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        resp   = ec2_client.describe_instances(InstanceIds=instance_ids)
        states = {}
        for res in resp["Reservations"]:
            for inst in res["Instances"]:
                states[inst["InstanceId"]] = inst["State"]["Name"]

        done  = sum(1 for s in states.values() if s == "terminated")
        other = {iid: s for iid, s in states.items() if s != "terminated"}

        log.info("  terminated=%d/%d  remaining=%s",
                 done, len(instance_ids),
                 ", ".join(f"{iid}:{s}" for iid, s in other.items()) or "—")

        if done == len(instance_ids):
            log.info("All instances terminated.")
            return

        time.sleep(10)

    log.warning("Timed out waiting for termination.  Check the AWS console.")


def delete_security_group(ec2_client, sg_id: str, dry_run: bool) -> None:
    """
    Attempt to delete the security group.  This will fail if any running
    instances still reference it — we handle that gracefully.
    EC2 typically needs 30–60 s after instance termination before an SG
    can be deleted (the ENI detach takes a moment).
    """
    if dry_run:
        log.info("[DRY RUN] Would delete security group %s", sg_id)
        return

    log.info("Deleting security group %s …", sg_id)
    for attempt in range(1, 7):
        try:
            ec2_client.delete_security_group(GroupId=sg_id)
            log.info("Security group %s deleted.", sg_id)
            return
        except ec2_client.exceptions.ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code == "DependencyViolation":
                log.debug("SG still in use (attempt %d/6) — waiting 15 s …", attempt)
                time.sleep(15)
            elif code == "InvalidGroup.NotFound":
                log.info("Security group %s already deleted.", sg_id)
                return
            else:
                log.warning("Could not delete SG %s: %s", sg_id, exc)
                return
    log.warning("Could not delete SG %s after retries — remove it manually.", sg_id)


def archive_state(dry_run: bool) -> None:
    """
    Move instances.json to instances_<timestamp>.json.done so it is no
    longer picked up by monitor/deploy but is preserved for reference.
    """
    if dry_run:
        log.info("[DRY RUN] Would archive %s", cfg.STATE_FILE)
        return

    if not cfg.STATE_FILE.exists():
        return

    ts      = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = cfg.STATE_FILE.parent / f"instances_{ts}.done.json"
    cfg.STATE_FILE.rename(archive)
    log.info("State file archived to %s", archive)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def teardown(
    tags_filter:    list[str] | None,
    collect:        bool,
    collect_files:  list[str],
    collect_dir:    Path,
    keep_sg:        bool,
    dry_run:        bool,
    force:          bool = False,
) -> None:
    """Full teardown flow: collect → gate check → terminate → wait → delete SG → archive state."""
    import boto3

    state, instances = _load_state(tags_filter)
    ec2              = boto3.client("ec2", region_name=cfg.REGION)

    log.info("=" * 60)
    log.info("TEARDOWN  |  project=%s  region=%s", state.get("project"), cfg.REGION)
    log.info("          |  instances=%d  dry_run=%s",
             len(instances), dry_run)
    log.info("=" * 60)

    if not instances:
        log.warning("No instances matched filter %s — nothing to do.", tags_filter)
        return

    # -- Step 1: Collect results (before terminating!) ------------------------─
    if collect and collect_files:
        log.info("-- Step 1 / 5: Collecting result files …")
        collect_results(instances, collect_files, collect_dir)
    else:
        log.info("-- Step 1 / 5: Skipping file collection (use --collect to enable).")

    # -- Step 1b: Pre-termination gate ----------------------------------------─
    # Verify every required artifact was actually downloaded locally before we
    # destroy the instances.  A silent SFTP failure would otherwise cause
    # permanent data loss.  Pass --force to override (e.g. after a manual
    # recovery or if the instance is known to be in a bad state).
    if collect and not force:
        log.info("-- Step 2 / 5: Pre-termination artifact gate …")
        missing_by_tag = verify_collected(instances, collect_dir)
        if missing_by_tag:
            summary = "; ".join(
                f"{tag}: {files}" for tag, files in missing_by_tag.items()
            )
            raise RuntimeError(
                f"Artifact gate FAILED — missing files: {summary}\n"
                "Instances have NOT been terminated.\n"
                "Re-run with --collect to retry, or --force to override."
            )
        log.info("-- Step 2 / 5: Gate passed — all artifacts verified locally.")
    elif force:
        log.warning("-- Step 2 / 5: Gate BYPASSED via --force — skipping artifact check.")
    else:
        log.info("-- Step 2 / 5: Gate skipped (collection not enabled).")

    # -- Step 3: Terminate instances ------------------------------------------─
    log.info("-- Step 3 / 5: Terminating instances …")
    ids = [i["instance_id"] for i in instances]
    terminate_instances(ec2, ids, dry_run)

    # -- Step 4: Wait for terminated ------------------------------------------─
    if not dry_run:
        log.info("-- Step 4 / 5: Waiting for termination …")
        wait_for_terminated(ec2, ids)

    # -- Step 4: Clean up security group --------------------------------------─
    sg_id      = state.get("sg_id", "")
    sg_created = state.get("sg_created", False)

    if sg_id and sg_created and not keep_sg:
        log.info("-- Step 5 / 5: Deleting security group %s (created by standup.py) …", sg_id)
        delete_security_group(ec2, sg_id, dry_run)
    else:
        if keep_sg:
            log.info("-- Step 5 / 5: Keeping security group (--keep-sg).")
        elif not sg_created:
            log.info("-- Step 5 / 5: SG was pre-existing — not deleting.")
        else:
            log.info("-- Step 5 / 5: No SG to clean up.")

    # -- Archive state --------------------------------------------------------─
    # Only archive if we tore down ALL instances (not a subset)
    all_tags   = {i["tag"] for i in json.loads(cfg.STATE_FILE.read_text() if cfg.STATE_FILE.exists() else "{}").get("instances", [])}
    torn_tags  = {i["tag"] for i in instances}
    if torn_tags >= all_tags:
        archive_state(dry_run)
    else:
        log.info("Partial teardown — leaving %s intact.", cfg.STATE_FILE.name)

    log.info("=" * 60)
    log.info("TEARDOWN COMPLETE%s", "  (dry run — no actual changes made)" if dry_run else "")
    log.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Terminate EC2 training instances and clean up resources.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tags",          nargs="+", default=None,
                        help="Only terminate instances with these tags.")
    parser.add_argument("--collect",       action="store_true",
                        help="Download result files before terminating.")
    parser.add_argument("--collect-files", nargs="+",
                        default=["*.pt", "*.csv", "*.png", "*.log"],
                        help="Glob patterns for files to collect (relative to ~/).")
    parser.add_argument("--collect-dir",   default="aws_results",
                        help="Local directory to save collected files.")
    parser.add_argument("--keep-sg",       action="store_true",
                        help="Do not delete the security group even if standup.py created it.")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Show what would be done without actually terminating anything.")
    parser.add_argument("--force",         action="store_true",
                        help="Bypass the pre-termination artifact gate. "
                             "Use only when you intentionally want to terminate "
                             "without verifying all artifacts were collected.")
    args = parser.parse_args()

    try:
        teardown(
            tags_filter=args.tags,
            collect=args.collect,
            collect_files=args.collect_files,
            collect_dir=Path(args.collect_dir),
            keep_sg=args.keep_sg,
            dry_run=args.dry_run,
            force=args.force,
        )
    except Exception as exc:
        log.exception("Teardown failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
