"""
aws/deploy.py — Upload files and start a remote process on EC2 instances.

This script reads aws/instances.json (written by standup.py) and for each
targeted instance:
  1. Waits until SSH becomes available (the instance is 'running' but sshd
     may still be starting, especially right after launch).
  2. Uploads the specified local files via SFTP (paramiko).
  3. Optionally runs a one-time setup command (e.g. pip install).
  4. Starts the main training command in the background via nohup so it
     survives SSH disconnection.

Usage (from project root):
    # Upload the runner + package and start it
    python aws/deploy.py \\
        --files scripts/run_tinycnn.py \\
        --cmd   "python run_tinycnn.py" \\
        --log   tinycnn.log

    # Upload multiple files, install deps first, then run
    python aws/deploy.py \\
        --files scripts/run_resnet.py requirements.txt \\
        --setup "pip install --quiet -r requirements.txt" \\
        --cmd   "python run_resnet.py" \\
        --log   resnet.log

    # Target only specific instances by their tag
    python aws/deploy.py \\
        --tags  tinycnn resnet \\
        --files scripts/run_{tag}.py \\
        --cmd   "python run_{tag}.py" \\
        --log   "{tag}.log"

    # {tag} in --cmd / --log is replaced with each instance's tag at deploy time.

Prerequisites:
    - standup.py has been run and aws/instances.json exists.
    - paramiko is installed:  pip install paramiko
"""

import argparse
import json
import logging
import os
import socket
import sys
import time
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
log = logging.getLogger("deploy")
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("paramiko").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# SSH / SFTP helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_state(tags_filter: list[str] | None) -> list[dict]:
    """
    Read aws/instances.json and return instance records.
    If tags_filter is provided, return only matching entries.
    """
    if not cfg.STATE_FILE.exists():
        raise FileNotFoundError(
            f"{cfg.STATE_FILE} not found.  Run python aws/standup.py first."
        )
    state = json.loads(cfg.STATE_FILE.read_text())
    instances = state.get("instances", [])

    if tags_filter:
        instances = [i for i in instances if i["tag"] in tags_filter]
        if not instances:
            raise ValueError(
                f"No instances found matching tags {tags_filter}.  "
                f"Available: {[i['tag'] for i in state['instances']]}"
            )
    return instances


def _wait_for_ssh(ip: str, tag: str,
                  max_attempts: int = 40,
                  interval_s: int = 15) -> None:
    """
    Poll TCP port 22 until it accepts a connection or max_attempts is reached.
    The instance is 'running' in AWS but sshd may not be listening yet —
    cloud-init can take 1–3 minutes to finish on a fresh boot.
    """
    log.info("[%s] Waiting for SSH on %s:%d …", tag, ip, cfg.SSH_PORT)
    for attempt in range(1, max_attempts + 1):
        try:
            sock = socket.create_connection((ip, cfg.SSH_PORT), timeout=5)
            sock.close()
            log.info("[%s] SSH is up (attempt %d/%d)", tag, attempt, max_attempts)
            return
        except OSError:
            log.debug("[%s] SSH not yet available (attempt %d/%d) — retrying in %ds …",
                      tag, attempt, max_attempts, interval_s)
            time.sleep(interval_s)

    raise TimeoutError(
        f"[{tag}] SSH on {ip}:{cfg.SSH_PORT} did not become available "
        f"after {max_attempts} attempts ({max_attempts * interval_s}s)."
    )


def _make_ssh_client(ip: str, tag: str):
    """
    Create and return an authenticated paramiko SSHClient.
    Uses key-based auth only (no password) — consistent with how EC2 works.
    """
    try:
        import paramiko
    except ImportError:
        raise ImportError(
            "paramiko is required for deploy.py.  "
            "Install it with:  pip install paramiko"
        )

    key_path = Path(cfg.KEY_PATH)
    if not key_path.exists():
        raise FileNotFoundError(
            f"Private key not found at {key_path}.  "
            f"Check EC2_KEY_PATH in aws/.env."
        )

    log.debug("[%s] Loading private key from %s", tag, key_path)
    # Try RSA first, then Ed25519 (covers most key types AWS supports)
    try:
        pkey = paramiko.RSAKey.from_private_key_file(str(key_path))
    except paramiko.ssh_exception.SSHException:
        pkey = paramiko.Ed25519Key.from_private_key_file(str(key_path))

    client = paramiko.SSHClient()
    # Automatically add the host to known_hosts — acceptable for short-lived
    # training instances where the host key changes on every launch.
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    log.info("[%s] Connecting to %s@%s:%d …", tag, cfg.SSH_USER, ip, cfg.SSH_PORT)
    client.connect(
        hostname=ip,
        port=cfg.SSH_PORT,
        username=cfg.SSH_USER,
        pkey=pkey,
        timeout=30,
        banner_timeout=60,   # Deep Learning AMI can be slow on first boot
        auth_timeout=30,
    )
    log.info("[%s] SSH connection established.", tag)
    return client


def _run_remote(client, cmd: str, tag: str, check: bool = True,
                pty: bool = True) -> tuple[int, str, str]:
    """
    Execute a command on the remote instance and return (exit_code, stdout, stderr).
    Streams stdout/stderr to the local log so you can watch progress in real time.
    If check=True, raise RuntimeError on non-zero exit.

    pty=True allocates a pseudo-terminal (good for streaming live output from a
    blocking command like pip install).  pty=False MUST be used when launching
    a background/detached process: a PTY delivers SIGHUP to the whole session
    when the channel closes, which would kill the backgrounded job.
    """
    log.debug("[%s] Remote exec: %s", tag, cmd)
    _, stdout, stderr = client.exec_command(cmd, get_pty=pty, timeout=300)

    # Read output line by line so it appears in logs immediately
    out_lines = []
    for line in iter(stdout.readline, ""):
        line = line.rstrip()
        if line:
            log.debug("[%s] remote> %s", tag, line)
            out_lines.append(line)

    exit_code  = stdout.channel.recv_exit_status()
    err_output = stderr.read().decode(errors="replace").strip()

    if err_output:
        log.debug("[%s] remote stderr> %s", tag, err_output)

    if check and exit_code != 0:
        raise RuntimeError(
            f"[{tag}] Command failed (exit {exit_code}): {cmd}\n"
            f"stderr: {err_output}"
        )

    return exit_code, "\n".join(out_lines), err_output


def _upload_files(client, local_files: list[str], remote_dir: str, tag: str) -> None:
    """
    Upload local_files to remote_dir on the instance via SFTP.
    Creates remote_dir if it doesn't exist.

    Note: SFTP does NOT expand '~' (that's a shell feature).  The SFTP session
    already starts in the user's home directory, so we strip a leading '~/'
    and use a home-relative POSIX path.  We also split on '/' explicitly
    (never os.sep) because the remote host is Linux even when we run on Windows.
    """
    # Normalize the remote directory to a POSIX path SFTP understands.
    # "~/training" -> "training" (relative to home, where SFTP starts)
    # "/opt/foo"   -> "/opt/foo" (absolute, kept as-is)
    rdir = remote_dir.strip()
    if rdir.startswith("~/"):
        rdir = rdir[2:]
    elif rdir == "~":
        rdir = "."

    sftp = client.open_sftp()
    try:
        # Create remote directory tree (mkdir -p equivalent), POSIX-split.
        if rdir not in (".", ""):
            is_absolute = rdir.startswith("/")
            parts = [p for p in rdir.split("/") if p]
            current = "/" if is_absolute else ""
            for part in parts:
                current = f"{current}{part}" if current in ("", "/") else f"{current}/{part}"
                try:
                    sftp.mkdir(current)
                    log.debug("[%s] Created remote dir: %s", tag, current)
                except IOError:
                    pass   # directory already exists

        remote_prefix = "" if rdir in (".", "") else f"{rdir}/"
        for local_path in local_files:
            src  = Path(local_path)
            dest = f"{remote_prefix}{src.name}"
            log.info("[%s] Uploading %s -> %s", tag, src, dest)
            sftp.put(str(src), dest)
            log.info("[%s] Upload complete: %s (%d bytes)", tag, src.name, src.stat().st_size)
    finally:
        sftp.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Per-instance deploy flow
# ═══════════════════════════════════════════════════════════════════════════════

def deploy_to_instance(
    instance: dict,
    local_files: list[str],
    remote_dir: str,
    setup_cmd: str | None,
    main_cmd: str,
    log_file: str,
) -> None:
    """
    Full deploy sequence for a single instance:
      1. Wait for SSH
      2. Upload files
      3. Run setup command (blocking — e.g. pip install)
      4. Launch main command in background via nohup
    """
    tag = instance["tag"]
    ip  = instance["public_ip"]

    if not ip:
        raise ValueError(f"[{tag}] Instance has no public IP — was it launched correctly?")

    # Expand {tag} placeholder in commands (so you can pass --cmd "python train.py --task {tag}")
    main_cmd = main_cmd.replace("{tag}", tag)
    log_file = log_file.replace("{tag}", tag)

    log.info("[%s] -- Starting deploy to %s ------------------------------", tag, ip)

    _wait_for_ssh(ip, tag)

    client = _make_ssh_client(ip, tag)
    try:
        # -- Upload files ----------------------------------------------------─
        if local_files:
            _upload_files(client, local_files, remote_dir, tag)

        # -- Setup command (blocking) ----------------------------------------─
        if setup_cmd:
            log.info("[%s] Running setup: %s", tag, setup_cmd)
            _run_remote(client, setup_cmd, tag)
            log.info("[%s] Setup complete.", tag)

        # -- Main command (non-blocking, fully detached) ----------------------
        # We cd to remote_dir first so relative paths (data/, checkpoints/) work.
        #
        # Two problems must be solved when launching a long-running remote job:
        #   1. The job must survive the SSH channel closing.  setsid + nohup +
        #      '< /dev/null' fully detach it into a new session with no
        #      controlling terminal, immune to the SIGHUP sent on channel close.
        #   2. The exec channel itself must close *immediately* so we are not
        #      blocked waiting for the (long-lived) job to finish.  We wrap the
        #      launch in a brace group whose stdin/stdout/stderr are ALL
        #      redirected away from the channel ('< /dev/null > /dev/null 2>&1').
        #      The PID is written to a file (~/<tag>.pid) instead of the channel,
        #      so once the group backgrounds the job and exits, the channel has
        #      no remaining writers and closes right away.
        # Must run WITHOUT a PTY (pty=False): a PTY would deliver SIGHUP to the
        # whole session on close, killing the job.
        pid_file = f"{log_file}.pid"
        bg_cmd = (
            f"cd {remote_dir} && "
            f"{{ setsid nohup {main_cmd} > ~/{log_file} 2>&1 & echo $! > ~/{pid_file}; }} "
            f"< /dev/null > /dev/null 2>&1"
        )
        log.info("[%s] Launching: %s", tag, main_cmd)
        log.info("[%s] Log file : ~/%s", tag, log_file)
        _run_remote(client, bg_cmd, tag, pty=False)

        # Read back the PID that the launch wrote to the pid file.
        _, pid_out, _ = _run_remote(client, f"cat ~/{pid_file} 2>/dev/null",
                                    tag, check=False, pty=False)
        pid = pid_out.strip().split("\n")[-1] if pid_out.strip() else "?"
        log.info("[%s] Process started with PID %s", tag, pid)

        # Give the process a moment to start, then confirm it is still alive
        # and that the log file was actually created.  This catches immediate
        # crashes (bad import, syntax error) right away instead of later.
        time.sleep(4)
        _, check_out, _ = _run_remote(
            client,
            f"ps -p {pid} > /dev/null 2>&1 && echo ALIVE || echo DEAD; "
            f"test -f ~/{log_file} && echo LOG_OK || echo LOG_MISSING",
            tag, check=False, pty=False,
        )
        status = check_out.replace("\r", " ").replace("\n", " ")
        if "DEAD" in status:
            # Pull whatever landed in the log so the failure is visible
            _, tail_out, _ = _run_remote(
                client, f"tail -n 20 ~/{log_file} 2>/dev/null",
                tag, check=False, pty=False,
            )
            log.error("[%s] Process died immediately. Log tail:\n%s", tag, tail_out)
            raise RuntimeError(f"[{tag}] Training process exited right after launch.")
        log.info("[%s] Confirmed alive (PID %s), log created. Monitor: python aws/monitor.py",
                 tag, pid)

    finally:
        client.close()
        log.debug("[%s] SSH connection closed.", tag)

    log.info("[%s] -- Deploy complete --------------------------------------", tag)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def deploy(
    local_files: list[str],
    main_cmd: str,
    log_file: str       = "{tag}.log",
    remote_dir: str     = "~/training",
    setup_cmd: str | None = None,
    tags_filter: list[str] | None = None,
    parallel: bool      = True,
) -> None:
    """
    Deploy to all (or selected) instances from instances.json.
    Set parallel=True to deploy concurrently (faster for many instances).
    """
    instances = _load_state(tags_filter)
    log.info("Deploying to %d instance(s): %s",
             len(instances), [i["tag"] for i in instances])

    if parallel and len(instances) > 1:
        import threading
        errors = {}

        def _deploy_thread(inst):
            try:
                deploy_to_instance(inst, local_files, remote_dir,
                                   setup_cmd, main_cmd, log_file)
            except Exception as exc:
                errors[inst["tag"]] = exc
                log.error("[%s] Deploy failed: %s", inst["tag"], exc)

        threads = [threading.Thread(target=_deploy_thread, args=(i,), daemon=True)
                   for i in instances]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            raise RuntimeError(f"Deploy failed on {list(errors.keys())}: {errors}")
    else:
        for inst in instances:
            deploy_to_instance(inst, local_files, remote_dir,
                               setup_cmd, main_cmd, log_file)

    log.info("=" * 60)
    log.info("ALL DEPLOYS COMPLETE")
    log.info("Monitor progress:  python aws/monitor.py")
    log.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload files and start training on EC2 instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--files",      nargs="*",  default=[],
                        help="Local file paths to upload (space-separated).")
    parser.add_argument("--cmd",        required=True,
                        help="Command to run remotely (use {tag} for per-instance substitution).")
    parser.add_argument("--log",        default="{tag}.log",
                        help="Remote log filename ({tag} is replaced per instance).")
    parser.add_argument("--remote-dir", default="~/training",
                        help="Remote directory to upload files to and cd into.")
    parser.add_argument("--setup",      default=None,
                        help="Optional setup command to run before the main cmd (blocking).")
    parser.add_argument("--tags",       nargs="+", default=None,
                        help="Only deploy to instances with these tags.")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Deploy sequentially instead of in parallel.")
    args = parser.parse_args()

    try:
        deploy(
            local_files=args.files,
            main_cmd=args.cmd,
            log_file=args.log,
            remote_dir=args.remote_dir,
            setup_cmd=args.setup,
            tags_filter=args.tags,
            parallel=not args.no_parallel,
        )
    except Exception as exc:
        log.exception("Deploy failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
