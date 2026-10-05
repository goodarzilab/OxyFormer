#!/mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python
"""Merge precondition for goodarzilab/OxyFormer (claim: integration-tests).

hanig-orchestrate's merge_unit.py runs this pinned file inside a disposable candidate merge of a unit's PR head
into the exact dev commit, and treats exit status 0 as a pass. It runs the whole pytest suite with the run's
environment, CPU only, the same way .github/workflows/ci.yml does, so a merge is refused when the merged tree's
tests fail even if each branch passed on its own. Exit status 5 (no tests collected) counts as a pass, as in CI.
"""
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

PYTHON = "/mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python"

# The suite runs on a compute node through Slurm, never on the login node (owner instruction, 2026-10-05).
# The candidate merge sits in node-local /tmp on the login node, so it is staged through shared storage and copied
# into the compute node's own /tmp, keeping paths as short as the original checkout (some tests bound socket path
# length). The job records pytest's exit status in the shared staging directory and exits with that same status; the
# merge passes only when the recorded status is 0 or 5 AND srun returned that same status, so neither a Slurm or
# launch failure that runs no tests nor a rewritten status file can pass a failing run. Termination signals are
# forwarded to srun, which cancels the job step and lets its trap remove the compute-node copy; the shared copy is
# removed in finally, and staging directories whose owning process is gone are swept. SIGKILL cannot be intercepted:
# an uncatchable kill may leave a copy until the next sweep (shared) or the node's /tmp cleanup (compute node).
# The Slurm time limit is generous because the merge operator's --verification-timeout bounds the whole run.
SHARED_TMP = "/mnt/weka/home/hgoodarzi/oxyformer-swarm/verify-tmp"
HOST = socket.gethostname()
JOB = (
    'set -u; d=$(mktemp -d /tmp/v-XXXXXX); trap \'rm -rf "$d"\' EXIT HUP INT TERM; '
    'cp -a "$STAGED" "$d/tree" && cd "$d/tree" || exit 97; '
    'env -u RESULT -u STAGED "$PYTHON" -B -m pytest -q; rc=$?; echo "$rc" > "$RESULT"; exit $rc'
)


def _owner_alive(path):
    try:
        host, pid = Path(path, "owner").read_text().split()
        if host != HOST:
            return True  # another host's verification: never touch it
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except (OSError, ValueError):
        return False


def _sweep_stale():
    for name in os.listdir(SHARED_TMP):
        path = os.path.join(SHARED_TMP, name)
        if name.startswith("merge-verify-") and not _owner_alive(path):
            shutil.rmtree(path, ignore_errors=True)


os.makedirs(SHARED_TMP, exist_ok=True)
_sweep_stale()
work = tempfile.mkdtemp(prefix="merge-verify-", dir=SHARED_TMP)
Path(work, "owner").write_text(f"{HOST} {os.getpid()}\n")
child = None


def _stop(signum, frame):
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)  # srun cancels the job step; the job's trap removes its copy
    raise SystemExit(1)


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGHUP, _stop)
status = srun_status = None
try:
    staged = os.path.join(work, "tree")
    result = os.path.join(work, "pytest-exit-status")
    shutil.copytree(os.getcwd(), staged, symlinks=True)
    srun = ["srun", "--partition=standard", "--account=root", "--nodes=1", "--ntasks=1", "--cpus-per-task=8",
            "--mem=32G", "--time=12:00:00", "--export=ALL", "--chdir=/tmp", "--job-name=oxyformer-merge-verify"]
    env = dict(os.environ, PYTHONPATH="src", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               STAGED=staged, PYTHON=PYTHON, RESULT=result)
    env.pop("SLURM_EXIT_ERROR", None)
    child = subprocess.Popen(srun + ["bash", "-c", JOB], env=env)
    srun_status = child.wait()
    try:
        status = int(Path(result).read_text().strip())
    except (OSError, ValueError):
        status = None  # pytest never reported: launch, allocation or copy failure
finally:
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, signal.SIG_IGN)  # cleanup must not be interrupted
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            child.kill()
    shutil.rmtree(work, ignore_errors=True)
sys.exit(0 if status in (0, 5) and srun_status == status else 1)
