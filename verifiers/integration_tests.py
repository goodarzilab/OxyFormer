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
import subprocess
import sys
import tempfile
import time

PYTHON = "/mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python"

# The suite runs on a compute node through Slurm, never on the login node (owner instruction, 2026-10-05).
# The candidate merge sits in node-local /tmp on the login node, so it is staged through shared storage and copied
# into the compute node's own /tmp, keeping paths as short as the original checkout (some tests bound socket path
# length). The job records pytest's own exit status in the shared staging directory; the verdict is taken from that
# record, never from srun's status, so a Slurm or launch failure that runs no tests cannot pass. The Slurm time limit
# is generous because the merge operator's --verification-timeout is the real bound on the whole run.
SHARED_TMP = "/mnt/weka/home/hgoodarzi/oxyformer-swarm/verify-tmp"
STALE_SECONDS = 24 * 3600
JOB = (
    'set -u; d=$(mktemp -d /tmp/v-XXXXXX); trap \'rm -rf "$d"\' EXIT HUP INT TERM; '
    'cp -a "$STAGED" "$d/tree" && cd "$d/tree" || exit 97; '
    '"$PYTHON" -B -m pytest -q; rc=$?; echo "$rc" > "$RESULT"; exit $rc'
)


def _stop(signum, frame):
    raise SystemExit(1)  # unwind through finally so the staged copy is removed


def _sweep_stale():
    now = time.time()
    for name in os.listdir(SHARED_TMP):
        path = os.path.join(SHARED_TMP, name)
        try:
            if name.startswith("merge-verify-") and now - os.stat(path).st_mtime > STALE_SECONDS:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGHUP, _stop)
os.makedirs(SHARED_TMP, exist_ok=True)
_sweep_stale()
work = tempfile.mkdtemp(prefix="merge-verify-", dir=SHARED_TMP)
status = None
try:
    staged = os.path.join(work, "tree")
    result = os.path.join(work, "pytest-exit-status")
    shutil.copytree(os.getcwd(), staged, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
    srun = ["srun", "--partition=standard", "--account=root", "--nodes=1", "--ntasks=1", "--cpus-per-task=8",
            "--mem=32G", "--time=12:00:00", "--export=ALL", "--chdir=/tmp", "--job-name=oxyformer-merge-verify"]
    env = dict(os.environ, PYTHONPATH="src", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               STAGED=staged, PYTHON=PYTHON, RESULT=result)
    env.pop("SLURM_EXIT_ERROR", None)
    subprocess.run(srun + ["bash", "-c", JOB], env=env)
    try:
        with open(result) as handle:
            status = int(handle.read().strip())
    except (OSError, ValueError):
        status = None  # pytest never reported: launch, allocation or copy failure
finally:
    shutil.rmtree(work, ignore_errors=True)
sys.exit(0 if status in (0, 5) else 1)
