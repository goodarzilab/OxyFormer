#!/mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python
"""Merge precondition for goodarzilab/OxyFormer (claim: integration-tests).

hanig-orchestrate's merge_unit.py runs this pinned file inside a disposable candidate merge of a unit's PR head
into the exact dev commit, and treats exit status 0 as a pass. It runs the whole pytest suite with the run's
environment, CPU only, the same way .github/workflows/ci.yml does, so a merge is refused when the merged tree's
tests fail even if each branch passed on its own. Exit status 5 (no tests collected) counts as a pass, as in CI.
"""
import os
import shutil
import subprocess
import sys
import tempfile

PYTHON = "/mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python"

# The suite runs on a compute node through Slurm, never on the login node (owner instruction, 2026-10-05).
# The candidate merge sits in node-local /tmp on the login node, so it is staged through shared storage and then copied
# into the compute node's own /tmp, keeping paths as short as the original checkout (some tests bound socket path
# length). srun blocks until the job ends and returns its exit status. The merge operator's --verification-timeout bounds
# the whole run.
SHARED_TMP = "/mnt/weka/home/hgoodarzi/oxyformer-swarm/verify-tmp"
JOB = (
    'set -u; d=$(mktemp -d /tmp/v-XXXXXX); cp -a "$STAGED" "$d/tree" && cd "$d/tree" && '
    '"$PYTHON" -B -m pytest -q -p no:cacheprovider; rc=$?; cd /; rm -rf "$d"; exit $rc'
)
os.makedirs(SHARED_TMP, exist_ok=True)
work = tempfile.mkdtemp(prefix="merge-verify-", dir=SHARED_TMP)
try:
    staged = os.path.join(work, "tree")
    shutil.copytree(os.getcwd(), staged, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
    srun = ["srun", "--partition=standard", "--account=root", "--nodes=1", "--ntasks=1", "--cpus-per-task=8",
            "--mem=32G", "--time=01:30:00", "--export=ALL", "--chdir=/tmp", "--job-name=oxyformer-merge-verify"]
    env = dict(os.environ, PYTHONPATH="src", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               STAGED=staged, PYTHON=PYTHON)
    run = subprocess.run(srun + ["bash", "-c", JOB], env=env)
finally:
    shutil.rmtree(work, ignore_errors=True)
sys.exit(0 if run.returncode in (0, 5) else 1)
