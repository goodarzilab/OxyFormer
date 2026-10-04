#!/mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python
"""Merge precondition for goodarzilab/OxyFormer (claim: integration-tests).

hanig-orchestrate's merge_unit.py runs this pinned file inside a disposable candidate merge of a unit's PR head
into the exact dev commit, and treats exit status 0 as a pass. It runs the whole pytest suite with the run's
environment, CPU only, the same way .github/workflows/ci.yml does, so a merge is refused when the merged tree's
tests fail even if each branch passed on its own. Exit status 5 (no tests collected) counts as a pass, as in CI.
"""
import os
import subprocess
import sys

PYTHON = "/mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python"

env = dict(os.environ, PYTHONPATH="src", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
try:
    run = subprocess.run([PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider"], env=env, timeout=840)
except subprocess.TimeoutExpired:
    print("integration-tests: pytest exceeded 840 s", file=sys.stderr)
    sys.exit(1)
sys.exit(0 if run.returncode in (0, 5) else 1)
