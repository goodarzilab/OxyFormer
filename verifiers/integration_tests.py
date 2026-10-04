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

# No time limit of its own: the merge operator's --verification-timeout bounds the run.
env = dict(os.environ, PYTHONPATH="src", CUDA_VISIBLE_DEVICES="")
run = subprocess.run([PYTHON, "-m", "pytest", "-q"], env=env)
sys.exit(0 if run.returncode in (0, 5) else 1)
