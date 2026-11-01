#!/mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python
"""Merge precondition for goodarzilab/OxyFormer (claim: integration-tests).

hanig-orchestrate's merge_unit.py runs this pinned file inside a disposable candidate merge of a unit's PR head
into the exact dev commit, and treats exit status 0 as a pass. It runs the whole pytest suite with the run's
environment, CPU only, the same way .github/workflows/ci.yml does, so a merge is refused when the merged tree's
tests fail even if each branch passed on its own. Exit status 5 (no tests collected) counts as a pass, as in CI.
Inside the disposable copy the candidate's index entries are read from its metadata, the metadata is discarded, and a
fresh repository is built entry by entry from the working files (--no-filters, literal paths, empty template, empty HOME)
before pytest, so HEAD matches the tree byte for byte as in CI; commit author, subject and history are synthetic, so tests
must not depend on commit metadata or history (CI on the real commit covers that).
"""
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PYTHON = "/mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python"

# The suite runs on a compute node through Slurm, never on the login node (owner instruction, 2026-10-05).
# The candidate merge sits in node-local /tmp on the login node, so it is staged through shared storage and copied
# into the compute node's own /tmp, keeping paths as short as the original checkout (some tests bound socket path
# length). The job records pytest's exit status in the shared staging directory and exits with that same status; the
# merge passes only when the recorded status is 0 or 5 AND srun returned that same status, so neither a Slurm or
# launch failure that runs no tests nor a rewritten status file can pass a failing run. Termination signals are
# forwarded to srun, which cancels the job step and lets its trap remove the compute-node copy; the shared copy is
# removed in finally, and this host's staging directories whose owner process is gone are swept (one without an owner
# record yet is left alone for a day, since it may still be starting; PID reuse can only delay a sweep). Slurm job
# variables are cleared before pytest so the suite sees the same environment as an in-place run. SIGKILL cannot be intercepted:
# an uncatchable kill may leave a copy until the next sweep (shared) or the node's /tmp cleanup (compute node).
# The verifier's environment is the merge operator's; options that point at login-node-local files (for example an
# inherited PYTEST_ADDOPTS -c path) are the operator's responsibility. The Slurm time limit is generous because the
# merge operator's --verification-timeout bounds the whole run.
SHARED_TMP = "/mnt/weka/home/hgoodarzi/oxyformer-swarm/verify-tmp"
UNOWNED_GRACE_SECONDS = 24 * 3600
HOST = socket.gethostname()
JOB = (
    'set -u; d=$(mktemp -d /tmp/v-XXXXXX); trap \'chmod -R u+w "$d" 2>/dev/null; rm -rf "$d"\' EXIT HUP INT QUIT TERM USR1 USR2 ALRM XCPU XFSZ VTALRM PROF; '
    # The staged root's CONTENTS are copied (a symlinked root is never followed into). HEAD must then describe the tree
    # under test, as CI's checkout does (HEAD-anchored inputs such as configs/approvals.yaml are read with expected_bytes
    # from HEAD): reconstruct() below, run from this same pinned file, reads the candidate's tracked entries from the copied
    # metadata, discards that metadata, and builds a fresh repository whose index is written entry by entry from the
    # working files with --no-filters and literal paths, so no attribute, filter, template, hook, signing, encoding or
    # pathspec rule can alter a byte or reach outside the copy. Commit author, subject and history are synthetic; tests
    # must not read commit metadata or history (CI on the real commit covers that).
    'mkdir -p "$d/tree" && cp -a "$OXYFORMER_VERIFY_STAGED/." "$d/tree/" && cd "$d/tree" || exit 97; '
    '"$OXYFORMER_VERIFY_PYTHON" "$OXYFORMER_VERIFY_RECONSTRUCT" "$d/tree" || exit 96; '
    'py=$OXYFORMER_VERIFY_PYTHON; out=$OXYFORMER_VERIFY_RESULT; '
    'if [ -n "${OXYFORMER_VERIFY_BASH_ENV+x}" ]; then export BASH_ENV=$OXYFORMER_VERIFY_BASH_ENV; fi; '
    'unset OXYFORMER_VERIFY_PYTHON OXYFORMER_VERIFY_RESULT OXYFORMER_VERIFY_STAGED OXYFORMER_VERIFY_BASH_ENV OXYFORMER_VERIFY_RECONSTRUCT; '
    'for v in $(compgen -e); do case "$v" in SLURM_*|SRUN_*) unset "$v";; esac; done; '
    '"$py" -m pytest -q; rc=$?; echo "$rc" > "$out"; exit $rc'
)


def _orphaned(path):
    """Only this host's verifications are judged; a directory without an owner record may be mid-creation."""
    try:
        host, pid = Path(path, "owner").read_text().split()
    except (OSError, ValueError):
        try:
            return time.time() - os.stat(path).st_mtime > UNOWNED_GRACE_SECONDS
        except OSError:
            return False
    if host != HOST:
        return False  # verifiers run on the login node; never judge another host's process table
    try:
        os.kill(int(pid), 0)
        return False
    except ProcessLookupError:
        return True
    except (OSError, ValueError):
        return False


def _sweep_stale():
    for name in os.listdir(SHARED_TMP):
        path = os.path.join(SHARED_TMP, name)
        if name.startswith("merge-verify-") and _orphaned(path):
            shutil.rmtree(path, ignore_errors=True)


def _git(repo, home, *args, stdin=None):
    env = {"PATH": os.defpath, "HOME": home, "XDG_CONFIG_HOME": os.path.join(home, ".config"),
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_ATTR_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0", "GIT_AUTHOR_NAME": "merge-verifier",
           "GIT_AUTHOR_EMAIL": "merge-verifier@localhost", "GIT_COMMITTER_NAME": "merge-verifier",
           "GIT_COMMITTER_EMAIL": "merge-verifier@localhost", "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
           "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z", "LC_ALL": "C"}
    # No content-conversion override here: status must judge the copied checkout under its own settings (for example
    # core.autocrlf=true legitimately checks out LF blobs as CRLF); blobs are hashed with --no-filters regardless.
    cmd = ["git", "--no-optional-locks", "-c", "core.hooksPath=" + os.devnull, "-c", "commit.gpgsign=false",
           "-c", "gpg.format=openpgp", "-c", "core.fsmonitor=false", "-c", "protocol.allow=never",
           "--literal-pathspecs", *args]
    proc = subprocess.run(cmd, cwd=repo, env=env, input=stdin, capture_output=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"reconstruct: git {args[0]} failed: {proc.stderr.decode(errors='replace')[:300]}")
    return proc.stdout


def reconstruct(tree):
    """Make HEAD describe the working tree byte for byte, without git's content-conversion layer.

    The candidate checkout is produced by the merge operator as ``git merge --no-commit`` into the exact target commit,
    so its index IS the candidate tree and the working files equal the index; both are checked here. The stage-0 index
    entries are read through the copied metadata (a self-contained ``.git`` directory or an ordinary linked-worktree
    pointer file; a symlinked ``.git`` is refused), that local metadata is discarded (permissions repaired without
    following symlinks; an external gitdir behind a pointer file is never touched), and a fresh repository (empty
    template, no inherited configuration, an empty HOME so no per-user attributes apply) receives an index written from
    the working files: regular files and symlinks hashed with --no-filters, literal paths. Then write-tree, commit-tree
    and update-ref. Submodules are outside this verifier's envelope and are refused explicitly.
    """
    tree = os.path.realpath(tree)
    meta = os.path.join(tree, ".git")
    if os.path.islink(meta) or not (os.path.isdir(meta) or os.path.isfile(meta)):
        raise SystemExit("reconstruct: the copied metadata must be a .git directory or an ordinary pointer file")
    home = tempfile.mkdtemp(prefix="verify-home-")
    try:
        # Content-based (git status refreshes stat information in memory, so a copied tree with fresh inodes compares
        # by content): any modified, deleted or untracked non-ignored path means the index is not the tree under test.
        # --no-optional-locks: status refreshes stat information in memory only and never writes an index (relevant when
        # the copied metadata is a pointer file naming an external gitdir). porcelain v1 -z records are "XY path", with
        # renames and copies ("R"/"C" in X) followed by a second NUL-terminated source path that is not a record.
        status = _git(tree, home, "status", "--porcelain=v1", "--untracked-files=all", "--no-renames", "-z")
        drift = []
        fields = status.split(b"\0")
        i = 0
        while i < len(fields):
            rec = fields[i]
            i += 1
            if not rec:
                continue
            if rec[:1] in (b"R", b"C"):
                i += 1   # skip the source path field of a rename/copy record
            # X is index vs HEAD (the staged candidate merge, expected); Y is worktree vs index.
            if rec[1:2] != b" " or rec[:2] == b"??":
                drift.append(rec[3:])
        if drift:
            raise SystemExit("reconstruct: the index does not describe the working tree (the merge operator stages the "
                             "candidate); differing paths: " + ", ".join(x.decode(errors="replace") for x in drift[:10]))
        raw = _git(tree, home, "ls-files", "-s", "-z", "--full-name")
        entries = []
        for record in raw.split(b"\0"):
            if not record:
                continue
            meta_fields, path = record.split(b"\t", 1)
            mode, sha, stage = meta_fields.split(b" ")
            if stage != b"0":
                raise SystemExit("reconstruct: unmerged index entry " + path.decode(errors="replace"))
            if mode == b"160000":
                raise SystemExit("reconstruct: submodules are outside this verifier's envelope: " + path.decode(errors="replace"))
            entries.append((mode, sha, path))
        if os.path.isdir(meta):
            for root, dirs, files in os.walk(meta, topdown=True):
                for name in dirs + files:
                    full = os.path.join(root, name)
                    if not os.path.islink(full):
                        try:
                            os.chmod(full, 0o700 if name in dirs else 0o600)
                        except OSError:
                            pass
            os.chmod(meta, 0o700)
            shutil.rmtree(meta)
        else:
            os.remove(meta)   # a linked-worktree pointer file: the external gitdir is left exactly as it was
        _git(tree, home, "init", "-q", "--template=")
        index = bytearray()
        for mode, sha, path in entries:
            full = os.path.join(tree, path.decode("utf-8", "surrogateescape"))
            if mode == b"120000":
                if os.path.islink(full):
                    target = os.readlink(full).encode("utf-8", "surrogateescape")
                else:   # core.symlinks=false checkout: the link target is the file's content
                    with open(full, "rb") as fh:
                        target = fh.read()
                blob = _git(tree, home, "hash-object", "-w", "-t", "blob", "--no-filters", "--stdin", stdin=target).strip()
            elif mode in (b"100644", b"100755"):
                with open(full, "rb") as fh:
                    blob = _git(tree, home, "hash-object", "-w", "-t", "blob", "--no-filters", "--stdin", stdin=fh.read()).strip()
            else:
                raise SystemExit("reconstruct: unsupported index mode " + mode.decode())
            index += mode + b" " + blob + b"\t" + path + b"\0"
        _git(tree, home, "update-index", "-z", "--index-info", stdin=bytes(index))
        tree_id = _git(tree, home, "write-tree").strip()
        commit = _git(tree, home, "commit-tree", tree_id, "-m", "candidate merge under verification").strip()
        _git(tree, home, "update-ref", "HEAD", commit.decode())
    finally:
        shutil.rmtree(home, ignore_errors=True)


if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
    reconstruct(sys.argv[1])
    sys.exit(0)


work = child = None


def _stop(signum, frame):
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)  # srun cancels the job step; the job's trap removes its copy
    raise SystemExit(1)


HANDLED = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGQUIT, signal.SIGUSR1, signal.SIGUSR2,
           signal.SIGALRM, signal.SIGXCPU, signal.SIGXFSZ, signal.SIGVTALRM, signal.SIGPROF)
for _sig in HANDLED:
    signal.signal(_sig, _stop)
status = srun_status = None
try:
    os.makedirs(SHARED_TMP, exist_ok=True)
    _sweep_stale()
    work = tempfile.mkdtemp(prefix="merge-verify-", dir=SHARED_TMP)
    Path(work, "owner").write_text(f"{HOST} {os.getpid()}\n")
    staged = os.path.join(work, "tree")
    result = os.path.join(work, "pytest-exit-status")
    reconstruct_script = os.path.join(work, "reconstruct.py")
    shutil.copyfile(os.path.abspath(__file__), reconstruct_script)   # the same pinned bytes run the reconstruction
    shutil.copytree(os.getcwd(), staged, symlinks=True)
    srun = ["srun", "--partition=standard", "--account=root", "--nodes=1", "--ntasks=1", "--cpus-per-task=8",
            "--mem=32G", "--time=12:00:00", "--export=ALL", "--chdir=/tmp", "--job-name=oxyformer-merge-verify"]
    # Job parameters use a private prefix and are removed before pytest, so the suite sees the original environment.
    env = dict(os.environ, PYTHONPATH="src", CUDA_VISIBLE_DEVICES="", OXYFORMER_VERIFY_STAGED=staged,
               OXYFORMER_VERIFY_PYTHON=PYTHON, OXYFORMER_VERIFY_RESULT=result,
               OXYFORMER_VERIFY_RECONSTRUCT=reconstruct_script)
    env.pop("SLURM_EXIT_ERROR", None)
    job_env = {k: v for k, v in env.items() if k != "BASH_ENV"}  # the wrapper shell must not source startup files
    if "BASH_ENV" in env:
        job_env["OXYFORMER_VERIFY_BASH_ENV"] = env["BASH_ENV"]
    # Block handled signals while srun starts so a signal cannot arrive before the child is tracked.
    signal.pthread_sigmask(signal.SIG_BLOCK, HANDLED)
    try:
        # The child must not inherit the blocked mask, or srun could not receive the forwarded SIGTERM.
        child = subprocess.Popen(srun + ["bash", "--noprofile", "--norc", "-c", JOB], env=job_env,
                                 preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_UNBLOCK, HANDLED))
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, HANDLED)
    srun_status = child.wait()
    try:
        status = int(Path(result).read_text().strip())
    except (OSError, ValueError):
        status = None  # pytest never reported: launch, allocation or copy failure
finally:
    for sig in HANDLED:
        signal.signal(sig, signal.SIG_IGN)  # cleanup must not be interrupted
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            child.kill()
    if work is not None:
        shutil.rmtree(work, ignore_errors=True)
sys.exit(0 if status in (0, 5) and srun_status == status else 1)
