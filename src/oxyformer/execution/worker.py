"""Run a CLI stage to process exit before the parent checks and publishes it.

This is a completion boundary, not filesystem confinement. The worker writes
only its unsealed result. A separate Linux subreaper waits for the scientific
interpreter to close its descriptors and finish finalizers, then reaps adopted
descendants before the parent validates and publishes. It confines no writes.
"""
import atexit
import ctypes
import importlib
import os
from pathlib import Path
import subprocess
import signal
import sys

from oxyformer.contracts import StageRequest, StageResult
from oxyformer.provenance import require
from .identity import code_identity, verify_module_origins
from .integrity import read_regular, verify_inputs
from .paths import atomic_write, isolated_caches

WORKER_RESULT = '_execution/worker-result.json'


def execute(request, module_name, repo):
    """Wait for the stage interpreter, including its finalizers, before return."""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(Path(repo) / 'src'))
    process = subprocess.Popen([sys.executable, '-m', 'oxyformer.execution.worker',
                                str(Path(request.output_dir) / '_execution/request.json'),
                                str(repo), module_name], cwd=repo, env=environment)
    try:
        process.wait()
    except BaseException:
        # Do not let an interrupted caller publish/check while its worker is
        # still active. The lifecycle process forwards termination and reaps.
        process.send_signal(signal.SIGTERM)
        while True:
            try:
                process.wait()
                break
            except KeyboardInterrupt:
                continue
        raise
    require(process.returncode == 0, f'stage worker exited with status {process.returncode}')
    return StageResult.from_json(read_regular(Path(request.output_dir) / WORKER_RESULT))


def _reap_descendants():
    while True:
        try:
            os.waitpid(-1, 0)
        except InterruptedError:
            continue
        except ChildProcessError:
            return


def supervise(request_path, repository, module_name):
    # Adoption is set before any stage work starts. This dedicated process does
    # not import the stage or own its resource-tracker pipes. Waiting here lets
    # the scientific interpreter perform normal shutdown before helper reaping.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        error = ctypes.get_errno()
        raise OSError(error, 'cannot enable child-subreaper: ' + os.strerror(error))
    stage = None
    interrupted = []

    def terminate(signum, frame):
        interrupted.append(signum)
        if stage is not None:
            stage.send_signal(signum)

    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    stage = subprocess.Popen([sys.executable, '-m', 'oxyformer.execution.worker',
                              '--stage-process', request_path, repository, module_name], cwd=repository)
    if interrupted:
        stage.send_signal(interrupted[-1])
    code = stage.wait()
    # Only the stage knows its helpers' exit conventions (grep uses 1 for no
    # matches). Reap every helper for completion, without judging its result.
    _reap_descendants()
    return 1 if interrupted or code != 0 else 0


def stage_main(request_path, repository, module_name):
    request = StageRequest.from_json(read_regular(request_path))
    try:
        require(code_identity(repository, request.output_dir) == request.code_identity,
                'worker code identity changed')
        verify_module_origins(repository)
        verify_inputs(request)
        caches = isolated_caches(request.output_dir)
        caches.__enter__()
        atexit.register(caches.__exit__, None, None, None)
        try:
            module = importlib.import_module(module_name)
        except (ImportError, FileNotFoundError) as exc:
            result = StageResult(request_hash=request.content_hash, status='blocked', artifacts=(),
                                 message=str(exc).strip() or type(exc).__name__)
        else:
            verify_module_origins(repository)
            require(callable(getattr(module, 'run_stage', None)), 'stage has no run_stage(StageRequest)')
            result = module.run_stage(request)
            verify_module_origins(repository)
        require(isinstance(result, StageResult), 'stage did not return StageResult')
    except BaseException as exc:
        result = StageResult(request_hash=request.content_hash, status='fail', artifacts=(),
                             message=str(exc).strip() or type(exc).__name__)
    atomic_write(request.output_dir, WORKER_RESULT, result.to_json())
    return 0


def main():
    args = sys.argv[1:]
    if args and args[0] == '--stage-process':
        return stage_main(*args[1:])
    return supervise(*args)


if __name__ == '__main__':
    raise SystemExit(main())
