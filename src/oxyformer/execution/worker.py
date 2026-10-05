"""Run a CLI stage to process exit before the parent checks and publishes it.

This is a completion boundary, not filesystem confinement. The worker writes
only its unsealed result; the parent validates it after normal Python thread,
child-process and temporary-object cleanup has finished.
"""
import atexit
import importlib
import os
from pathlib import Path
import subprocess
import sys

from oxyformer.contracts import StageRequest, StageResult
from oxyformer.provenance import require
from .identity import verify_module_origins
from .paths import atomic_write, isolated_caches

WORKER_RESULT = '_execution/worker-result.json'


def execute(request, module_name, repo):
    """Wait for the stage interpreter, including its finalizers, before return."""
    environment = dict(os.environ, PYTHONPATH=str(Path(repo) / 'src'))
    process = subprocess.run([sys.executable, '-m', 'oxyformer.execution.worker',
                              str(Path(request.output_dir) / '_execution/request.json'),
                              str(repo), module_name], cwd=repo, env=environment)
    require(process.returncode == 0, f'stage worker exited with status {process.returncode}')
    return StageResult.from_json((Path(request.output_dir) / WORKER_RESULT).read_text())


def _wait_children():
    # Registered before importing the stage so its later atexit callbacks run
    # first. Python joins non-daemon threads before atexit. Reap any unjoined
    # direct subprocesses too; a mere run_stage return is not process completion.
    while True:
        try:
            os.waitpid(-1, 0)
        except InterruptedError:
            continue
        except ChildProcessError:
            return


def main():
    request_path, repository, module_name = sys.argv[1:]
    request = StageRequest.from_json(Path(request_path).read_text())
    try:
        request.verify_inputs()
        caches = isolated_caches(request.output_dir)
        caches.__enter__()
        atexit.register(caches.__exit__, None, None, None)
        atexit.register(_wait_children)
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


if __name__ == '__main__':
    raise SystemExit(main())
