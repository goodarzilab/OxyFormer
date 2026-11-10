"""Optional, failure-isolated run telemetry. Never part of scientific state."""
from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import signal
import threading
from contextlib import contextmanager
import subprocess
import sys
import tempfile
import time

import torch

from oxyformer.training.checkpoint import capture_rng, restore_rng

_SECRET_FIELD = re.compile(r"(?:api[_-]?key|(?:access|auth|refresh|bearer|session|client|identity|id)[_-]?token|authorization|^token$|^key$|password|secret|credential)", re.I)


def sanitize(value):
    if isinstance(value, dict):
        return {sanitize(str(sanitize(k))): ('[redacted]' if _SECRET_FIELD.search(str(k)) else sanitize(v))
                for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        key = os.environ.get('WANDB_API_KEY')
        if not key:
            return value
        if key.isdecimal():
            canonical = key.lstrip('0') or '0'
            value = re.sub(r'[0-9]+', lambda match: '[redacted]'
                           if (match[0].lstrip('0') or '0') == canonical else match[0], value)
        return value.replace(key, '[redacted]')
    if isinstance(value, (int, float)):
        key = os.environ.get('WANDB_API_KEY')
        if key and ((key.isdecimal() and value == int(key)) or key in str(value)):
            return '[redacted]'
    return value


def noop(event):
    pass


def validate_hook(hook):
    if hook is not None and hook is not noop and type(hook) is not WandbLogger:
        raise TypeError('run_logger must be a WandbLogger, noop, or None')


def inspect_config(hook, config):
    validate_hook(hook)
    if hook is None or hook is noop:
        return
    unknown = set(vars(config)) - CONFIG_SCHEMA[type(config).__name__].keys()
    if unknown:
        hook.logging_ok = False
        hook.refusal = 'refused: undeclared config key ' + sorted(unknown)[0]


def emit(hook, event, device='cpu'):
    validate_hook(hook)
    if hook is None or hook is noop:
        return
    rng = None
    try:
        rng = capture_rng(device)
        hook(sanitize(event))
    except Exception:
        hook.logging_ok = False
    finally:
        if rng is not None:
            try:
                restore_rng(rng, device)
            except Exception:
                hook.logging_ok = False


def step_event(hook, *, model, loss, learning_rate, samples, started, device, **context):
    if hook is None or hook is noop:
        return
    try:
        gradients = [p.grad.detach().double().norm() for p in model.parameters()
                     if p.grad is not None]
        norm = torch.stack(gradients).norm().item() if gradients else 0.
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
            memory = torch.cuda.max_memory_allocated(device)
        else:
            memory = 0
        event = dict(event='train_step', train_loss=None if loss is None else float(loss.detach()),
            learning_rate=learning_rate, gradient_norm=norm,
            samples_per_second=samples / max(time.monotonic() - started, 1e-12),
            gpu_peak_bytes=memory, **context)
        emit(hook, event, device)
    except Exception:
        if isinstance(hook, WandbLogger):
            hook.logging_ok = False


# Every declared runtime key has an explicit projection policy. New fields fail closed.
CONFIG_SCHEMA = {
    'FitConfig': dict.fromkeys(('entity_graph inner fold policy policy_covariates treatment_design '
        'feature_kinds families county_field exposure_assignment_level output_dir family '
        'population_field settings study stopping_ids ssl_epochs predecessor stop_request '
        'max_batches slice_seconds checkpoint_margin_seconds device deadline_timing').split(), 'included'),
    'PretrainConfig': dict.fromkeys(('settings output_dir predecessor slice_seconds '
        'checkpoint_margin_seconds max_batches stop_request controller_state device finish_logging').split(), 'included')}
CONFIG_SCHEMA['FitConfig']['data'] = 'hashed'
for _schema in CONFIG_SCHEMA.values():
    _schema['run_logger'] = 'excluded'


def stop_state(request, *, reason=None):
    if reason == 'requested':
        return 'honoured'
    # Never invoke a requested property: deadline proxies can mutate accounting.
    return 'requested' if request is not None and vars(request).get('requested') is True else 'none'


def project_config(config):
    schema = CONFIG_SCHEMA[type(config).__name__]
    names = {f.name for f in fields(config)} | set(getattr(config, '__dict__', {}))
    if names != schema.keys():
        raise ValueError('unclassified configuration keys: ' + str(names ^ schema.keys()))
    values = {}
    for name, policy in schema.items():
        if policy in ('excluded', 'hashed'):
            continue
        value = getattr(config, name)
        if name == 'stop_request':
            value = stop_state(value)
        elif name == 'predecessor':
            value = getattr(value, 'checkpoint', value)
        elif name == 'deadline_timing' and value is not None:
            value = value.report()
        values[name] = value.to_dict() if hasattr(value, 'to_dict') else value
    return values


def window_start(hook, device):
    if hook is not None and hook is not noop and torch.device(device).type == 'cuda':
        try:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            if isinstance(hook, WandbLogger):
                hook.logging_ok = False
    return time.monotonic()


GIT_TIMEOUT = 10
FINISH_TIMEOUT = 180


class _FinishTimeout(BaseException):
    pass


@contextmanager
def _finish_deadline():
    # Linux main-thread delivery only. Refuse telemetry in threaded callers or
    # while another component owns SIGALRM, rather than replacing its timer.
    if threading.current_thread() is not threading.main_thread():
        raise _FinishTimeout()
    previous = signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL) != (0., 0.):
        raise _FinishTimeout()
    def expired(*args):
        raise _FinishTimeout()
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, FINISH_TIMEOUT)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def git_revision():
    process = None
    try:
        process = subprocess.Popen(['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[3], stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE, text=True, start_new_session=True)
        output, _ = process.communicate(timeout=GIT_TIMEOUT)
        if process.returncode:
            return 'unavailable: git error'
        return output.strip()
    except subprocess.TimeoutExpired:
        return 'unavailable: timeout'
    except (OSError, subprocess.SubprocessError):
        return 'unavailable: git error'
    finally:
        if process is not None:
            # Kill the session even if the leader exited but a descendant kept
            # the output pipe open. Reap the leader without draining that pipe.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            if process.stdout is not None:
                process.stdout.close()


def run_config(config, identity, *, data, split, seed, defer_git=False):
    values = project_config(config)
    values['data'] = dict(pin=data.content_hash, lineage=data.lineage.to_dict())
    if hasattr(data, 'sources'):
        values['data']['manifest'] = data.to_dict()
    values['split'] = split.to_dict()
    git_sha = None if defer_git else git_revision()
    values = sanitize(values)
    serialized = json.dumps(values, sort_keys=True, separators=(',', ':'))
    return dict(config=values, config_snapshot_sha256=sha256(serialized.encode()).hexdigest(),
        recipe_hash=identity.config_hash, identity=identity.to_dict(), git_sha=git_sha,
        seed=seed, fold=config.settings.fold if not hasattr(config, 'fold') else config.fold,
        hardware=dict(device=config.device, node=socket.gethostname(),
                      gpu=torch.cuda.get_device_name(config.device) if torch.device(config.device).type == 'cuda' else None),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), attempt=config.output_dir)


def start(hook, config, identity, *, data, split, seed, phase):
    if hook is None or hook is noop:
        return
    try:
        inspect_config(hook, config)
        if hook.refusal:
            emit(hook, dict(event='status', status=hook.refusal), config.device)
            emit(hook, dict(event='run_start', refused=True, status=hook.refusal), config.device)
            return
        metadata = run_config(config, identity, data=data, split=split, seed=seed, defer_git=True)
        emit(hook, dict(event='run_start', phase=phase, **metadata), config.device)
    except Exception:
        if isinstance(hook, WandbLogger):
            hook.logging_ok = False


def finish(hook, device='cpu'):
    emit(hook, {'event': 'run_finish'}, device)


class WandbLogger:
    def __init__(self, directory, *, sync=True):
        self.directory = Path(directory)
        self.sync = sync
        self.run = None
        self.logging_ok = True
        self.sync_ok = None
        self.finished = False
        self.init_attempted = False
        self.spool = None
        self.attempt_dir = None
        self.status_path = None
        self._events = []
        self.refusal = None

    def __call__(self, event):
        if self.finished:
            return
        if event['event'] == 'run_finish':
            rng = None
            try:
                rng = capture_rng('cpu')
                with _finish_deadline():
                    self._finish()
            except _FinishTimeout:
                self.logging_ok = False
                self.finished = True
            finally:
                if rng is not None:
                    restore_rng(rng, 'cpu')
            return
        # No filesystem, SDK or provenance subprocess can block a fitting slice.
        try:
            self._events.append(json.dumps(sanitize(event)))
        except Exception:
            self.logging_ok = False
            raise

    def _prepare_directory(self):
        if self.attempt_dir is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.attempt_dir = Path(tempfile.mkdtemp(prefix='attempt-', dir=self.directory))
            self.status_path = self.attempt_dir / 'logging-status.json'

    def _seal(self):
        rows = []
        revision = None
        queried = False
        for text in self._events:
            event = json.loads(text)
            if event.get('event') == 'run_start' and 'git_sha' in event and event['git_sha'] is None:
                if not queried:
                    revision, queried = git_revision(), True
                    if revision.startswith('unavailable:'):
                        self.logging_ok = False
                event['git_sha'] = revision
                event['git_revision'] = revision
            rows.append(json.dumps(sanitize(event)))
        if not rows:
            self.logging_ok = False
        header = json.dumps(dict(schema='oxyformer-events-v1', capture_ok=self.logging_ok))
        digest = sha256()
        pending = self.attempt_dir / 'events-pending.jsonl'
        with pending.open('w') as stream:
            for line in (header, *rows):
                stream.write(line + '\n')
                digest.update((line + '\n').encode())
            stream.write(json.dumps(dict(sealed=True, count=len(rows), sha256=digest.hexdigest())) + '\n')
        final = self.attempt_dir / 'events-complete.jsonl'
        os.replace(pending, final)
        self.spool = final

    def _read_spool(self):
        lines = self.spool.read_text().splitlines()
        if len(lines) < 3:
            raise ValueError('unsealed event spool')
        header, footer = json.loads(lines[0]), json.loads(lines[-1])
        rows = lines[1:-1]
        digest = sha256(''.join(line + '\n' for line in lines[:-1]).encode()).hexdigest()
        if (header.get('schema') != 'oxyformer-events-v1' or
                type(header.get('capture_ok')) is not bool or footer.get('sealed') is not True or
                type(footer.get('count')) is not int or footer['count'] != len(rows) or
                footer.get('sha256') != digest):
            raise ValueError('incomplete event spool')
        events = [sanitize(json.loads(row)) for row in rows]
        if not events or any(type(event) is not dict or not isinstance(event.get('event'), str)
                             for event in events):
            raise ValueError('invalid event spool')
        return events, header['capture_ok']

    def _deliver(self):
        if self.init_attempted:
            return
        self.init_attempted = True
        events, captured = self._read_spool()
        self.logging_ok = self.logging_ok and captured
        if any(event.get('refused') is True for event in events):
            self.logging_ok = False
            return
        import wandb
        first = events[0]
        self.run = wandb.init(entity='goodarzilab', project='oxyformer-v2',
            mode='offline', dir=str(self.attempt_dir),
            config=first if first['event'] == 'run_start' else None,
            settings=wandb.Settings(console='off', disable_git=True,
                                    disable_code=True, disable_job_creation=True))
        for event in events:
            self.run.log(event)

    def _status(self, *, pending=False):
        if self.status_path is None:
            return
        try:
            temporary = self.status_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(
                logging_ok=None if pending else self.logging_ok, sync_ok=self.sync_ok)))
            os.replace(temporary, self.status_path)
        except OSError:
            self.logging_ok = False

    def _finish(self):
        # The training callers have already persisted their scientific checkpoint.
        try:
            self._prepare_directory()
            self._status(pending=True)
            if self.spool is None:
                self._seal()
            if not self.refusal:
                self._deliver()
        except Exception:
            self.logging_ok = False
        closed = False
        try:
            if self.run is not None:
                self.run.finish()
                closed = True
                if self.sync:
                    if not os.environ.get('WANDB_API_KEY'):
                        raise RuntimeError('Environment credential required')
                    offline_dir = str(Path(self.run.dir).parent)
                    result = subprocess.run([sys.executable, '-m', 'wandb', 'sync',
                        '--entity', 'goodarzilab', '--project', 'oxyformer-v2', offline_dir],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=120, check=False)
                    self.sync_ok = result.returncode == 0
        except Exception:
            if not closed:
                self.logging_ok = False
            self.sync_ok = False
        finally:
            self.finished = True
        self._status()

    @classmethod
    def replay(cls, spool, *, sync=True):
        """Recover a sealed spool; prior capture failures remain incomplete."""
        spool = Path(spool)
        hook = cls(spool.parent, sync=sync)
        hook.spool = spool
        finish(hook)
        return hook
