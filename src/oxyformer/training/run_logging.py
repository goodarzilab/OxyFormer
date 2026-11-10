"""Optional, failure-isolated run telemetry. Never part of scientific state."""
from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time

import torch

from oxyformer.training.checkpoint import capture_rng, restore_rng

_SECRET_FIELD = re.compile(r"(?:api[_-]?key|access[_-]?token|auth[_-]?token|^token$|^key$|password|secret|credential)", re.I)


def sanitize(value):
    if isinstance(value, dict):
        return {sanitize(str(k)): ('[redacted]' if _SECRET_FIELD.search(str(k)) else sanitize(v))
                for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        key = os.environ.get('WANDB_API_KEY')
        return value.replace(key, '[redacted]') if key else value
    return value


def noop(event):
    pass


def emit(hook, event, device='cpu'):
    if hook is None:
        return
    rng = capture_rng(device)
    try:
        hook(sanitize(event))
    except Exception:
        if isinstance(hook, WandbLogger):
            hook.logging_ok = False
    finally:
        restore_rng(rng, device)


def step_event(hook, *, model, loss, learning_rate, samples, started, device, **context):
    if hook is None:
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
    if hook is not None and torch.device(device).type == 'cuda':
        try:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            if isinstance(hook, WandbLogger):
                hook.logging_ok = False
    return time.monotonic()


def run_config(config, identity, *, data, split, seed):
    values = project_config(config)
    values['data'] = dict(pin=data.content_hash, lineage=data.lineage.to_dict())
    if hasattr(data, 'sources'):
        values['data']['manifest'] = data.to_dict()
    values['split'] = split.to_dict()
    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[3], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        git_sha = None
    values = sanitize(values)
    serialized = json.dumps(values, sort_keys=True, separators=(',', ':'))
    return dict(config=values, config_snapshot_sha256=sha256(serialized.encode()).hexdigest(),
        recipe_hash=identity.config_hash, identity=identity.to_dict(), git_sha=git_sha,
        seed=seed, fold=config.settings.fold if not hasattr(config, 'fold') else config.fold,
        hardware=dict(device=config.device, node=socket.gethostname(),
                      gpu=torch.cuda.get_device_name(config.device) if torch.device(config.device).type == 'cuda' else None),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), attempt=config.output_dir)


def start(hook, config, identity, *, data, split, seed, phase):
    if hook is None:
        return
    try:
        metadata = run_config(config, identity, data=data, split=split, seed=seed)
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

    def __call__(self, event):
        if self.finished:
            return
        if event['event'] == 'run_finish':
            self._finish()
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.spool is None:
            fd, name = tempfile.mkstemp(prefix='events-', suffix='.jsonl', dir=self.directory)
            os.close(fd)
            self.spool = Path(name)
        with self.spool.open('a') as stream:
            stream.write(json.dumps(sanitize(event)) + '\n')

    def _deliver(self):
        if self.init_attempted:
            return
        self.init_attempted = True
        import wandb
        with self.spool.open() as stream:
            first = json.loads(next(stream))
            self.run = wandb.init(entity='goodarzilab', project='oxyformer-v2',
                mode='offline', dir=str(self.directory),
                config=first if first['event'] == 'run_start' else None,
                settings=wandb.Settings(console='off', disable_git=True,
                                        disable_code=True, disable_job_creation=True))
            self.run.log(first)
            for line in stream:
                self.run.log(json.loads(line))

    def _status(self, *, pending=False):
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / 'logging-status.json').write_text(json.dumps(
                dict(logging_ok=None if pending else self.logging_ok, sync_ok=self.sync_ok)))
        except OSError:
            self.logging_ok = False

    def _finish(self):
        # Both callers have persisted their checkpoint before reaching this path.
        self._status(pending=True)
        try:
            if self.spool is None:
                self.logging_ok = False
            else:
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
        """Recover a retained event spool without rerunning scientific work."""
        spool = Path(spool)
        hook = cls(spool.parent, sync=sync)
        hook.spool = spool
        finish(hook)
        return hook
