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


def run_config(config, identity, *, data, split, seed):
    excluded = {'run_logger', 'data', 'deadline_timing', 'stop_request', 'predecessor'}
    values = {}
    for field in fields(config):
        if field.name in excluded:
            continue
        value = getattr(config, field.name)
        values[field.name] = value.to_dict() if hasattr(value, 'to_dict') else value
    predecessor = config.predecessor
    predecessor = getattr(predecessor, 'checkpoint', predecessor)
    values['resume'] = None if predecessor is None else predecessor.to_dict()
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
    if isinstance(hook, WandbLogger):
        emit(hook, {'event': 'run_finish'}, device)


class WandbLogger:
    def __init__(self, directory, *, sync=True):
        self.directory = Path(directory)
        self.sync = sync
        self.run = None
        self.logging_ok = True
        self.sync_ok = None
        self.finished = False

    def __call__(self, event):
        if self.finished:
            return
        if event['event'] == 'run_finish':
            self._finish()
            return
        if self.run is None:
            mode = os.environ.get('WANDB_MODE', 'offline')
            if mode not in ('offline', 'dryrun', 'disabled') and not os.environ.get('WANDB_API_KEY'):
                raise RuntimeError('Environment credential required')
            import wandb
            self.directory.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(entity='goodarzilab', project='oxyformer-v2',
                mode=mode, dir=str(self.directory),
                config=event if event['event'] == 'run_start' else None,
                settings=wandb.Settings(console='off', disable_git=True,
                                        disable_code=True, disable_job_creation=True))
        self.run.log(event)

    def _finish(self):
        closed = False
        try:
            if self.run is None:
                self.logging_ok = False
                return
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
            try:
                (self.directory / 'logging-status.json').write_text(json.dumps(
                    dict(logging_ok=self.logging_ok, sync_ok=self.sync_ok)))
            except OSError:
                self.logging_ok = False
