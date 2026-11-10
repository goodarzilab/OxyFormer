from dataclasses import replace
import json
import shutil
from pathlib import Path
import random
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oxyformer.training import run_logging as logging
from oxyformer.training.checkpoint import capture_rng


class FakeSDK:
    def __init__(self):
        self.events = []
        self.kwargs = None
        self.closed = False

    def init(self, **kwargs):
        self.kwargs = kwargs
        return self

    def Settings(self, **kwargs):
        return kwargs

    @property
    def dir(self):
        return str(Path(self.kwargs['dir']) / 'offline-run' / 'files')

    def log(self, event):
        self.events.append(event)

    def finish(self):
        self.closed = True


def test_default_offline_and_sync_is_separate_fact(tmp_path, monkeypatch):
    sdk = FakeSDK()
    monkeypatch.setenv('WANDB_API_KEY', '0123456789' * 4)
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    monkeypatch.delenv('WANDB_MODE', raising=False)
    calls = []
    def sync(command, **kwargs):
        assert sdk.closed
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(subprocess, 'run', sync)
    hook = logging.WandbLogger(tmp_path)
    logging.emit(hook, dict(event='run_start', recipe_hash='a' * 64))
    logging.finish(hook)
    assert sdk.kwargs['mode'] == 'offline'
    assert sdk.kwargs['entity'] == 'goodarzilab' and sdk.kwargs['project'] == 'oxyformer-v2'
    assert hook.logging_ok and hook.sync_ok is False
    assert json.loads((tmp_path / 'logging-status.json').read_text()) == dict(logging_ok=True, sync_ok=False)
    assert calls[0][0][1:4] == ['-m', 'wandb', 'sync']
    assert calls[0][1]['timeout'] == 120


@pytest.mark.parametrize('failure', ['init', 'log', 'finish', 'sync'])
def test_failure_isolation_and_secret_free_diagnostics(tmp_path, monkeypatch, failure, capsys):
    sdk = FakeSDK()
    key = '0123456789' * 4
    monkeypatch.setenv('WANDB_API_KEY', key)
    def broken(*args, **kwargs):
        raise RuntimeError(key)
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    if failure == 'sync':
        monkeypatch.setattr(subprocess, 'run', broken)
    else:
        monkeypatch.setattr(sdk, failure, broken)
    hook = logging.WandbLogger(tmp_path)
    logging.emit(hook, dict(event='run_start'))
    logging.finish(hook)
    assert hook.finished
    assert not hook.logging_ok or hook.sync_ok is False
    if failure == 'finish':
        assert hook.logging_ok is False
    assert key not in capsys.readouterr().out + capsys.readouterr().err
    assert key not in (tmp_path / 'logging-status.json').read_text()


def test_rng_identity_even_if_callback_draws_and_raises():
    before = capture_rng()
    def callback(event):
        random.random()
        np.random.random()
        torch.rand(4)
        raise RuntimeError('injected')
    logging.emit(callback, dict(event='evaluation'))
    after = capture_rng()
    assert before['python'] == after['python'] and before['numpy'] == after['numpy']
    assert torch.equal(before['torch'], after['torch'])
    assert before['cuda'] == after['cuda']
    logging.emit(None, {})
    logging.emit(logging.noop, {})
    assert torch.equal(after['torch'], torch.get_rng_state())


def assert_events(events, phases):
    for phase in phases:
        steps = [e for e in events if e['event'] == 'train_step' and e['phase'] == phase]
        assert steps
        for step in steps:
            assert {'train_loss', 'learning_rate', 'gradient_norm', 'samples_per_second',
                    'gpu_peak_bytes', 'epoch', 'step'} <= step.keys()
            assert step['samples_per_second'] > 0 and step['gradient_norm'] >= 0
            assert step['gpu_peak_bytes'] == 0
        assert any(e['event'] == 'evaluation' and e['phase'] == phase for e in events)
    starts = [e for e in events if e['event'] == 'run_start']
    assert starts
    for start in starts:
        assert {'config', 'config_snapshot_sha256', 'recipe_hash', 'identity', 'git_sha',
                'seed', 'fold', 'hardware', 'slurm_job_id', 'attempt'} <= start.keys()
        assert len(start['recipe_hash']) == 64 and len(start['git_sha']) == 40
        assert start['config']['data']['pin']
    assert any(e['event'] == 'checkpoint' and e['checkpoint']['payload']['sha256'] for e in events)


def test_secret_hygiene_in_run_tree_and_repository(tmp_path, monkeypatch):
    key = '0123456789' * 4
    monkeypatch.setenv('WANDB_API_KEY', key)
    events = []
    logging.emit(events.append, dict(event='config', nested=dict(api_key=key), value=key))
    (tmp_path / 'events.json').write_text(json.dumps(events))
    assert events[0]['nested']['api_key'] == '[redacted]'
    root = Path(__file__).resolve().parents[1]
    paths = [root / p for p in subprocess.check_output(
        ['git', 'ls-files'], cwd=root, text=True).splitlines()]
    paths += [p for p in tmp_path.rglob('*') if p.is_file()]
    # Credential assignments, not bare Git SHAs or SHA-256 provenance.
    pattern = re.compile(rb'(?:api[_-]?key|token|secret)[\s"\x27:=]+[a-fA-F0-9]{40}\b', re.I)
    for path in paths:
        if path.is_file():
            content = path.read_bytes()
            assert key.encode() not in content, path
            assert not pattern.search(content), path


def test_real_offline_sdk(tmp_path, monkeypatch):
    pytest.importorskip('wandb', reason='optional SDK absent in CI CPU environment')
    monkeypatch.setenv('WANDB_MODE', 'offline')
    key = '0123456789' * 4
    monkeypatch.setenv('WANDB_API_KEY', key)
    hook = logging.WandbLogger(tmp_path, sync=False)
    logging.emit(hook, dict(event='run_start', seed=1103))
    logging.emit(hook, dict(event='train_step', train_loss=1.))
    logging.finish(hook)
    assert hook.logging_ok and hook.finished and hook.sync_ok is None
    assert list(tmp_path.rglob('*.wandb'))
    for path in tmp_path.rglob('*'):
        if path.is_file():
            assert key.encode() not in path.read_bytes(), path


def check_invariance(monkeypatch, config, trace, run, tmp_path, phase, phases):
    import oxyformer.training.checkpoint as checkpoint
    # Paths and ZIP dates affect archive bytes independently of numerics.
    monkeypatch.setattr(checkpoint.zipfile.time, 'localtime',
                        lambda *args: (2026, 1, 1, 0, 0, 0, 3, 1, 0))
    sdk = FakeSDK()
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    results = []
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for hook in (None, logging.noop, logging.WandbLogger(tmp_path / 'logs', sync=False)):
            trace.clear()
            artifact = run(replace(config, run_logger=hook))
            results.append((list(trace), artifact.sha256))
            shutil.rmtree(config.output_dir)
    finally:
        torch.use_deterministic_algorithms(deterministic)
    for losses, digest in results[1:]:
        assert digest == results[0][1]
        assert len(losses) == len(results[0][0])
        assert all(torch.equal(x, y) for x, y in zip(losses, results[0][0]))
    assert config.run_logger is None
    assert_events(sdk.events, phases)
    logged = [e['train_loss'] for e in sdk.events if e['event'] == 'train_step'
              and e['phase'] == phase and e['train_loss'] is not None]
    assert logged == [float(loss) for loss in results[0][0]]


def test_sync_requires_environment_credential(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'wandb', FakeSDK())
    monkeypatch.delenv('WANDB_API_KEY', raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail('Must not fall back to persisted credentials')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    hook = logging.WandbLogger(tmp_path)
    logging.emit(hook, dict(event='run_start'))
    logging.finish(hook)
    assert hook.logging_ok and hook.sync_ok is False
