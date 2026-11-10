from dataclasses import replace
import json
import multiprocessing
import shutil
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oxyformer.training import run_logging as logging
from oxyformer.training.checkpoint import capture_rng


_manager = None


def shared_manager():
    global _manager
    if _manager is None:
        _manager = multiprocessing.get_context('fork').Manager()
    return _manager


def shared_list(values=()):
    return shared_manager().list(values)


class FakeSDK:
    def __init__(self):
        self.state = shared_manager().Namespace()
        self.state.events = []
        self.state.kwargs = None
        self.state.closed = False

    @property
    def events(self):
        return self.state.events

    @property
    def kwargs(self):
        return self.state.kwargs

    @property
    def closed(self):
        return self.state.closed

    def init(self, **kwargs):
        self.state.kwargs = kwargs
        return self

    def Settings(self, **kwargs):
        return kwargs

    @property
    def dir(self):
        return str(Path(self.kwargs['dir']) / 'offline-run' / 'files')

    def log(self, event):
        self.state.events = self.events + [event]

    def finish(self):
        self.state.closed = True


def test_default_offline_and_sync_is_separate_fact(tmp_path, monkeypatch):
    sdk = FakeSDK()
    monkeypatch.setenv('WANDB_API_KEY', '0123456789' * 4)
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    monkeypatch.delenv('WANDB_MODE', raising=False)
    calls = shared_list()
    def sync(command, **kwargs):
        assert sdk.closed
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(subprocess, 'run', sync)
    hook = logging.WandbLogger(tmp_path)
    logging.emit(hook, dict(event='run_start', recipe_hash='a' * 64))
    assert sdk.kwargs is None  # SDK work cannot consume the fitting slice.
    logging.finish(hook)
    assert sdk.kwargs['mode'] == 'offline'
    assert sdk.kwargs['entity'] == 'goodarzilab' and sdk.kwargs['project'] == 'oxyformer-v2'
    assert hook.logging_ok and hook.sync_ok is False
    assert json.loads(hook.status_path.read_text()) == dict(logging_ok=True, sync_ok=False)
    assert calls[0][0][1:4] == ['-m', 'wandb', 'sync']
    assert calls[0][1]['timeout'] == 120
    monkeypatch.delenv('WANDB_API_KEY')
    calls[:] = []
    hook = logging.WandbLogger(tmp_path)
    logging.emit(hook, dict(event='run_start'))
    logging.finish(hook)
    assert not calls and hook.logging_ok and hook.sync_ok is False


@pytest.mark.parametrize('failure', ['init', 'log', 'finish', 'sync'])
def test_failure_isolation_and_secret_free_diagnostics(tmp_path, monkeypatch, failure, capfd):
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
        monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    hook = logging.WandbLogger(tmp_path)
    logging.emit(hook, dict(event='run_start'))
    logging.finish(hook)
    assert hook.finished
    assert not hook.logging_ok or hook.sync_ok is False
    if failure != 'sync':
        assert hook.logging_ok is False
    if failure == 'finish':
        assert hook.logging_ok is False
    captured = capfd.readouterr()
    assert key not in captured.out + captured.err
    assert key not in hook.status_path.read_text()


def test_rng_identity_even_if_capture_draws_and_raises(tmp_path, monkeypatch):
    before = capture_rng()
    def callback(self, event):
        random.random()
        np.random.random()
        torch.rand(4)
        raise RuntimeError('injected')
    monkeypatch.setattr(logging.WandbLogger, '__call__', callback)
    logging.emit(logging.WandbLogger(tmp_path), dict(event='evaluation'))
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
    hook = logging.WandbLogger(tmp_path / 'logs', sync=False)
    logging.emit(hook, dict(event='config', nested=dict(api_key=key), value=key, mapping={key: 'fixture'}))
    events = [json.loads(row) for row in hook._events]
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
    monkeypatch.setattr(checkpoint.zipfile.time, 'localtime',
                        lambda *args: time.struct_time((2026, 1, 1, 0, 0, 0, 3, 1, 0)))
    sdk = FakeSDK()
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    results = []
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        hooks = (None, logging.noop, logging.WandbLogger(tmp_path / 'logs', sync=False),
                 logging.WandbLogger(tmp_path / 'outage', sync=False))
        for index, hook in enumerate(hooks):
            if index == 3:
                def unavailable(**kwargs):
                    raise RuntimeError('service unavailable')
                monkeypatch.setattr(sdk, 'init', unavailable)
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
    if phase == 'nuisance':
        assert sdk.events[0]['config']['deadline_timing']['compute_deadline'] == config.deadline_timing.deadline
    logged = [e['train_loss'] for e in sdk.events if e['event'] == 'train_step'
              and e['phase'] == phase and e['train_loss'] is not None]
    assert logged == [float(loss) for loss in results[0][0]]


def test_finish_rejects_generic_callable():
    with pytest.raises(TypeError, match='WandbLogger'):
        logging.finish([].append)


def test_closed_config_schema():
    from dataclasses import fields, make_dataclass
    from oxyformer.training.fit import FitConfig
    from oxyformer.training.pretrain import PretrainConfig
    for cls in (FitConfig, PretrainConfig):
        assert set(logging.CONFIG_SCHEMA[cls.__name__]) == {f.name for f in fields(cls)}
        assert set(logging.CONFIG_SCHEMA[cls.__name__].values()) <= {'included', 'hashed', 'redacted', 'excluded'}
    Unknown = make_dataclass('PretrainConfig', [('unclassified', int)])
    with pytest.raises(ValueError, match='unclassified'):
        logging.project_config(Unknown(1))


def test_stop_snapshot_is_observational():
    from oxyformer.training.checkpoint import CheckpointRequest
    request = CheckpointRequest()
    assert logging.stop_state(None) == logging.stop_state(request) == 'none'
    request.request()
    assert logging.stop_state(request) == 'requested'
    assert logging.stop_state(request, reason='requested') == 'honoured'
    assert request.requested


def test_allocator_windows(tmp_path, monkeypatch):
    allocator = SimpleNamespace(current=512, peak=8192, resets=0)
    restore = logging.restore_rng
    monkeypatch.setattr(logging, 'capture_rng', lambda device: capture_rng())
    monkeypatch.setattr(logging, 'restore_rng', lambda state, device: restore(state, 'cpu'))
    def reset(device):
        allocator.peak = allocator.current
        allocator.resets += 1
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', reset)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: allocator.peak)
    hook = logging.WandbLogger(tmp_path, sync=False)
    model = torch.nn.Linear(1, 1)
    for peak in (1024, 768):
        started = logging.window_start(hook, 'cuda:0')
        allocator.peak = max(allocator.peak, peak)
        logging.step_event(hook, model=model, loss=None, learning_rate=.01,
                           samples=2, started=started, device='cuda:0')
    events = [json.loads(row) for row in hook._events]
    assert allocator.resets == 2
    assert [e['gpu_peak_bytes'] for e in events] == [1024, 768]


def test_online_environment_cannot_enable_network_or_retry(tmp_path, monkeypatch):
    sdk = FakeSDK()
    attempts = shared_list()
    def unavailable(**kwargs):
        attempts.append(kwargs['mode'])
        raise RuntimeError('service unavailable')
    sdk.init = unavailable
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    monkeypatch.setenv('WANDB_MODE', 'online')
    hook = logging.WandbLogger(tmp_path, sync=False)
    for _ in range(3):
        logging.emit(hook, dict(event='run_start'))
    logging.finish(hook)
    assert list(attempts) == ['offline']
    assert not hook.logging_ok and hook.finished


@pytest.mark.parametrize('phase', ['fit', 'ssl'])
def test_real_offline_training_payloads(tmp_path, monkeypatch, phase):
    pytest.importorskip('wandb', reason='real SDK runs only in the separate clone')
    monkeypatch.setenv('WANDB_MODE', 'online')
    key = '0123456789' * 4
    monkeypatch.setenv('WANDB_API_KEY', key)
    hook = logging.WandbLogger(tmp_path / 'logs', sync=False)
    if phase == 'fit':
        from test_fit import make_case
        from oxyformer.training.fit import fit_fold
        spec, split, manifest, config = make_case(tmp_path / 'attempt')
        artifact = fit_fold(spec, split, manifest, replace(config, run_logger=hook), 1103).checkpoint
    else:
        from test_pretrain import make_case
        from oxyformer.training.pretrain import pretrain
        view, split, config = make_case(tmp_path / 'attempt')
        artifact = pretrain(view, split, replace(config, run_logger=hook), 1103)
    assert artifact.sha256
    assert hook.logging_ok and hook.finished and hook.sync_ok is None
    assert list((tmp_path / 'logs').rglob('*.wandb'))
    for path in tmp_path.rglob('*'):
        if path.is_file():
            assert key.encode() not in path.read_bytes(), path


def test_unknown_runtime_attribute_is_not_silently_dropped(tmp_path):
    from test_pretrain import make_case
    _, _, config = make_case(tmp_path)
    config.__dict__['new_runtime_key'] = 1
    with pytest.raises(ValueError, match='new_runtime_key'):
        logging.project_config(config)


def test_sdk_import_failure_persists_status(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'wandb', None)
    directory = tmp_path / 'fresh'
    hook = logging.WandbLogger(directory, sync=False)
    logging.emit(hook, dict(event='run_start'))
    logging.finish(hook)
    status = json.loads(hook.status_path.read_text())
    assert status['logging_ok'] is False and hook.finished


def test_slow_sdk_failure_cannot_consume_ssl_slice(tmp_path, monkeypatch):
    from test_pretrain import make_case
    from oxyformer.training.pretrain import pretrain
    import oxyformer.training.checkpoint as checkpoint
    monkeypatch.setattr(checkpoint.zipfile.time, 'localtime',
                        lambda *args: time.struct_time((2026, 1, 1, 0, 0, 0, 3, 1, 0)))
    view, split, config = make_case(tmp_path / 'attempt')
    config = replace(config, slice_seconds=30., checkpoint_margin_seconds=1., max_batches=1)
    clock = [0.]
    monkeypatch.setattr(logging.time, 'monotonic', lambda: clock[0])
    sdk = FakeSDK()
    def slow_failure(**kwargs):
        clock[0] += 35.
        raise RuntimeError('slow offline init failure')
    sdk.init = slow_failure
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    artifacts = []
    for hook in (None, logging.WandbLogger(tmp_path / 'logs', sync=False)):
        clock[0] = 0.
        artifacts.append(pretrain(view, split, replace(config, run_logger=hook), 1103))
        shutil.rmtree(config.output_dir)
    assert [a.step for a in artifacts] == [1, 1]
    assert artifacts[0].sha256 == artifacts[1].sha256


def test_held_out_metric_values_match_checkpoint(tmp_path):
    from test_fit import make_case
    from oxyformer.training.fit import fit_fold
    from oxyformer.training.checkpoint import load_checkpoint
    spec, split, manifest, config = make_case(tmp_path / 'attempt')
    hook = logging.WandbLogger(tmp_path / 'logs', sync=False)
    artifact = fit_fold(spec, split, manifest, replace(config, run_logger=hook), 1103).checkpoint
    events = [json.loads(row) for row in hook._events]
    results = load_checkpoint(artifact, artifact.identity)['controller']['results']
    evaluations = [e for e in events if e['event'] == 'evaluation' and e['phase'] == 'held_out']
    assert evaluations
    for event in evaluations:
        result = next(r for r in results if r['kind'] == event['kind'] and
                      r['fold'] == event['inner_fold'] and r['grid'] == event['grid'])
        assert event['metrics'] == list(result['metrics'])
        assert all(type(value) is float for value in event['metrics'])


def test_retained_spool_can_be_replayed(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'wandb', None)
    key = '0123456789' * 4
    monkeypatch.setenv('WANDB_API_KEY', key)
    hook = logging.WandbLogger(tmp_path, sync=False)
    logging.emit(hook, dict(event='run_start', api_key=key))
    logging.emit(hook, dict(event='train_step', train_loss=1.))
    logging.finish(hook)
    assert not hook.logging_ok and hook.spool.is_file()
    assert key not in hook.spool.read_text()
    sdk = FakeSDK()
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    recovered = logging.WandbLogger.replay(hook.spool, sync=False)
    assert recovered.logging_ok and recovered.finished and sdk.closed
    assert sdk.events == [dict(event='run_start', api_key='[redacted]'),
                          dict(event='train_step', train_loss=1.)]


def test_slow_spool_failure_cannot_consume_ssl_slice(tmp_path, monkeypatch):
    from test_pretrain import make_case
    from oxyformer.training.pretrain import pretrain
    import oxyformer.training.checkpoint as checkpoint
    monkeypatch.setattr(checkpoint.zipfile.time, 'localtime',
                        lambda *args: time.struct_time((2026, 1, 1, 0, 0, 0, 3, 1, 0)))
    view, split, config = make_case(tmp_path / 'attempt')
    config = replace(config, slice_seconds=30., checkpoint_margin_seconds=1., max_batches=1)
    clock, failed = [0.], shared_list([False])
    monkeypatch.setattr(logging.time, 'monotonic', lambda: clock[0])
    original = Path.open
    def slow_write(path, mode='r', *args, **kwargs):
        if path.name.startswith('events-') and mode in ('a', 'w') and not failed[0]:
            failed[0] = True
            clock[0] += 35.
            raise OSError('slow log filesystem')
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', slow_write)
    monkeypatch.setitem(sys.modules, 'wandb', FakeSDK())
    artifacts = []
    for hook in (None, logging.WandbLogger(tmp_path / 'logs', sync=False)):
        clock[0] = 0.
        artifacts.append(pretrain(view, split, replace(config, run_logger=hook), 1103))
        shutil.rmtree(config.output_dir)
    assert failed[0]
    assert [a.step for a in artifacts] == [1, 1]
    assert artifacts[0].sha256 == artifacts[1].sha256


def test_numeric_runtime_key_and_refresh_token_are_redacted(tmp_path, monkeypatch):
    key, refresh = '1234567890' * 4, 'abcdef0123' * 4
    monkeypatch.setenv('WANDB_API_KEY', key)
    sdk = FakeSDK()
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    hook = logging.WandbLogger(tmp_path, sync=False)
    logging.emit(hook, dict(event='run_start', identifier=int(key), refresh_token=refresh))
    logging.finish(hook)
    for path in tmp_path.rglob('*'):
        if path.is_file():
            assert key not in path.read_text(errors='replace')
            assert refresh not in path.read_text(errors='replace')
    assert sdk.kwargs['config']['identifier'] == '[redacted]'
    assert sdk.kwargs['config']['refresh_token'] == '[redacted]'


def test_replay_preserves_prior_capture_failure(tmp_path, monkeypatch):
    original = logging.WandbLogger.__call__
    def dropping(self, event):
        if event.get('event') == 'train_step' and event.get('step') == 1:
            raise OSError('one lost event')
        return original(self, event)
    monkeypatch.setattr(logging.WandbLogger, '__call__', dropping)
    hook = logging.WandbLogger(tmp_path, sync=False)
    logging.emit(hook, dict(event='run_start'))
    logging.emit(hook, dict(event='train_step', step=1))
    logging.emit(hook, dict(event='train_step', step=2))
    monkeypatch.setitem(sys.modules, 'wandb', None)
    logging.finish(hook)
    assert not hook.logging_ok
    sdk = FakeSDK()
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    replay = logging.WandbLogger.replay(hook.spool, sync=False)
    assert replay.finished and not replay.logging_ok
    assert len(sdk.events) == 2


def test_shared_logging_base_keeps_attempt_statuses(tmp_path, monkeypatch):
    first = logging.WandbLogger(tmp_path, sync=False)
    logging.emit(first, dict(event='run_start'))
    monkeypatch.setitem(sys.modules, 'wandb', None)
    logging.finish(first)
    first_status = getattr(first, 'status_path', tmp_path / 'logging-status.json')
    assert json.loads(first_status.read_text())['logging_ok'] is False
    second = logging.WandbLogger(tmp_path, sync=False)
    logging.emit(second, dict(event='run_start'))
    monkeypatch.setitem(sys.modules, 'wandb', FakeSDK())
    logging.finish(second)
    second_status = getattr(second, 'status_path', tmp_path / 'logging-status.json')
    assert first_status != second_status
    assert json.loads(first_status.read_text())['logging_ok'] is False
    assert json.loads(second_status.read_text())['logging_ok'] is True


def test_selection_evaluations_match_checkpoint(tmp_path):
    from test_fit import make_case
    from oxyformer.training.fit import fit_fold
    from oxyformer.training.checkpoint import load_checkpoint
    spec, split, manifest, config = make_case(tmp_path / 'attempt')
    hook = logging.WandbLogger(tmp_path / 'logs', sync=False)
    artifact = fit_fold(spec, split, manifest, replace(config, run_logger=hook), 1103).checkpoint
    events = [json.loads(row) for row in hook._events]
    results = [r for r in load_checkpoint(artifact, artifact.identity)['controller']['results']
               if r['kind'] == 'origin']
    assert any(r['selection_metrics'] != r['metrics'] for r in results)
    selected = [e for e in events if e['event'] == 'evaluation' and e['phase'] == 'selection']
    assert len(selected) == len(results) > 0
    for event in selected:
        result = next(r for r in results if r['fold'] == event['inner_fold'] and r['grid'] == event['grid'])
        assert event['metrics'] == list(result['selection_metrics'])
        assert event['original_ids'] == list(result['selection_ids'])


def test_slow_git_provenance_failure_cannot_consume_ssl_slice(tmp_path, monkeypatch):
    from test_pretrain import make_case
    from oxyformer.training.pretrain import pretrain
    import oxyformer.training.checkpoint as checkpoint
    monkeypatch.setattr(checkpoint.zipfile.time, 'localtime',
                        lambda *args: time.struct_time((2026, 1, 1, 0, 0, 0, 3, 1, 0)))
    view, split, config = make_case(tmp_path / 'attempt')
    config = replace(config, slice_seconds=30., checkpoint_margin_seconds=1., max_batches=1)
    clock, queried = [0.], shared_list([False])
    monkeypatch.setattr(logging.time, 'monotonic', lambda: clock[0])
    def slow_git():
        queried[0] = True
        clock[0] += 35.
        return 'unavailable: injected slow provenance'
    monkeypatch.setattr(logging, 'git_revision', slow_git)
    monkeypatch.setitem(sys.modules, 'wandb', FakeSDK())
    artifacts = []
    for hook in (None, logging.noop, logging.WandbLogger(tmp_path / 'logs', sync=False)):
        clock[0] = 0.
        artifacts.append(pretrain(view, split, replace(config, run_logger=hook), 1103))
        shutil.rmtree(config.output_dir)
    assert queried[0]
    assert [a.step for a in artifacts] == [1, 1, 1]
    assert len({a.sha256 for a in artifacts}) == 1


def test_capture_needs_no_filesystem_or_sdk_before_finish(tmp_path, monkeypatch):
    hook = logging.WandbLogger(tmp_path, sync=False)
    sdk = FakeSDK()
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    def unavailable(*args, **kwargs):
        raise OSError('filesystem unavailable during training')
    with monkeypatch.context() as context:
        context.setattr(Path, 'open', unavailable)
        logging.emit(hook, dict(event='run_start'))
        logging.emit(hook, dict(event='train_step', step=1))
        assert hook.logging_ok and not hook.init_attempted and hook.spool is None
    logging.finish(hook)
    assert hook.logging_ok and len(sdk.events) == 2


@pytest.mark.parametrize('damage', ['truncate', 'drop_row', 'capture_flag'])
def test_unsealed_or_modified_spool_cannot_publish_complete(tmp_path, monkeypatch, damage):
    monkeypatch.setitem(sys.modules, 'wandb', None)
    hook = logging.WandbLogger(tmp_path / 'logs', sync=False)
    logging.emit(hook, dict(event='run_start'))
    logging.emit(hook, dict(event='train_step', step=1))
    logging.finish(hook)
    lines = hook.spool.read_text().splitlines()
    if damage == 'truncate':
        lines.pop()
    elif damage == 'drop_row':
        lines.pop(1)
    else:
        header = json.loads(lines[0])
        header['capture_ok'] = not header['capture_ok']
        lines[0] = json.dumps(header)
    damaged = tmp_path / 'damaged.jsonl'
    damaged.write_text('\n'.join(lines) + '\n')
    sdk = FakeSDK()
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    replay = logging.WandbLogger.replay(damaged, sync=False)
    assert replay.finished and not replay.logging_ok and sdk.kwargs is None


def test_leading_zero_numeric_key_is_redacted(tmp_path, monkeypatch):
    key = '0' + '1234567890' * 3 + '123456789'
    monkeypatch.setenv('WANDB_API_KEY', key)
    sdk = FakeSDK()
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    hook = logging.WandbLogger(tmp_path, sync=False)
    logging.emit(hook, dict(event='run_start', mapping={int(key): 'fixture'}))
    logging.finish(hook)
    assert sdk.kwargs['config']['mapping'] == {'[redacted]': 'fixture'}


@pytest.mark.parametrize('phase', ['fit', 'ssl'])
def test_generic_hook_rejected_before_training(tmp_path, phase):
    def callback(event):
        pytest.fail('generic callback ran')
    if phase == 'fit':
        from test_fit import make_case
        from oxyformer.training.fit import fit_fold
        spec, split, data, config = make_case(tmp_path / 'attempt')
        run = lambda: fit_fold(spec, split, data, replace(config, run_logger=callback), 1103)
    else:
        from test_pretrain import make_case
        from oxyformer.training.pretrain import pretrain
        data, split, config = make_case(tmp_path / 'attempt')
        run = lambda: pretrain(data, split, replace(config, run_logger=callback), 1103)
    with pytest.raises(TypeError, match='WandbLogger'):
        run()
    assert not Path(config.output_dir).exists()


def test_decimal_string_credentials_redacted(tmp_path, monkeypatch):
    key = '0' + '1234567890' * 3 + '123456789'
    monkeypatch.setenv('WANDB_API_KEY', key)
    forms = [key, str(int(key)), '000' + str(int(key))]
    sdk = FakeSDK()
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    hook = logging.WandbLogger(tmp_path, sync=False)
    logging.emit(hook, dict(event='run_start', values=forms,
                           mapping={value: value for value in forms}))
    logging.finish(hook)
    assert sdk.kwargs['config']['values'] == ['[redacted]'] * 3
    assert sdk.kwargs['config']['mapping'] == {'[redacted]': '[redacted]'}
    for path in tmp_path.rglob('*'):
        if path.is_file():
            assert all(form.encode() not in path.read_bytes() for form in forms)


@pytest.mark.parametrize('phase', ['fit', 'ssl'])
def test_entrypoint_refuses_unknown_config_before_copy(tmp_path, monkeypatch, phase):
    monkeypatch.setitem(sys.modules, 'wandb', FakeSDK())
    hook = logging.WandbLogger(tmp_path / 'logs', sync=False)
    if phase == 'fit':
        from test_fit import make_case
        from oxyformer.training.fit import fit_fold
        spec, split, data, config = make_case(tmp_path / 'attempt')
        config = replace(config, run_logger=hook)
        config.__dict__['new_runtime_key'] = 'do not export'
        artifact = fit_fold(spec, split, data, config, 1103).checkpoint
    else:
        from test_pretrain import make_case
        from oxyformer.training.pretrain import pretrain
        data, split, config = make_case(tmp_path / 'attempt')
        config = replace(config, run_logger=hook)
        config.__dict__['new_runtime_key'] = 'do not export'
        artifact = pretrain(data, split, config, 1103)
    assert artifact.sha256 and not hook.logging_ok
    events = [json.loads(row) for row in hook.spool.read_text().splitlines()[1:-1]]
    assert any(e.get('status') == 'refused: undeclared config key new_runtime_key' for e in events)
    assert any(e['event'] == 'run_start' and e.get('refused') is True for e in events)
    assert not any('do not export' in str(e) for e in events)


def test_git_timeout_kills_owned_process_group(tmp_path, monkeypatch):
    import os
    import time
    executable = tmp_path / 'git'
    marker = tmp_path / 'pid'
    executable.write_text('#!/usr/bin/env python3\nimport os,time\n'
                          f'open({str(marker)!r}, "w").write(str(os.getpid()))\n'
                          'time.sleep(100)\n')
    executable.chmod(0o700)
    monkeypatch.setenv('PATH', str(tmp_path) + ':' + os.environ['PATH'])
    monkeypatch.setattr(logging, 'GIT_TIMEOUT', .2, raising=False)
    started = time.monotonic()
    revision = logging.git_revision()
    assert time.monotonic() - started < 1
    assert revision.startswith('unavailable: ')
    pid = int(marker.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_ssl_skipped_batches_have_no_train_step(tmp_path, monkeypatch):
    from test_pretrain import make_case
    from oxyformer.training.pretrain import pretrain
    import oxyformer.training.pretrain as module
    view, split, config = make_case(tmp_path / 'attempt')
    hook = logging.WandbLogger(tmp_path / 'logs', sync=False)
    original = module.balanced_loss
    calls = [0]
    def skip_second(losses, families):
        calls[0] += 1
        return None if calls[0] == 2 else original(losses, families)
    monkeypatch.setattr(module, 'balanced_loss', skip_second)
    norms = []
    original_step = torch.optim.AdamW.step
    def record_update(optimizer, *args, **kwargs):
        gradients = [p.grad.detach().double().norm() for group in optimizer.param_groups
                     for p in group['params'] if p.grad is not None]
        norms.append(torch.stack(gradients).norm().item())
        return original_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, 'step', record_update)
    monkeypatch.setitem(sys.modules, 'wandb', FakeSDK())
    pretrain(view, split, replace(config, run_logger=hook, max_batches=3), 1103)
    events = [json.loads(row) for row in hook._events]
    steps = [e for e in events if e['event'] == 'train_step']
    skipped = [e for e in events if e['event'] == 'batch_skipped']
    assert len(steps) == len(norms) == 2
    assert len(skipped) == 1 and skipped[0]['processed_batches'] == 2
    assert [e['processed_batches'] for e in steps] == [1, 3]
    assert [e['gradient_norm'] for e in steps] == norms
    assert all(e['train_loss'] is not None for e in steps)


@pytest.mark.parametrize('action', ['_prepare_directory', '_seal', '_deliver', '_status'])
@pytest.mark.parametrize('failure', ['raise', 'stall'])
@pytest.mark.parametrize('phase', ['fit', 'ssl'])
def test_finish_external_failures_preserve_checkpoint_and_result(tmp_path, monkeypatch, action, failure, phase):
    import time
    import oxyformer.training.checkpoint as checkpoint
    monkeypatch.setattr(checkpoint.zipfile.time, 'localtime',
                        lambda *args: time.struct_time((2026, 1, 1, 0, 0, 0, 3, 1, 0)))
    monkeypatch.setattr(logging, 'FINISH_TIMEOUT', .1)
    if phase == 'fit':
        from test_fit import make_case
        from oxyformer.training.fit import fit_fold
        spec, split, data, config = make_case(tmp_path / 'attempt')
        run = lambda config: fit_fold(spec, split, data, config, 1103)
    else:
        from test_pretrain import make_case
        from oxyformer.training.pretrain import pretrain
        data, split, config = make_case(tmp_path / 'attempt')
        run = lambda config: pretrain(data, split, config, 1103)
    off = run(config)
    off_json = off.to_dict()
    shutil.rmtree(config.output_dir)
    hook = logging.WandbLogger(tmp_path / 'logs', sync=False)
    entered = shared_list()
    def unavailable(*args, **kwargs):
        entered.append(time.monotonic())
        if failure == 'stall':
            time.sleep(30)
        raise OSError('external action unavailable')
    monkeypatch.setattr(logging.WandbLogger, action, unavailable)
    monkeypatch.setitem(sys.modules, 'wandb', FakeSDK())
    on = run(replace(config, run_logger=hook))
    assert on.to_dict() == off_json
    assert entered and time.monotonic() - entered[-1] < 1
    assert not hook.logging_ok and hook.finished


def test_start_steps_and_evaluations_are_memory_only(tmp_path, monkeypatch):
    import builtins
    import socket
    from test_fit import make_case
    _, split, data, config = make_case(tmp_path / 'attempt')
    hook = logging.WandbLogger(tmp_path / 'logs', sync=False)
    config = replace(config, run_logger=hook, device='cpu')
    identity = SimpleNamespace(config_hash='a' * 64, to_dict=lambda: {})
    model = torch.nn.Linear(1, 1)
    model(torch.ones(1, 1)).sum().backward()
    sdk = FakeSDK()
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    def forbidden(*args, **kwargs):
        raise AssertionError('external action during capture')
    with monkeypatch.context() as context:
        for owner, name in [(builtins, 'open'), (Path, 'open'), (Path, 'mkdir'),
                            (subprocess, 'Popen'), (subprocess, 'run'),
                            (subprocess, 'check_output'), (socket, 'create_connection'),
                            (sdk, 'init'), (sdk, 'log'), (sdk, 'finish')]:
            context.setattr(owner, name, forbidden)
        logging.start(hook, config, identity, data=data, split=split, seed=1103, phase='fold')
        logging.step_event(hook, model=model, loss=torch.tensor(1.), learning_rate=.01,
                           samples=1, started=0., device='cpu', step=0, epoch=0)
        logging.emit(hook, dict(event='evaluation', metrics=[1.]))
    assert hook.logging_ok and not hook.init_attempted and hook.spool is None
    assert [json.loads(row)['event'] for row in hook._events] == ['run_start', 'train_step', 'evaluation']


def test_refused_snapshot_cannot_be_published_by_replay(tmp_path, monkeypatch):
    sdk = FakeSDK()
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    hook = logging.WandbLogger(tmp_path, sync=False)
    hook.logging_ok = False
    hook.refusal = 'refused: undeclared config key extra'
    logging.emit(hook, dict(event='run_start', refused=True, status=hook.refusal))
    logging.finish(hook)
    replay = logging.WandbLogger.replay(hook.spool, sync=False)
    assert replay.finished and not replay.logging_ok
    assert sdk.kwargs is None and not sdk.events


@pytest.mark.parametrize('action', ['init', 'log', 'finish'])
def test_sdk_stalls_are_bounded(tmp_path, monkeypatch, action):
    import time
    sdk = FakeSDK()
    def stalled(*args, **kwargs):
        time.sleep(30)
    monkeypatch.setattr(sdk, action, stalled)
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    monkeypatch.setattr(logging, 'FINISH_TIMEOUT', .1)
    hook = logging.WandbLogger(tmp_path, sync=False)
    logging.emit(hook, dict(event='run_start'))
    started = time.monotonic()
    logging.finish(hook)
    assert time.monotonic() - started < 1
    assert hook.finished and not hook.logging_ok


def test_refusal_survives_missing_flagged_start(tmp_path, monkeypatch):
    from test_pretrain import make_case
    from oxyformer.training.pretrain import pretrain
    sdk = FakeSDK()
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    view, split, config = make_case(tmp_path / 'attempt')
    hook = logging.WandbLogger(tmp_path / 'logs', sync=False)
    config = replace(config, run_logger=hook)
    config.__dict__['extra'] = 'not exported'
    original = logging.WandbLogger.__call__
    def dropping(self, event):
        if event.get('event') == 'run_start' and event.get('refused'):
            raise OSError('lost start')
        return original(self, event)
    monkeypatch.setattr(logging.WandbLogger, '__call__', dropping)
    pretrain(view, split, config, 1103)
    monkeypatch.setattr(logging.WandbLogger, '__call__', original)
    replay = logging.WandbLogger.replay(hook.spool, sync=False)
    assert not replay.logging_ok and sdk.kwargs is None


@pytest.mark.parametrize('native', ['hash', 'wait'])
def test_native_sdk_stall_is_bounded(tmp_path, monkeypatch, native):
    import ctypes
    import hashlib
    import time
    sdk = FakeSDK()
    def stalled(**kwargs):
        if native == 'hash':
            hashlib.pbkdf2_hmac('sha256', b'fixture', b'fixture', 3000000)
        else:
            ctypes.CDLL(None).system(b'sleep 1')
        raise RuntimeError('native stall')
    monkeypatch.setattr(sdk, 'init', stalled)
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    monkeypatch.setattr(logging, 'FINISH_TIMEOUT', .05)
    hook = logging.WandbLogger(tmp_path, sync=False)
    logging.emit(hook, dict(event='run_start'))
    started = time.monotonic()
    logging.finish(hook)
    assert time.monotonic() - started < .5
    assert hook.finished and not hook.logging_ok


def test_finish_does_not_access_parent_rng(tmp_path, monkeypatch):
    import time
    monkeypatch.setitem(sys.modules, 'wandb', FakeSDK())
    hook = logging.WandbLogger(tmp_path, sync=False)
    logging.emit(hook, dict(event='run_start'))
    original = logging.capture_rng
    accesses = shared_list()
    def stalled(*args, **kwargs):
        accesses.append(True)
        time.sleep(.4)
        return original(*args, **kwargs)
    monkeypatch.setattr(logging, 'capture_rng', stalled)
    monkeypatch.setattr(logging, 'FINISH_TIMEOUT', .05)
    started = time.monotonic()
    logging.finish(hook)
    assert time.monotonic() - started < .5
    assert not accesses


@pytest.mark.parametrize('form', ['exact', 'integer', 'zeros'])
def test_credential_directory_refuses_delivery(tmp_path, monkeypatch, form):
    key = '0' + '1234567890' * 3 + '123456789'
    monkeypatch.setenv('WANDB_API_KEY', key)
    value = {'exact': key, 'integer': str(int(key)), 'zeros': '000' + str(int(key))}[form]
    sdk = FakeSDK()
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    directory = tmp_path / value
    hook = logging.WandbLogger(directory, sync=False)
    logging.emit(hook, dict(event='run_start'))
    logging.finish(hook)
    assert sdk.kwargs is None and not hook.logging_ok
    assert not directory.exists()


def test_non_string_extra_config_key_refuses_telemetry(tmp_path, monkeypatch):
    from test_pretrain import make_case
    from oxyformer.training.pretrain import pretrain
    monkeypatch.setitem(sys.modules, 'wandb', FakeSDK())
    view, split, config = make_case(tmp_path / 'attempt')
    hook = logging.WandbLogger(tmp_path / 'logs', sync=False)
    config = replace(config, run_logger=hook)
    config.__dict__[1] = 'cache-only metadata'
    config.__dict__['extra'] = 'another unknown key'
    artifact = pretrain(view, split, config, 1103)
    assert artifact.sha256 and not hook.logging_ok
    assert 'refused: undeclared config key' in hook.spool.read_text()
    assert 'cache-only metadata' not in hook.spool.read_text()


def test_spool_footer_redacts_numeric_credential(tmp_path, monkeypatch):
    key = '0' * 39 + '2'
    monkeypatch.setenv('WANDB_API_KEY', key)
    monkeypatch.setitem(sys.modules, 'wandb', FakeSDK())
    # Keep this test on the footer path: pytest's numbered directory can itself
    # contain the credential's decimal form and correctly refuse all delivery.
    while True:
        directory = Path(tempfile.mkdtemp(prefix='footer-', dir='/tmp'))
        if logging.sanitize(str(directory)) == str(directory):
            break
        shutil.rmtree(directory)
    try:
        hook = logging.WandbLogger(directory, sync=False)
        logging.emit(hook, dict(event='run_start'))
        logging.emit(hook, dict(event='train_step'))
        logging.finish(hook)
        footer = json.loads(hook.spool.read_text().splitlines()[-1])
        assert footer.get('count') != int(key)
        assert not hook.logging_ok
    finally:
        shutil.rmtree(directory)


def test_finish_timeout_cannot_orphan_git_group(tmp_path, monkeypatch):
    import os
    import signal
    executable = tmp_path / 'git'
    marker = tmp_path / 'git-pid'
    executable.write_text('#!/usr/bin/env python3\nimport os,time\n'
                          f'open({str(marker)!r}, "w").write(str(os.getpid()))\n'
                          'time.sleep(100)\n')
    executable.chmod(0o700)
    monkeypatch.setenv('PATH', str(tmp_path) + ':' + os.environ['PATH'])
    monkeypatch.setattr(logging, 'GIT_TIMEOUT', 1.)
    monkeypatch.setattr(logging, 'FINISH_TIMEOUT', .2)
    monkeypatch.setitem(sys.modules, 'wandb', None)
    hook = logging.WandbLogger(tmp_path / 'logs', sync=False)
    logging.emit(hook, dict(event='run_start', git_sha=None))
    started = time.monotonic()
    logging.finish(hook)
    assert time.monotonic() - started < 2.5
    assert marker.exists(), 'Git double must have entered'
    pid = int(marker.read_text())
    alive = False
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        pass
    finally:
        if alive:
            os.killpg(pid, signal.SIGKILL)
    assert not alive, 'finish timeout left an owned Git process behind'
