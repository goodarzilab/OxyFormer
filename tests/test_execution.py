"""Offline synthetic execution and campaign acceptance. No scheduler or GPU."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import zipfile

import pytest
import yaml

from oxyformer.cli import main
from oxyformer.contracts import StageResult
from oxyformer.execution.campaign import expand_campaign, validate_plan
from oxyformer.execution.identity import code_identity, scientific_fingerprint
from oxyformer.execution.paths import atomic_json, safe_extract
from oxyformer.execution.runner import dependency_variable, resolve_dependencies, run
from oxyformer.provenance import ArtifactLineage, ArtifactRecord, ContractError, file_hash


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def commit(repo):
    git(repo, 'add', '.')
    git(repo, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-qm', 'Synthetic fixture')
    return git(repo, 'rev-parse', 'HEAD')


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.delenv('SWARM_UNIT_DIR', raising=False)
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'configs/execution').mkdir(parents=True)
    (repo / 'src').mkdir()
    (repo / 'src/science.py').write_text('value = 1\n')
    (repo / 'README.md').write_text('fixture\n')
    (repo / 'configs/execution/stages.yaml').write_text(yaml.safe_dump({
        'schema_version': 1, 'stages': {'dummy': {'module': 'oxyformer.dummy',
        'acquisition_receipts': {'data-unit': 'receipts.json'}}}}))
    (repo / 'configs/approvals.yaml').write_text('schema_version: 1\napproved_by: fixture\n')
    git(repo, 'init', '-q')
    commit(repo)
    out = tmp_path / 'attempt'
    out.mkdir()
    (out / 'code_commit.txt').write_text(git(repo, 'rev-parse', 'HEAD') + '\n')
    monkeypatch.setitem(sys.modules, 'oxyformer.dummy', SimpleNamespace(run_stage=dummy))
    return repo, out


def dummy(request):
    request.verify_inputs()
    out = Path(request.output_dir)
    assert Path(os.environ['HF_HOME']).is_relative_to(out)
    task = json.loads(Path(request.task_path).read_text())
    for path in task.get('outputs', ['value.json']):
        atomic_json(out, path, {'value': 1})
    lineage = ArtifactLineage(source_hashes=(sha256(b'fixture').hexdigest(),), unit_ids=('dummy',),
                              parent_hashes=(), split_hash=None, config_hash=request.config_hash,
                              model_hash=None, environment=(('python', 'fixture'),), seed=None,
                              parameter_count=None)
    outputs = task.get('outputs', ['value.json'])
    return StageResult(request_hash=request.content_hash, status='pass', message='fixture',
                       artifacts=tuple(ArtifactRecord(path=p, sha256=file_hash(out / p),
                                                      lineage=lineage, kind='fixture') for p in outputs))


def task_file(out, **changes):
    task = {'id': 'dummy', 'stage': 'dummy', 'needs': {}, 'outputs': ['value.json']}
    task.update(changes)
    path = out / 'input-task.json'
    path.write_text(json.dumps(task))
    return path


def test_dummy_stage_atomic_records_and_cache_isolation(runtime, monkeypatch):
    repo, out = runtime
    monkeypatch.setenv('HF_HOME', '/unrelated/cache')
    assert run('dummy', out, repo, task_file=task_file(out)).status == 'pass'
    assert os.environ['HF_HOME'] == '/unrelated/cache'
    result = json.loads((out / '_execution/result.json').read_text())
    assert result['payload']['status'] == 'pass'
    env = json.loads((out / '_execution/environment.json').read_text())
    assert env['executable'] == sys.executable and env['packages']
    with pytest.raises(FileExistsError):
        atomic_json(out, '_execution/result.json', {})


def test_dependency_normalization(tmp_path):
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    assert dependency_variable('atlas-east.north') == 'SWARM_DEP_ATLAS_EAST_NORTH'
    assert resolve_dependencies(['atlas-east.north'], {'SWARM_DEP_ATLAS_EAST_NORTH': str(upstream)}) == {
        'atlas-east.north': upstream}
    with pytest.raises(ContractError, match='missing dependency variable'):
        resolve_dependencies(['atlas-east.north'], {'SWARM_DEP_atlas-east.north': str(upstream)})
    with pytest.raises(ContractError, match='normalization collision'):
        resolve_dependencies(['a-b', 'a_b'], {'SWARM_DEP_A_B': str(upstream)})


@pytest.mark.parametrize('operation', [
    'write', 'truncate', 'unlink', 'rename', 'mkdir', 'chmod', 'utime',
    'subprocess_write', 'subprocess_chmod', 'native_write', 'dir_fd', 'inherited_fd',
])
def test_faulty_stage_cannot_mutate_upstream(runtime, tmp_path, monkeypatch, operation):
    import ctypes

    repo, out = runtime
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    source = upstream / 'data.json'
    source.write_text('{"fixture":1}')
    (upstream / 'receipts.json').write_text('{}')
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(upstream))
    before = source.read_bytes(), source.stat().st_mode, source.stat().st_mtime_ns
    directory_before = upstream.stat().st_mtime_ns
    inherited_fd = os.open(source, os.O_WRONLY)

    def faulty(request):
        path = Path(request.dependency_paths[0])
        if operation == 'write':
            path.write_text('corrupted')
        elif operation == 'truncate':
            os.truncate(path, 0)
        elif operation == 'unlink':
            path.unlink()
        elif operation == 'rename':
            path.rename(upstream / 'renamed')
        elif operation == 'mkdir':
            (upstream / 'unowned').mkdir()
        elif operation == 'chmod':
            path.chmod(0o600)
        elif operation == 'utime':
            os.utime(path, ns=(1, 1))
        elif operation == 'subprocess_write':
            subprocess.run([sys.executable, '-c',
                            'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("bad")',
                            str(path)], check=True)
        elif operation == 'subprocess_chmod':
            subprocess.run(['chmod', '600', str(path)], check=True)
        elif operation == 'native_write':
            libc = ctypes.CDLL(None, use_errno=True)
            fd = libc.open(os.fsencode(path), os.O_WRONLY | os.O_TRUNC)
            if fd < 0:
                raise OSError(ctypes.get_errno(), 'native dependency write denied')
            os.close(fd)
        elif operation == 'dir_fd':
            fd = os.open(upstream, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.chmod('data.json', 0o600, dir_fd=fd)
            finally:
                os.close(fd)
        else:
            os.write(inherited_fd, b'bad')
        return dummy(request)

    monkeypatch.setitem(sys.modules, 'oxyformer.dummy', SimpleNamespace(run_stage=faulty))
    try:
        result = run('dummy', out, repo, deps_env=True,
                     task_file=task_file(out, needs={'data-unit': ['data.json', 'receipts.json']}))
    finally:
        os.close(inherited_fd)
    assert result.status == 'fail'
    assert source.exists()
    assert (source.read_bytes(), source.stat().st_mode, source.stat().st_mtime_ns) == before
    assert upstream.stat().st_mtime_ns == directory_before
    assert sorted(p.name for p in upstream.iterdir()) == ['data.json', 'receipts.json']


def test_upstream_unchanged_and_output_overlap_rejected(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    source = upstream / 'data.json'
    source.write_text('{"fixture":1}')
    (upstream / 'receipts.json').write_text('{}')
    before = source.read_bytes(), source.stat().st_mode, source.stat().st_mtime_ns
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(upstream))
    task = task_file(out, needs={'data-unit': ['data.json', 'receipts.json']})
    assert run('dummy', out, repo, deps_env=True, task_file=task).status == 'pass'
    assert (source.read_bytes(), source.stat().st_mode, source.stat().st_mtime_ns) == before
    nested = upstream / 'child'
    nested.mkdir()
    (nested / 'code_commit.txt').write_text(git(repo, 'rev-parse', 'HEAD'))
    with pytest.raises(ContractError, match='overlaps an upstream'):
        run('dummy', nested, repo, deps_env=True, task_file=task)


@pytest.mark.parametrize('name', ['../outside', '/tmp/escape', 'nested/../../escape', './value', 'a\\b'])
def test_output_escape(runtime, name):
    repo, out = runtime
    with pytest.raises(ContractError, match='relative path'):
        run('dummy', out, repo, task_file=task_file(out, outputs=[name]))


def test_output_symlink_escape(runtime, tmp_path):
    repo, out = runtime
    (out / 'alias').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ContractError, match='escapes attempt'):
        run('dummy', out, repo, task_file=task_file(out, outputs=['alias/escaped']))


def test_result_symlink_escape(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.write_text('{"value":1}')
    def escaping(request):
        result = dummy(request)
        target = out / 'value.json'
        target.unlink()
        target.symlink_to(elsewhere)
        return result
    monkeypatch.setitem(sys.modules, 'oxyformer.dummy', SimpleNamespace(run_stage=escaping))
    result = run('dummy', out, repo, task_file=task_file(out))
    assert result.status == 'fail' and 'escapes output' in result.message


def test_isolated_stage_allows_owned_metadata_native_io_and_temporary_files(runtime, monkeypatch):
    repo, out = runtime

    def owned(request):
        import multiprocessing
        import numpy as np
        import tempfile

        root = Path(request.output_dir)
        data = np.memmap(root / 'array.bin', dtype='float64', mode='w+', shape=(3,))
        data[:] = [1, 2, 3]
        data.flush()
        del data
        shutil.copy2(root / 'array.bin', root / 'copy.bin')
        subprocess.run(['chmod', '600', str(root / 'copy.bin')], check=True)
        subprocess.run(['touch', str(root / 'copy.bin')], check=True)
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.chmod('copy.bin', 0o640, dir_fd=fd)
        finally:
            os.close(fd)
        with tempfile.NamedTemporaryFile() as stream:
            assert Path(stream.name).is_relative_to(root)
            stream.write(b'fixture')
        with tempfile.TemporaryFile() as stream:
            os.fchmod(stream.fileno(), 0o600)
        queue = multiprocessing.get_context('fork').Queue()
        queue.put('fixture')
        assert queue.get(timeout=5) == 'fixture'
        queue.close()
        queue.join_thread()
        return dummy(request)

    monkeypatch.setitem(sys.modules, 'oxyformer.dummy', SimpleNamespace(run_stage=owned))
    result = run('dummy', out, repo, task_file=task_file(out))
    assert result.status == 'pass', result.message
    assert (out / 'copy.bin').stat().st_mode & 0o777 == 0o640


def test_isolation_unavailable_blocks_before_stage(runtime, monkeypatch):
    from oxyformer.execution import isolation

    repo, out = runtime

    def unavailable(_):
        raise OSError('synthetic unavailable kernel protection')

    monkeypatch.setattr(isolation, 'restrict_writes', unavailable)
    result = run('dummy', out, repo, task_file=task_file(out))
    assert result.status == 'blocked'
    assert 'unavailable kernel protection' in result.message
    assert not (out / 'value.json').exists()


def test_stage_import_is_already_guarded(runtime, tmp_path, monkeypatch):
    from oxyformer.execution import runner

    repo, out = runtime
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    source = upstream / 'receipts.json'
    source.write_text('{}')
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(upstream))

    def faulty_import(_):
        source.write_text('corrupted during import')
        return SimpleNamespace(run_stage=dummy)

    monkeypatch.setattr(runner.importlib, 'import_module', faulty_import)
    result = run('dummy', out, repo, deps_env=True,
                 task_file=task_file(out, needs={'data-unit': ['receipts.json']}))
    assert result.status == 'fail'
    assert source.read_text() == '{}'


def test_code_commit_and_dirty_repo(runtime):
    repo, out = runtime
    (out / 'code_commit.txt').write_text('0' * 40)
    with pytest.raises(ContractError, match='HEAD'):
        code_identity(repo, out)
    (out / 'code_commit.txt').write_text(git(repo, 'rev-parse', 'HEAD'))
    (repo / 'src/science.py').write_text('value = 2\n')
    with pytest.raises(ContractError, match='tracked modifications'):
        code_identity(repo, out)


def test_missing_module_blocks_lazily(runtime):
    repo, out = runtime
    sys.modules.pop('oxyformer.dummy', None)
    result = run('dummy', out, repo, task_file=task_file(out))
    assert result.status == 'blocked'
    assert (out / '_execution/result.json').exists()


def test_cli_selects_task(runtime):
    repo, out = runtime
    path = out / 'tasks.json'
    path.write_text(json.dumps({'tasks': [{'id': 'selected', 'stage': 'dummy', 'outputs': ['value.json']}]}))
    assert main(['run-stage', '--stage', 'dummy', '--out', str(out), '--repo', str(repo),
                 '--deps-env', '--task', str(path), '--task-id', 'selected']) == 0
    assert json.loads((out / '_execution/task.json').read_text())['id'] == 'selected'


def locked_task(repo, out, upstream, monkeypatch):
    upstream.mkdir()
    (upstream / 'code_commit.txt').write_text(git(repo, 'rev-parse', 'HEAD'))
    lock = upstream / 'recipe_lock.json'
    def producer(request):
        result = dummy(request)
        lock.write_text(json.dumps({'scientific_fingerprint': scientific_fingerprint(repo)}))
        return replace(result, artifacts=(replace(result.artifacts[0], sha256=file_hash(lock)),))
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, 'oxyformer.dummy', SimpleNamespace(run_stage=producer))
        assert run('dummy', upstream, repo,
                   task_file=task_file(upstream, outputs=['recipe_lock.json'])).status == 'pass'
    monkeypatch.setenv('SWARM_DEP_CAMPAIGN_LOCK', str(upstream))
    return task_file(out, needs={'campaign-lock': ['recipe_lock.json']},
                     recipe_lock={'dependency': 'campaign-lock', 'path': 'recipe_lock.json',
                                  'sha256': file_hash(lock)})


@pytest.mark.parametrize('change', ['src/science.py', 'configs/new.yaml', 'docs/plan/protocol.md'])
def test_locked_recipe_rejects_scientific_drift(runtime, tmp_path, monkeypatch, change):
    repo, out = runtime
    task = locked_task(repo, out, tmp_path / 'lock', monkeypatch)
    path = repo / change
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('changed\n')
    (out / 'code_commit.txt').write_text(commit(repo))
    with pytest.raises(ContractError, match='recipe scientific code/config drift'):
        run('dummy', out, repo, deps_env=True, task_file=task)


def test_locked_recipe_permits_defined_documentation_change(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    task = locked_task(repo, out, tmp_path / 'lock', monkeypatch)
    (repo / 'README.md').write_text('documentation update\n')
    (out / 'code_commit.txt').write_text(commit(repo))
    assert run('dummy', out, repo, deps_env=True, task_file=task).status == 'pass'


def test_continuation_ownership_and_consecutive_steps(runtime, tmp_path, monkeypatch):
    repo, old = runtime
    first_task = task_file(old, id='first', continuation={'owner': 'work-1', 'step': 0, 'predecessor': None})
    assert run('dummy', old, repo, task_file=first_task).status == 'pass'
    monkeypatch.setenv('SWARM_DEP_FIRST', str(old))
    for owner, step, expected in [('wrong-owner', 1, 'ownership'), ('work-1', 2, 'consecutive'), ('work-1', 1, None)]:
        out = tmp_path / f'next-{owner}-{step}'
        out.mkdir()
        (out / 'code_commit.txt').write_text(git(repo, 'rev-parse', 'HEAD'))
        task = task_file(out, id='next', needs={'first': ['_execution/task.json', '_execution/request.json',
                                                   '_execution/result.json', 'value.json']},
                         continuation={'owner': owner, 'step': step, 'predecessor': 'first'})
        if expected:
            with pytest.raises(ContractError, match=expected):
                run('dummy', out, repo, task_file=task, deps_env=True)
        else:
            assert run('dummy', out, repo, task_file=task, deps_env=True).status == 'pass'


def test_continuation_requires_declared_predecessor(runtime):
    repo, out = runtime
    task = task_file(out, continuation={'owner': 'work', 'step': 1, 'predecessor': 'sibling'})
    with pytest.raises(ContractError, match='explicit dependency'):
        run('dummy', out, repo, task_file=task, deps_env=True)


@pytest.mark.parametrize('attack', ['traversal', 'absolute', 'symlink', 'hardlink', 'device', 'duplicate'])
def test_safe_tar_rejects_unsafe_members(tmp_path, attack):
    archive = tmp_path / 'fixture.tar'
    with tarfile.open(archive, 'w') as tar:
        member = tarfile.TarInfo({'traversal': '../escape', 'absolute': '/escape'}.get(attack, 'file'))
        if attack == 'symlink':
            member.type, member.linkname = tarfile.SYMTYPE, '/tmp/escape'
        elif attack == 'hardlink':
            member.type, member.linkname = tarfile.LNKTYPE, '../escape'
        elif attack == 'device':
            member.type = tarfile.CHRTYPE
        tar.addfile(member)
        if attack == 'duplicate':
            tar.addfile(member)
    with pytest.raises(ContractError):
        safe_extract(archive, tmp_path, 'unpacked')
    assert not (tmp_path / 'unpacked').exists()


def test_safe_extract_selection_budget_and_upstream_preserved(tmp_path):
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.writestr('chosen/data', 'tiny')
        handle.writestr('other/data', 'other')
    before = file_hash(archive)
    target = safe_extract(archive, tmp_path, 'unpacked', members=['chosen/data'], max_bytes=4)
    assert (target / 'chosen/data').read_text() == 'tiny'
    assert not (target / 'other').exists()
    assert file_hash(archive) == before
    with pytest.raises(ContractError, match='byte limit'):
        safe_extract(archive, tmp_path, 'oversized', max_bytes=4)
    assert not (tmp_path / 'oversized').exists()


@pytest.fixture
def spec():
    return {'schema_version': 1, 'id': 'screen-01', 'kind': 'screening',
            'prerequisites': ['stage-runner', 'coverage-harness'],
            'inputs': {'campaign-lock': ['recipe_lock.json']},
            'recipe_lock': {'dependency': 'campaign-lock', 'path': 'recipe_lock.json', 'sha256': 'a' * 64},
            'work': [{'id': 'fold-0', 'stage': 'primary', 'parameters': {'fold': 0, 'seed': 1103},
                      'outputs': ['continuation.tar', 'progress.json'],
                      'slices': [{'gpus': 1, 'wall_seconds': 14400}, {'gpus': 2, 'wall_seconds': 7200}]},
                     {'id': 'fold-1', 'stage': 'primary', 'parameters': {'fold': 1, 'seed': 1103},
                      'outputs': ['continuation.tar', 'progress.json'],
                      'slices': [{'gpus': 1, 'wall_seconds': 3600}]}],
            'collector': {'stage': 'campaign-collect', 'outputs': ['summary.json'], 'wall_seconds': 3600}}


def test_campaign_reproducible_bounded_and_merge_barriers(spec):
    plan = expand_campaign(spec, {})
    assert plan == expand_campaign(deepcopy(spec), {})
    assert len(plan['expected_leaves']) == 3
    assert len(set(u['id'] for u in plan['units'])) == 4
    for unit in plan['units']:
        assert len(unit['id']) <= 32 and unit['gpu_hours'] <= 4
        assert {'stage-runner', 'coverage-harness', 'campaign-lock'} <= set(unit['needs'])
        assert {'--partition=standard', '--account=root', '--qos=normal'} <= set(unit['sbatch'])
        assert '--array' not in ' '.join(unit['sbatch'])
        command = unit['command']
        assert 'git clone --depth 1 --branch dev' in command
        assert 'PYTHONPATH="$SWARM_UNIT_DIR/src/src"' in command
        assert 'envs/oxyformer/bin/python -m oxyformer.cli run-stage' in command
        assert not any(token in command for token in ['sbatch ', 'srun ', 'crontab ', 'systemctl ', 'release_lock'])
    first, second = plan['tasks'][:2]
    assert second['continuation']['predecessor'] == first['id']
    assert second['continuation']['owner'] == first['continuation']['owner']
    assert first['id'] in plan['units'][1]['needs']
    assert set(plan['expected_leaves']) <= set(plan['units'][-1]['needs'])


def test_collector_omitted_dependency_is_rejected(spec):
    plan = expand_campaign(spec, {})
    plan['units'][-1]['needs'].remove(plan['expected_leaves'][0])
    with pytest.raises(ContractError, match='collector omitted required leaf dependency'):
        validate_plan(plan, {})


def test_omitted_leaf_cannot_hide_by_editing_expected_list(spec):
    plan = expand_campaign(spec, {})
    removed = plan['expected_leaves'].pop()
    plan['units'] = [u for u in plan['units'] if u['id'] != removed]
    plan['units'][-1]['needs'].remove(removed)
    with pytest.raises(ContractError, match='collector omitted|required campaign leaves'):
        validate_plan(plan, {})


def test_cycles_rejected(spec):
    plan = expand_campaign(spec, {})
    plan['units'][0]['needs'].append(plan['units'][1]['id'])
    with pytest.raises(ContractError, match='cycle'):
        validate_plan(plan, {})


def test_continuation_plan_ownership_mutation(spec):
    plan = expand_campaign(spec, {})
    plan['tasks'][1]['continuation']['owner'] = 'different-work'
    with pytest.raises(ContractError, match='ownership drift'):
        validate_plan(plan, {})


@pytest.mark.parametrize('mutation,error', [
    ('too_many', 'forty'), ('too_long', 'four GPU-hours'), ('multi_gpu', 'four GPU-hours'),
    ('template', 'unresolved template'), ('collision', 'normalization collision'), ('escape', 'relative path')])
def test_invalid_campaigns(spec, mutation, error):
    if mutation == 'too_many':
        spec['work'][0]['slices'] *= 21
    elif mutation == 'too_long':
        spec['work'][0]['slices'][0]['wall_seconds'] = 14401
    elif mutation == 'multi_gpu':
        spec['work'][0]['slices'][0]['gpus'] = 2
    elif mutation == 'template':
        spec['id'] = 'run-{fold}'
    elif mutation == 'collision':
        spec['prerequisites'] += ['code-a', 'code_a']
    elif mutation == 'escape':
        spec['work'][0]['outputs'] = ['../escape']
    with pytest.raises(ContractError, match=error):
        expand_campaign(spec, {})


@pytest.mark.parametrize('kind', ['final-coverage', 'anchor', 'refit-audit'])
def test_campaign_allocation_required(spec, kind):
    spec['kind'] = kind
    with pytest.raises(ContractError, match='missing owner campaign allocation'):
        expand_campaign(spec, {})
    approvals = {'owner_decisions': {'campaign_allocations': {spec['id']: {'kind': kind, 'gpu_hours': 9}}}}
    assert expand_campaign(spec, approvals)['units']


def test_unit_mutation_cannot_inject_scheduler_or_array(spec):
    for field, value in [('command', 'sbatch something'), ('sbatch', ['--array=1-40']), ('gpu_hours', 99)]:
        plan = expand_campaign(spec, {})
        plan['units'][0][field] = value
        with pytest.raises(ContractError, match='drift'):
            validate_plan(plan, {})


def test_failed_upstream_cannot_feed_another_stage(runtime, tmp_path, monkeypatch):
    repo, upstream = runtime
    def failed(request):
        return StageResult(request_hash=request.content_hash, status='fail', artifacts=(), message='gate failed')
    monkeypatch.setitem(sys.modules, 'oxyformer.dummy', SimpleNamespace(run_stage=failed))
    assert run('dummy', upstream, repo, task_file=task_file(upstream)).status == 'fail'
    out = tmp_path / 'consumer'
    out.mkdir()
    (out / 'code_commit.txt').write_text(git(repo, 'rev-parse', 'HEAD'))
    monkeypatch.setenv('SWARM_DEP_GATE', str(upstream))
    with pytest.raises(ContractError, match='did not pass'):
        run('dummy', out, repo, deps_env=True,
            task_file=task_file(out, needs={'gate': ['_execution/result.json']}))


def test_stage_required_dependency_cannot_be_removed(runtime):
    repo, out = runtime
    registry = repo / 'configs/execution/stages.yaml'
    registry.write_text(yaml.safe_dump({'schema_version': 1, 'stages': {
        'dummy': {'module': 'oxyformer.dummy', 'needs': {'required': ['data.json']}}}}))
    (out / 'code_commit.txt').write_text(commit(repo))
    with pytest.raises(ContractError, match='stage-required dependency'):
        run('dummy', out, repo, deps_env=True, task_file=task_file(out))


def test_locked_approvals_cannot_be_replaced(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    task = locked_task(repo, out, tmp_path / 'lock', monkeypatch)
    other = tmp_path / 'changed-approvals.yaml'
    other.write_text('schema_version: 1\napproved_by: different\n')
    with pytest.raises(ContractError, match='locked approvals'):
        run('dummy', out, repo, deps_env=True, task_file=task, approvals=other)


def test_stage_dependency_without_receipt_is_rejected(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    upstream = tmp_path / 'incomplete-stage'
    upstream.mkdir()
    (upstream / 'data.json').write_text('{}')
    monkeypatch.setenv('SWARM_DEP_INCOMPLETE', str(upstream))
    with pytest.raises(ContractError, match='stage receipt missing'):
        run('dummy', out, repo, deps_env=True,
            task_file=task_file(out, needs={'incomplete': ['data.json']}))


def test_slurm_minute_rounding_is_included_in_gpu_bound():
    from oxyformer.execution.campaign import resources
    with pytest.raises(ContractError, match='four GPU-hours'):
        resources(7, 2057)  # Slurm grants 35 minutes: 4.0833 GPU-hours.
    flags, gpu_hours = resources(2, 61)
    assert '--time=00:02:00' in flags
    assert gpu_hours == pytest.approx(2 * 120 / 3600)


def test_safe_tar_accepts_dot_prefix_without_traversal(tmp_path):
    archive = tmp_path / 'payload.tar'
    with tarfile.open(archive, 'w') as tar:
        directory = tarfile.TarInfo('.')
        directory.type = tarfile.DIRTYPE
        tar.addfile(directory)
        member = tarfile.TarInfo('./nested/data')
        member.size = 4
        tar.addfile(member, io.BytesIO(b'tiny'))
    target = safe_extract(archive, tmp_path, 'unpacked')
    assert (target / 'nested/data').read_bytes() == b'tiny'


def test_campaign_lock_uses_unit_id_distinct_from_stage_name(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    actual = yaml.safe_load((Path(__file__).parents[1] / 'configs/execution/stages.yaml').read_text())
    lock_settings = deepcopy(actual['stages']['campaign-lock'])
    assert 'tract-gate' in lock_settings['needs']
    assert 'tract-support-gate' not in lock_settings['needs']
    lock_settings['module'] = 'oxyformer.dummy'
    stages = {'campaign-lock': lock_settings}
    for stage, unit in [('tract-support-gate', 'tract-gate'), ('simulation-smoke', 'simulation-smoke')]:
        stages[stage] = {'module': 'oxyformer.dummy', 'outputs': lock_settings['needs'][unit]}
    (repo / 'configs/execution/stages.yaml').write_text(yaml.safe_dump({'schema_version': 1, 'stages': stages}))
    head = commit(repo)
    (out / 'code_commit.txt').write_text(head)
    for stage, unit in [('tract-support-gate', 'tract-gate'), ('simulation-smoke', 'simulation-smoke')]:
        upstream = tmp_path / unit
        upstream.mkdir()
        (upstream / 'code_commit.txt').write_text(head)
        assert run(stage, upstream, repo).status == 'pass'
        monkeypatch.setenv(dependency_variable(unit), str(upstream))
    assert run('campaign-lock', out, repo, deps_env=True).status == 'pass'


def test_acquisition_exemption_requires_declared_source_receipt(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'data.json').write_text('{}')
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(source))
    with pytest.raises(ContractError, match='acquisition receipt must be a declared input'):
        run('dummy', out, repo, deps_env=True,
            task_file=task_file(out, needs={'data-unit': ['data.json']}))


def test_task_cannot_exempt_an_incomplete_stage(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'data.json').write_text('{}')
    (source / 'receipts.json').write_text('{}')
    monkeypatch.setenv('SWARM_DEP_INCOMPLETE', str(source))
    with pytest.raises(ContractError, match='stage receipt missing'):
        run('dummy', out, repo, deps_env=True, task_file=task_file(
            out, needs={'incomplete': ['data.json', 'receipts.json']},
            acquisition_receipts={'incomplete': 'receipts.json'}))


def test_safe_tar_dot_prefix_does_not_hide_traversal_or_duplicates(tmp_path):
    for index, names in enumerate([['./../escape'], ['./same', 'same']]):
        archive = tmp_path / f'bad-{index}.tar'
        with tarfile.open(archive, 'w') as tar:
            for name in names:
                tar.addfile(tarfile.TarInfo(name))
        with pytest.raises(ContractError):
            safe_extract(archive, tmp_path, f'unpacked-{index}')
        assert not (tmp_path / f'unpacked-{index}').exists()


def test_cli_import_from_pristine_repo_keeps_bytecode_ignored(runtime, tmp_path):
    import inspect
    repo, out = runtime
    original = Path(__file__).parents[1]
    shutil.copytree(original / 'src/oxyformer', repo / 'src/oxyformer',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(original / '.gitignore', repo / '.gitignore')
    imports = '''from pathlib import Path
import os
import json
from hashlib import sha256
from oxyformer.contracts import StageResult
from oxyformer.provenance import ArtifactLineage, ArtifactRecord, file_hash
from oxyformer.execution.paths import atomic_json
'''
    (repo / 'src/oxyformer/dummy.py').write_text(imports + inspect.getsource(dummy) + '\nrun_stage = dummy\n')
    (out / 'code_commit.txt').write_text(commit(repo))
    task = task_file(out)
    env = dict(os.environ, PYTHONPATH=str(repo / 'src'), CUDA_VISIBLE_DEVICES='')
    process = subprocess.run([sys.executable, '-m', 'oxyformer.cli', 'run-stage',
                              '--stage', 'dummy', '--repo', str(repo), '--out', str(out),
                              '--task', str(task)], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert process.returncode == 0, process.stdout + process.stderr
    assert list((repo / 'src/oxyformer/__pycache__').glob('*.pyc'))
    assert git(repo, 'status', '--porcelain', '--untracked-files=all') == ''


def test_locked_primary_stage_cannot_omit_recipe(runtime):
    repo, out = runtime
    original = Path(__file__).parents[1]
    settings = yaml.safe_load((original / 'configs/execution/stages.yaml').read_text())['stages']['primary']
    settings['module'] = 'oxyformer.dummy'
    (repo / 'configs/execution/stages.yaml').write_text(yaml.safe_dump({
        'schema_version': 1, 'stages': {'primary': settings}}))
    (out / 'code_commit.txt').write_text(commit(repo))
    with pytest.raises(ContractError, match='requires a recipe lock'):
        run('primary', out, repo, task_file=task_file(out, stage='primary'))


@pytest.mark.parametrize('seconds,allocation', [((360, 720, 13320), 4.0), ((360, 720, 1080), 0.6)])
def test_exact_campaign_allocation_is_not_rejected_by_float_sum(spec, seconds, allocation):
    spec['kind'] = 'final-coverage'
    spec['work'] = spec['work'][:1]
    spec['work'][0]['slices'] = [{'gpus': 1, 'wall_seconds': value} for value in seconds]
    approvals = {'owner_decisions': {'campaign_allocations': {spec['id']: {'kind': 'final-coverage', 'gpu_hours': allocation}}}}
    assert expand_campaign(spec, approvals)['units']
    approvals['owner_decisions']['campaign_allocations'][spec['id']]['gpu_hours'] = allocation - 1e-9
    with pytest.raises(ContractError, match='allocation does not cover'):
        expand_campaign(spec, approvals)


def test_json_task_preserves_exponent_number_types(runtime):
    repo, out = runtime
    parameters = {'lr': 1e-5, 'large': 1e20, 'numeric_label': '1e-05'}
    assert run('dummy', out, repo, task_file=task_file(out, parameters=parameters)).status == 'pass'
    actual = json.loads((out / '_execution/task.json').read_text())['parameters']
    assert actual == parameters
    assert isinstance(actual['lr'], float) and isinstance(actual['large'], float)


@pytest.mark.parametrize('field', ['outputs', 'parameters'])
def test_single_brace_campaign_templates_rejected(spec, field):
    if field == 'outputs':
        spec['work'][0]['outputs'] = ['result-{fold}.json']
    else:
        spec['work'][0]['parameters']['fold'] = '{fold:02d}'
    with pytest.raises(ContractError, match='unresolved template'):
        expand_campaign(spec, {})


def test_cli_invalid_repo_is_blocked_instead_of_a_traceback(tmp_path, monkeypatch):
    monkeypatch.delenv('SWARM_UNIT_DIR', raising=False)
    repo, out = tmp_path / 'not-a-repo', tmp_path / 'attempt'
    repo.mkdir()
    out.mkdir()
    assert main(['run-stage', '--stage', 'dummy', '--repo', str(repo), '--out', str(out)]) == 2


def test_campaign_task_requires_lock_even_for_generic_stage(runtime):
    repo, out = runtime
    with pytest.raises(ContractError, match='requires a recipe lock'):
        run('dummy', out, repo, task_file=task_file(out, campaign='screen-01'))


def test_cli_malformed_task_types_are_blocked(runtime):
    repo, out = runtime
    task = task_file(out, outputs=None)
    assert main(['run-stage', '--stage', 'dummy', '--repo', str(repo), '--out', str(out),
                 '--task', str(task)]) == 2


def test_json_content_keeps_types_regardless_of_filename(tmp_path):
    from oxyformer.execution.runner import read_mapping
    task = tmp_path / 'task.yaml'
    task.write_text(json.dumps({'lr': 1e-5}))
    assert read_mapping(task)['lr'] == 1e-5


def test_invalid_json_never_falls_back_to_yaml(tmp_path):
    from oxyformer.execution.runner import read_mapping
    task = tmp_path / 'task.json'
    task.write_text('lr: 1.0e-5\n')
    with pytest.raises(ContractError, match='invalid JSON'):
        read_mapping(task)


@pytest.mark.parametrize('upstream_locked,change,error', [
    (True, 'src/science.py', 'dependency scientific code/config drift'),
    (True, 'README.md', None),
    (False, 'src/science.py', None),
])
def test_locked_campaign_checks_upstream_science(runtime, tmp_path, monkeypatch,
                                                upstream_locked, change, error):
    repo, upstream = runtime
    if upstream_locked:
        old_task = locked_task(repo, upstream, tmp_path / 'old-lock', monkeypatch)
        value = json.loads(old_task.read_text())
        value.update(id='upstream', campaign='old-campaign')
        old_task.write_text(json.dumps(value))
    else:
        old_task = task_file(upstream, id='upstream')
    assert run('dummy', upstream, repo, deps_env=True, task_file=old_task).status == 'pass'
    before = {str(p): file_hash(p) for p in upstream.rglob('*') if p.is_file()}
    (repo / change).write_text('changed\n')
    head = commit(repo)
    consumer = tmp_path / 'consumer'
    consumer.mkdir()
    (consumer / 'code_commit.txt').write_text(head)
    lock_task = locked_task(repo, consumer, tmp_path / 'new-lock', monkeypatch)
    lock_ref = json.loads(lock_task.read_text())['recipe_lock']
    spec = {'schema_version': 1, 'id': 'new-campaign', 'kind': 'screening',
            'prerequisites': [],
            'inputs': {'campaign-lock': ['recipe_lock.json'], 'upstream': ['value.json']},
            'recipe_lock': lock_ref,
            'work': [{'id': 'work', 'stage': 'dummy', 'outputs': ['value.json'],
                      'slices': [{'gpus': 0, 'wall_seconds': 60}]}],
            'collector': {'stage': 'dummy', 'outputs': ['summary.json'], 'wall_seconds': 60}}
    generated = expand_campaign(spec, {})
    selected = consumer / 'generated-task.json'
    selected.write_text(json.dumps(generated['tasks'][0]))
    monkeypatch.setenv('SWARM_DEP_UPSTREAM', str(upstream))
    if error:
        with pytest.raises(ContractError, match=error):
            run('dummy', consumer, repo, deps_env=True, task_file=selected)
        assert not (consumer / 'value.json').exists()
    else:
        assert run('dummy', consumer, repo, deps_env=True, task_file=selected).status == 'pass'
    assert {str(p): file_hash(p) for p in upstream.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('existing', [False, True])
def test_build_tasks_cli_publishes_into_new_or_existing_directory(tmp_path, spec, existing):
    root = Path(__file__).parents[1]
    spec_file = tmp_path / 'spec.json'
    approvals_file = tmp_path / 'approvals.yaml'
    spec_file.write_text(json.dumps(spec))
    approvals_file.write_bytes((root / 'configs/approvals.yaml').read_bytes())
    out = tmp_path / 'plans' / 'campaign'
    if existing:
        out.mkdir(parents=True)
    command = [sys.executable, str(root / 'scripts/build_tasks.py'), '--spec', str(spec_file),
               '--approvals', str(approvals_file), '--out', str(out)]
    env = dict(os.environ, PYTHONPATH=str(root / 'src'), CUDA_VISIBLE_DEVICES='')
    process = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True)
    assert process.returncode == 0, process.stdout + process.stderr
    plan = json.loads((out / 'expanded_units.json').read_text())
    assert validate_plan(plan, {}) == expand_campaign(spec, {})
    assert json.loads((out / 'task_manifest.json').read_text())['tasks'] == plan['tasks']
    before = {str(p): file_hash(p) for p in out.iterdir()}
    repeated = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True)
    assert repeated.returncode != 0
    assert {str(p): file_hash(p) for p in out.iterdir()} == before


@pytest.mark.parametrize('kind', ['final-coverage', 'anchor', 'refit-audit'])
def test_builder_rejects_unanchored_owner_allocation(tmp_path, spec, kind):
    root = Path(__file__).parents[1]
    spec['kind'] = kind
    spec['id'] = 'synthetic-unapproved'
    approvals = {'schema_version': 1, 'approved_by': 'fixture', 'owner_decisions': {
        'campaign_allocations': {spec['id']: {'kind': kind, 'gpu_hours': 9}}}}
    spec_file = tmp_path / 'spec.json'
    spec_file.write_text(json.dumps(spec))
    alternate = tmp_path / 'alternate.yaml'
    alternate.write_text(yaml.safe_dump(approvals))
    out = tmp_path / 'plan'
    process = subprocess.run([sys.executable, str(root / 'scripts/build_tasks.py'),
        '--spec', str(spec_file), '--approvals', str(alternate), '--out', str(out)],
        env=dict(os.environ, PYTHONPATH=str(root / 'src'), CUDA_VISIBLE_DEVICES=''),
        text=True, capture_output=True)
    assert process.returncode != 0, 'Builder accepted an allocation outside its owner record'
    assert 'authoritative owner approvals' in process.stderr
    assert not out.exists()


@pytest.mark.parametrize('exception', [RuntimeError(), AssertionError(), FileNotFoundError('stage output')])
def test_executed_stage_exception_always_publishes_failure(runtime, monkeypatch, exception):
    repo, out = runtime
    def failing(request):
        raise exception
    monkeypatch.setitem(sys.modules, 'oxyformer.dummy', SimpleNamespace(run_stage=failing))
    result = run('dummy', out, repo, task_file=task_file(out))
    assert result.status == 'fail'
    assert result.message
    receipt = StageResult.from_json((out / '_execution/result.json').read_text())
    assert receipt == result


@pytest.mark.parametrize('name', ['config.json', 'task.json', 'request.json',
                                  'environment.json', 'identity.json',
                                  'code_commit.txt', 'result.json'])
def test_parent_controls_cannot_be_replaced_by_stage(runtime, monkeypatch, name):
    repo, out = runtime
    def faulty(request):
        result = dummy(request)
        path = out / name if name == 'code_commit.txt' else out / '_execution' / name
        path.write_text(result.to_json() if name == 'result.json' else 'corrupted control')
        return result
    monkeypatch.setitem(sys.modules, 'oxyformer.dummy', SimpleNamespace(run_stage=faulty))
    result = run('dummy', out, repo, task_file=task_file(out))
    assert result.status == 'fail'
    assert StageResult.from_json((out / '_execution/result.json').read_text()).status == 'fail'
    quarantine = list(out.glob('.execution-rejected-*'))
    assert len(quarantine) == 1
    preserved = quarantine[0] / (name if name == 'code_commit.txt' else 'entry/' + name)
    if name == 'result.json':
        assert StageResult.from_json(preserved.read_text()).status == 'pass'
    else:
        assert preserved.read_text() == 'corrupted control'
    assert (out / 'code_commit.txt').read_text().strip() == git(repo, 'rev-parse', 'HEAD')
    assert json.loads((out / '_execution/identity.json').read_text())['head'] == git(repo, 'rev-parse', 'HEAD')


@pytest.mark.parametrize('operation', ['delete', 'replace', 'hardlink', 'mode',
                                       'namespace_alias', 'namespace_external'])
def test_control_namespace_rejection_preserves_external_targets(runtime, tmp_path, monkeypatch, operation):
    repo, out = runtime
    external = tmp_path / 'external'
    external.mkdir()
    (external / 'sentinel').write_text('unchanged')
    def faulty(request):
        result = dummy(request)
        execution = out / '_execution'
        path = execution / 'identity.json'
        if operation == 'delete':
            path.unlink()
        elif operation == 'replace':
            replacement = out / 'replacement'
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        elif operation == 'hardlink':
            os.link(path, out / 'identity-alias')
        elif operation == 'mode':
            path.chmod(0o400)
        else:
            execution.rename(out / 'moved-execution')
            target = external if operation == 'namespace_external' else out / 'moved-execution'
            execution.symlink_to(target, target_is_directory=True)
            if operation == 'namespace_alias':
                (execution / 'result.json').write_text(result.to_json())
        return result
    monkeypatch.setitem(sys.modules, 'oxyformer.dummy', SimpleNamespace(run_stage=faulty))
    result = run('dummy', out, repo, task_file=task_file(out))
    assert result.status == 'fail'
    assert not (out / '_execution').is_symlink()
    assert StageResult.from_json((out / '_execution/result.json').read_text()).status == 'fail'
    assert list(out.glob('.execution-rejected-*'))
    assert {p.name: p.read_text() for p in external.iterdir()} == {'sentinel': 'unchanged'}


def test_forged_identity_producer_fails_and_consumer_rejects(runtime, tmp_path, monkeypatch):
    repo, upstream = runtime
    other = tmp_path / 'other-repo'
    subprocess.run(['git', 'clone', '-q', '--no-hardlinks', str(repo), str(other)], check=True)
    (other / 'src/science.py').write_text('changed\n')
    other_head = commit(other)
    other_fingerprint = scientific_fingerprint(other)
    task = locked_task(repo, upstream, tmp_path / 'old-lock', monkeypatch)
    def faulty(request):
        result = dummy(request)
        path = upstream / '_execution/identity.json'
        value = json.loads(path.read_text())
        value['scientific_fingerprint'] = other_fingerprint
        path.write_text(json.dumps(value))
        return result
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, 'oxyformer.dummy', SimpleNamespace(run_stage=faulty))
        producer = run('dummy', upstream, repo, deps_env=True, task_file=task)
    assert producer.status == 'fail'
    evidence = next(upstream.glob('.execution-rejected-*/entry/identity.json'))
    assert json.loads(evidence.read_text())['scientific_fingerprint'] == other_fingerprint
    consumer = tmp_path / 'consumer'
    consumer.mkdir()
    (consumer / 'code_commit.txt').write_text(other_head)
    selected = locked_task(other, consumer, tmp_path / 'new-lock', monkeypatch)
    document = json.loads(selected.read_text())
    document['needs']['upstream'] = ['value.json']
    selected.write_text(json.dumps(document))
    monkeypatch.setenv('SWARM_DEP_UPSTREAM', str(upstream))
    with pytest.raises(ContractError, match='dependency stage did not pass'):
        run('dummy', consumer, other, deps_env=True, task_file=selected)


@pytest.mark.parametrize('zombie', [False, True])
def test_parent_with_unrelated_child_is_rejected_without_reaping(runtime, zombie):
    import time
    repo, out = runtime
    child = subprocess.Popen([sys.executable, '-c', '' if zombie else 'import time; time.sleep(30)'])
    try:
        if zombie:
            deadline = time.monotonic() + 5
            while Path(f'/proc/{child.pid}/stat').read_text().split(') ', 1)[1][0] != 'Z':
                assert time.monotonic() < deadline
                time.sleep(0.01)
        with pytest.raises(ContractError, match='childless'):
            run('dummy', out, repo, task_file=task_file(out))
        if zombie:
            assert os.waitpid(child.pid, os.WNOHANG) == (child.pid, 0)
            child.returncode = 0
        else:
            assert child.poll() is None
        assert not (out / 'value.json').exists()
    finally:
        if child.returncode is None:
            child.terminate()
            child.wait(timeout=5)


def test_multithreaded_parent_is_rejected(runtime):
    import threading
    repo, out = runtime
    stop = threading.Event()
    thread = threading.Thread(target=stop.wait)
    thread.start()
    try:
        with pytest.raises(ContractError, match='single-threaded'):
            run('dummy', out, repo, task_file=task_file(out))
        assert thread.is_alive()
        assert not (out / 'value.json').exists()
    finally:
        stop.set()
        thread.join()


def test_unproved_cleanup_does_not_publish_receipt(runtime, monkeypatch):
    from oxyformer.execution import runner
    from oxyformer.execution.isolation import CleanupIncomplete
    repo, out = runtime
    def unproved(*args):
        raise CleanupIncomplete('synthetic unresolved descendants')
    monkeypatch.setattr(runner, 'isolated_stage', unproved)
    with pytest.raises(CleanupIncomplete, match='unresolved descendants'):
        run('dummy', out, repo, task_file=task_file(out))
    assert not (out / '_execution/result.json').exists()


def _publication_case(mode, root):
    """Executed in a fresh, externally timed subprocess; all fixtures synthetic."""
    import signal
    import time
    with pytest.MonkeyPatch.context() as patch:
        repo, out = runtime.__wrapped__(root, patch)
        notification_times = []
        if mode == 'notification_timeout':
            from oxyformer.execution import isolation
            original_service = isolation.service_notification
            def empty_receive(*args):
                original_service(*args)
                started = time.monotonic()
                original_service(*args)  # eventually drains the pending queue
                notification_times.append(time.monotonic() - started)
            patch.setattr(isolation, 'service_notification', empty_receive)
        def stage(request):
            if mode == 'notification_timeout':
                (out / 'owned').write_text('fixture')
                (out / 'owned').chmod(0o600)
                time.sleep(0.4)
            elif mode == 'shared_memory':
                from multiprocessing import shared_memory
                memory = shared_memory.SharedMemory(create=True, size=4)
                memory.buf[:] = b'test'
                memory.close()
                memory.unlink()
            elif mode == 'spawn':
                import multiprocessing
                (out / 'spawn_fixture.py').write_text('def send(queue):\n    queue.put("fixture")\n')
                sys.path.insert(0, str(out))
                import spawn_fixture
                context = multiprocessing.get_context('spawn')
                queue = context.Queue()
                child = context.Process(target=spawn_fixture.send, args=(queue,))
                child.start()
                assert queue.get(timeout=5) == 'fixture'
                child.join(timeout=5)
                assert child.exitcode == 0
                queue.close()
                queue.join_thread()
            elif mode == 'eof_helper':
                read_fd, write_fd = os.pipe()
                helper = subprocess.Popen([sys.executable, '-c',
                    'import os,sys; os.read(int(sys.argv[1]),1)', str(read_fd)], pass_fds=(read_fd,))
                os.close(read_fd)
                (out / 'helper-pid').write_text(str(helper.pid))
                # The worker intentionally retains write_fd until its exit.
            else:
                child = os.fork()
                if child == 0:
                    try:
                        if mode == 'detached':
                            os.setsid()
                            if os.fork():
                                os._exit(0)
                        if mode in ('stubborn', 'detached'):
                            signal.signal(signal.SIGTERM, signal.SIG_IGN)
                        (out / 'helper-pid').write_text(str(os.getpid()))
                        if mode == 'late_writer':
                            time.sleep(0.1)
                            (out / '_execution/identity.json').write_text('late corruption')
                        elif mode == 'metadata_helper':
                            time.sleep(0.1)
                            (out / 'helper-pid').chmod(0o600)
                            (out / 'metadata-finished').write_text('yes')
                        elif mode != 'joined':
                            time.sleep(60)
                    finally:
                        os._exit(0)
                limit = time.monotonic() + 5
                while not (out / 'helper-pid').exists():
                    assert time.monotonic() < limit
                    time.sleep(0.01)
                if mode == 'joined':
                    os.waitpid(child, 0)
                if mode == 'aborted_worker':
                    os._exit(3)
            return dummy(request)
        patch.setitem(sys.modules, 'oxyformer.dummy', SimpleNamespace(run_stage=stage))
        started = time.monotonic()
        result = run('dummy', out, repo, task_file=task_file(out))
        elapsed = time.monotonic() - started
        assert elapsed < 10, (mode, elapsed)
        expected = 'fail' if mode in ('late_writer', 'aborted_worker') else 'pass'
        assert result.status == expected, (mode, result.status, result.message)
        pid_file = out / 'helper-pid'
        if pid_file.exists():
            assert not Path('/proc', pid_file.read_text()).exists(), 'helper survived publication'
        if mode == 'metadata_helper':
            assert (out / 'metadata-finished').read_text() == 'yes'
        if mode == 'notification_timeout':
            assert any(0.02 <= value < 0.25 for value in notification_times), notification_times
            assert max(notification_times) < 0.25
        assert not Path(f'/proc/{os.getpid()}/task/{os.getpid()}/children').read_text().strip()
        print(json.dumps({'mode': mode, 'status': result.status, 'elapsed_seconds': elapsed}), flush=True)


@pytest.mark.parametrize('mode', ['shared_memory', 'spawn', 'joined', 'eof_helper',
                                  'inherited_socket', 'stubborn', 'detached',
                                  'late_writer', 'metadata_helper', 'aborted_worker',
                                  'notification_timeout'])
def test_parent_publication_lifecycle_completes_without_live_helpers(tmp_path, mode):
    import signal
    script = ('import runpy,sys; from pathlib import Path; '
              'runpy.run_path(sys.argv[1])["_publication_case"](sys.argv[2],Path(sys.argv[3]))')
    process = subprocess.Popen([sys.executable, '-c', script, str(Path(__file__).resolve()),
                                mode, str(tmp_path)], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        # Kill only this fixture's observed process tree, including setsid helpers.
        descendants = []
        def collect(pid):
            try:
                children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
            except FileNotFoundError:
                return
            for child in map(int, children):
                collect(child)
                descendants.append(child)
        collect(process.pid)
        for pid in [*descendants, process.pid]:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        output, _ = process.communicate(timeout=5)
        pytest.fail(f'{mode} exceeded external 20-second timeout: {output}')
    assert process.returncode == 0, output
    print(output, end='')
