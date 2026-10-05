"""Offline stage and campaign acceptance."""
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
from oxyformer.contracts import StageRequest, StageResult
from oxyformer.execution.campaign import expand_campaign, validate_plan
from oxyformer.execution.identity import code_identity, scientific_fingerprint
from oxyformer.execution.paths import atomic_json, safe_extract
from oxyformer.execution.runner import dependency_variable, resolve_dependencies, run as run_worker, verify_dependency_result
from oxyformer.provenance import ArtifactLineage, ArtifactRecord, ContractError, file_hash


SOURCE_NEEDS = {'data-unit': ['data.json', 'receipts.json']}

def fixture_module(repo, function):
    return SimpleNamespace(run_stage=function, __file__=str(repo / 'src/oxyformer/dummy.py'))


def install_stage(monkeypatch, repo, function):
    monkeypatch.setitem(sys.modules, 'oxyformer.dummy', fixture_module(repo, function))


def inline_fixture_stage(request, module_name, repo):
    """Inject small closure stages; real process lifecycle is covered by CLI tests."""
    import importlib
    from oxyformer.execution.paths import isolated_caches
    with isolated_caches(request.output_dir):
        try:
            module = importlib.import_module(module_name)
        except (ImportError, FileNotFoundError) as exc:
            return StageResult(request_hash=request.content_hash, status='blocked', artifacts=(), message=str(exc))
        return module.run_stage(request)


def run(*args, **kwargs):
    return run_worker(*args, execute=inline_fixture_stage, **kwargs)


def run_task(repo, out, path=None, *, deps_env=True, **task):
    return run('dummy', out, repo, deps_env=deps_env, task_file=path or task_file(out, **task))


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
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'configs/execution').mkdir(parents=True)
    (repo / 'src').mkdir()
    (repo / 'src/science.py').write_text('value = 1\n')
    (repo / 'src/oxyformer').mkdir()
    (repo / 'src/oxyformer/dummy.py').write_text('# synthetic module origin for injected stage fixtures\n')
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
    install_stage(monkeypatch, repo, dummy)
    return repo, out


def fixture_lineage(request, unit):
    return ArtifactLineage(source_hashes=(sha256(b'fixture').hexdigest(),), unit_ids=(unit,),
                           parent_hashes=(), split_hash=None, config_hash=request.config_hash,
                           model_hash=None, environment=(('python', 'fixture'),), seed=None,
                           parameter_count=None)


def dummy(request):
    request.verify_inputs()
    out = Path(request.output_dir)
    assert Path(os.environ['HF_HOME']).is_relative_to(out)
    task = read_json(Path(request.task_path))
    for path in task.get('outputs', ['value.json']):
        atomic_json(out, path, {'value': 1})
    lineage = fixture_lineage(request, 'dummy')
    outputs = task.get('outputs', ['value.json'])
    return StageResult(request_hash=request.content_hash, status='pass', message='fixture',
                       artifacts=tuple(ArtifactRecord(path=p, sha256=file_hash(out / p),
                                                      lineage=lineage, kind='fixture') for p in outputs))


def refused_dependency_probe(root):
    code = '''import sys
from pathlib import Path
from oxyformer.execution.runner import verify_dependency_result
from oxyformer.provenance import ContractError
try:
    verify_dependency_result(Path(sys.argv[1]))
except (ContractError, OSError) as exc:
    print(exc)
else:
    raise AssertionError('changed fingerprint accepted')
'''
    return bounded_python(code, root)


def bounded_python(code, *args):
    process = subprocess.run([sys.executable, '-c', code, *map(str, args)],
                             env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src')),
                             capture_output=True, text=True, timeout=3)
    assert process.returncode == 0, process.stdout + process.stderr
    return process


def read_json(path):
    return json.loads(path.read_text())


def read_stage_result(out):
    return StageResult.from_json((out / '_execution/result.json').read_text())


def initialize_attempt(repo, path):
    path.mkdir()
    (path / 'code_commit.txt').write_text(git(repo, 'rev-parse', 'HEAD'))


def task_file(out, **changes):
    task = {'id': 'dummy', 'stage': 'dummy', 'needs': {}, 'outputs': ['value.json']}
    task.update(changes)
    path = out / 'input-task.json'
    path.write_text(json.dumps(task))
    return path



def publish_source_fixture(repo, root, *, parent=None, head=None):
    """The source producer seals its own completed synthetic attempt."""
    from oxyformer.execution.integrity import FINGERPRINT, publish_result
    from oxyformer.execution.paths import atomic_write
    dependencies = {} if parent is None else {'data-unit': str(parent)}
    inputs = () if parent is None else (str(parent / FINGERPRINT),)
    config = atomic_json(root, '_execution/config.json', {'dependencies': dependencies})
    task = atomic_json(root, '_execution/task.json', {'id': 'data-unit', 'stage': 'source'})
    request = StageRequest(stage='source', config_path=str(config), config_hash=file_hash(config),
                           task_path=str(task), task_hash=file_hash(task), dependency_paths=inputs,
                           dependency_hashes=tuple(file_hash(p) for p in inputs), output_dir=str(root),
                           code_identity=head or git(repo, 'rev-parse', 'HEAD'))
    atomic_write(root, '_execution/request.json', request.to_json())
    lineage = fixture_lineage(request, 'data-unit')
    result = StageResult(request_hash=request.content_hash, status='pass', message='source fixture',
                         artifacts=tuple(ArtifactRecord(path=p, sha256=file_hash(root / p),
                                                        lineage=lineage, kind='source')
                                         for p in ['data.json', 'receipts.json']))
    published = publish_result(root, result)
    assert published.status == 'pass', published.message
    return published


def make_source_fixture(repo, source, monkeypatch=None):
    source.mkdir()
    seal_source_fixture(repo, source)
    if monkeypatch is not None:
        monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(source))
    return source


def seal_source_fixture(repo, source):
    (source / 'data.json').write_text('{}')
    (source / 'receipts.json').write_text('{}')
    publish_source_fixture(repo, source)


def test_dummy_stage_atomic_records_and_cache_isolation(runtime, monkeypatch):
    repo, out = runtime
    monkeypatch.setenv('HF_HOME', '/unrelated/cache')
    assert run_task(repo, out, deps_env=False).status == 'pass'
    assert os.environ['HF_HOME'] == '/unrelated/cache'
    result = read_json((out / '_execution/result.json'))
    assert result['payload']['status'] == 'pass'
    env = read_json((out / '_execution/environment.json'))
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


def test_upstream_unchanged_and_output_overlap_rejected(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    source = upstream / 'data.json'
    source.write_text('{"fixture":1}')
    (upstream / 'receipts.json').write_text('{}')
    before = source.read_bytes(), source.stat().st_mode, source.stat().st_mtime_ns
    publish_source_fixture(repo, upstream)
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(upstream))
    task = task_file(out, needs=SOURCE_NEEDS)
    assert run_task(repo, out, task).status == 'pass'
    assert (source.read_bytes(), source.stat().st_mode, source.stat().st_mtime_ns) == before
    nested = upstream / 'child'
    initialize_attempt(repo, nested)
    with pytest.raises(ContractError, match='overlaps an upstream'):
        run_task(repo, nested, task)


@pytest.mark.parametrize('name', ['../outside', '/tmp/escape', 'nested/../../escape', './value', 'a\\b'])
def test_output_escape(runtime, name):
    repo, out = runtime
    with pytest.raises(ContractError, match='relative path'):
        run_task(repo, out, deps_env=False, outputs=[name])


def test_output_symlink_escape(runtime, tmp_path):
    repo, out = runtime
    (out / 'alias').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ContractError, match='escapes attempt'):
        run_task(repo, out, deps_env=False, outputs=['alias/escaped'])


def test_result_symlink_escape(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    def escaping(request):
        result = dummy(request)
        target = out / 'value.json'
        target.unlink()
        elsewhere = tmp_path / 'elsewhere'
        elsewhere.write_text('{"value":1}')
        target.symlink_to(elsewhere)
        return result
    install_stage(monkeypatch, repo, escaping)
    result = run_task(repo, out, deps_env=False)
    assert result.status == 'fail' and 'escapes output' in result.message


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
    result = run_task(repo, out, deps_env=False)
    assert result.status == 'blocked'
    assert (out / '_execution/result.json').exists()


def test_cli_selects_task(runtime, monkeypatch):
    repo, out = runtime
    monkeypatch.setattr('oxyformer.execution.runner.run', run)
    monkeypatch.setattr('oxyformer.execution.identity.verify_module_origins', lambda *a, **k: None)
    path = out / 'tasks.json'
    path.write_text(json.dumps({'tasks': [{'id': 'selected', 'stage': 'dummy', 'outputs': ['value.json']}]}))
    assert main(['run-stage', '--stage', 'dummy', '--out', str(out), '--repo', str(repo),
                 '--deps-env', '--task', str(path), '--task-id', 'selected']) == 0
    assert read_json((out / '_execution/task.json'))['id'] == 'selected'


def locked_task(repo, out, upstream, monkeypatch):
    initialize_attempt(repo, upstream)
    lock = upstream / 'recipe_lock.json'
    def producer(request):
        result = dummy(request)
        lock.write_text(json.dumps({'scientific_fingerprint': scientific_fingerprint(repo)}))
        return replace(result, artifacts=(replace(result.artifacts[0], sha256=file_hash(lock)),))
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, 'oxyformer.dummy', fixture_module(repo, producer))
        assert run_task(repo, upstream, deps_env=False, id='campaign-lock', outputs=['recipe_lock.json']).status == 'pass'
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
        run_task(repo, out, task)


def test_locked_recipe_permits_defined_documentation_change(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    task = locked_task(repo, out, tmp_path / 'lock', monkeypatch)
    (repo / 'README.md').write_text('documentation update\n')
    (out / 'code_commit.txt').write_text(commit(repo))
    assert run_task(repo, out, task).status == 'pass'


def test_continuation_ownership_and_consecutive_steps(runtime, tmp_path, monkeypatch):
    repo, old = runtime
    first_task = task_file(old, id='first', continuation={'owner': 'work-1', 'step': 0, 'predecessor': None})
    assert run_task(repo, old, first_task, deps_env=False).status == 'pass'
    monkeypatch.setenv('SWARM_DEP_FIRST', str(old))
    for owner, step, expected in [('wrong-owner', 1, 'ownership'), ('work-1', 2, 'consecutive'), ('work-1', 1, None)]:
        out = tmp_path / f'next-{owner}-{step}'
        initialize_attempt(repo, out)
        task = task_file(out, id='next', needs={'first': ['_execution/task.json', '_execution/request.json',
                                                   '_execution/result.json', 'value.json']},
                         continuation={'owner': owner, 'step': step, 'predecessor': 'first'})
        if expected:
            with pytest.raises(ContractError, match=expected):
                run_task(repo, out, task)
        else:
            assert run_task(repo, out, task).status == 'pass'


def test_continuation_requires_declared_predecessor(runtime):
    repo, out = runtime
    task = task_file(out, continuation={'owner': 'work', 'step': 1, 'predecessor': 'sibling'})
    with pytest.raises(ContractError, match='explicit dependency'):
        run_task(repo, out, task)


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
        assert 'cd "$SWARM_UNIT_DIR/src"' in command
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
    install_stage(monkeypatch, repo, failed)
    assert run_task(repo, upstream, deps_env=False).status == 'fail'
    out = tmp_path / 'consumer'
    initialize_attempt(repo, out)
    monkeypatch.setenv('SWARM_DEP_GATE', str(upstream))
    with pytest.raises(ContractError, match='did not pass'):
        run_task(repo, out, needs={'gate': ['_execution/result.json']})


def test_stage_required_dependency_cannot_be_removed(runtime):
    repo, out = runtime
    registry = repo / 'configs/execution/stages.yaml'
    registry.write_text(yaml.safe_dump({'schema_version': 1, 'stages': {
        'dummy': {'module': 'oxyformer.dummy', 'needs': {'required': ['data.json']}}}}))
    (out / 'code_commit.txt').write_text(commit(repo))
    with pytest.raises(ContractError, match='stage-required dependency'):
        run_task(repo, out)


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
        run_task(repo, out, needs={'incomplete': ['data.json']})


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
        stages[stage] = {'unit_id': unit, 'module': 'oxyformer.dummy', 'outputs': lock_settings['needs'][unit]}
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
        run_task(repo, out, needs={'data-unit': ['data.json']})


def test_task_cannot_exempt_an_incomplete_stage(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'data.json').write_text('{}')
    (source / 'receipts.json').write_text('{}')
    monkeypatch.setenv('SWARM_DEP_INCOMPLETE', str(source))
    with pytest.raises(ContractError, match='stage receipt missing'):
        run_task(repo, out, needs={'incomplete': ['data.json', 'receipts.json']}, acquisition_receipts={'incomplete': 'receipts.json'})


def test_safe_tar_dot_prefix_does_not_hide_traversal_or_duplicates(tmp_path):
    for index, names in enumerate([['./../escape'], ['./same', 'same']]):
        archive = tmp_path / f'bad-{index}.tar'
        with tarfile.open(archive, 'w') as tar:
            for name in names:
                tar.addfile(tarfile.TarInfo(name))
        with pytest.raises(ContractError):
            safe_extract(archive, tmp_path, f'unpacked-{index}')
        assert not (tmp_path / f'unpacked-{index}').exists()


def test_cli_import_from_pristine_repo_keeps_code_roots_clean(runtime, tmp_path):
    repo, out = runtime
    prepare_cli_fixture(repo, out, 'run_stage = dummy\n')
    task = task_file(out)
    env = dict(os.environ, PYTHONPATH=str(repo / 'src'), CUDA_VISIBLE_DEVICES='')
    with (out / 'run.log').open('w') as log:
        process = subprocess.run([sys.executable, '-m', 'oxyformer.cli', 'run-stage',
                                  '--stage', 'dummy', '--repo', str(repo), '--out', str(out),
                                  '--task', str(task)], cwd=tmp_path, env=env, text=True,
                                 stdout=log, stderr=subprocess.STDOUT)
    assert process.returncode == 0, (out / 'run.log').read_text()
    assert not list((repo / 'src/oxyformer').rglob('*.pyc'))
    assert git(repo, 'status', '--porcelain', '--untracked-files=all') == ''
    assert verify_dependency_result(out).status == 'pass'
    with (out / 'run.log').open('a') as log:
        log.write('unexpected late log write')
    with pytest.raises(ContractError, match='fingerprint mismatch.*run.log'):
        verify_dependency_result(out)


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
    assert run_task(repo, out, deps_env=False, parameters=parameters).status == 'pass'
    actual = read_json((out / '_execution/task.json'))['parameters']
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
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')
    repo, out = tmp_path / 'not-a-repo', tmp_path / 'attempt'
    repo.mkdir()
    out.mkdir()
    assert main(['run-stage', '--stage', 'dummy', '--repo', str(repo), '--out', str(out)]) == 2


def test_campaign_task_requires_lock_even_for_generic_stage(runtime):
    repo, out = runtime
    with pytest.raises(ContractError, match='requires a recipe lock'):
        run_task(repo, out, deps_env=False, campaign='screen-01')


def test_cli_malformed_task_types_are_blocked(runtime, monkeypatch):
    repo, out = runtime
    monkeypatch.setattr('oxyformer.execution.identity.verify_module_origins', lambda *a, **k: None)
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
        value = read_json(old_task)
        value.update(id='upstream', campaign='old-campaign')
        old_task.write_text(json.dumps(value))
    else:
        old_task = task_file(upstream, id='upstream')
    assert run_task(repo, upstream, old_task).status == 'pass'
    before = {str(p): file_hash(p) for p in upstream.rglob('*') if p.is_file()}
    (repo / change).write_text('changed\n')
    head = commit(repo)
    consumer = tmp_path / 'consumer'
    consumer.mkdir()
    (consumer / 'code_commit.txt').write_text(head)
    lock_task = locked_task(repo, consumer, tmp_path / 'new-lock', monkeypatch)
    lock_ref = read_json(lock_task)['recipe_lock']
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
            run_task(repo, consumer, selected)
        assert not (consumer / 'value.json').exists()
    else:
        assert run_task(repo, consumer, selected).status == 'pass'
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
    plan = read_json((out / 'expanded_units.json'))
    assert validate_plan(plan, {}) == expand_campaign(spec, {})
    assert read_json((out / 'task_manifest.json'))['tasks'] == plan['tasks']
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
    install_stage(monkeypatch, repo, failing)
    result = run_task(repo, out, deps_env=False)
    assert result.status == 'fail'
    assert result.message
    receipt = read_stage_result(out)
    assert receipt == result


@pytest.mark.parametrize('mutation', ['bytes', 'chmod', 'added', 'removed', 'symlink'])
@pytest.mark.parametrize('exit_kind', ['pass', 'exception', 'system-exit'])
def test_upstream_tree_mutation_fails_and_blocks_later_consumer(
        runtime, tmp_path, monkeypatch, mutation, exit_kind):
    repo, out = runtime
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    (upstream / 'data.json').write_text('{}')
    (upstream / 'receipts.json').write_text('{}')
    # Selected-file hashes miss mutations of undeclared upstream entries.
    extra = upstream / 'extra'
    extra.mkdir()
    victim = extra / 'victim'
    victim.write_text('original')
    link = extra / 'link'
    link.symlink_to('victim')
    publish_source_fixture(repo, upstream)
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(upstream))
    needs = SOURCE_NEEDS
    changed = {'bytes': 'extra/victim', 'chmod': 'extra/victim',
               'added': 'extra/new', 'removed': 'extra/victim',
               'symlink': 'extra/link'}[mutation]
    def faulty(request):
        result = dummy(request)
        if mutation == 'bytes':
            victim.write_text('modified')
        elif mutation == 'chmod':
            victim.chmod(victim.stat().st_mode ^ 0o100)
        elif mutation == 'added':
            (extra / 'new').write_text('new')
        elif mutation == 'removed':
            victim.unlink()
        else:
            link.unlink()
            link.symlink_to('../data.json')
        if exit_kind == 'exception':
            raise RuntimeError('synthetic stage failure')
        if exit_kind == 'system-exit':
            raise SystemExit(7)
        return result
    install_stage(monkeypatch, repo, faulty)
    result = run_task(repo, out, needs=needs)
    assert result.status == 'fail'
    assert str(upstream / changed) in result.message
    assert read_stage_result(out) == result
    receipt = read_json((out / '_execution/dependency_check.json'))
    assert receipt['status'] == 'fail'
    assert changed in receipt['attempts'][str(upstream)]['changed_paths']
    # The runner must not repair upstream data or modes while reporting failure.
    if mutation == 'bytes':
        assert victim.read_text() == 'modified'
    elif mutation == 'chmod':
        assert victim.stat().st_mode & 0o100
    elif mutation == 'added':
        assert (extra / 'new').is_file()
    elif mutation == 'removed':
        assert not victim.exists()
    else:
        assert os.readlink(link) == '../data.json'
    later = tmp_path / 'later'
    initialize_attempt(repo, later)
    install_stage(monkeypatch, repo, dummy)
    with pytest.raises(ContractError, match='tainted|fingerprint'):
        run_task(repo, later, needs=needs)
    assert not (later / 'value.json').exists()


def test_read_only_upstream_tree_remains_usable(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    upstream = tmp_path / 'source'
    upstream.mkdir()
    (upstream / 'data.json').write_text('{}')
    (upstream / 'receipts.json').write_text('{}')
    (upstream / 'link').symlink_to('data.json')
    publish_source_fixture(repo, upstream)
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(upstream))
    needs = SOURCE_NEEDS
    before = {p.name: (p.lstat().st_mode, p.lstat().st_size,
                      os.readlink(p) if p.is_symlink() else (None if p.is_dir() else p.read_bytes()))
              for p in upstream.iterdir()}
    for attempt in [out, tmp_path / 'later']:
        attempt.mkdir(exist_ok=True)
        (attempt / 'code_commit.txt').write_text(git(repo, 'rev-parse', 'HEAD'))
        assert run_task(repo, attempt, needs=needs).status == 'pass'
    after = {p.name: (p.lstat().st_mode, p.lstat().st_size,
                     os.readlink(p) if p.is_symlink() else (None if p.is_dir() else p.read_bytes()))
             for p in upstream.iterdir()}
    assert after == before


def test_tree_fingerprint_binds_types_modes_bytes_links_and_all_entries(tmp_path):
    from oxyformer.execution.integrity import fingerprint_tree, changed_paths
    import stat
    root = tmp_path / 'tree'
    root.mkdir()
    (root / 'empty').mkdir()
    file = root / 'file'
    file.write_bytes(b'abcd')
    (root / 'link').symlink_to('file')
    os.mkfifo(root / 'pipe')
    before = fingerprint_tree(root)
    assert set(before) == {'.', 'empty', 'file', 'link', 'pipe'}
    assert before['file'] == {'type': stat.S_IFREG, 'mode': stat.S_IMODE(file.stat().st_mode),
                              'size': 4, 'sha256': sha256(b'abcd').hexdigest(), 'target': None}
    assert before['empty']['type'] == stat.S_IFDIR
    assert before['link']['type'] == stat.S_IFLNK and before['link']['target'] == 'file'
    assert before['pipe']['type'] == stat.S_IFIFO
    assert fingerprint_tree(root) == before
    file.unlink()
    file.mkdir()
    assert 'file' in changed_paths(before, fingerprint_tree(root))


def test_tree_fingerprint_errors_are_explicit_and_path_specific(tmp_path, monkeypatch):
    from oxyformer.execution.integrity import fingerprint_tree, changed_paths
    (tmp_path / 'file').write_text('content')
    before = fingerprint_tree(tmp_path)
    original = os.open
    def denied(path, flags, *args, **kwargs):
        if Path(path) == tmp_path / 'file':
            raise PermissionError('synthetic unreadable entry')
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(os, 'open', denied)
    after = fingerprint_tree(tmp_path)
    assert 'synthetic unreadable entry' in after['file']['error']
    assert changed_paths(before, after) == ['file']


@pytest.mark.parametrize('raises', [False, True])
def test_post_execution_check_names_proc_fd_chmod(runtime, tmp_path, monkeypatch, raises):
    repo, out = runtime
    upstream = tmp_path / 'source'
    upstream.mkdir()
    (upstream / 'data.json').write_text('{}')
    (upstream / 'receipts.json').write_text('{}')
    victim = upstream / 'undeclared'
    victim.write_text('unchanged bytes')
    before = file_hash(victim)
    publish_source_fixture(repo, upstream)
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(upstream))
    def faulty(request):
        result = dummy(request)
        with victim.open('rb') as stream:
            os.chmod(f'/proc/self/fd/{stream.fileno()}', victim.stat().st_mode ^ 0o100)
        if raises:
            raise RuntimeError('synthetic failure after chmod')
        return result
    install_stage(monkeypatch, repo, faulty)
    result = run_task(repo, out, needs=SOURCE_NEEDS)
    assert result.status == 'fail'
    assert str(victim) in result.message
    receipt = read_json((out / '_execution/dependency_check.json'))
    assert receipt['attempts'][str(upstream)]['status'] == 'tainted'
    assert receipt['attempts'][str(upstream)]['changed_paths'] == ['undeclared']
    assert file_hash(victim) == before and victim.stat().st_mode & 0o100
    assert not (tmp_path / '.oxyformer-integrity').exists()


@pytest.mark.parametrize('when', ['before-consumer', 'during-consumer'])
def test_transitive_upstream_metadata_cannot_escape_detection(runtime, tmp_path, monkeypatch, when):
    repo, middle = runtime
    source = make_source_fixture(repo, tmp_path / 'source', monkeypatch)
    assert run_task(repo, middle, id='middle', needs=SOURCE_NEEDS).status == 'pass'
    later = tmp_path / 'later'
    initialize_attempt(repo, later)
    monkeypatch.setenv('SWARM_DEP_MIDDLE', str(middle))
    task = task_file(later, needs={'middle': ['value.json']})
    victim = source / 'data.json'
    if when == 'before-consumer':
        victim.chmod(victim.stat().st_mode ^ 0o100)
        with pytest.raises(ContractError, match='fingerprint'):
            run_task(repo, later, task)
        assert not (later / 'value.json').exists()
    else:
        def faulty(request):
            result = dummy(request)
            victim.chmod(victim.stat().st_mode ^ 0o100)
            return result
        install_stage(monkeypatch, repo, faulty)
        result = run_task(repo, later, task)
        assert result.status == 'fail'
        assert str(victim) in result.message


def test_consumer_binds_published_fingerprint_digest(runtime, tmp_path, monkeypatch):
    from oxyformer.execution.integrity import FINGERPRINT
    repo, out = runtime
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'data.json').write_text('{}')
    (source / 'receipts.json').write_text('{}')
    published = publish_source_fixture(repo, source)
    expected = next(a.sha256 for a in published.artifacts if a.path == FINGERPRINT)
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(source))
    assert run_task(repo, out, needs=SOURCE_NEEDS).status == 'pass'
    request = StageRequest.from_json((out / '_execution/request.json').read_text())
    assert dict(zip(request.dependency_paths, request.dependency_hashes))[str(source / FINGERPRINT)] == expected
    # Alter the publication record itself; its original result digest must win.
    (source / FINGERPRINT).write_text('{}')
    later = tmp_path / 'later'
    initialize_attempt(repo, later)
    with pytest.raises(ContractError, match='fingerprint hash mismatch'):
        run_task(repo, later, needs=SOURCE_NEEDS)


def test_unsealed_acquisition_cannot_become_a_new_baseline(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'data.json').write_text('{}')
    (source / 'receipts.json').write_text('{}')
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(source))
    with pytest.raises(ContractError, match='stage receipt missing'):
        run_task(repo, out, needs=SOURCE_NEEDS)
    assert not (source / '_execution').exists()


def test_restored_upstream_state_is_identical_under_fingerprint_contract(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    source = make_source_fixture(repo, tmp_path / 'source')
    victim = source / 'data.json'
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(source))
    def faulty(request):
        original = victim.read_bytes()
        victim.write_bytes(b'[]')
        assert victim.read_bytes() == b'[]'
        victim.write_bytes(original)
        return dummy(request)
    install_stage(monkeypatch, repo, faulty)
    result = run_task(repo, out, needs=SOURCE_NEEDS)
    # Fingerprints compare input states, not a history of transient writes.
    assert result.status == 'pass', result.message
    assert victim.read_bytes() == b'{}'


@pytest.mark.parametrize('mutation', ['chmod', 'undeclared_bytes'])
def test_preflight_cannot_rebase_a_changed_dependency(runtime, tmp_path, monkeypatch, mutation):
    import oxyformer.execution.runner as runner
    repo, out = runtime
    source = tmp_path / 'source'
    source.mkdir()
    victim = source / 'data.json'
    victim.write_text('{}')
    (source / 'receipts.json').write_text('{}')
    extra = source / 'undeclared.txt'
    extra.write_text('before')
    publish_source_fixture(repo, source)
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(source))
    original = runner.verify_published_tree
    def interleave(root, result, expected_hash=None):
        verified = original(root, result, expected_hash)
        if root == source:
            if mutation == 'chmod':
                victim.chmod(victim.stat().st_mode ^ 0o100)
            else:
                extra.write_text('after')
        return verified
    monkeypatch.setattr(runner, 'verify_published_tree', interleave)
    result = run_task(repo, out, needs=SOURCE_NEEDS)
    assert result.status == 'fail'
    assert str(victim if mutation == 'chmod' else extra) in result.message


@pytest.mark.parametrize('record_index', [0, -1])
def test_rewritten_upstream_result_is_rejected_by_later_consumers(runtime, tmp_path, monkeypatch, record_index):
    repo, out = runtime
    source = make_source_fixture(repo, tmp_path / 'source', monkeypatch)
    needs = SOURCE_NEEDS
    def faulty(request):
        result = dummy(request)
        record = source / '_execution/result.json'
        published = StageResult.from_json(record.read_text())
        artifacts = list(published.artifacts)
        selected = artifacts[record_index]
        artifacts[record_index] = replace(selected, lineage=replace(selected.lineage,
                                          parent_hashes=('b' * 64,)))
        record.write_text(replace(published, artifacts=artifacts).to_json())
        return result
    install_stage(monkeypatch, repo, faulty)
    result = run_task(repo, out, needs=needs)
    assert result.status == 'fail'
    assert str(source / '_execution/result.json') in result.message
    later = tmp_path / 'later'
    initialize_attempt(repo, later)
    install_stage(monkeypatch, repo, dummy)
    with pytest.raises(ContractError, match='publication|fingerprint|result record'):
        run_task(repo, later, needs=needs)


def test_tempfile_cache_does_not_cross_attempts(runtime, tmp_path, monkeypatch):
    import tempfile
    repo, first = runtime
    monkeypatch.setattr(tempfile, 'tempdir', None)
    def scratch_stage(request):
        scratch = Path(tempfile.mkdtemp())
        assert scratch.is_relative_to(Path(request.output_dir))
        return dummy(request)
    install_stage(monkeypatch, repo, scratch_stage)
    assert run_task(repo, first, deps_env=False).status == 'pass'
    shutil.rmtree(first)
    later = tmp_path / 'later'
    initialize_attempt(repo, later)
    result = run_task(repo, later, deps_env=False)
    assert result.status == 'pass', result.message


def test_cli_rejects_modules_imported_from_another_checkout(runtime, tmp_path):
    repo, out = runtime
    prepare_cli_fixture(repo, out, 'run_stage = dummy\n')
    other = tmp_path / 'other'
    shutil.copytree(repo, other)
    wrong = other / 'src/oxyformer/dummy.py'
    wrong.write_text(wrong.read_text().replace("{'value': 1}", "{'value': 2}"))
    process = subprocess.run([sys.executable, '-m', 'oxyformer.cli', 'run-stage',
                              '--stage', 'dummy', '--repo', str(repo), '--out', str(out),
                              '--task', str(task_file(out))], cwd=other / 'src',
                             env=dict(os.environ, PYTHONPATH=str(repo / 'src'), CUDA_VISIBLE_DEVICES=''),
                             capture_output=True, text=True)
    assert process.returncode == 2, process.stdout + process.stderr
    assert 'outside --repo' in process.stderr
    assert not (out / 'value.json').exists()


def test_forty_leaf_limit_includes_cpu_slices(spec):
    # The owner mandate bounds total leaves, not only GPU leaves.
    spec['work'] = spec['work'][:1]
    spec['work'][0]['slices'] = [{'gpus': 1, 'wall_seconds': 60}] * 39 + [{'gpus': 0, 'wall_seconds': 60}]
    assert len(expand_campaign(spec, {})['expected_leaves']) == 40
    spec['work'][0]['slices'].append({'gpus': 1, 'wall_seconds': 60})
    with pytest.raises(ContractError, match='forty leaves'):
        expand_campaign(spec, {})


@pytest.mark.parametrize('name', ['_execution/result.json', '_execution/fingerprint.json'])
def test_late_publication_control_modes_are_verified(runtime, tmp_path, monkeypatch, name):
    repo, out = runtime
    source = make_source_fixture(repo, tmp_path / 'source')
    path = source / name
    path.chmod(path.stat().st_mode ^ 0o100)
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(source))
    with pytest.raises(ContractError, match='fingerprint mismatch'):
        run_task(repo, out, needs=SOURCE_NEEDS)


def prepare_cli_fixture(repo, out, stage_body):
    import inspect
    original = Path(__file__).parents[1]
    shutil.copytree(original / 'src/oxyformer', repo / 'src/oxyformer', dirs_exist_ok=True,
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
    (repo / 'src/oxyformer/dummy.py').write_text(imports + inspect.getsource(read_json) + '\n' + inspect.getsource(fixture_lineage) + '\n' + inspect.getsource(dummy) + '\n' + stage_body)
    (out / 'code_commit.txt').write_text(commit(repo))


def run_cli_fixture(repo, out, stage_body, *, needs=None, entrypoint=None, timeout=30):
    """Execute real copied CLI code and a synthetic stage in a fresh interpreter."""
    prepare_cli_fixture(repo, out, stage_body)
    command = [sys.executable, '-m', 'oxyformer.cli'] if entrypoint is None else [sys.executable, '-c', entrypoint]
    command += ['run-stage', '--stage', 'dummy', '--repo', str(repo), '--out', str(out),
                '--task', str(task_file(out, needs=needs or {})), '--deps-env']
    process = subprocess.Popen(command, cwd=repo, env=dict(os.environ, PYTHONPATH=str(repo / 'src'),
                               CUDA_VISIBLE_DEVICES=''), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        import signal
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


@pytest.mark.xfail(strict=True, reason='ARC-1339: general outside-write confinement is an owner-deferred host-runtime limitation')
def test_cli_undeclared_outside_write_fails(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    outside = tmp_path / 'outside.json'
    monkeypatch.setenv('FIXTURE_OUTSIDE', str(outside))
    process = run_cli_fixture(repo, out, '''def run_stage(request):
    Path(os.environ['FIXTURE_OUTSIDE']).write_text('undeclared output')
    return dummy(request)
''')
    assert process.returncode == 1, process.stdout + process.stderr
    result = read_stage_result(out)
    assert str(outside) in result.message


@pytest.mark.parametrize('control', ['result.json', 'fingerprint.json'])
def test_identical_control_rewrite_accepts_later_consumers(runtime, tmp_path, monkeypatch, control):
    repo, out = runtime
    source = make_source_fixture(repo, tmp_path / 'source', monkeypatch)
    needs = SOURCE_NEEDS
    victim = source / '_execution' / control
    def faulty(request):
        result = dummy(request)
        victim.write_bytes(victim.read_bytes())
        return result
    install_stage(monkeypatch, repo, faulty)
    result = run_task(repo, out, needs=needs)
    assert result.status == 'pass', result.message
    later = tmp_path / 'later'
    initialize_attempt(repo, later)
    install_stage(monkeypatch, repo, dummy)
    assert run_task(repo, later, needs=needs).status == 'pass'


def test_cli_finalizes_temporary_directories_before_publication(runtime):
    repo, out = runtime
    process = run_cli_fixture(repo, out, '''import tempfile
scratch = None
def run_stage(request):
    global scratch
    scratch = tempfile.TemporaryDirectory()
    (Path(scratch.name) / 'scratch').write_text('temporary work')
    return dummy(request)
''')
    assert process.returncode == 0, process.stdout + process.stderr
    assert verify_dependency_result(out).status == 'pass'


@pytest.mark.parametrize('worker_kind', ['thread', 'subprocess'])
def test_cli_waits_for_background_mutation_before_post_check(runtime, tmp_path, monkeypatch, worker_kind):
    repo, out = runtime
    source = make_source_fixture(repo, tmp_path / 'source')
    victim = source / 'data.json'
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(source))
    monkeypatch.setenv('FIXTURE_WORKER_KIND', worker_kind)
    # Delay publication outside the stage to expose its unjoined worker.
    entrypoint = '''import time
from pathlib import Path
import oxyformer.cli as cli
import oxyformer.execution.runner as runner
original = runner.publish_result
def wait_for_worker(root, result, **kwargs):
    deadline = time.monotonic() + 5
    while not (Path(root) / 'worker-finished').exists():
        assert time.monotonic() < deadline
        time.sleep(.01)
    return original(root, result, **kwargs)
runner.publish_result = wait_for_worker
raise SystemExit(cli.main())
'''
    process = run_cli_fixture(repo, out, '''import threading
import time
import subprocess
import sys
def run_stage(request):
    result = dummy(request)
    def background():
        time.sleep(.2)
        (Path(os.environ['SWARM_DEP_DATA_UNIT']) / 'data.json').write_text('[]')
        (Path(request.output_dir) / 'worker-finished').write_text('done')
    if os.environ['FIXTURE_WORKER_KIND'] == 'thread':
        threading.Thread(target=background).start()
    else:
        code = "import time,os; from pathlib import Path; time.sleep(.2); (Path(os.environ['SWARM_DEP_DATA_UNIT'])/'data.json').write_text('[]'); Path(os.environ['FIXTURE_FINISHED']).write_text('done')"
        subprocess.Popen([sys.executable, '-c', code], env=dict(os.environ, FIXTURE_FINISHED=str(Path(request.output_dir) / 'worker-finished')))
    return result
''', needs=SOURCE_NEEDS, entrypoint=entrypoint)
    assert process.returncode == 1, process.stdout + process.stderr
    result = read_stage_result(out)
    assert result.status == 'fail' and str(victim) in result.message
    check = read_json((out / '_execution/dependency_check.json'))
    assert check['attempts'][str(source)]['status'] == 'tainted'


@pytest.mark.parametrize('operation', ['rewrite_restore_mtime', 'copy2'])
@pytest.mark.parametrize('control', ['result.json', 'fingerprint.json'])
def test_control_rewrites_bind_fingerprinted_properties_only(runtime, tmp_path, monkeypatch, operation, control):
    repo, out = runtime
    source = make_source_fixture(repo, tmp_path / 'source', monkeypatch)
    needs = SOURCE_NEEDS
    victim = source / '_execution' / control
    before = victim.stat()
    def faulty(request):
        result = dummy(request)
        if operation == 'copy2':
            backup = Path(request.output_dir) / 'control-backup'
            shutil.copy2(victim, backup)
            shutil.copy2(backup, victim)
        else:
            victim.write_bytes(victim.read_bytes())
            os.utime(victim, ns=(before.st_atime_ns, before.st_mtime_ns))
        return result
    install_stage(monkeypatch, repo, faulty)
    result = run_task(repo, out, needs=needs)
    assert result.status == 'pass', result.message
    assert victim.stat().st_mtime_ns == before.st_mtime_ns
    assert victim.stat().st_ctime_ns != before.st_ctime_ns
    later = tmp_path / 'later'
    initialize_attempt(repo, later)
    install_stage(monkeypatch, repo, dummy)
    assert run_task(repo, later, needs=needs).status == 'pass'

    # Late records bind bytes and modes; refusal must not repair them.
    changed = tmp_path / 'changed'
    initialize_attempt(repo, changed)
    def fingerprinted_change(request):
        result = dummy(request)
        if operation == 'copy2':
            victim.chmod(victim.stat().st_mode ^ 0o100)
        else:
            victim.write_bytes(victim.read_bytes() + b' ')
        return result
    install_stage(monkeypatch, repo, fingerprinted_change)
    result = run_task(repo, changed, needs=needs)
    assert result.status == 'fail' and str(victim) in result.message
    check = read_json((changed / '_execution/dependency_check.json'))
    assert check['attempts'][str(source)]['status'] == 'tainted'
    assert '_execution/' + control in check['attempts'][str(source)]['changed_paths']
    refused = tmp_path / 'refused'
    initialize_attempt(repo, refused)
    install_stage(monkeypatch, repo, dummy)
    with pytest.raises(ContractError, match='fingerprint|record'):
        run_task(repo, refused, needs=needs)


def test_nested_worker_mutation_cannot_outlive_publication(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    source = make_source_fixture(repo, tmp_path / 'source')
    victim = source / 'data.json'
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(source))
    # Capture waits for the finite grandchild after its parent exits.
    body = '''import subprocess
import sys
def run_stage(request):
    result = dummy(request)
    grandchild = "import os,time; from pathlib import Path; time.sleep(5); (Path(os.environ['SWARM_DEP_DATA_UNIT'])/'data.json').write_text('[]')"
    helper = 'import subprocess,sys; subprocess.Popen([sys.executable, "-c", ' + repr(grandchild) + '])'
    subprocess.Popen([sys.executable, '-c', helper])
    return result
'''
    process = run_cli_fixture(repo, out, body, needs=SOURCE_NEEDS)
    assert victim.read_text() == '[]', process.stdout + process.stderr
    result = read_stage_result(out)
    assert process.returncode == 1 and result.status == 'fail', result.to_json()
    assert str(victim) in result.message


def test_worker_allows_normal_resource_tracker_shutdown(runtime):
    repo, out = runtime
    process = run_cli_fixture(repo, out, '''from multiprocessing.shared_memory import SharedMemory
def run_stage(request):
    scratch = SharedMemory(create=True, size=1)
    scratch.close()
    scratch.unlink()
    return dummy(request)
''', timeout=10)
    assert process.returncode == 0, process.stdout + process.stderr
    assert verify_dependency_result(out).status == 'pass'


def test_changed_fingerprint_fifo_is_refused_without_blocking(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    source = make_source_fixture(repo, tmp_path / 'source')
    victim = source / '_execution/fingerprint.json'
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(source))
    def faulty(request):
        result = dummy(request)
        victim.unlink()
        os.mkfifo(victim)
        raise RuntimeError('stage failed after replacing control')
    install_stage(monkeypatch, repo, faulty)
    result = run_task(repo, out, needs=SOURCE_NEEDS)
    assert result.status == 'fail' and str(victim) in result.message
    # The bounded probe must refuse this FIFO without a writer.
    process = refused_dependency_probe(source)
    assert 'fingerprint' in process.stdout


def test_worker_accepts_expected_nonzero_housekeeping_status(runtime):
    repo, out = runtime
    process = run_cli_fixture(repo, out, '''import subprocess
def run_stage(request):
    result = dummy(request)
    log = Path(request.output_dir) / 'clean.log'
    log.write_text('')
    subprocess.Popen(['/bin/sh', '-c', 'sleep 1; grep -q stale "$1"', 'fixture', str(log)])
    return result
''')
    assert process.returncode == 0, process.stdout + process.stderr
    assert verify_dependency_result(out).status == 'pass'


def test_cli_keeps_declared_run_log_hash_valid(runtime):
    repo, out = runtime
    entrypoint = '''import os,sys
from pathlib import Path
from oxyformer.cli import main
out = Path(sys.argv[sys.argv.index('--out') + 1])
with (out / 'run.log').open('w') as log:
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
raise SystemExit(main())
'''
    process = run_cli_fixture(repo, out, '''from dataclasses import replace
def run_stage(request):
    result = dummy(request)
    log = ArtifactRecord(path='run.log', sha256=file_hash(Path(request.output_dir) / 'run.log'),
                         lineage=result.artifacts[0].lineage, kind='log')
    return replace(result, artifacts=(*result.artifacts, log))
''', entrypoint=entrypoint)
    assert process.returncode == 0, (out / 'run.log').read_text()
    assert verify_dependency_result(out).status == 'pass'


def test_changing_fingerprint_to_fifo_cannot_skip_post_check(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    source = make_source_fixture(repo, tmp_path / 'source', monkeypatch)
    process = run_cli_fixture(repo, out, """def run_stage(request):
    result = dummy(request)
    victim = Path(os.environ['SWARM_DEP_DATA_UNIT']) / '_execution/fingerprint.json'
    victim.unlink()
    os.mkfifo(victim)
    return result
""", needs=SOURCE_NEEDS, timeout=5)
    assert process.returncode == 1, process.stdout + process.stderr
    result = read_stage_result(out)
    assert result.status == 'fail' and str(source / '_execution/fingerprint.json') in result.message
    check = read_json((out / '_execution/dependency_check.json'))
    assert check['attempts'][str(source)]['status'] == 'tainted'


@pytest.mark.parametrize('control', ['result.json', 'request.json'])
def test_preflight_control_fifo_swap_is_nonblocking(runtime, tmp_path, control):
    repo, _ = runtime
    source = make_source_fixture(repo, tmp_path / 'source')
    code = '''import os,sys
from pathlib import Path
from oxyformer.execution import runner
from oxyformer.provenance import ContractError
original = runner.dependency_file
def swap(root, relative):
    path = original(root, relative)
    if relative == '_execution/' + sys.argv[2]:
        path.unlink()
        os.mkfifo(path)
    return path
runner.dependency_file = swap
try:
    runner.verify_dependency_result(Path(sys.argv[1]))
except (ContractError, OSError) as exc:
    print(exc)
else:
    raise AssertionError('FIFO accepted')
'''
    process = bounded_python(code, str(source), control)
    assert control in process.stdout


@pytest.mark.parametrize('relative', ['_execution/fingerprint.json', 'data.json'])
def test_transitive_fifo_is_refused_before_input_hashing(runtime, tmp_path, monkeypatch, relative):
    repo, middle = runtime
    source = make_source_fixture(repo, tmp_path / 'source', monkeypatch)
    assert run_task(repo, middle, id='middle', needs=SOURCE_NEEDS).status == 'pass'
    victim = source / relative
    victim.unlink()
    os.mkfifo(victim)
    process = refused_dependency_probe(middle)
    assert relative in process.stdout


@pytest.mark.parametrize('consumer_kind', ['collector', 'leaf'])
def test_generated_collector_binds_each_expected_producer(runtime, tmp_path, monkeypatch, consumer_kind):
    repo, out = runtime
    lock_task = locked_task(repo, out, tmp_path / 'lock', monkeypatch)
    lock_ref = read_json(lock_task)['recipe_lock']
    spec = {'schema_version': 1, 'id': 'campaign-a', 'kind': 'screening',
            'prerequisites': [], 'inputs': {'campaign-lock': ['recipe_lock.json']},
            'recipe_lock': lock_ref,
            'work': [{'id': name, 'stage': 'dummy', 'outputs': ['value.json'],
                      'slices': [{'gpus': 0, 'wall_seconds': 60}]} for name in ['a', 'b']],
            'collector': {'stage': 'dummy', 'outputs': ['summary.json'], 'wall_seconds': 60}}
    plan = expand_campaign(spec, {})
    other = expand_campaign(dict(spec, id='campaign-b'), {})
    attempts = {}
    for task in [*plan['tasks'][:-1], other['tasks'][0]]:
        attempt = tmp_path / task['id']
        initialize_attempt(repo, attempt)
        task_path = attempt / 'input-task.json'
        task_path.write_text(json.dumps(task))  # exact generated task, unedited
        assert run_task(repo, attempt, task_path).status == 'pass'
        attempts[task['id']] = attempt
    for unit in plan['expected_leaves']:
        monkeypatch.setenv(dependency_variable(unit), str(attempts[unit]))
    selected = plan['tasks'][-1]
    if consumer_kind == 'leaf':
        downstream = dict(spec, id='campaign-c', inputs={**spec['inputs'],
            **{unit: ['value.json'] for unit in plan['expected_leaves']}})
        selected = expand_campaign(downstream, {})['tasks'][0]
    good_task = out / 'generated-collector.json'
    good_task.write_text(json.dumps(selected))
    assert run_task(repo, out, good_task).status == 'pass'
    wrong = tmp_path / 'wrong-collector'
    initialize_attempt(repo, wrong)
    wrong_task = wrong / 'generated-collector.json'
    wrong_task.write_text(good_task.read_text())
    monkeypatch.setenv(dependency_variable(plan['expected_leaves'][-1]),
                       str(attempts[other['tasks'][0]['id']]))
    with pytest.raises(ContractError, match='producer identity'):
        run_task(repo, wrong, wrong_task)
    assert not (wrong / 'summary.json').exists()


def test_ignored_untracked_stage_code_is_refused(runtime):
    repo, out = runtime
    git(repo, 'rm', '--cached', 'src/oxyformer/dummy.py')
    with (repo / '.git/info/exclude').open('a') as stream:
        stream.write('\n/src/oxyformer/dummy.py\n')
    process = run_cli_fixture(repo, out, 'run_stage = dummy\n')
    assert git(repo, 'ls-files', 'src/oxyformer/dummy.py') == ''
    assert git(repo, 'check-ignore', 'src/oxyformer/dummy.py') == 'src/oxyformer/dummy.py'
    assert process.returncode == 2, process.stdout + process.stderr
    assert not (out / 'value.json').exists()


def test_archive_fifo_change_cannot_skip_failure_receipt(runtime, tmp_path, monkeypatch):
    repo, out = runtime
    source = tmp_path / 'source'
    initialize_attempt(repo, source)
    def archive_producer(request):
        result = dummy(request)
        archive = Path(request.output_dir) / 'payload.tar'
        with tarfile.open(archive, 'w') as tar:
            member = tarfile.TarInfo('tiny.txt')
            member.size = 2
            tar.addfile(member, io.BytesIO(b'ok'))
        return replace(result, artifacts=(replace(result.artifacts[0], sha256=file_hash(archive)),))
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, 'oxyformer.dummy', fixture_module(repo, archive_producer))
        assert run_task(repo, source, deps_env=False, id='source', outputs=['payload.tar']).status == 'pass'
    monkeypatch.setenv('SWARM_DEP_SOURCE', str(source))
    process = run_cli_fixture(repo, out, '''from oxyformer.execution.paths import safe_extract
def run_stage(request):
    result = dummy(request)
    victim = Path(os.environ['SWARM_DEP_SOURCE']) / 'payload.tar'
    victim.unlink()
    os.mkfifo(victim)
    safe_extract(victim, request.output_dir, 'unpacked')
    return result
''', needs={'source': ['payload.tar']}, timeout=5)
    assert process.returncode == 1, process.stdout + process.stderr
    result = read_stage_result(out)
    assert result.status == 'fail' and str(source / 'payload.tar') in result.message


def test_output_inside_transitive_attempt_is_refused_before_writing(runtime, tmp_path, monkeypatch):
    from oxyformer.execution.integrity import fingerprint_tree
    repo, middle = runtime
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'data.json').write_text('{}')
    (source / 'receipts.json').write_text('{}')
    nested = source / 'handoff'
    initialize_attempt(repo, nested)
    # Publish the proposed output/task first, so only the consumer may mutate.
    task = task_file(nested, needs={'middle': ['value.json']})
    publish_source_fixture(repo, source)
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(source))
    assert run_task(repo, middle, id='middle', needs=SOURCE_NEEDS).status == 'pass'
    monkeypatch.setenv('SWARM_DEP_MIDDLE', str(middle))
    before = fingerprint_tree(source)
    with pytest.raises(ContractError, match='overlap|upstream'):
        run_task(repo, nested, task)
    assert fingerprint_tree(source) == before
    assert not (nested / '_execution').exists()


@pytest.mark.parametrize('relative', ['src/oxyformer/helper.py', 'scripts/helper.py',
                                      'src/oxyformer/notes.md', 'scripts/notes.md',
                                      'src/oxyformer/helper.pyc'])
@pytest.mark.parametrize('derive', ['commit', 'recipe'])
def test_identity_rejects_every_ignored_file_in_code_roots(runtime, relative, derive):
    repo, out = runtime
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('unrecorded code or resource\n')
    with (repo / '.git/info/exclude').open('a') as stream:
        stream.write('\n/' + relative + '\n')
    with pytest.raises(ContractError, match='untracked.*' + relative):
        code_identity(repo, out) if derive == 'commit' else scientific_fingerprint(repo)


@pytest.mark.parametrize('flag', ['--assume-unchanged', '--skip-worktree'])
@pytest.mark.parametrize('derive', ['commit', 'recipe'])
def test_identity_checks_disk_bytes_even_when_index_suppresses_status(runtime, flag, derive):
    repo, out = runtime
    git(repo, 'update-index', flag, 'src/oxyformer/dummy.py')
    (repo / 'src/oxyformer/dummy.py').write_text('unrecorded = True\n')
    assert git(repo, 'status', '--porcelain') == ''
    with pytest.raises(ContractError, match='tracked modifications.*src/oxyformer/dummy.py'):
        code_identity(repo, out) if derive == 'commit' else scientific_fingerprint(repo)


@pytest.mark.parametrize('relative', ['src/oxyformer/notes.md', 'scripts/notes.md', 'docs/helper.py'])
def test_recipe_fingerprint_covers_resources_under_import_roots(runtime, relative):
    repo, out = runtime
    note = repo / relative
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text('one')
    commit(repo)
    before = scientific_fingerprint(repo)
    note.write_text('two')
    commit(repo)
    assert scientific_fingerprint(repo) != before


@pytest.mark.parametrize('special', ['fifo', 'socket', 'directory', 'symlink'])
def test_safe_extract_checks_archive_type_before_open(tmp_path, special):
    import socket
    archive = tmp_path / 'payload.tar'
    sock = None
    if special == 'fifo':
        os.mkfifo(archive)
    elif special == 'socket':
        sock = socket.socket(socket.AF_UNIX)
        sock.bind(str(archive))
    elif special == 'directory':
        archive.mkdir()
    else:
        archive.symlink_to('/dev/null')
    code = '''import sys
from oxyformer.execution.paths import safe_extract
from oxyformer.provenance import ContractError
try:
    safe_extract(sys.argv[1], sys.argv[2], 'extracted')
except ContractError as exc:
    assert sys.argv[1] in str(exc), str(exc)
else:
    raise AssertionError('nonregular archive accepted')
'''
    try:
        process = bounded_python(code, str(archive), str(tmp_path))
        assert not (tmp_path / 'extracted').exists()
    finally:
        if sock is not None:
            sock.close()


def test_output_containing_transitive_attempt_is_refused_before_writing(runtime, tmp_path, monkeypatch):
    from oxyformer.execution.integrity import fingerprint_tree
    repo, middle = runtime
    out = tmp_path / 'consumer'
    initialize_attempt(repo, out)
    task = task_file(out, needs={'middle': ['value.json']})
    source = make_source_fixture(repo, out / 'source')
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(source))
    assert run_task(repo, middle, id='middle', needs=SOURCE_NEEDS).status == 'pass'
    monkeypatch.setenv('SWARM_DEP_MIDDLE', str(middle))
    before = fingerprint_tree(out)
    with pytest.raises(ContractError, match='overlap.*' + str(source)):
        run_task(repo, out, task)
    assert fingerprint_tree(out) == before


def test_builder_alternate_approval_fifo_is_nonblocking(tmp_path):
    approval = tmp_path / 'approval.yaml'
    os.mkfifo(approval)
    process = subprocess.run([sys.executable, str(Path(__file__).parents[1] / 'scripts/build_tasks.py'),
                              '--spec', str(tmp_path / 'unused.json'), '--approvals', str(approval),
                              '--out', str(tmp_path / 'out')], capture_output=True, text=True,
                             env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src')),
                             timeout=3)
    assert process.returncode != 0
    assert str(approval) in process.stderr and 'regular' in process.stderr
    assert not (tmp_path / 'out').exists()


def test_module_namespace_cannot_extend_beyond_checked_checkout(runtime, tmp_path):
    from oxyformer.execution.identity import verify_module_origins
    repo, _ = runtime
    namespace = SimpleNamespace(__path__=[str(repo / 'src/oxyformer'), str(tmp_path)])
    with pytest.raises(ContractError, match='outside --repo'):
        verify_module_origins(repo, [namespace])


def test_locked_recipe_checks_scientific_identity_of_transitive_attempts(runtime, tmp_path, monkeypatch):
    repo, ancestor = runtime
    old_task = locked_task(repo, ancestor, tmp_path / 'old-lock', monkeypatch)
    value = read_json(old_task)
    old_task.write_text(json.dumps(dict(value, id='ancestor')))
    assert run_task(repo, ancestor, old_task).status == 'pass'
    middle = tmp_path / 'middle'
    initialize_attempt(repo, middle)
    monkeypatch.setenv('SWARM_DEP_ANCESTOR', str(ancestor))
    assert run_task(repo, middle, id='middle', needs={'ancestor': ['value.json']}).status == 'pass'
    (repo / 'src/science.py').write_text('new_science = 2\n')
    head = commit(repo)
    consumer = tmp_path / 'consumer'
    consumer.mkdir()
    (consumer / 'code_commit.txt').write_text(head)
    path = locked_task(repo, consumer, tmp_path / 'new-lock', monkeypatch)
    task = read_json(path)
    task['needs']['middle'] = ['value.json']
    path.write_text(json.dumps(task))
    monkeypatch.setenv('SWARM_DEP_MIDDLE', str(middle))
    with pytest.raises(ContractError, match='dependency scientific code/config drift'):
        run_task(repo, consumer, path)
    assert not (consumer / 'value.json').exists()


@pytest.mark.parametrize('kind', ['final-coverage', 'anchor', 'refit-audit'])
def test_builder_refuses_modified_checkout_approvals(runtime, tmp_path, spec, kind):
    repo, _ = runtime
    original = Path(__file__).parents[1]
    (repo / 'scripts').mkdir()
    shutil.copyfile(original / 'scripts/build_tasks.py', repo / 'scripts/build_tasks.py')
    commit(repo)
    spec.update(id='unapproved', kind=kind)
    spec_file = tmp_path / 'spec.json'
    spec_file.write_text(json.dumps(spec))
    # Only this synthetic checkout's approvals are modified, never owner data.
    (repo / 'configs/approvals.yaml').write_text(yaml.safe_dump({
        'schema_version': 1, 'approved_by': 'fixture', 'owner_decisions': {
            'campaign_allocations': {'unapproved': {'kind': kind, 'gpu_hours': 9}}}}))
    process = subprocess.run([sys.executable, str(repo / 'scripts/build_tasks.py'),
                              '--spec', str(spec_file), '--out', str(tmp_path / 'plan')],
                             env=dict(os.environ, PYTHONPATH=str(original / 'src')),
                             capture_output=True, text=True, timeout=10)
    assert process.returncode != 0, 'builder accepted modified owner allocations'
    assert 'approvals' in process.stderr and 'HEAD' in process.stderr
    assert not (tmp_path / 'plan').exists()


@pytest.mark.parametrize('overlap', [False, True])
def test_code_identity_does_not_refresh_upstream_git_index(runtime, tmp_path, monkeypatch, overlap):
    from oxyformer.execution.integrity import fingerprint_tree
    repo, out = runtime
    source = tmp_path / 'source'
    source.mkdir()
    clone = source / 'src'
    shutil.move(repo, clone)
    if overlap:
        out = source / 'handoff'
        initialize_attempt(clone, out)
    task = task_file(out, needs=SOURCE_NEEDS)
    (source / 'data.json').write_text('{}')
    (source / 'receipts.json').write_text('{}')
    git(clone, 'status', '--porcelain')
    publish_source_fixture(clone, source)
    victim = clone / 'src/science.py'
    meta = victim.stat()
    os.utime(victim, ns=(meta.st_atime_ns, meta.st_mtime_ns + 1000000000))
    before = fingerprint_tree(source)
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(source))
    if overlap:
        with pytest.raises(ContractError, match='overlaps an upstream'):
            run_task(clone, out, task)
    else:
        assert run_task(clone, out, task).status == 'pass'
    assert fingerprint_tree(source) == before


def test_stage_cannot_publish_in_attempt_symlink_as_regular_artifact(runtime, monkeypatch):
    repo, out = runtime
    def faulty(request):
        result = dummy(request)
        (out / 'value.json').rename(out / 'stored.json')
        (out / 'value.json').symlink_to('stored.json')
        return result
    monkeypatch.setitem(sys.modules, 'oxyformer.dummy', SimpleNamespace(run_stage=faulty))
    result = run_task(repo, out, deps_env=False)
    assert result.status == 'fail' and str(out / 'value.json') in result.message


@pytest.mark.parametrize('directory_link', [False, True])
def test_dependency_file_rejects_symlink_before_resolving(tmp_path, directory_link):
    from oxyformer.execution.runner import dependency_file
    (tmp_path / 'stored').mkdir()
    (tmp_path / 'stored/value.json').write_text('{}')
    if directory_link:
        (tmp_path / 'link').symlink_to('stored', target_is_directory=True)
        relative = 'link/value.json'
    else:
        (tmp_path / 'value.json').symlink_to('stored/value.json')
        relative = 'value.json'
    with pytest.raises(ContractError, match='regular|symlink'):
        dependency_file(tmp_path, relative)


def test_dependency_root_symlink_is_not_a_directory_input(tmp_path):
    (tmp_path / 'attempt').mkdir()
    (tmp_path / 'alias').symlink_to('attempt', target_is_directory=True)
    with pytest.raises(ContractError, match='symlink|directory'):
        resolve_dependencies(['source'], {'SWARM_DEP_SOURCE': str(tmp_path / 'alias')})


def test_regular_reader_rejects_symlink_ancestor(tmp_path):
    from oxyformer.execution.integrity import read_regular
    (tmp_path / 'stored').mkdir()
    (tmp_path / 'stored/data.json').write_text('{}')
    (tmp_path / 'alias').symlink_to('stored', target_is_directory=True)
    with pytest.raises(ContractError, match='symlink|directory'):
        read_regular(tmp_path / 'alias/data.json')


@pytest.mark.parametrize('control', ['request.json', 'environment.json', 'identity.json'])
def test_stage_cannot_change_persisted_control_identity(runtime, monkeypatch, control):
    repo, out = runtime
    victim = out / '_execution' / control
    def faulty(request):
        result = dummy(request)
        value = read_json(victim)
        if control == 'request.json':
            value['payload']['code_identity'] = '0' * 40
        else:
            value['changed'] = True
        victim.write_text(json.dumps(value))
        return result
    monkeypatch.setitem(sys.modules, 'oxyformer.dummy', SimpleNamespace(run_stage=faulty))
    result = run_task(repo, out, deps_env=False)
    assert result.status == 'fail' and str(victim) in result.message
    assert read_stage_result(out).status == 'fail'


@pytest.mark.parametrize('control', ['dependency_check.json', 'result.json', 'fingerprint.json', '_execution'])
@pytest.mark.parametrize('kind', ['file', 'fifo', 'directory', 'symlink'])
@pytest.mark.parametrize('mutate', [False, True])
def test_reserved_receipt_collision_preserves_upstream_failure(runtime, tmp_path, monkeypatch, control, kind, mutate):
    repo, out = runtime
    source = make_source_fixture(repo, tmp_path / 'source')
    victim = source / 'data.json'
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(source))
    def faulty(request):
        result = dummy(request)
        path = out / control if control == '_execution' else out / '_execution' / control
        if control == '_execution': path.rename(out / 'saved-controls')
        if kind == 'file': path.write_text('{}')
        elif kind == 'fifo': os.mkfifo(path)
        elif kind == 'directory': path.mkdir()
        else: path.symlink_to(source / 'receipts.json')
        if mutate:
            victim.write_text('[]')
            raise RuntimeError('stage diagnostic failure')
        return result
    monkeypatch.setitem(sys.modules, 'oxyformer.dummy', SimpleNamespace(run_stage=faulty))
    result = run_task(repo, out, needs=SOURCE_NEEDS)
    assert result.status == 'fail'
    collision = out / control if control == '_execution' else out / '_execution' / control
    assert str(victim if mutate else collision) in result.message
    receipt = read_json((out / '_execution/dependency_check.json'))
    assert ('data.json' in receipt['attempts'][str(source)]['changed_paths']) == mutate
    assert read_stage_result(out).status == 'fail'
    assert (source / 'receipts.json').read_text() == '{}'


def test_pass_without_artifacts_is_rejected_by_merged_contract():
    with pytest.raises(ContractError, match='passing stage must declare artifacts'):
        StageResult(request_hash='0' * 64, status='pass', artifacts=(), message='validation only')


def test_unwritable_control_directory_cannot_suppress_upstream_receipts(runtime, tmp_path, monkeypatch):
    if os.geteuid() == 0:
        pytest.skip('this reproduction requires ordinary Unix permissions')
    repo, out = runtime
    source = tmp_path / 'source'
    (source / 'extra').mkdir(parents=True)
    victim = source / 'extra/victim'
    victim.write_text('before')
    seal_source_fixture(repo, source)
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(source))
    try:
        process = run_cli_fixture(repo, out, '''def run_stage(request):
    result = dummy(request)
    (Path(os.environ['SWARM_DEP_DATA_UNIT']) / 'extra/victim').write_text('changed')
    (Path(request.output_dir) / '_execution').chmod(0o500)
    return result
''', needs=SOURCE_NEEDS)
        assert victim.read_text() == 'changed', process.stdout + process.stderr
        assert process.returncode == 1, process.stdout + process.stderr
        result = read_stage_result(out)
        assert result.status == 'fail' and str(victim) in result.message
        receipt = read_json((out / '_execution/dependency_check.json'))
        assert 'extra/victim' in receipt['attempts'][str(source)]['changed_paths']
    finally:
        # Restore only this failed consumer's directory so pytest can clean up.
        (out / '_execution').chmod(0o700)


@pytest.mark.parametrize('receipt', ['dependency_check.json', 'result.json'])
def test_receipt_write_error_after_execution_is_failed_with_changed_paths(runtime, tmp_path, monkeypatch, receipt):
    repo, out = runtime
    source = make_source_fixture(repo, tmp_path / 'source', monkeypatch)
    entrypoint = """import sys
from oxyformer.execution import runner, integrity
from oxyformer.cli import main
original = integrity._replace_control
def unavailable(root, relative, text):
    if relative == '_execution/RECEIPT':
        raise PermissionError('synthetic receipt write denial: ' + relative)
    return original(root, relative, text)
runner._replace_control = integrity._replace_control = unavailable
raise SystemExit(main(sys.argv[1:]))
""".replace('RECEIPT', receipt)
    process = run_cli_fixture(repo, out, """def run_stage(request):
    result = dummy(request)
    (Path(os.environ['SWARM_DEP_DATA_UNIT']) / 'data.json').write_text('changed')
    return result
""", needs=SOURCE_NEEDS, entrypoint=entrypoint)
    assert process.returncode == 1, process.stderr
    assert str(source / 'data.json') in process.stderr
    assert 'synthetic receipt write denial' in process.stderr
    assert 'blocked:' not in process.stderr


def test_unwritable_control_file_recovery_never_chmods_upstream(tmp_path):
    from oxyformer.execution.integrity import _repair_control_directory
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    victim = upstream / 'file'
    victim.write_text('upstream')
    victim.chmod(0o400)
    out = tmp_path / 'attempt'
    control = out / '_execution'
    control.mkdir(parents=True)
    own = control / 'request.json'
    own.write_text('{}')
    own.chmod(0o400)
    (control / 'file-link').symlink_to(victim)
    (control / 'directory-link').symlink_to(upstream, target_is_directory=True)
    os.link(victim, control / 'hardlink')
    nested = control / 'nested'
    nested.mkdir()
    nested.chmod(0)
    try:
        _repair_control_directory(out)
        assert own.stat().st_mode & 0o600 == 0o600
        assert nested.stat().st_mode & 0o700 == 0o700
        assert victim.stat().st_mode & 0o777 == 0o400
        assert victim.read_text() == 'upstream'
    finally:
        nested.chmod(0o700)


@pytest.mark.parametrize('kind', ['file', 'directory'])
def test_read_only_isolated_cache_is_an_honest_passing_stage(runtime, monkeypatch, kind):
    repo, out = runtime
    def cached(request):
        result = dummy(request)
        cache = Path(os.environ['HF_HOME']) / 'download'
        if kind == 'file':
            cache.write_text('cached')
            cache.chmod(0o444)
        else:
            cache.mkdir()
            cache.chmod(0o555)
        return result
    install_stage(monkeypatch, repo, cached)
    result = run_task(repo, out, deps_env=False)
    assert result.status == 'pass', result.message


@pytest.mark.parametrize('location', ['upstream', 'cache'])
def test_deep_tree_preserves_detection_and_publication(runtime, tmp_path, monkeypatch, location):
    repo, out = runtime
    source = make_source_fixture(repo, tmp_path / 'source', monkeypatch)
    created = []
    def stage(request):
        result = dummy(request)
        path = source if location == 'upstream' else Path(os.environ['HF_HOME'])
        for _ in range(1150):
            path = path / 'd'
            path.mkdir()
            created.append(path)
        if location == 'cache':
            path.chmod(0o555)
        return result
    install_stage(monkeypatch, repo, stage)
    try:
        result = run_task(repo, out, needs=SOURCE_NEEDS)
        assert read_stage_result(out) == result
        if location == 'upstream':
            assert result.status == 'fail' and str(created[-1]) in result.message
            check = read_json(out / '_execution/dependency_check.json')
            assert created[-1].relative_to(source).as_posix() in check['attempts'][str(source)]['changed_paths']
            with pytest.raises(ContractError, match='fingerprint mismatch'):
                verify_dependency_result(source)
        else:
            assert result.status == 'pass', result.message
            assert verify_dependency_result(out) == result
    finally:
        for path in reversed(created):
            path.rmdir()


def test_rewritten_upstream_publication_fails_changer_and_transitive_collector(runtime, tmp_path, monkeypatch):
    from oxyformer.execution.integrity import FINGERPRINT, RESULT, publication_tree
    from oxyformer.provenance import canonical_json
    repo, out = runtime
    source = tmp_path / 'source'
    source.mkdir()
    victim = source / 'undeclared'
    victim.write_text('before')
    seal_source_fixture(repo, source)
    monkeypatch.setenv('SWARM_DEP_DATA_UNIT', str(source))
    middle = tmp_path / 'middle'
    initialize_attempt(repo, middle)
    assert run_task(repo, middle, id='middle', needs=SOURCE_NEEDS).status == 'pass'
    monkeypatch.setenv('SWARM_DEP_MIDDLE', str(middle))
    def rewritten(request):
        result = dummy(request)
        victim.write_text('changed')
        fingerprint = read_json(source / FINGERPRINT)
        fingerprint['entries'] = publication_tree(source)
        (source / FINGERPRINT).write_text(canonical_json(fingerprint))
        producer = read_stage_result(source)
        producer = replace(producer, artifacts=tuple(
            replace(a, sha256=file_hash(source / FINGERPRINT)) if a.path == FINGERPRINT else a
            for a in producer.artifacts))
        (source / RESULT).write_text(producer.to_json())
        return result
    install_stage(monkeypatch, repo, rewritten)
    failed = run_task(repo, out, id='changing', needs={'middle': ['value.json']})
    assert failed.status == 'fail' and str(victim) in failed.message
    assert read_stage_result(out) == failed
    check = read_json(out / '_execution/dependency_check.json')
    assert check['status'] == 'fail'
    assert check['attempts'][str(source)]['status'] == 'tainted'
    assert 'undeclared' in check['attempts'][str(source)]['changed_paths']
    install_stage(monkeypatch, repo, dummy)
    monkeypatch.setenv('SWARM_DEP_CHANGING', str(out))
    # Refuse both the failed changer and the previously passing middle whose
    # persisted request binds the ancestor's original publication identity.
    for unit, reason in [('changing', 'did not pass'), ('middle', 'input hash mismatch')]:
        collector = tmp_path / ('collector-' + unit)
        initialize_attempt(repo, collector)
        with pytest.raises(ContractError, match=reason):
            run_task(repo, collector, needs={unit: ['value.json']})
        assert not (collector / '_execution').exists()
        assert not (collector / 'value.json').exists()


def test_deep_valid_dependency_lineage_does_not_exhaust_python_stack(runtime, tmp_path):
    repo, _ = runtime
    head = git(repo, 'rev-parse', 'HEAD')
    parent = None
    for index in range(1100):
        root = tmp_path / str(index)
        root.mkdir()
        for name in SOURCE_NEEDS['data-unit']:
            (root / name).write_text('{}')
        publish_source_fixture(repo, root, parent=parent, head=head)
        parent = root
    assert verify_dependency_result(parent).status == 'pass'


@pytest.mark.parametrize('commit_change', [True, False])
def test_stage_cannot_replace_recorded_checkout_identity(runtime, monkeypatch, commit_change):
    import runpy
    repo, out = runtime
    original = git(repo, 'rev-parse', 'HEAD')
    def replacement(request):
        result = dummy(request)
        (repo / 'src/science.py').write_text('value = 2\n')
        if commit_change:
            replacement_head = commit(repo)
            (out / 'code_commit.txt').write_text(replacement_head)
        value = runpy.run_path(str(repo / 'src/science.py'))['value']
        (out / 'value.json').write_text(json.dumps({'value': value}))
        return replace(result, artifacts=(replace(result.artifacts[0], sha256=file_hash(out / 'value.json')),))
    install_stage(monkeypatch, repo, replacement)
    result = run_task(repo, out, deps_env=False)
    assert (git(repo, 'rev-parse', 'HEAD') != original) == commit_change
    assert read_json(out / 'value.json') == {'value': 2}
    request = StageRequest.from_json((out / '_execution/request.json').read_text())
    assert request.code_identity == original
    assert read_json(out / '_execution/identity.json')['head'] == original
    assert result.status == 'fail', 'replacement checkout passed under the original recorded identity'
    assert read_stage_result(out) == result
    if commit_change:
        assert 'code identity changed' in result.message
        assert original in result.message and git(repo, 'rev-parse', 'HEAD') in result.message
    else:
        assert 'tracked modifications' in result.message and 'src/science.py' in result.message


def test_builder_existing_expansion_cannot_leave_mixed_task_manifest(tmp_path, spec):
    root = Path(__file__).parents[1]
    spec_file = tmp_path / 'spec.json'
    spec_file.write_text(json.dumps(spec))
    out = tmp_path / 'plan'
    out.mkdir()
    old = {'previous_campaign': True}
    (out / 'expanded_units.json').write_text(json.dumps(old))
    process = subprocess.run([sys.executable, str(root / 'scripts/build_tasks.py'),
        '--spec', str(spec_file), '--out', str(out)],
        env=dict(os.environ, PYTHONPATH=str(root / 'src'), CUDA_VISIBLE_DEVICES=''),
        capture_output=True, text=True, timeout=30)
    assert process.returncode != 0
    assert read_json(out / 'expanded_units.json') == old
    assert not (out / 'task_manifest.json').exists(), 'new tasks were published beside an older expansion'
