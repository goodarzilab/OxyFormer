"""Hermetic receipt reuse and resumed admission through the real runner."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest import raises
import yaml

from oxyformer.contracts import StageResult
from oxyformer.execution.identity import scientific_fingerprint
from oxyformer.execution.integrity import FINGERPRINT, fingerprint_tree
from oxyformer.execution.runner import dependency_variable, run as run_worker
from oxyformer.provenance import ArtifactRecord, ContractError, canonical_json, file_hash
from test_campaign_profiles import ADMISSION_SHAPE_CASES, admission_shape_fixture
from test_execution import (publication_authority, runtime, source, assert_pass, commit,
    new_attempt, task_file, read_json, read_check, inline_fixture_stage, lineage_for, SOURCE_NEEDS)


@pytest.fixture
def verification_chain(runtime, source, tmp_path, monkeypatch):
    """Real publications/budgets, synthetic bytes, and no training or data reads."""
    from oxyformer.validation import coverage
    from test_coverage import successful_records
    repo, _ = runtime
    stages = ('dummy', 'simulation-smoke', 'campaign-estimate', 'campaign-collect', 'campaign-lock', 'campaign-setup-probe')
    (repo / 'configs/execution/stages.yaml').write_text(yaml.safe_dump({'schema_version': 1,
        'stages': {name: {'module': 'oxyformer.dummy'} for name in stages}}))
    commit(repo)
    roots = {'data-unit': source}
    draws, _ = successful_records(1)
    parameters = {'mode': 'profile', 'wall_seconds': 600., 'draws': draws,
        'scenario': {'name': 'null_effect', 'effect': 'null', 'active_mechanisms': ['null']},
        'recipe': {'fixture': 'verification-only'}}

    def stage(request, *unused):
        task = read_json(Path(request.task_path))
        config = read_json(Path(request.config_path))
        if request.stage == 'campaign-estimate':
            records = [coverage.read_leaf_measurements(roots[f'calibration-{i}'] / 'budget_record.json')
                       for i in range(6)]
            return coverage.publish(request, {'budget_estimate.json': {'calibration': {'records': [
                {key: record[key] for key in ('record_hash', 'budget_generation')}
                | {'task_id': record['binding']['task_id'], **{key: record['profile'][key]
                    for key in ('scientific_fingerprint', 'environment_hash')}} for record in records]}}},
                status='pass', message='synthetic estimate')
        if request.stage != 'simulation-smoke':
            if task['id'] == 'heavy':
                path = Path(request.output_dir) / 'blob.bin'
                path.write_bytes(b'x' * (8 * 1024 * 1024))
                return StageResult(request_hash=request.content_hash, status='pass', message='synthetic bytes',
                    artifacts=(ArtifactRecord(path=path.name, sha256=file_hash(path), kind='fixture',
                        lineage=lineage_for(request, task['id'])),))
            return inline_fixture_stage(request, 'oxyformer.dummy', repo)
        started = coverage.time.monotonic()
        chain = task.get('continuation')
        history = coverage.load_budget(request, config, chain['predecessor'])['leaves'] if chain and chain['predecessor'] else []
        timing = {'recipe_hash': coverage.digest(parameters['recipe']),
            'complete_repetition_seconds': [], 'setup_seconds': 0., 'device': 'cpu',
            'environment': {'synthetic': True}, 'production_equivalent': False,
            'scientific_fingerprint': scientific_fingerprint(repo), 'environment_hash': 'a' * 64}
        extra = ()
        if chain:
            path = Path(request.output_dir) / 'continuation.tar'
            path.write_bytes(b'synthetic opaque continuation\n' * (128 * 1024))
            extra = (ArtifactRecord(path=path.name, sha256=file_hash(path), kind='coverage_continuation',
                lineage=lineage_for(request, task['id'])),)
            timing['leaves'] = [*history, {'task_id': task['id'], 'step': chain['step'],
                **dict.fromkeys(('setup_seconds', 'compute_seconds', 'import_seconds', 'export_seconds'), 0.)}]
        values = {'result.json': {'records': [], 'draws': draws,
            'recipe_hash': timing['recipe_hash'], 'mode': 'profile',
            'scenario': coverage.SCMConfig(**parameters['scenario']).to_dict()['payload']}}
        values.update({name: {'fixture': 'synthetic profile output'} for name in ('timing.json', 'runtime.json')
            if name in task['outputs']})
        result = coverage.publish(request, values,
            status='pass', message='synthetic checkpoint', extra_artifacts=extra)
        summary = coverage.summarize([], draws, production_equivalent=False, null_scenario=True)
        return coverage.record_measurements(request, result, started, task, timing, summary)

    results = {}

    def produce(unit, stage_name='dummy', needs=None, *, executor=None, expect_pass=True, **fields):
        out = new_attempt(repo, tmp_path / unit)
        task = {'id': unit, 'stage': stage_name, 'needs': needs or {}, **fields}
        for dep in task['needs']:
            monkeypatch.setenv(dependency_variable(dep), str(roots[dep]))
        result = run_worker(stage_name, out, repo, deps_env=True,
            task_file=task_file(out, **task), execute=executor or stage)
        if expect_pass:
            assert_pass(result)
        roots[unit] = out
        results[unit] = result
        return out

    produce('heavy', outputs=['blob.bin'])
    produce('frame', needs=SOURCE_NEEDS, outputs=['value.json'])
    for i in range(6):
        produce(f'calibration-{i}', 'simulation-smoke', {'heavy': ['blob.bin']},
            parameters=parameters, outputs=['result.json', 'budget_record.json'])
    refs = [{'dependency': f'calibration-{i}', 'path': 'budget_record.json'} for i in range(6)]
    produce('estimate', 'campaign-estimate', {'frame': ['value.json'],
        **{ref['dependency']: [ref['path']] for ref in refs}}, parameters={'setup_calibration': refs},
        outputs=['budget_estimate.json'])

    def leaf(step, *, direct=False, outputs=None):
        needs = {'frame': ['value.json'], 'estimate': ['budget_estimate.json']}
        if step:
            needs[f'leaf-{step-1}'] = ['_execution/task.json', '_execution/request.json',
                '_execution/result.json', 'result.json', 'budget_record.json', 'continuation.tar']
        if direct:
            needs.update(SOURCE_NEEDS)
        return produce(f'leaf-{step}', 'simulation-smoke', needs, parameters=parameters,
            continuation={'owner': 'synthetic-chain', 'step': step,
                'predecessor': f'leaf-{step-1}' if step else None},
            outputs=outputs or ['result.json', 'budget_record.json', 'continuation.tar'])
    return SimpleNamespace(leaf=leaf, roots=roots, produce=produce, repo=repo, source=source,
        parameters=parameters, results=results)


def test_five_leaf_receipt_reuse_bounds_verification_work(verification_chain, tmp_path):
    from oxyformer.execution.runner import verify_only
    chain = verification_chain
    rows, full_rows = [], []
    for step in range(5):
        root = chain.leaf(step)
        rows.append(read_check(root)['timings'])
        full_rows.append(verify_only(root, tmp_path / f'full-depth-{step+1}', full_walk=True)['timings'])
    assert rows[0]['verified_roots'] >= 10
    assert [row['verified_roots'] for row in full_rows] == [10, 11, 12, 13, 14]
    assert all(row['verified_roots'] == 3 and row['ancestors_reused'] >= 8
        and row['ancestors_rewalked'] == 0 and not row['reuse_miss'] for row in rows[1:])
    assert all(row['measurement_ancestors_reused'] == 7 for row in rows[1:])
    hashed = lambda row: sum(v.get('hashing', {}).get('bytes', 0) for v in row['roots'].values())
    assert max(map(hashed, rows[1:])) < 1.2 * hashed(rows[0])
    assert hashed(rows[4]) <= 1.1 * hashed(rows[1])
    full = full_rows[-1]
    reused = verify_only(chain.roots['leaf-4'], tmp_path / 'reused-replay')['timings']
    assert full['verified_roots'] >= 14 and reused['verified_roots'] == 3
    assert hashed(reused) < .5 * hashed(full)
    print('SYNTHETIC_VERIFICATION_TIMINGS=' + json.dumps([
        {'depth': i + 1, 'seconds': row['total_seconds'], 'bytes': hashed(row),
         'verified_roots': row['verified_roots'], 'reused': row.get('ancestors_reused', 0),
         'full_walk_seconds': full_rows[i]['total_seconds'],
         'full_walk_bytes': hashed(full_rows[i]), 'full_walk_roots': full_rows[i]['verified_roots']}
        for i, row in enumerate(rows)]))


def test_closure_reads_each_direct_parent_identity_once(verification_chain, monkeypatch):
    from oxyformer.execution import runner
    chain = verification_chain
    for step in range(4):
        predecessor = chain.leaf(step)
    direct = {name: chain.roots[name] for name in ('frame', 'estimate')}
    policy = runner.DependencyReuse(direct)
    policy.add_receipt(predecessor)
    reads = {root / '_execution/task.json': 0 for root in direct.values()}
    original = runner.read_mapping

    def observed(path):
        if Path(path) in reads:
            reads[Path(path)] += 1
        return original(path)

    monkeypatch.setattr(runner, 'read_mapping', observed)
    policy.validate_closure()
    assert not policy.rewalk
    assert set(reads.values()) == {1}, reads


@pytest.mark.parametrize('consumer', ['resumed', 'direct', 'collector'])
def test_receipt_boundary_and_collector_backstop(verification_chain, tmp_path, monkeypatch, consumer):
    chain = verification_chain
    chain.leaf(0)
    (chain.source / 'data.json').write_text('{"changed_after_seal":true}')
    if consumer == 'resumed':
        result = chain.leaf(1)
        assert read_check(result)['timings']['ancestors_reused'] >= 8
        return  # This undetected transitive-only drift is the declared boundary.
    if consumer == 'direct':
        with raises(ContractError, match='(fingerprint|input hash) mismatch'):
            chain.leaf(1, direct=True)
        assert not (tmp_path / 'leaf-1' / FINGERPRINT).exists()
    else:
        with raises(ContractError, match='(fingerprint|input hash) mismatch'):
            chain.produce('collector', 'campaign-collect', {'leaf-0': ['result.json']}, outputs=['value.json'])
        assert not (tmp_path / 'collector' / FINGERPRINT).exists()


@pytest.mark.parametrize('name', ['_execution/task.json', '_execution/result.json'])
def test_unused_transitive_control_payload_is_attested(verification_chain, tmp_path, name):
    chain = verification_chain
    chain.leaf(0)
    (chain.source / name).write_text('{"unused_payload_changed":true}')
    resumed = chain.leaf(1)
    assert read_check(resumed)['timings']['ancestors_rewalked'] == 0
    with raises(ContractError):
        chain.produce('collector', 'campaign-collect', {'leaf-1': ['result.json']}, outputs=['value.json'])
    assert not (tmp_path / 'collector' / FINGERPRINT).exists()


@pytest.mark.parametrize('fault', ['tampered', 'missing', 'multiple_missing', 'malformed', 'unbound'])
def test_invalid_predecessor_dependency_receipt_rewalks_shared_ancestor(verification_chain, monkeypatch, fault):
    from oxyformer.execution import runner
    chain = verification_chain
    original_replace, original_json = runner._replace_control, runner.atomic_json

    def replace_control(root, name, value):
        if Path(root).name == 'leaf-0' and name == runner.DEPENDENCY_CHECK and fault != 'unbound':
            check = json.loads(value)
            if fault in ('missing', 'multiple_missing'):
                del check['attempts'][str(chain.source)]
                if fault == 'multiple_missing':
                    del check['attempts'][str(chain.roots['heavy'])]
            elif fault == 'malformed':
                check['attempts'][str(chain.source)] = None
            else:
                check['attempts'][str(chain.source)]['fingerprint']['data.json']['sha256'] = 'f' * 64
            value = canonical_json(check)
        return original_replace(root, name, value)

    def write_json(root, name, value):
        path = original_json(root, name, value)
        if Path(root).name == 'leaf-0' and name == '_execution/dependencies.json' and fault == 'unbound':
            # The receipt remains publication-bound, but its snapshot is not
            # named/hash-bound by the producer request. It cannot authorize reuse.
            return original_json(root, '_execution/unbound-snapshot.json', {})
        return path

    with monkeypatch.context() as injection:
        injection.setattr(runner, '_replace_control', replace_control)
        injection.setattr(runner, 'atomic_json', write_json)
        chain.leaf(0)
    leaf = chain.leaf(1)
    timing = read_check(leaf)['timings']
    assert timing['ancestors_rewalked'] >= 1
    assert timing['roots'][str(chain.source)]['hashing']['bytes'] > 0
    assert read_check(leaf)['attempts'][str(chain.source)]['fingerprint'] == fingerprint_tree(chain.source,
        include=read_check(leaf)['attempts'][str(chain.source)]['fingerprint']['.']['published'])
    if fault == 'multiple_missing':
        assert timing['reuse_miss'] and timing['ancestors_rewalked'] > timing['predecessor_publications_added']
        assert read_json(leaf / 'budget_record.json')['leaves'][-1]['verification']['status'] == 'reuse-miss'


def test_postseal_predecessor_receipt_tamper_never_authorizes_reuse(verification_chain):
    chain = verification_chain
    predecessor = chain.leaf(0)
    path = predecessor / '_execution/dependency_check.json'
    path.chmod(path.stat().st_mode | 0o200)
    path.write_text('{}')
    with raises(ContractError, match='fingerprint mismatch'):
        chain.leaf(1)


@pytest.mark.parametrize('field', ['changed_paths', 'tainted_paths'])
@pytest.mark.parametrize('value', [False, None, 0, '', {}])
def test_falsey_receipt_lists_force_full_verification(verification_chain, tmp_path, monkeypatch, field, value):
    from oxyformer.execution import runner
    chain = verification_chain
    original = runner._replace_control

    def replace_control(root, name, text):
        if Path(root).name == 'leaf-0' and name == runner.DEPENDENCY_CHECK:
            check = json.loads(text)
            check['attempts'][str(chain.source)][field] = value
            text = canonical_json(check)
        return original(root, name, text)

    with monkeypatch.context() as injection:
        injection.setattr(runner, '_replace_control', replace_control)
        chain.leaf(0)
    (chain.source / 'data.json').write_text('{"changed_after_seal":true}')
    with raises(ContractError, match='(fingerprint|input hash) mismatch'):
        chain.leaf(1)
    assert not (tmp_path / 'leaf-1' / FINGERPRINT).exists()


def test_receipt_missing_both_snapshot_and_attempt_for_nested_ancestor(verification_chain, tmp_path, monkeypatch):
    from oxyformer.execution import runner
    chain = verification_chain
    chain.produce('middle', needs=SOURCE_NEEDS, outputs=['value.json'])
    chain.produce('outer', needs={'middle': ['value.json']}, outputs=['value.json'])
    original_json, original_replace = runner.atomic_json, runner._replace_control

    def write_json(root, name, value):
        if Path(root).name == 'p0' and name == '_execution/dependencies.json':
            value = deepcopy(value)
            del value[str(chain.source)]
        return original_json(root, name, value)

    def replace_control(root, name, text):
        if Path(root).name == 'p0' and name == runner.DEPENDENCY_CHECK:
            value = json.loads(text)
            del value['attempts'][str(chain.source)]
            text = canonical_json(value)
        return original_replace(root, name, text)

    fields = {'parameters': chain.parameters,
        'outputs': ['result.json', 'budget_record.json', 'continuation.tar']}
    with monkeypatch.context() as injection:
        injection.setattr(runner, 'atomic_json', write_json)
        injection.setattr(runner, '_replace_control', replace_control)
        chain.produce('p0', 'simulation-smoke', {'outer': ['value.json']},
            continuation={'owner': 'closure-chain', 'step': 0, 'predecessor': None}, **fields)
    (chain.source / 'data.json').write_text('{"changed_after_seal":true}')
    with raises(ContractError, match='(fingerprint|input hash) mismatch'):
        chain.produce('p1', 'simulation-smoke', {'outer': ['value.json'], 'p0': [
            '_execution/task.json', '_execution/request.json', '_execution/result.json',
            'result.json', 'budget_record.json', 'continuation.tar']},
            continuation={'owner': 'closure-chain', 'step': 1, 'predecessor': 'p0'}, **fields)
    assert not (tmp_path / 'p1' / FINGERPRINT).exists()


def test_null_receipt_file_entry_rewalks_unchanged_source(verification_chain, monkeypatch):
    from oxyformer.execution import runner
    chain = verification_chain
    original_json, original_replace = runner.atomic_json, runner._replace_control

    def write_json(root, name, value):
        if Path(root).name == 'leaf-0' and name == '_execution/dependencies.json':
            value = deepcopy(value)
            value[str(chain.source)]['data.json'] = None
        return original_json(root, name, value)

    def replace_control(root, name, text):
        if Path(root).name == 'leaf-0' and name == runner.DEPENDENCY_CHECK:
            value = json.loads(text)
            value['attempts'][str(chain.source)]['fingerprint']['data.json'] = None
            text = canonical_json(value)
        return original_replace(root, name, text)

    with monkeypatch.context() as injection:
        injection.setattr(runner, 'atomic_json', write_json)
        injection.setattr(runner, '_replace_control', replace_control)
        chain.leaf(0)
    leaf = chain.leaf(1)
    timing = read_check(leaf)['timings']
    assert timing['ancestors_rewalked'] >= 1
    assert timing['roots'][str(chain.source)]['hashing']['bytes'] > 0


@pytest.mark.parametrize('ancestor', ['source', 'heavy'])
def test_diagnostic_scratch_cannot_enter_attested_transitive_ancestor(verification_chain, ancestor):
    from oxyformer.execution.runner import verify_only
    chain = verification_chain
    chain.leaf(0)
    resumed = chain.leaf(1)
    root = chain.source if ancestor == 'source' else chain.roots['heavy']
    scratch = root / 'diagnostic-scratch'
    with raises(ContractError, match='overlaps.*attempt|overlaps.*dependency|overlaps.*ancestor'):
        verify_only(resumed, scratch)
    assert not scratch.exists()


@pytest.fixture
def probe_pipeline(verification_chain, request):
    """Actual runner receipts and synthetic profile bytes; no training or lock bypass."""
    from oxyformer.execution.integrity import RESULT, verify_published_tree
    from oxyformer.validation import campaign_probe, coverage
    chain = verification_chain
    outputs = ['result.json', 'budget_record.json', 'continuation.tar', 'timing.json', 'runtime.json']
    sources = {}
    for step in (0, 1):
        root = chain.leaf(step, outputs=outputs)
        terminal = StageResult.from_json((root / RESULT).read_text())
        tree = verify_published_tree(root, terminal)
        sources[str(step+1)] = {'reference': {'dependency': f'leaf-{step}', 'path': 'budget_record.json'},
            'publication_hash': tree[FINGERPRINT]['sha256'], 'archive_hash': tree['continuation.tar']['sha256'],
            'shape': campaign_probe.publication_shape(tree), 'outputs': outputs}
    record = read_json(chain.roots['leaf-1'] / 'budget_record.json')
    scenario = chain.parameters['scenario']['name']
    missing = getattr(request, 'param', 1) == 'missing'
    copies = 1 if missing else getattr(request, 'param', 1)
    profile = {'draws': chain.parameters['draws'], 'leaf_measurements': record['leaves'],
        'setup_profile_sources': {} if missing else sources, 'assumed_usable_per_leaf': 100. * copies,
        'seconds_per_repetition': 100.}
    expected = campaign_probe.expected_predecessor_shape(profile, outputs, draw_count=copies)
    lock = {'campaign_id': 'synthetic-admission', 'scientific_fingerprint': scientific_fingerprint(chain.repo),
        'environment_hash': 'a' * 64, 'estimate_input': {'dependency': 'estimate', 'path': 'budget_estimate.json'},
        'estimate_hash': file_hash(chain.roots['estimate'] / 'budget_estimate.json'),
        'admission_receipt': {'dependency': 'admission', 'path': 'admission.json'},
        'budget': {'profiles': {scenario: profile}, 'expected_predecessor_shapes': {scenario: expected}}}
    lock_hash = coverage.digest(lock)
    base = {'frame': ['value.json'], 'estimate': ['budget_estimate.json'],
        'leaf-1': list(campaign_probe.PROFILE_INPUTS), 'lock': list(campaign_probe.LOCK_OUTPUTS)}
    first = {'parameters': {'scenario': chain.parameters['scenario']}, 'needs': base}
    plans = [{'tasks': [first], 'spec': {'inputs': base}}]

    def lock_producer(request, *unused):
        return coverage.publish(request, {'recipe_lock.json': lock,
            'expanded_units.json': {'lock_hash': lock_hash, 'plans': plans},
            'task_manifest.json': {}, 'budget.json': lock['budget'], 'gate.json': {}},
            status='pass', message='synthetic immutable lock for verification, no scientific admission')

    chain.produce('lock', 'campaign-lock', {k: v for k, v in base.items() if k != 'lock'},
        executor=lock_producer, parameters={}, outputs=list(campaign_probe.LOCK_OUTPUTS))
    plan = campaign_probe.setup_admission_plan(lock, plans, 'lock', {})

    def execute(request, *unused):
        task, config = read_json(Path(request.task_path)), read_json(Path(request.config_path))
        if task['parameters']['mode'] == 'admission-predecessor':
            return campaign_probe.run_stage(request)
        # Each value belongs to an independently executed request/publication.
        value = campaign_probe.probe(request, config, task,
            measurer=lambda: 100. if task['parameters'].get('first_draw') else 600.)
        return coverage.publish(request, {'setup_probe.json': value}, status='pass', message='synthetic setup draw')

    def run(task, *, expect_pass=True):
        fields = deepcopy(task)
        unit, stage, needs = fields.pop('id'), fields.pop('stage'), fields.pop('needs')
        return chain.produce(unit, stage, needs, executor=execute, expect_pass=expect_pass, **fields)
    return SimpleNamespace(chain=chain, plan=plan, run=run, lock=lock)


@pytest.mark.parametrize('probe_pipeline', [1, 3], indirect=True)
def test_two_depth_probe_pipeline_uses_fresh_and_grand_predecessor_receipts(probe_pipeline):
    pipeline = probe_pipeline
    for task in pipeline.plan['tasks'][:-1]:
        pipeline.run(task)
    receipts = {}
    for key, ref in pipeline.plan['receipt_inputs'].items():
        root = pipeline.chain.roots[ref['dependency']]
        value = read_json(root / 'setup_probe.json')
        assert value['setup_draws'] == [600., 100.] and value['setup_seconds'] == 100.
        receipts[key] = value
        if key != 'fresh':
            assert value['verification']['ancestors_reused'] > 0
            assert value['verification']['ancestors_rewalked'] == 0
            assert not value['verification']['reuse_miss']
            assert value['predecessor_shape']['status'] == 'matched'
    assert [receipts[key]['depth'] for key in ('fresh', 'resumed_1', 'resumed_2')] == [0, 1, 2]
    assert pipeline.plan['tasks'][-1]['parameters']['probe_inputs'] == pipeline.plan['receipt_inputs']
    assert all('--partition=standard' in unit['sbatch'] and unit['gpu_hours'] == 0 for unit in pipeline.plan['units'])


@pytest.mark.parametrize('probe_pipeline', ['missing'], indirect=True)
def test_resumed_admission_requires_same_tree_early_profiles(probe_pipeline):
    pipeline = probe_pipeline
    for task in pipeline.plan['tasks'][:-1]:
        if task['parameters']['mode'] == 'admission-predecessor':
            root = pipeline.run(task, expect_pass=False)
            result = pipeline.chain.results[task['id']]
            assert result.status != 'pass' and 'same-tree profile leaves unavailable' in result.message
            assert not (root / 'continuation.tar').exists() and not (root / FINGERPRINT).exists()
            return
        pipeline.run(task)
    pytest.fail('synthetic predecessor builder missing')


@pytest.mark.parametrize('depth', [1, 2])
def test_resumed_probe_refuses_cross_tree_profile(probe_pipeline, monkeypatch, depth):
    from oxyformer.validation import coverage
    pipeline = probe_pipeline
    for task in pipeline.plan['tasks'][:-1]:
        p = task['parameters']
        if p['mode'] == 'admission-predecessor' and p['depth'] == depth:
            original = coverage.load_budget
            def changed(*args, **kwargs):
                record = original(*args, **kwargs)
                if args[2] == p['profile_input']['dependency']:
                    record = deepcopy(record)
                    record['profile']['scientific_fingerprint']['sha256'] = 'f' * 64
                return record
            monkeypatch.setattr(coverage, 'load_budget', changed)
            root = pipeline.run(task, expect_pass=False)
            result = pipeline.chain.results[task['id']]
            assert result.status != 'pass' and 'same-tree profile' in result.message
            assert not (root / 'continuation.tar').exists() and not (root / FINGERPRINT).exists()
            return
        pipeline.run(task)
    pytest.fail('expected synthetic predecessor task')


@pytest.mark.parametrize('depth', [1, 2])
def test_resumed_probe_refuses_reuse_miss(probe_pipeline, monkeypatch, depth):
    from oxyformer.validation import campaign_probe
    pipeline = probe_pipeline
    original = campaign_probe.publish
    def malformed_receipt(request, values, **kwargs):
        value = values.get('setup_probe.json', {})
        if value.get('mode') == 'admission-predecessor' and value['depth'] == depth:
            values = deepcopy(values)
            values['setup_probe.json']['dependency_receipt']['check']['attempts'] = {}
        return original(request, values, **kwargs)
    monkeypatch.setattr(campaign_probe, 'publish', malformed_receipt)
    for task in pipeline.plan['tasks'][:-1]:
        p = task['parameters']
        if p['mode'] == 'admission-resumed' and p['depth'] == depth:
            root = pipeline.run(task, expect_pass=False)
            result = pipeline.chain.results[task['id']]
            assert result.status != 'pass' and 'reuse-miss' in result.message
            assert not (root / FINGERPRINT).exists()
            return
        pipeline.run(task)
    pytest.fail('expected resumed probe task')




@pytest.mark.parametrize('admission_shape_fixture', ADMISSION_SHAPE_CASES, indirect=True)
def test_resumed_receipt_synthetic_shape_matches_r5_lock(admission_shape_fixture):
    from oxyformer.validation import campaign_probe
    req, config, task, deps, tree, receipt, status = admission_shape_fixture
    publication = lambda root: (None, tree)
    if status == 'admission-shape-mismatch':
        with raises(ContractError, match=status):
            campaign_probe.resumed_probe_receipt(task, deps, publication)
    else:
        assert campaign_probe.resumed_probe_receipt(task, deps, publication) == (
            deps['predecessor'], receipt['dependency_receipt'])
