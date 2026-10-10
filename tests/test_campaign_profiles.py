"""Synthetic-only adapter, estimate and production-task contracts; no fitting."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from oxyformer.data.entity_graph import EntityGraph
from oxyformer.execution.runner import read_mapping
from oxyformer.design.splits import dependence_groups
from oxyformer.provenance import ContractError, file_hash
from oxyformer.training import nested_cv
from oxyformer.validation import campaign, campaign_estimate, coverage, real_frame, smoke_inputs
from oxyformer.validation.scm import SCMConfig
from test_campaign import STAMPS, request, write
from test_coverage import successful_records
from test_design import make_inputs

ROOT = Path(__file__).parents[1]


def real_fixture():
    """Two artificial counties with real-input types, missing X and subblocks."""
    v = make_inputs()
    rows = tuple(replace(r, original_id=c + r.original_id, tract_id=c + r.tract_id,
        assignment_geography=c + r.assignment_geography, county=c,
        latitude=r.latitude + (2 if c == 'c2' else 0))
        for c in ('c1', 'c2') for r in v['geography'].rows)
    atlas = tuple(replace(r, tract_id=c + r.tract_id) for c in ('c1', 'c2') for r in v['atlas'].rows)
    ids = tuple(r.original_id for r in rows)
    graph = EntityGraph(original_ids=ids, links=())
    m = v['data_manifest']
    schema = tuple(replace(col, nullable=True) if col.name in ('y', 'female_share') else col for col in m.schema)
    manifest = replace(m, schema=schema, original_ids=ids, entity_graph_hash=graph.content_hash,
                       lineage=replace(m.lineage, unit_ids=ids))
    v.update(data_manifest=manifest, entity_graph=graph,
        geography=replace(v['geography'], rows=rows, data_manifest_hash=manifest.content_hash),
        atlas=replace(v['atlas'], rows=atlas, expected_tract_ids=ids),
        covariates=replace(v['covariates'], original_ids=ids, values=((None,),) * len(ids),
            lineage=replace(v['covariates'].lineage, unit_ids=ids, parent_hashes=(manifest.content_hash,))),
        approvals=read_mapping(ROOT / 'configs/approvals.yaml'))
    return v


@pytest.fixture(scope='module')
def built():
    values = real_fixture()
    return values, real_frame.build_inputs(values, read_mapping(ROOT / 'configs/design.yaml'))


def test_real_frame_preserves_x_geography_missingness_and_whole_clusters(built):
    v, (endpoint, frame, support, audit) = built
    assert frame.original_ids == endpoint.outer.original_ids
    assert frame.x == ((None,),) * len(frame.original_ids)
    assert endpoint.data.covariates(frame.columns).values == v['covariates'].values
    assert set(endpoint.data.column('y')) == {None}
    rows = {r.original_id: r for r in v['geography'].rows}
    assert frame.coordinates == tuple((rows[i].latitude, rows[i].longitude) for i in frame.original_ids)
    assert frame.geography_ids == tuple(rows[i].assignment_geography for i in frame.original_ids)
    for group in dependence_groups(
            v['geography'].rows, v['entity_graph']):
        present = set(group).intersection(frame.original_ids)
        assert not present or present == set(group)
    assert audit['cluster_sizes'] == dict(Counter(frame.cluster_ids))
    assert not set(frame.original_ids).intersection(endpoint.outer.design_ids)
    assert endpoint.treatment_design.knots == support.spline_knots
    for fold in range(5):
        endpoint.validate(fold)
    assert endpoint.outer.seed_ids == (1103, 2207, 3301)


def test_real_frame_can_bind_generated_observations_and_all_production_settings(built, tmp_path):
    _, (endpoint, frame, _, _) = built
    from oxyformer.validation.generators import ObservedRecords
    # Synthetic observations, no truth computation required for adapter checks.
    sample = ObservedRecords(frame=frame, a=(5.,) * len(frame.original_ids),
        y=(50.,) * len(frame.original_ids), measured_columns=(),
        measured_x=((),) * len(frame.original_ids),
        flag_available=(True,) * len(frame.original_ids),
        survey_included=(True,) * len(frame.original_ids),
        biomarker_available=(True,) * len(frame.original_ids),
        registered_events=(None,) * len(frame.original_ids),
        observed_denominator=(None,) * len(frame.original_ids))
    rebound = coverage.bind_observations(endpoint, sample)
    for fold in range(5):
        config = rebound.configuration(fold, tmp_path / str(fold),
                                       **nested_cv._settings({'nested_cv': {}}, rebound))
        assert config.ssl_epochs == 30 and config.settings.frozen_epochs == 150
        assert set(config.inner.split.fold_ids) == {0, 1, 2}
        assert not set(config.inner.split.original_ids).intersection(
            set(rebound.outer.original_ids) - set(rebound.outer.training_ids(fold)))
    assert rebound.data.covariates(frame.columns).values == endpoint.data.covariates(frame.columns).values


def test_real_frame_refuses_missing_atlas_and_failed_primary_gate():
    values = real_fixture()
    config = read_mapping(ROOT / 'configs/design.yaml')
    atlas = values['atlas']
    values['atlas'] = replace(atlas, rows=atlas.rows[1:], missing_tract_ids=(atlas.rows[0].tract_id,),
                              coverage_complete=False)
    with pytest.raises(ContractError, match='incomplete atlas'):
        real_frame.build_inputs(values, config)
    values['atlas'] = replace(atlas, rows=tuple(replace(r, inhabited_elevation_m=1.) for r in atlas.rows))
    with pytest.raises(ContractError, match='no tract target'):
        real_frame.build_inputs(values, config)


def test_generated_tasks_use_all_registered_scenarios_and_no_lock(built):
    _, (endpoint, frame, _, _) = built
    tasks = real_frame.profile_tasks(endpoint, frame)['tasks']
    profiles, estimate = tasks[:4], tasks[4]
    assert len(profiles) == 4
    assert {p['parameters']['scenario']['name'] for p in profiles} == {
        r['name'] for r in campaign._registry()['scenarios']}
    hashes = set()
    for task in profiles:
        p = task['parameters']
        assert task['stage'] == 'simulation-smoke' and p['mode'] == 'profile'
        assert set(task['needs']) == {'real-frame-inputs'}
        campaign.validate_recipe(p['recipe'])
        hashes.add(coverage.digest(p['recipe']))
        assert p['recipe']['endpoint_hash'] == endpoint.content_hash
        assert p['recipe']['frame_hash'] == frame.content_hash
        assert len(p['draws']) == 1 and p['wall_seconds'] == 41400
    assert len(hashes) == 1
    assert estimate['parameters']['gpus'] == 0
    assert set(estimate['needs']) == {'real-frame-inputs', *(p['id'] for p in profiles)}
    assert all('recipe_lock' not in task and 'campaign' not in task for task in tasks)


@pytest.fixture
def estimate_request(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'fingerprint', lambda: deepcopy(STAMPS))
    endpoint, frame = smoke_inputs.build_inputs()
    tasks = real_frame.profile_tasks(endpoint, frame)['tasks']
    deps = {'real-frame-inputs': tmp_path / 'inputs'}
    write(deps['real-frame-inputs'] / 'endpoint.json', endpoint.to_dict())
    write(deps['real-frame-inputs'] / 'frame.json', frame.to_dict())
    for task in tasks[:4]:
        p = task['parameters']
        _, records = successful_records(1)
        records[0]['draw'] = p['draws'][0]
        if p['scenario']['effect'] != 'null':
            records[0]['truth'] = 0.
        root = deps[task['id']] = tmp_path / task['id']
        write(root / 'result.json', {'mode': 'profile', 'scenario': p['scenario'],
            'recipe_hash': coverage.digest(p['recipe']), 'draws': p['draws'], 'records': records})
        write(root / 'timing.json', {'production_equivalent': True, 'complete': True, 'all_successful': True,
            'recipe_hash': coverage.digest(p['recipe']), 'device': 'cpu', 'gpu_seconds': 0,
            'complete_repetition_seconds': [10.], 'wall_seconds': 13., 'setup_seconds': 2.,
            'measurement_scope': 'complete_stage_return', **STAMPS})
    req = request(tmp_path / 'estimate', tasks[4], deps)
    # There is deliberately no approvals or allocation key at all.
    config = json.loads(Path(req.config_path).read_text())
    del config['approvals']
    write(Path(req.config_path), config)
    return replace(req, config_hash=file_hash(req.config_path))


def update_task(req, edit):
    task = json.loads(Path(req.task_path).read_text())
    edit(task['parameters'])
    write(Path(req.task_path), task)
    return replace(req, task_hash=file_hash(req.task_path))


def test_estimate_never_expands_locks_admits_or_requires_allocation(estimate_request, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('estimate invoked an allocation-gated campaign operation')
    for name in ('build_lock', 'expand_campaign', '_admit_locked_leaves'):
        monkeypatch.setattr(campaign, name, forbidden)
    # Keep the partial-batch arithmetic fixture at its original four-hour proposal.
    estimate_request = update_task(estimate_request, lambda p: p.update(wall_seconds=14400))
    result = campaign_estimate.run_stage(estimate_request)
    assert result.status == 'pass', result.message
    result.verify(estimate_request)
    root = Path(estimate_request.output_dir)
    assert {p.name for p in root.iterdir()} == {'budget_estimate.json', 'artifact_manifest.json'}
    budget = coverage.read_json(root / 'budget_estimate.json')
    assert budget['estimate_only'] and not budget['admitted']
    assert budget['total_repetitions'] == 4000 and budget['final_coverage_gpu_hours'] == 0
    # 2*(setup=2 + 654*(repetition=10 + publication=1)) <= 14400.
    assert budget['common_feasible_repetitions_per_leaf'] == 654
    assert budget['projected_leaf_count'] == 8
    assert budget['final_coverage_cpu_wall_hours'] == pytest.approx(88032 / 3600)
    assert budget['final_coverage_cpu_core_hours'] == pytest.approx(8 * 88032 / 3600)
    assert budget['feasible_under_requested_wall_and_leaf_cap']


def test_infeasible_estimate_reports_needed_wall_without_admission(estimate_request):
    req = update_task(estimate_request, lambda p: p.update(wall_seconds=10))
    budget = campaign_estimate.estimate_budget(req)
    assert budget['common_feasible_repetitions_per_leaf'] == 0
    assert budget['leaf_count_at_requested_wall'] is None
    assert not budget['feasible_under_requested_wall_and_leaf_cap']
    assert budget['projected_repetitions_per_leaf'] == 100
    assert budget['projected_leaf_count'] == 40
    assert budget['projected_required_leaf_wall_seconds'] == 2204
    assert not budget['admitted']


@pytest.mark.parametrize('change', ['scenario', 'code', 'environment', 'recipe', 'incomplete', 'smoke'])
def test_estimate_rejects_incomplete_or_mismatched_profiles(estimate_request, change):
    req = estimate_request
    config = coverage.read_json(req.config_path)
    root = Path(config['dependencies']['profile-null-effect'])
    path = root / ('result.json' if change == 'scenario' else 'timing.json')
    value = coverage.read_json(path)
    if change == 'scenario':
        value['scenario']['beta'] = 2.
    elif change == 'code':
        value['scientific_fingerprint']['sha256'] = 'd' * 64
    elif change == 'environment':
        value['environment_hash'] = 'd' * 64
    elif change == 'recipe':
        value['recipe_hash'] = 'd' * 64
    elif change == 'incomplete':
        value['complete'] = False
    else:
        value['production_equivalent'] = False
    write(path, value)
    req = replace(req, dependency_hashes=tuple(map(file_hash, req.dependency_paths)))
    result = campaign_estimate.run_stage(req)
    assert result.status == 'blocked' and not result.artifacts
    assert not list(Path(req.output_dir).iterdir())


def test_estimate_requires_all_four_profiles_and_1000_repetitions(estimate_request):
    req = update_task(estimate_request, lambda p: p['profiles'].pop('nonlinear'))
    with pytest.raises(ContractError, match='all registered'):
        campaign_estimate.estimate_budget(req)
    # The original request is separately reconstructed by pytest for each test.


def test_estimate_cannot_lower_minimum(estimate_request):
    req = update_task(estimate_request, lambda p: p.update(final_repetitions=999))
    with pytest.raises(ContractError, match='final repetitions'):
        campaign_estimate.estimate_budget(req)


def test_profile_resources_are_explicit_and_match_estimate():
    tasks = real_frame.profile_tasks(*smoke_inputs.build_inputs())['tasks']
    estimate = tasks[4]['parameters']
    expected = {k: estimate[k] for k in ('cpus_per_task', 'gpus', 'wall_seconds')}
    assert expected == {'cpus_per_task': 8, 'gpus': 0, 'wall_seconds': 41400}
    for task in tasks[:5]:
        assert task.get('resources') == expected
        assert task['parameters']['wall_seconds'] == task['resources']['wall_seconds']


@pytest.mark.parametrize('fixture_name', ['smoke', 'real_fixture'])
def test_existing_profile_task_bytes_match_launch_base(fixture_name, built):
    # SHA256 of canonical bytes at dev/launch base a778a7c; excludes only the
    # appended probe. This pins every CPU/estimate field, including draw seeds.
    pair = smoke_inputs.build_inputs() if fixture_name == 'smoke' else built[1][:2]
    expected = {
        'smoke': '1fb64eaf6987e8fb62c6df3c95b53f0482e2fe7cfcc07e5db214a20fd35e7377',
        'real_fixture': '3c6f161d31f47ffe9a04e13c9fd264dcc8d1acdc84f200cb71608738c2ac6490',
    }
    assert coverage.digest(real_frame.profile_tasks(*pair)['tasks'][:5]) == expected[fixture_name]


def test_gpu_probe_appends_identical_work_with_explicit_cuda_and_bounded_resources(built):
    tasks = real_frame.profile_tasks(*built[1][:2])['tasks']
    assert len(tasks) == 6
    cpu, estimate, gpu = tasks[0], tasks[4], tasks[5]
    assert gpu['id'] == 'profile-null-effect-gpu'
    assert gpu['stage'] == 'simulation-smoke'
    assert gpu['needs'] == cpu['needs'] and gpu['outputs'] == cpu['outputs']
    assert gpu['resources'] == {'cpus_per_task': 8, 'gpus': 1, 'wall_seconds': 13500}
    assert gpu['resources']['gpus'] * (gpu['resources']['wall_seconds'] + 900) <= 4 * 3600
    expected = {**cpu['parameters'], 'wall_seconds': 13500, 'device': 'cuda'}
    assert gpu['parameters'] == expected
    assert gpu['id'] not in estimate['needs']
    assert gpu['id'] not in str(estimate['parameters'])
    # Editing execution metadata on the appended task cannot alter the CPU task.
    gpu['parameters']['draws'][0]['seed'] += 1
    assert gpu['parameters']['draws'] != cpu['parameters']['draws']


@pytest.fixture
def gpu_profile_request(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'fingerprint', lambda: deepcopy(STAMPS))
    endpoint, frame = smoke_inputs.build_inputs()
    task = real_frame.profile_tasks(endpoint, frame)['tasks'][-1]
    inputs = tmp_path / 'inputs'
    write(inputs / 'endpoint.json', endpoint.to_dict())
    write(inputs / 'frame.json', frame.to_dict())
    return request(tmp_path / 'probe', task, {'real-frame-inputs': inputs})


def test_gpu_probe_refuses_unavailable_cuda_before_fitting(gpu_profile_request, monkeypatch):
    import torch
    monkeypatch.setenv('OXYFORMER_DEVICE', 'cuda')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(coverage, 'execute_draw', lambda *args: pytest.fail('started draw without CUDA'))
    result = coverage.run_stage(gpu_profile_request)
    assert result.status == 'blocked' and 'unavailable' in result.message
    assert not result.artifacts


def test_gpu_probe_refuses_cpu_runtime_before_fitting(gpu_profile_request, monkeypatch):
    from oxyformer.training import fit
    import torch
    monkeypatch.setattr(fit, 'resolve_device', lambda device='auto':
                        torch.device('cuda:0' if device == 'cuda' else 'cpu'))
    monkeypatch.setattr(coverage, 'execute_draw', lambda *args: pytest.fail('started CPU draw'))
    result = coverage.run_stage(gpu_profile_request)
    assert result.status == 'blocked' and 'required device' in result.message
    assert not result.artifacts


@pytest.mark.parametrize('outcome', ['success', 'incomplete', 'numerical_failure', 'late_fit', 'late_publication'])
def test_gpu_profile_timing_and_budget_are_honest(gpu_profile_request, monkeypatch, outcome):
    from oxyformer.training import fit
    import torch
    clock = [0.]
    monkeypatch.setattr(coverage.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(fit, 'resolve_device', lambda device='auto': torch.device('cuda:0'))
    environment = {'device': 'cuda:0', 'cuda_runtime': 'test-runtime', 'gpu': 'synthetic-device'}
    monkeypatch.setattr(fit, 'fit_environment', lambda device='auto': tuple(environment.items()))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    def execute(draw, frame, scenario, template, recipe, root, deadline):
        assert deadline == 13500
        clock[0] += 13501 if outcome == 'late_fit' else 10
        _, records = successful_records(1)
        record = {**records[0], 'draw': draw, 'wall_seconds': clock[0]}
        if outcome in ('incomplete', 'numerical_failure'):
            record.update(status=outcome, estimates={}, reason='budget exhausted or numerical failure')
        return record
    original_publish = coverage.publish
    def publish(*args, **kwargs):
        result = original_publish(*args, **kwargs)
        if outcome == 'late_publication':
            clock[0] += 13501
        return result
    monkeypatch.setattr(coverage, 'execute_draw', execute)
    monkeypatch.setattr(coverage, 'publish', publish)
    result = coverage.run_stage(gpu_profile_request)
    assert result.status == ('pass' if outcome == 'success' else 'fail'), result.message
    result.verify(gpu_profile_request)
    timing = coverage.read_json(Path(gpu_profile_request.output_dir) / 'timing.json')
    assert timing['device'] == 'cuda:0' and timing['environment'] == environment
    assert timing['gpu_seconds'] == timing['wall_seconds'] > 0
    assert timing['complete'] == (outcome == 'success')
    assert timing['budget_exceeded'] == outcome.startswith('late_')
    assert len(timing['complete_repetition_seconds']) == (0 if outcome in ('incomplete', 'numerical_failure') else 1)


@pytest.mark.parametrize('unaccounted', [False, True])
def test_real_frame_dispatch_excludes_accounted_dem_and_preserves_reservation(tmp_path, unaccounted):
    from oxyformer.design.splits import reserve_design
    from test_tract_tasks import collected_design_request
    values = real_fixture()
    reservation = reserve_design(values['geography'].rows, values['entity_graph'])
    sealed = set(reservation.design_ids)
    missing = (reservation.design_ids[0], next(r.original_id for r in values['geography'].rows
                                               if r.original_id not in sealed))
    unknown = next(r.original_id for r in values['geography'].rows if r.original_id not in {*missing, *sealed})
    req = collected_design_request(tmp_path, values, missing=missing, accounted=missing,
        absent=(unknown,) if unaccounted else (), stage='real-frame-inputs')
    prepared, config = real_frame.prepare_inputs(req)
    if unaccounted:
        result = real_frame.run_stage(req)
        assert result.status == 'blocked' and 'incomplete atlas coverage' in result.message
        assert not result.artifacts
        return
    endpoint, frame, support, audit = real_frame.build_inputs(prepared, config)
    assert not set(missing).intersection(frame.original_ids)
    assert set(endpoint.outer.design_ids) == sealed
    assert set(support.design_ids) == sealed
    assert audit['coverage']['complete'] and audit['coverage']['accounted_missing_dem_tracts'] == 2
    assert dict(audit['exclusions'])[missing[1]] == 'atlas_missing_dem_coverage'
    assert {r['original_id'] for r in audit['allocation_exclusions']} == set(missing)
    assert audit['input_rows'] == audit['evaluation_rows'] + len(audit['design_ids']) + len(audit['exclusions'])
    exposures = dict(zip(endpoint.data.manifest.original_ids, endpoint.data.column('a')))
    assert all(exposures[oid] is None for oid in missing)
    retained = [r.exposure_mmhg for r in prepared['atlas'].rows if r.tract_id in sealed]
    assert endpoint.treatment_design.center == pytest.approx(sum(retained) / len(retained))
    for fold in range(5):
        endpoint.validate(fold)
        assert not set(missing).intersection(endpoint.outer.training_ids(fold))
    from oxyformer.validation.generators import ObservedRecords
    n = len(frame.original_ids)
    sample = ObservedRecords(frame=frame, a=(5.,) * n, y=(50.,) * n, measured_columns=(),
        measured_x=((),) * n, flag_available=(True,) * n, survey_included=(True,) * n,
        biomarker_available=(True,) * n, registered_events=(None,) * n, observed_denominator=(None,) * n)
    rebound = coverage.bind_observations(endpoint, sample)
    for fold in range(5):
        rebound.configuration(fold, tmp_path / str(fold), **nested_cv._settings({'nested_cv': {}}, rebound))
    result = real_frame.run_stage(req)
    assert result.status == 'pass', result.message
    result.verify(req)
