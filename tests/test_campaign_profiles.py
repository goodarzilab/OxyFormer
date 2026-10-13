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
    assert frame.original_ids == tuple(i for i, role in zip(endpoint.outer.frame_original_ids,
        endpoint.outer.roles) if role != 'inert')
    assert tuple(i for i, w in zip(frame.original_ids, frame.weights) if w) == endpoint.outer.original_ids
    assert audit['evaluation_rows'] == len(endpoint.outer.original_ids)
    assert audit['simulation_rows'] == len(frame.original_ids)
    assert audit['training_only_rows'] > 0
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
    values['atlas'] = atlas
    values['geography'] = replace(values['geography'], rows=tuple(
        replace(r, label_available=False) for r in values['geography'].rows))
    with pytest.raises(ContractError, match='no tract target'):
        real_frame.build_inputs(values, config)


def test_generated_tasks_use_all_registered_scenarios_and_no_lock(built):
    _, (endpoint, frame, _, _) = built
    tasks = real_frame.profile_tasks(endpoint, frame, campaign_mode='registered')['tasks']
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
    tasks = real_frame.profile_tasks(endpoint, frame, campaign_mode='registered')['tasks']
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
    tasks = real_frame.profile_tasks(*smoke_inputs.build_inputs(), campaign_mode='registered')['tasks']
    estimate = tasks[4]['parameters']
    expected = {k: estimate[k] for k in ('cpus_per_task', 'gpus', 'wall_seconds')}
    assert expected == {'cpus_per_task': 8, 'gpus': 0, 'wall_seconds': 41400}
    for task in tasks[:5]:
        assert task.get('resources') == expected
        assert task['parameters']['wall_seconds'] == task['resources']['wall_seconds']


@pytest.mark.parametrize('fixture_name', ['smoke', 'real_fixture'])
def test_profile_task_bytes_pin_smoke_and_amended_real_frame(fixture_name, built):
    # Smoke retains the dev/launch-base a778a7c bytes. The real fixture binds
    # A2 support/target/split identities. Both pins cover every CPU/estimate
    # field, including draw seeds, and exclude only the appended GPU probe.
    pair = smoke_inputs.build_inputs() if fixture_name == 'smoke' else built[1][:2]
    expected = {
        'smoke': '1fb64eaf6987e8fb62c6df3c95b53f0482e2fe7cfcc07e5db214a20fd35e7377',
        'real_fixture': 'ec44a0c499cfd354bc3a617661514bcd3d96779d284a73f9b7ea158fefbc15d3',
    }
    assert coverage.digest(real_frame.profile_tasks(*pair, campaign_mode='registered')['tasks'][:5]) == expected[fixture_name]


def test_gpu_probe_appends_identical_work_with_explicit_cuda_and_bounded_resources(built):
    tasks = real_frame.profile_tasks(*built[1][:2], campaign_mode='registered')['tasks']
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
    task = real_frame.profile_tasks(endpoint, frame, campaign_mode='registered')['tasks'][-1]
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
def test_real_frame_dispatch_excludes_accounted_dem_and_preserves_reservation(tmp_path, monkeypatch, unaccounted):
    from oxyformer.design.splits import reserve_design
    from test_tract_tasks import collected_design_request, install_coverage_approval
    install_coverage_approval(tmp_path, monkeypatch)
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


def test_real_frame_reports_fixed_training_rows_and_binds_their_atlas_exposure(built):
    values, (original, _, _, _) = built
    oid = original.outer.original_ids[0]
    covariates = values['covariates']
    changed = replace(covariates, values=tuple((0.,) if i == oid else x
        for i, x in zip(covariates.original_ids, covariates.values)))
    endpoint, frame, support, audit = real_frame.build_inputs(
        {**values, 'covariates': changed}, read_mapping(ROOT / 'configs/design.yaml'))
    roles = dict(zip(endpoint.outer.frame_original_ids, endpoint.outer.roles))
    fixed = coverage.fixed_exposures(endpoint, frame)
    assert roles[oid] == 'training' and oid in fixed
    assert all(roles[i] == 'training' for i in fixed)
    assert audit['fixed_exposure_rows'] == len(fixed)
    assert audit['sampled_exposure_rows'] + len(fixed) == audit['simulation_rows']
    assert dict(zip(frame.original_ids, frame.weights))[oid] == 0.
    from oxyformer.validation.generators import ObservedRecords
    doses = dict(zip(endpoint.data.manifest.original_ids, endpoint.data.column(endpoint.data.manifest.exposure_field)))
    n = len(frame.original_ids)
    observed = ObservedRecords(frame=frame, a=tuple(doses[i] for i in frame.original_ids),
        y=(50.,) * n, measured_columns=(), measured_x=((),) * n,
        flag_available=(True,) * n, survey_included=(True,) * n,
        biomarker_available=(True,) * n, registered_events=(None,) * n, observed_denominator=(None,) * n)
    rebound = coverage.bind_observations(endpoint, observed)
    assert rebound.data.column(endpoint.data.manifest.exposure_field) == endpoint.data.column(endpoint.data.manifest.exposure_field)
    wrong = replace(observed, a=tuple(a + 1 if i == oid else a for i, a in zip(frame.original_ids, observed.a)))
    with pytest.raises(ContractError, match='fixed exposure differs from atlas value'):
        coverage.bind_observations(endpoint, wrong)
    locations = real_frame.profile_tasks(endpoint, frame, campaign_mode='registered')['tasks'][0]['parameters']['recipe']['inference']['county_locations']
    geo = {r.original_id: r for r in endpoint.geography.rows}
    for county, location in locations.items():
        rows = [geo[i] for i in endpoint.outer.original_ids if geo[i].county == county]
        assert location == [sum(r.latitude for r in rows) / len(rows), sum(r.longitude for r in rows) / len(rows)]


def derived_support_fixture(built):
    """Policy-only fixture for registry comparisons, not for fitting."""
    _, (endpoint, frame, _, _) = built
    targets = set(endpoint.outer.original_ids)
    keys = dict(zip(frame.original_ids, frame.support_keys))
    training = [i for i in frame.original_ids if i not in targets]
    replacements = {keys[endpoint.outer.original_ids[0]]: ((0., 3.), (7., 10.)),
                    keys[training[0]]: ((0., 1.),), keys[training[1]]: ()}
    policy = replace(endpoint.policy, components_by_key=tuple(
        (key, replacements.get(key, components)) for key, components in endpoint.policy.components_by_key))
    return replace(endpoint, policy=policy), frame


def test_real_frame_declarations_derive_sampled_facts_and_keep_synthetic_explicit(built):
    from oxyformer.validation.scm import sampled_support_facts
    endpoint, frame = derived_support_fixture(built)
    fixed = coverage.fixed_exposures(endpoint, frame)
    facts = sampled_support_facts(endpoint.policy, frame, fixed_exposures=fixed)
    assert len(facts['gap_ids']) == len(facts['no_positive_width_shift_ids']) == len(fixed) == 1
    assert not set(fixed).intersection(facts['sampled_ids'])
    original_policy = endpoint.policy.to_json()
    profiles = real_frame.profile_tasks(endpoint, frame, campaign_mode='registered')['tasks'][:4]
    for task, scenario in zip(profiles, campaign.registered_scenarios(endpoint, frame)):
        assert task['parameters']['scenario'] == scenario
        assert scenario['support_gaps'] and scenario['heterogeneous_eligibility']
        SCMConfig(**scenario).validate_policy(endpoint.policy, frame, fixed_exposures=fixed)
    assert endpoint.policy.to_json() == original_policy
    gap_key = dict(zip(frame.original_ids, frame.support_keys))[endpoint.outer.original_ids[0]]
    assert not endpoint.policy.shift_mask([2.], [gap_key])[0]
    manifest = endpoint.data.manifest
    synthetic = replace(endpoint, data=replace(endpoint.data, manifest=replace(manifest,
        spec=replace(manifest.spec, endpoint='synthetic-suite-a'))))
    for scenario in campaign.registered_scenarios(synthetic, frame):
        assert not scenario['support_gaps'] and not scenario['heterogeneous_eligibility']
        with pytest.raises(ContractError, match='support_gaps switch'):
            SCMConfig(**scenario).validate_policy(synthetic.policy, frame, fixed_exposures=fixed)


def test_estimate_and_lock_compare_derived_real_frame_scenarios(built, tmp_path, monkeypatch):
    from test_campaign import setup_lock
    endpoint, frame = derived_support_fixture(built)
    tasks = real_frame.profile_tasks(endpoint, frame, campaign_mode='registered')['tasks']
    deps = {'real-frame-inputs': tmp_path / 'real-inputs'}
    write(deps['real-frame-inputs'] / 'endpoint.json', endpoint.to_dict())
    write(deps['real-frame-inputs'] / 'frame.json', frame.to_dict())
    monkeypatch.setattr(campaign, 'fingerprint', lambda: deepcopy(STAMPS))
    for task in tasks[:4]:
        p = task['parameters']
        _, records = successful_records(1)
        records[0]['draw'] = p['draws'][0]
        root = deps[task['id']] = tmp_path / task['id']
        write(root / 'result.json', {'mode': 'profile', 'scenario': p['scenario'],
            'recipe_hash': coverage.digest(p['recipe']), 'draws': p['draws'], 'records': records})
        write(root / 'timing.json', {'production_equivalent': True, 'complete': True, 'all_successful': True,
            'recipe_hash': coverage.digest(p['recipe']), 'device': 'cpu', 'gpu_seconds': 0,
            'complete_repetition_seconds': [10.], 'wall_seconds': 13., 'setup_seconds': 2.,
            'measurement_scope': 'complete_stage_return', **STAMPS})
    req = request(tmp_path / 'estimate-derived', tasks[4], deps)
    assert campaign_estimate.estimate_budget(req)['total_repetitions'] == 4000
    # Exercise registry admission without fitting or timing this policy-only fixture.
    monkeypatch.setattr(campaign, '_admit_locked_leaves', lambda *args: [])
    _, lock_task = setup_lock(tmp_path / 'lock-template')
    p = lock_task['parameters']
    p.update({key: tasks[4]['parameters'][key] for key in ('endpoint_input', 'frame_input', 'recipe')})
    scenario = tasks[0]['parameters']['scenario']
    p.update(scenario_family=[scenario], profiles={'null_effect': tasks[4]['parameters']['profiles']['null_effect']})
    lock_req = request(tmp_path / 'lock-derived', lock_task, deps)
    config = coverage.read_json(lock_req.config_path)
    lock, _, _ = campaign.build_lock(lock_req, config, lock_task)
    assert lock['scenario_family'] == [scenario]
    explicit = SCMConfig(name='null_effect', effect='null', active_mechanisms=('null',)).to_dict()['payload']
    lock_task['parameters']['scenario_family'] = [explicit]
    with pytest.raises(ContractError, match='unregistered final scenario'):
        campaign.build_lock(lock_req, config, lock_task)
    stale = coverage.read_json(deps['profile-null-effect'] / 'result.json')
    stale['scenario'] = explicit
    write(deps['profile-null-effect'] / 'result.json', stale)
    req = replace(req, dependency_hashes=tuple(map(file_hash, req.dependency_paths)))
    with pytest.raises(ContractError, match='profile scenario mismatch'):
        campaign_estimate.estimate_budget(req)


def screening_approvals():
    return read_mapping(ROOT / 'configs/approvals.yaml')


@pytest.mark.parametrize('change', ['absent', 'null', 'cell', 'count', 'budget', 'gpu', 'scenarios', 'deferral', 'boolean'])
def test_screening_requires_complete_owner_decision(tmp_path, monkeypatch, change):
    approvals = screening_approvals()
    decision = approvals['owner_decisions']['screening_campaign']
    if change == 'absent':
        del approvals['owner_decisions']['screening_campaign']
    elif change == 'null':
        approvals['owner_decisions']['screening_campaign'] = None
    elif change == 'cell':
        decision['recipe']['reference_cap'] = 64
    elif change == 'count':
        decision['repetitions_per_scenario'] = 1000
    elif change == 'budget':
        decision['budget_h100_hours'] = 3601
    elif change == 'gpu':
        decision['gpu_concurrency_max'] = 5
    elif change == 'scenarios':
        decision['scenarios'].pop()
    elif change == 'deferral':
        decision['final_cells'] = 'approved'
    else:
        decision['recipe']['seed_count'] = True
    path = write(tmp_path / 'approvals.json', approvals)
    monkeypatch.setattr(campaign, 'OWNER_APPROVALS', path)
    with pytest.raises(ContractError, match='screening'):
        real_frame.profile_tasks(*smoke_inputs.build_inputs(), campaign_mode='screening')
    # Registered callers remain opt-in and do not consume screening overrides.
    registered = real_frame.profile_tasks(*smoke_inputs.build_inputs(), campaign_mode='registered')
    assert 'study' not in registered['tasks'][0]['parameters']['recipe']


def test_screening_profiles_bind_exact_cell_without_changing_registered_values():
    endpoint, frame = smoke_inputs.build_inputs()
    manifest = real_frame.profile_tasks(endpoint, frame, campaign_mode='screening')
    assert manifest['campaign_label'] == campaign.SCREENING_LABEL
    assert len(manifest['tasks']) == 5
    for task in manifest['tasks']:
        recipe = task['parameters']['recipe']
        assert recipe['study'] == dict(reference_cap=32, frozen_epochs=30, grid_points=2, seed_count=1)
        assert recipe['nested_cv'] == dict(ssl_epochs=30, frozen_epochs=150, batch_size=256, precision='fp32')
        assert task['resources'] == dict(cpus_per_task=8, gpus=1, wall_seconds=14400)
        campaign.validate_recipe(recipe)
    assert campaign._registry()['minimum_final_repetitions'] == coverage.MIN_REPETITIONS == 1000
    assert 'final_repetitions' not in manifest['tasks'][-1]['parameters']


@pytest.fixture
def screening_request(tmp_path, monkeypatch):
    import torch
    from oxyformer.training import fit
    monkeypatch.setattr(campaign, 'fingerprint', lambda: deepcopy(STAMPS))
    monkeypatch.setattr(campaign, 'resolve_device', lambda: torch.device('cuda:0'))
    monkeypatch.setattr(fit, 'resolve_device', lambda device='auto': torch.device('cuda:0'))
    endpoint, frame = smoke_inputs.build_inputs()
    tasks = real_frame.profile_tasks(endpoint, frame, campaign_mode='screening')['tasks']
    deps = {'real-frame-inputs': tmp_path / 'inputs'}
    write(deps['real-frame-inputs'] / 'endpoint.json', endpoint.to_dict())
    write(deps['real-frame-inputs'] / 'frame.json', frame.to_dict())
    for task in tasks[:4]:
        p = task['parameters']
        metadata = campaign.screening_metadata(p['recipe'])
        _, records = successful_records(1)
        records[0]['draw'] = p['draws'][0]
        root = deps[task['id']] = tmp_path / task['id']
        write(root / 'result.json', {**metadata, 'mode': 'profile', 'scenario': p['scenario'],
            'recipe_hash': coverage.digest(p['recipe']), 'draws': p['draws'], 'records': records})
        write(root / 'timing.json', {**metadata, 'production_equivalent': True, 'complete': True, 'all_successful': True,
            'recipe_hash': coverage.digest(p['recipe']), 'device': 'cuda:0', 'gpu_seconds': 1005.,
            'complete_repetition_seconds': [1000.], 'wall_seconds': 1005., 'setup_seconds': 4.,
            'measurement_scope': 'complete_stage_return', **STAMPS})
    return request(tmp_path / 'screen-estimate', tasks[-1], deps, approvals=screening_approvals())


def test_screening_estimate_reports_100_draws_and_exact_cell(screening_request):
    result = campaign_estimate.run_stage(screening_request)
    assert result.status == 'pass', result.message
    result.verify(screening_request)
    estimate = coverage.read_json(Path(screening_request.output_dir) / 'budget_estimate.json')
    assert estimate['total_repetitions'] == 100
    assert estimate['repetitions_per_scenario'] == 25
    assert estimate['repetitions_per_leaf'] == 7
    assert estimate['projected_leaf_count'] == 16
    assert estimate['screening_gpu_hours'] <= 3600
    assert estimate['feasible_under_requested_wall_and_leaf_cap']
    assert estimate['study'] == campaign.SCREENING_CELL
    assert not estimate['certifies_production_coverage'] and not estimate['admitted']
    for path in Path(screening_request.output_dir).glob('*.json'):
        assert coverage.read_json(path)['campaign_label'] == campaign.SCREENING_LABEL


@pytest.mark.parametrize('hours,expected', [(35, 25), (36, 24), (40, 22), (899, 1), (900, 0), (901, 0)])
def test_screening_uniform_reduction_never_raises_ceiling(hours, expected):
    decision = campaign.screening_decision()
    profiles = {name: dict(setup_seconds=0., seconds_per_repetition=hours * 3600,
        publication_verification_seconds=0., profile_repetitions=1) for name in decision['scenarios']}
    result = campaign_estimate.screening_plan(profiles, decision, 14400, 1.)
    assert result['repetitions_per_scenario'] == expected
    assert result['total_repetitions'] == 4 * expected
    assert result['screening_gpu_hours'] <= 3600
    assert result['limits']['run_gpu_hours'] == 3600
    assert result['repetitions_reduced'] == (expected < 25)
    assert set(result['scenario_gpu_hours'].values()) == {hours * expected}
    assert not result['feasible_under_requested_wall_and_leaf_cap']
    if expected < 25:
        assert 4 * hours * (expected + 1) + 600 / 3600 > 3600


@pytest.mark.parametrize('drift', ['approval', 'approval_null', 'approval_scalar', 'cell', 'profile_cell',
                                   'screening_repetitions', 'repetitions_per_leaf'])
def test_screening_estimate_refuses_unapproved_or_mismatched_cell(screening_request, drift):
    req = screening_request
    if drift in ('approval', 'approval_null', 'approval_scalar'):
        value = coverage.read_json(req.config_path)
        value['approvals'] = {'approval': {}, 'approval_null': None, 'approval_scalar': []}[drift]
        write(Path(req.config_path), value)
        req = replace(req, config_hash=file_hash(req.config_path))
    elif drift in ('screening_repetitions', 'repetitions_per_leaf'):
        req = update_task(req, lambda p: p.update({drift: 1000}))
    elif drift == 'cell':
        req = update_task(req, lambda p: p['recipe']['study'].update(grid_points=4))
    else:
        config = coverage.read_json(req.config_path)
        path = Path(config['dependencies']['profile-null-effect']) / 'timing.json'
        value = coverage.read_json(path)
        value['study']['seed_count'] = 3
        write(path, value)
        req = replace(req, dependency_hashes=tuple(map(file_hash, req.dependency_paths)))
    result = campaign_estimate.run_stage(req)
    assert result.status == 'blocked' and not result.artifacts


def screening_lock_request(screening_request, tmp_path):
    task = coverage.read_json(screening_request.task_path)
    p = task['parameters']
    # Use named inputs, not dependency ordering.
    config = coverage.read_json(screening_request.config_path)
    root = Path(config['dependencies']['real-frame-inputs'])
    endpoint = nested_cv.PreparedEndpoint.from_json((root / 'endpoint.json').read_text())
    from oxyformer.validation.scm import CovariateFrame
    frame = CovariateFrame.from_json((root / 'frame.json').read_text())
    p.update(campaign_id='approved-screen', scenario_family=campaign.registered_scenarios(endpoint, frame),
        evaluation_namespace='screening-only-test', retry_rules=deepcopy(coverage.RETRY_RULES),
        prior_locks=[], cpu_wall_hours_budget=200)
    task.update(id='campaign-lock', stage='campaign-lock')
    return request(tmp_path / 'screen-lock', task,
        {k: Path(v) for k, v in config['dependencies'].items()}, approvals=config['approvals'])


def test_screening_lock_only_expands_bounded_screening_and_four_lanes(screening_request, tmp_path):
    req = screening_lock_request(screening_request, tmp_path)
    result = campaign.run_stage(req)
    assert result.status == 'pass', result.message
    result.verify(req)
    root = Path(req.output_dir)
    lock = coverage.read_json(root / 'recipe_lock.json')
    plans = coverage.read_json(root / 'expanded_units.json')['plans']
    assert list(lock['instances']) == ['screening'] and len(plans) == 1
    assert sum(len(b['draws']) for b in lock['batches']) == 100
    assert {b['mode'] for b in lock['batches']} == {'screening'}
    assert lock['study'] == campaign.SCREENING_CELL
    assert lock['budget']['run_gpu_hours_ceiling'] == 3600
    assert lock['budget']['concurrency_limits']['gpu'] == 4
    plan = plans[0]
    gpu = plan['units'][:-1]
    assert all('export OXYFORMER_DEVICE=cuda' in u['command'] for u in plan['units'])
    assert all('--cpus-per-task=8' in u['sbatch'] for u in plan['units'])
    collector = plan['units'][-1]
    assert collector['pool'] == 'gpu' and '--gpus=1' in collector['sbatch']
    assert lock['budget']['gpu_hours'] == pytest.approx(sum(u['gpu_hours'] for u in plan['units']))
    assert lock['budget']['collector_gpu_hours'] == pytest.approx(1/6)
    assert len(gpu) == 16
    assert all(u['gpu_hours'] <= 4 and '--gpus=1' in u['sbatch'] and u['pool'] == 'gpu' for u in gpu)
    ids = {u['id'] for u in gpu}
    for index, unit in enumerate(gpu):
        local_needs = set(unit['needs']) & ids
        assert local_needs == ({gpu[index - 4]['id']} if index >= 4 else set())
    # Explore every possible completed prefix in the four independent chains.
    from itertools import product
    for finished in product(range(5), repeat=4):
        done = {gpu[lane + step * 4]['id'] for lane, length in enumerate(finished) for step in range(length)}
        ready = [u for u in gpu if u['id'] not in done and (set(u['needs']) & ids) <= done]
        assert len(ready) <= 4
        collector_ready = (set(collector['needs']) & ids) <= done
        assert len(ready) + int(collector_ready) <= 4
    for path in root.glob('*.json'):
        assert coverage.read_json(path)['campaign_label'] == campaign.SCREENING_LABEL
    # Removing a scheduling edge is detectable by the independent plan rederivation.
    from oxyformer.execution.campaign import validate_plan
    broken = deepcopy(plan)
    broken['units'][4]['needs'].remove(gpu[0]['id'])
    with pytest.raises(ContractError, match='dependency drift'):
        validate_plan(broken, screening_approvals())


def test_screening_lock_refuses_overlong_repetitions(screening_request, tmp_path):
    req = screening_lock_request(screening_request, tmp_path)
    config = coverage.read_json(req.config_path)
    for name, directory in config['dependencies'].items():
        if name.startswith('profile-'):
            path = Path(directory) / 'timing.json'
            value = coverage.read_json(path)
            value.update(complete_repetition_seconds=[40 * 3600], wall_seconds=40 * 3600 + 5,
                         gpu_seconds=40 * 3600 + 5)
            write(path, value)
    req = replace(req, dependency_hashes=tuple(map(file_hash, req.dependency_paths)))
    assert campaign.run_stage(req).status == 'blocked'
    assert not list(Path(req.output_dir).glob('*.json'))


def test_screening_draw_passes_exact_cell_to_production_fitting(tmp_path, monkeypatch):
    from oxyformer.training.fit import RecipeStudy
    endpoint, frame = smoke_inputs.build_inputs()
    task = real_frame.profile_tasks(endpoint, frame, campaign_mode='screening')['tasks'][0]
    p = task['parameters']
    observed = []
    def estimate(bound, recipe, root, deadline, *, study=None):
        observed.append(study)
        assert type(study) is RecipeStudy
        assert (study.reference_cap, study.frozen_epochs, study.grid_points, study.seed_count) == (32, 30, 2, 1)
        assert study.seeds == (1103,)
        for fold in range(5):
            config = bound.configuration(fold, root / str(fold), study=study,
                **nested_cv._settings(recipe, bound))
            assert config.frozen_epochs == 30
            assert len(config.grid) == 2
            assert config.study.reference_cap == 32
            assert config.ssl_epochs == 30
        return None
    monkeypatch.setattr(coverage, 'estimate_repetition', estimate)
    record = coverage.execute_draw(p['draws'][0], frame, SCMConfig(**p['scenario']), endpoint,
                                  p['recipe'], tmp_path, float('inf'))
    assert observed, record
    assert record['status'] == 'incomplete'
    assert record['reason'] == 'budget exhausted', record


def test_real_frame_automatically_uses_declared_screening_mode(built):
    manifest = real_frame.profile_tasks(*built[1][:2])
    assert manifest['campaign_mode'] == 'screening'
    assert len(manifest['tasks']) == 5
    assert all(t['parameters']['recipe']['study'] == campaign.SCREENING_CELL for t in manifest['tasks'])
    # The established synthetic CUDA probe remains registered and opt-in.
    synthetic = real_frame.profile_tasks(*smoke_inputs.build_inputs())
    assert synthetic['tasks'][-1]['id'] == 'profile-null-effect-gpu'
    assert 'study' not in synthetic['tasks'][-1]['parameters']['recipe']
