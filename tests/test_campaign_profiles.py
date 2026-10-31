"""Synthetic-only adapter, estimate and production-task contracts; no fitting."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime
import json
from pathlib import Path

import pytest

from oxyformer.data.entity_graph import EntityGraph
from oxyformer.execution.runner import read_mapping
from oxyformer.design.splits import dependence_groups
from oxyformer.provenance import ContractError, file_hash
from oxyformer.training import nested_cv
from oxyformer.validation import campaign, campaign_estimate, campaign_probe, coverage, real_frame, smoke_inputs
from oxyformer.validation.scm import SCMConfig
from test_campaign import STAMPS, request, write, seal_budget_fixture, refresh_profile, profile_directory
from test_coverage import successful_records, run_coverage
from test_design import make_inputs

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def profile_publication_store(tmp_path, monkeypatch):
    monkeypatch.setenv('OXYFORMER_PUBLICATION_STORE', str(tmp_path / '.publications'))


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
    monkeypatch.setenv("OXYFORMER_PUBLICATION_STORE", str(tmp_path / ".publications"))
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
        seal_budget_fixture(root, task['id'], task)
    # Preserve this arithmetic fixture's factor, explicitly approved by its fixture.
    tasks[4]['parameters']['profile_safety_factor'] = 2.
    req = request(tmp_path / 'estimate', tasks[4], deps)
    # Planning requires the sizing decision, but no campaign allocation.
    config = json.loads(Path(req.config_path).read_text())
    del config['approvals']['owner_decisions']['campaign_allocations']
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
    assert budget['owner_safety_factor_decision'] == {**campaign.screening_safety_factor(), 'profile_safety_factor': 2.}
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
    req = refresh_profile(req)
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
def test_profile_task_bytes_pin_smoke_and_amended_real_frame(fixture_name, built):
    # Smoke retains the dev/launch-base a778a7c bytes. The real fixture binds
    # A2 support/target/split identities. Both pins cover every CPU/estimate
    # field, including draw seeds, and exclude only the appended GPU probe.
    pair = smoke_inputs.build_inputs() if fixture_name == 'smoke' else built[1][:2]
    expected = {
        'smoke': '1fb64eaf6987e8fb62c6df3c95b53f0482e2fe7cfcc07e5db214a20fd35e7377',
        'real_fixture': 'ec44a0c499cfd354bc3a617661514bcd3d96779d284a73f9b7ea158fefbc15d3',
    }
    tasks = real_frame.profile_tasks(*pair)['tasks'][:5]
    assert tasks[4]['parameters']['profile_safety_factor'] == campaign.screening_safety_factor()['profile_safety_factor']
    # Only the estimate's factor changes; retain the prior full-manifest pin.
    tasks[4]['parameters']['profile_safety_factor'] = 2.
    assert coverage.digest(tasks) == expected[fixture_name]


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
    result = run_coverage(gpu_profile_request)
    assert result.status == 'blocked' and 'unavailable' in result.message
    assert not result.artifacts


def test_gpu_probe_refuses_cpu_runtime_before_fitting(gpu_profile_request, monkeypatch):
    from oxyformer.training import fit
    import torch
    monkeypatch.setattr(fit, 'resolve_device', lambda device='auto':
                        torch.device('cuda:0' if device == 'cuda' else 'cpu'))
    monkeypatch.setattr(coverage, 'execute_draw', lambda *args: pytest.fail('started CPU draw'))
    result = run_coverage(gpu_profile_request)
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
        assert deadline == 13470  # 13500 seconds minus the publication reserve.
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
    result = run_coverage(gpu_profile_request)
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
    locations = real_frame.profile_tasks(endpoint, frame)['tasks'][0]['parameters']['recipe']['inference']['county_locations']
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
    profiles = real_frame.profile_tasks(endpoint, frame)['tasks'][:4]
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
    tasks = real_frame.profile_tasks(endpoint, frame)['tasks']
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
        seal_budget_fixture(root, task['id'], task)
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
    req = refresh_profile(req)
    with pytest.raises(ContractError, match='profile scenario mismatch'):
        campaign_estimate.estimate_budget(req)


def test_profile_chain_accounts_for_every_leaf_and_refuses_partial(tmp_path, monkeypatch):
    from test_coverage import chain_fixture, chain_request
    task, inputs = chain_fixture(tmp_path)
    task["parameters"]["mode"] = "profile"
    task["parameters"]["recipe"]["nested_cv"] = {}
    task["parameters"].pop("study")
    clock = [0.]
    monkeypatch.setattr(coverage.time, "monotonic", lambda: clock[0])
    def stamp():
        clock[0] += 2.
        return deepcopy(STAMPS)
    monkeypatch.setattr(campaign, "fingerprint", stamp)
    original_read, original_write, original_publish = coverage._read_continuation, coverage._write_continuation, coverage.publish
    def read(*a, **k):
        clock[0] += 3.
        return original_read(*a, **k)
    def write(*a, **k):
        clock[0] += 4.
        return original_write(*a, **k)
    def publish(*a, **k):
        clock[0] += 5.
        return original_publish(*a, **k)
    monkeypatch.setattr(coverage, "_read_continuation", read)
    monkeypatch.setattr(coverage, "_write_continuation", write)
    monkeypatch.setattr(coverage, "publish", publish)
    calls = []
    def estimate(*a, **k):
        calls.append(1)
        clock[0] += 10.
        return None if len(calls) == 1 else {m: {"value": 0., "se": 1.} for m in coverage.METHODS}
    monkeypatch.setattr(coverage, "estimate_repetition", estimate)
    first = chain_request(tmp_path / "first", task, {"inputs": inputs})
    assert run_coverage(first).status == "pass"
    timing = coverage.read_json(Path(first.output_dir) / "timing.json")
    assert not timing["complete"]
    def profile(req):
        config = json.loads(Path(req.config_path).read_text())
        result = coverage.read_json(Path(req.output_dir) / "result.json")
        unit = coverage.read_json(req.task_path)["id"]
        reader = chain_request(tmp_path / ("reader-" + str(len(calls))), task, {unit: Path(req.output_dir)})
        config = json.loads(Path(reader.config_path).read_text())
        return campaign._profile(reader, config, {"dependency": unit, "path": "result.json"},
                                 result["scenario"], task["parameters"]["recipe"], STAMPS)
    with pytest.raises(ContractError, match="complete production"):
        profile(first)
    task = deepcopy(task)
    task.update(id="chain-1", continuation={"owner": "coverage-test", "step": 1, "predecessor": "chain-0"})
    second = chain_request(tmp_path / "second", task, {"inputs": inputs, "chain-0": Path(first.output_dir)})
    assert run_coverage(second).status == "pass"
    timing = coverage.read_json(Path(second.output_dir) / "timing.json")
    assert timing["complete"] and timing["wall_seconds"] == 45.
    assert timing["complete_repetition_seconds"] == [20.]
    assert [leaf["wall_seconds"] for leaf in timing["leaves"]] == [21., 24.]
    measured = profile(second)
    assert measured["seconds_per_repetition"] == 20.
    assert measured["leaf_overheads"] == {"setup_seconds": 2., "import_seconds": 3., "export_seconds": 4., "publication_seconds": 5.}


@pytest.mark.parametrize("case", ["unequal_cumulative_compute", "setup_exceeds_import_export"])
def test_chained_profile_uses_cumulative_max_and_full_overhead(tmp_path, case):
    from test_campaign import setup_lock
    req, task = setup_lock(tmp_path)
    result_path = tmp_path / "profile/result.json"
    result = coverage.read_json(result_path)
    for record in result["records"]:
        record["wall_seconds"] = 10.  # Last-leaf time is not cumulative compute.
    write(result_path, result)
    path = profile_directory(req) / "timing.json"
    timing = coverage.read_json(path)
    if case == "unequal_cumulative_compute":
        leaves = [{"task_id": f"profile-{i}", "step": i, "compute_seconds": compute,
            "setup_seconds": 1., "import_seconds": 0., "export_seconds": 0., "publication_seconds": 0.,
            "wall_seconds": compute + 1.} for i, compute in enumerate((90., 20.))]
        # Bootstrap capacity also pays setup: 90 compute + 1 setup fits 110.
        timing.update(wall_seconds=112., setup_seconds=2., admitted_budget_seconds=110.)
    else:
        leaves = [{"task_id": "profile-0", "step": 0, "compute_seconds": 110., "setup_seconds": 10.,
            "import_seconds": 2., "export_seconds": 2., "publication_seconds": 1., "wall_seconds": 125.}]
        timing.update(wall_seconds=125., setup_seconds=10., admitted_budget_seconds=600.)
    timing.update(measurement_scope="complete_chain_return", leaves=leaves, complete_repetition_seconds=[100., 10.])
    write(path, timing)
    req = refresh_profile(req)
    profile = campaign._profile(req, json.loads(Path(req.config_path).read_text()),
        {"dependency": "profile", "path": "result.json"}, result["scenario"], task["parameters"]["recipe"], STAMPS)
    assert profile["seconds_per_repetition"] == 100.
    if case == "unequal_cumulative_compute":
        # Four repetitions distinguish the cumulative maximum while respecting
        # the positive-capacity guard (360 s unit-plus-write minimum).
        assert campaign.slice_count(profile, 4, 2491., 1.) == 3
    else:
        assert profile["leaf_overheads"] == {"setup_seconds": 10., "import_seconds": 2.,
                                            "export_seconds": 2., "publication_seconds": 1.}


def test_month1_profiles_bind_owner_cell_and_four_sequential_standard_chains(built):
    approvals = read_mapping(ROOT / 'configs/approvals.yaml')
    manifest = real_frame.screening_profile_tasks(*built[1][:2], approvals)
    tasks = manifest['tasks']
    assert len(tasks) == 161 and tasks[-1]['stage'] == 'campaign-estimate'
    assert not manifest['certifies_release_criteria']
    decision = campaign.screening_decision()
    for index, task in enumerate(tasks[:-1]):
        step = index % 40
        assert task['continuation']['step'] == step
        assert task['continuation']['predecessor'] == (tasks[index-1]['id'] if step else None)
        assert task['parameters']['study'] == decision['recipe']
        assert task['parameters']['recipe']['study'] == decision['recipe']
        assert task['resources'] == {'cpus_per_task': 8, 'gpus': 1, 'wall_seconds': 14400, 'partition': 'standard'}
        assert task['parameters']['wall_seconds'] == 14100
        allocation = coverage.leaf_allocation(task['parameters']['wall_seconds'])
        assert allocation['allocation_seconds'] == task['resources']['wall_seconds']
        assert allocation['setup_seconds'] >= 300
        assert allocation['finalization_seconds'] >= 300
        assert 'continuation.tar' in task['outputs'] and coverage.BUDGET_RECORD in task['outputs']
        if step:
            assert set(tasks[index-1]['outputs']) <= set(task['needs'][tasks[index-1]['id']])
    assert {r['dependency'] for r in tasks[-1]['parameters']['profiles'].values()} == {
        tasks[i]['id'] for i in (39, 79, 119, 159)}
    assert not any(t['stage'] in ('campaign-lock', 'coverage') for t in tasks)


@pytest.mark.parametrize('field', ['scenarios', 'recipe', 'budget_h100_hours', 'final_cells',
    'gpu_concurrency_max', 'gpu_concurrency_exempt_partitions', 'month1_sizing'])
def test_month1_decision_refuses_missing_fields(field):
    approvals = read_mapping(ROOT / 'configs/approvals.yaml')
    approvals['owner_decisions']['screening_campaign'].pop(field)
    with pytest.raises(ContractError, match='screening'):
        campaign.screening_decision(approvals)


@pytest.mark.parametrize('field,value', [('leaf_bound_scope', 'all_scenarios'),
    ('leaves_per_scenario_max', 41), ('gpu_hours_per_leaf_max', True),
    ('repetitions_per_scenario_max', 26), ('sizing_rule', ''), ('continuation', None), ('deferred', '')])
def test_month1_decision_refuses_malformed_sizing(field, value):
    approvals = read_mapping(ROOT / 'configs/approvals.yaml')
    approvals['owner_decisions']['screening_campaign']['month1_sizing'][field] = value
    with pytest.raises(ContractError, match='month1_sizing'):
        campaign.screening_decision(approvals)


@pytest.fixture
def screening_rates(request):
    return getattr(request, 'param', (100., 20000., 50000., 200000.))


@pytest.fixture
def screening_request(tmp_path, monkeypatch, screening_rates):
    import torch
    from oxyformer.contracts import StageResult
    from oxyformer.execution.integrity import verify_published_tree
    monkeypatch.setattr(campaign, 'fingerprint', lambda: deepcopy(STAMPS))
    monkeypatch.setattr(campaign, 'resolve_device', lambda: torch.device('cuda:0'))
    approvals = read_mapping(ROOT / 'configs/approvals.yaml')
    endpoint, frame = smoke_inputs.build_inputs()
    tasks = real_frame.screening_profile_tasks(endpoint, frame, approvals)['tasks']
    deps = {'real-frame-inputs': tmp_path / 'inputs'}
    write(deps['real-frame-inputs'] / 'endpoint.json', endpoint.to_dict())
    write(deps['real-frame-inputs'] / 'frame.json', frame.to_dict())
    for index, rate in zip((39, 79, 119, 159), screening_rates):
        task = tasks[index]
        p = task['parameters']
        _, records = successful_records(1)
        records[0]['draw'] = p['draws'][0]
        records[0]['wall_seconds'] = rate
        root = deps[task['id']] = tmp_path / task['id']
        write(root / 'result.json', {'mode': 'profile', 'scenario': p['scenario'],
            'recipe_hash': coverage.digest(p['recipe']), 'draws': p['draws'], 'records': records})
        leaves = [{'task_id': t['id'], 'step': i, 'compute_seconds': rate / 40,
            'setup_seconds': 100., 'import_seconds': 2., 'export_seconds': 3.,
            'publication_seconds': 5., 'wall_seconds': rate / 40 + 110.}
            for i, t in enumerate(tasks[index-39:index+1])]
        write(root / 'timing.json', {'production_equivalent': False, 'complete': True, 'all_successful': True,
            'recipe_hash': coverage.digest(p['recipe']), 'device': 'cuda:0', 'gpu_seconds': rate + 4400.,
            'complete_repetition_seconds': [rate], 'wall_seconds': rate + 4400., 'setup_seconds': 4000.,
            'measurement_scope': 'complete_chain_return', 'leaves': leaves, **STAMPS})
        for name in ('runtime.json', 'coverage_measurements.json', 'artifact_manifest.json'):
            write(root / name, {'fixture': 'synthetic'})
        (root / 'continuation.tar').write_bytes(b'profile continuation fixture\n' * 1024)
        terminal_timing = coverage.read_json(root / 'timing.json')
        snapshots = {}
        for step, early_task in enumerate(tasks[index-39:index-37]):
            early = deps[early_task['id']] = tmp_path / early_task['id']
            value = coverage.read_json(root / 'result.json')
            value['records'] = []
            write(early / 'result.json', value)
            timing = deepcopy(terminal_timing)
            timing.update(leaves=deepcopy(leaves[:step+1]), complete=False, all_successful=False,
                complete_repetition_seconds=[], wall_seconds=sum(row['wall_seconds'] for row in leaves[:step+1]),
                setup_seconds=sum(row['setup_seconds'] for row in leaves[:step+1]))
            write(early / 'timing.json', timing)
            for name in ('runtime.json', 'coverage_measurements.json', 'artifact_manifest.json'):
                write(early / name, {'fixture': 'synthetic'})
            (early / 'continuation.tar').write_bytes(b'profile continuation fixture\n' * 1024)
            seal_budget_fixture(early, early_task['id'], early_task, complete_controls=True)
            snapshots[str(early)] = verify_published_tree(early,
                StageResult.from_json((early / '_execution/result.json').read_text()))
        seal_budget_fixture(root, task['id'], task, complete_controls=True, dependency_snapshots=snapshots)
    estimate_task = tasks[-1]
    estimate_task['parameters'].pop('calibration_probes')
    estimate_task['needs'].pop('campaign-calibration')
    estimate_task['parameters']['setup_calibration'] = [
        {'dependency': ref['dependency'], 'path': coverage.BUDGET_RECORD}
        for ref in estimate_task['parameters']['profiles'].values()]
    return request(tmp_path / 'screening-estimate', estimate_task, deps, approvals)


@pytest.mark.parametrize('foreign_owner', [False, True])
def test_profile_setup_sources_bind_terminal_profile_publications(screening_request, tmp_path, foreign_owner):
    import shutil
    config = coverage.read_json(screening_request.config_path)
    task = coverage.read_json(screening_request.task_path)
    scenario, refs = next(iter(task['parameters']['probe_profiles'].items()))
    unit = refs['1']['dependency']
    original = Path(config['dependencies'][unit])
    foreign = tmp_path / 'another-profile-attempt'
    foreign.mkdir()
    for path in original.iterdir():
        if path.is_file() and path.suffix in ('.json', '.tar'):
            shutil.copyfile(path, foreign / path.name)
    producer = coverage.read_json(original / '_execution/task.json')
    if foreign_owner:
        producer['continuation']['owner'] = 'another-profile-chain'
    archive = foreign / 'continuation.tar'
    archive.write_bytes(archive.read_bytes() * 3)
    seal_budget_fixture(foreign, unit, producer, complete_controls=True)
    deps = {key: Path(value) for key, value in config['dependencies'].items()}
    deps[unit] = foreign
    revised = request(tmp_path / 'cross-profile-estimate', task, deps, config['approvals'])
    with pytest.raises(ContractError, match='same.*profile|profile.*chain'):
        campaign_estimate.estimate_budget(revised)
    assert not list(Path(revised.output_dir).iterdir())


def screening_lock_request(req, tmp_path):
    outcome = campaign_estimate.run_stage(req)
    assert outcome.status == 'pass', outcome.message
    original = coverage.read_json(req.task_path)
    parameters = {**original['parameters'], 'campaign_id': 'month1',
        'estimate_input': {'dependency': 'campaign-estimate', 'path': 'budget_estimate.json'},
        'evaluation_namespace': 'month1-evaluation', 'retry_rules': coverage.RETRY_RULES,
        'prior_locks': [], 'cpu_wall_hours_budget': 1000.,
        'admission_receipt': {'dependency': 'campaign-admit', 'path': 'admission.json'}}
    config = coverage.read_json(req.config_path)
    deps = {k: Path(v) for k, v in config['dependencies'].items()}
    deps['campaign-estimate'] = Path(req.output_dir)
    return request(tmp_path / 'screening-lock', {'id': 'campaign-lock', 'stage': 'campaign-lock',
        'parameters': parameters}, deps, config['approvals'])


def test_month1_estimate_uses_maximal_independent_chained_counts_and_admits_nothing(screening_request, monkeypatch):
    import math
    def forbidden(*a, **kw):
        pytest.fail('estimate admitted or expanded work')
    for name in ('build_lock', 'expand_campaign', '_admit_locked_leaves'):
        monkeypatch.setattr(campaign, name, forbidden)
    estimate = campaign_estimate.estimate_budget(screening_request)
    assert estimate['owner_safety_factor_decision'] == campaign.screening_safety_factor()
    assert estimate['profile_safety_factor'] == campaign.screening_safety_factor()['profile_safety_factor']
    assert estimate['estimate_only'] and not estimate['admitted'] and not estimate['certifies_release_criteria']
    assert len({r['repetitions'] for r in estimate['per_scenario'].values()}) == 4
    for name, row in estimate['per_scenario'].items():
        profile = estimate['profiles'][name]
        capacity = coverage.usable_leaf_compute(14100, profile['leaf_overheads']) - profile['resume_overhead_seconds']
        demand = profile['seconds_per_repetition'] * row['repetitions'] + profile['publication_verification_seconds']
        assert row['leaves'] == math.ceil(estimate['profile_safety_factor'] * demand / capacity) + 1
        assert row['leaves'] <= 40 and row['gpu_hours'] == row['leaves'] * 4
        assert row['repetitions'] == 25 or estimate['profile_safety_factor'] * (profile['seconds_per_repetition'] * (row['repetitions'] + 1) + profile['publication_verification_seconds']) > 39 * capacity
    assert estimate['gpu_hours'] <= 640 < estimate['budget_h100_hours']
    assert not list(Path(screening_request.output_dir).iterdir())


def test_month1_lock_matches_estimate_instances_and_global_concurrency(screening_request, tmp_path):
    req = screening_lock_request(screening_request, tmp_path)
    result = campaign.run_stage(req)
    assert result.status == 'pass', result.message
    result.verify(req)
    root = Path(req.output_dir)
    lock = coverage.read_json(root / 'recipe_lock.json')
    plans = coverage.read_json(root / 'expanded_units.json')['plans']
    estimate = coverage.read_json(Path(screening_request.output_dir) / 'budget_estimate.json')
    assert lock['estimate_hash'] == file_hash(Path(screening_request.output_dir) / 'budget_estimate.json')
    assert len(plans) == 4 and 'final' not in lock['instances']
    assert lock['budget']['concurrency_limits']['gpu'] == 4
    assert lock['budget']['per_scenario'] == estimate['per_scenario']
    assert lock['budget']['owner_safety_factor_decision'] == estimate['owner_safety_factor_decision']
    assert coverage.read_json(root / 'budget.json')['owner_safety_factor_decision'] == campaign.screening_safety_factor()
    # Each instance is a single path including its collector: four disjoint
    # paths imply at most four simultaneous GPUs under any topological schedule.
    for plan in plans:
        name = plan['spec']['work'][0]['parameters']['scenario']['name']
        assert len(plan['expected_leaves']) == estimate['per_scenario'][name]['leaves']
        assert len(plan['spec']['work']) == 1
        assert all(t['parameters']['mode'] == 'screening' for t in plan['tasks'][:-1])
        assert all(u['gpu_hours'] <= 4 and '--partition=standard' in u['sbatch'] and '--gpus=1' in u['sbatch']
                   and 'export OXYFORMER_DEVICE=cuda' in u['command'] for u in plan['units'][:-1])
        assert plan['units'][-1]['gpu_hours'] == 0
        assert 'export OXYFORMER_DEVICE=cuda' not in plan['units'][-1]['command']
        for previous, following in zip(plan['units'], plan['units'][1:]):
            assert previous['id'] in following['needs']
    for name in ('recipe_lock.json', 'expanded_units.json', 'task_manifest.json', 'budget.json', 'gate.json', 'artifact_manifest.json'):
        value = coverage.read_json(root / name)
        assert value['certifies_release_criteria'] is False
        assert value['final_cells'] == campaign.screening_decision()['final_cells']


@pytest.mark.parametrize('partition', ['preemptible', 'large_batch', None])
def test_month1_lock_and_estimate_refuse_nonstandard_partition(screening_request, tmp_path, partition):
    req = screening_lock_request(screening_request, tmp_path)
    req = update_task(req, lambda p: p.update(partition=partition))
    result = campaign.run_stage(req)
    assert result.status == 'blocked' and 'partition' in result.message
    req = update_task(screening_request, lambda p: p.update(partition=partition))
    assert campaign_estimate.run_stage(req).status == 'blocked'


@pytest.mark.parametrize('change', ['missing', 'count', 'safety', 'wall', 'approval', 'manual_count'])
def test_month1_lock_requires_unchanged_prior_estimate(screening_request, tmp_path, change):
    req = screening_lock_request(screening_request, tmp_path)
    if change == 'missing':
        req = update_task(req, lambda p: p.pop('estimate_input'))
    elif change == 'count':
        path = Path(screening_request.output_dir) / 'budget_estimate.json'
        value = coverage.read_json(path)
        value['per_scenario']['null_effect']['repetitions'] -= 1
        write(path, value)
        req = replace(req, dependency_hashes=tuple(map(file_hash, req.dependency_paths)))
    elif change == 'approval':
        config = coverage.read_json(req.config_path)
        config['approvals']['owner_decisions'].pop('screening_campaign')
        write(Path(req.config_path), config)
        req = replace(req, config_hash=file_hash(req.config_path))
    else:
        req = update_task(req, lambda p: p.update({'safety': {'profile_safety_factor': 3.},
            'wall': {'wall_seconds': 13500}, 'manual_count': {'screening_repetitions': 25}}[change]))
    result = campaign.run_stage(req)
    assert result.status == 'blocked' and not result.artifacts
    assert not list(Path(req.output_dir).iterdir())


@pytest.mark.parametrize('shape_percent', [None, 110, 120, 150, 151])
def test_month1_bound_cell_executes_and_cpu_collector_scopes_one_scenario(screening_request, tmp_path, monkeypatch, shape_percent):
    from test_coverage import chain_request
    from oxyformer.training.fit import RecipeStudy
    req = screening_lock_request(screening_request, tmp_path)
    result = campaign.run_stage(req)
    assert result.status == 'pass', result.message
    root = Path(req.output_dir)
    lock = coverage.read_json(root / 'recipe_lock.json')
    plans = coverage.read_json(root / 'expanded_units.json')['plans']
    plan = next(p for p in plans if p['spec']['work'][0]['parameters']['scenario']['name'] == 'null_effect')
    assert len(plan['expected_leaves']) == 2
    config = coverage.read_json(req.config_path)
    deps = {k: Path(v) for k, v in config['dependencies'].items()}
    deps['campaign-lock'] = root
    deps['campaign-admit'] = sealed_json_fixture(tmp_path / 'admit-fixture', 'campaign-admit',
        'campaign-admit', {'admission.json': {'admitted': True, 'lock_hash': coverage.digest(lock),
            'estimate_hash': lock['estimate_hash'], 'setup_policy': 'three-receipt-minima-v1',
            'admission_measurements': {'setup_minima': {'fresh': 0., 'resumed_1': 0., 'resumed_2': 0.},
                                       'allowance_seconds': 60.}}})
    calls = []
    def execute(draw, frame, scenario, template, recipe, destination, deadline, *, continuation, study):
        assert study == RecipeStudy(**campaign.SCREENING_CELL)
        assert recipe['study'] == campaign.SCREENING_CELL
        calls.append(draw)
        _, records = successful_records(1)
        return {**records[0], 'draw': draw, 'wall_seconds': .001}
    monkeypatch.setattr(coverage, 'execute_draw', execute)
    if shape_percent is not None:
        original_archive = coverage._write_continuation
        expected_shape = lock['budget']['expected_predecessor_shapes']['null_effect']
        def sized_archive(path, states, **kwargs):
            original_archive(path, states, **kwargs)
            if kwargs['chain']['step'] == 0:
                target = expected_shape['archive_bytes'] * shape_percent // 100
                assert path.stat().st_size <= target
                # Valid tar padding models publication-size variance without
                # changing the saved continuation state or scientific work.
                with path.open('ab') as stream:
                    stream.write(b'\0' * (target - path.stat().st_size))
        monkeypatch.setattr(coverage, '_write_continuation', sized_archive)
    for leaf in plan['tasks'][:-1]:
        leaf_req = chain_request(tmp_path / leaf['id'], leaf, deps)
        cfg = coverage.read_json(leaf_req.config_path)
        cfg['approvals'] = config['approvals']
        write(Path(leaf_req.config_path), cfg)
        leaf_req = replace(leaf_req, config_hash=file_hash(leaf_req.config_path))
        result = run_coverage(leaf_req)
        if shape_percent == 151 and leaf['continuation']['step'] == 1:
            assert result.status == 'blocked' and 'admission-shape-mismatch' in result.message
            stopped = coverage.read_json(Path(leaf_req.output_dir) / coverage.BUDGET_RECORD)
            assert stopped['diagnostic_only'] and stopped['status'] == 'admission-shape-mismatch'
            assert not (Path(leaf_req.output_dir) / '_execution/fingerprint.json').exists()
            return
        assert result.status == 'pass', result.message
        if shape_percent is not None and leaf['continuation']['step'] == 1:
            measured = coverage.read_json(Path(leaf_req.output_dir) / coverage.BUDGET_RECORD)['leaves'][-1]
            shape = measured['setup_safety']['admission_shape']
            assert shape['expected'] == expected_shape
            assert shape['status'] == ('matched' if shape_percent == 110 else 'admission-shape-exceeded')
        # The spare carries completed draws without fitting them again.
        assert calls == lock['batches'][0]['draws']
        value = coverage.read_json(Path(leaf_req.output_dir) / 'result.json')
        assert not value['certifies_release_criteria'] and not value['summary']['production_equivalent']
        deps[leaf['id']] = Path(leaf_req.output_dir)
    collector = request(tmp_path / 'month1-collector', plan['tasks'][-1], deps, config['approvals'])
    # CPU collection has a different hardware environment, but verifies the
    # unchanged scientific fingerprint and bound profile receipt identities.
    monkeypatch.setattr(campaign, 'fingerprint', lambda: {**deepcopy(STAMPS), 'environment_hash': 'd' * 64})
    collected = campaign.run_stage(collector)
    assert collected.status == 'pass', collected.message
    gate = coverage.read_json(Path(collector.output_dir) / 'gate.json')
    assert set(gate['scenario_summaries']) == {'null_effect'}
    assert not gate['certifies_production_coverage'] and not gate['certifies_release_criteria']
    assert coverage.read_json(Path(collector.output_dir) / 'result.json')['gate'] == gate
    assert gate['scenario_summaries']['null_effect']['declared_repetitions'] == 25


def test_month1_profile_binds_study_and_refuses_wrong_cell_before_fitting(tmp_path, monkeypatch):
    from test_coverage import chain_request
    endpoint, frame = smoke_inputs.build_inputs()
    approvals = read_mapping(ROOT / 'configs/approvals.yaml')
    task = real_frame.screening_profile_tasks(endpoint, frame, approvals)['tasks'][0]
    task['parameters'].pop('device')  # CPU synthetic preflight, no fit is allowed.
    task['parameters']['study']['seed_count'] = 3
    deps = {'real-frame-inputs': tmp_path / 'inputs'}
    write(deps['real-frame-inputs'] / 'endpoint.json', endpoint.to_dict())
    write(deps['real-frame-inputs'] / 'frame.json', frame.to_dict())
    req = request(tmp_path / 'wrong-study', task, deps, approvals)
    monkeypatch.setattr(coverage, 'execute_draw', lambda *a, **k: pytest.fail('wrong study was fitted'))
    result = run_coverage(req)
    assert result.status == 'blocked' and 'study' in result.message


@pytest.mark.parametrize('screening_rates', [(100., 20000., 50000., 500000.)], indirect=True)
def test_month1_refuses_when_one_complete_profile_cannot_fit_one_evaluation_repetition(screening_request):
    # A 300,000 s bootstrap measurement fits forty unsafetied profile slots,
    # but one evaluation repetition cannot fit the declared 2x safety factor.
    result = campaign_estimate.run_stage(screening_request)
    assert result.status == 'blocked' and 'one screening repetition' in result.message
    assert not result.artifacts and not list(Path(screening_request.output_dir).iterdir())


def test_month1_estimate_refuses_completed_nonterminal_profile_slot(screening_request, tmp_path):
    config = coverage.read_json(screening_request.config_path)
    task = coverage.read_json(screening_request.task_path)
    reference = task['parameters']['profiles']['null_effect']
    source = Path(config['dependencies'][reference['dependency']])
    producer = coverage.read_json(source / '_execution/task.json')
    old_id = producer['id']
    producer['id'] = 'profile-null-effect-0'
    producer['continuation'].update(step=0, predecessor=None)
    producer['needs'] = {'real-frame-inputs': producer['needs']['real-frame-inputs']}
    root = tmp_path / 'completed-first-slot'
    result = coverage.read_json(source / 'result.json')
    timing = coverage.read_json(source / 'timing.json')
    rate = timing['complete_repetition_seconds'][0]
    leaf = {**timing['leaves'][0], 'task_id': producer['id'], 'step': 0,
            'compute_seconds': rate, 'wall_seconds': rate + 110.}
    timing.update(leaves=[leaf], wall_seconds=rate + 110., gpu_seconds=rate + 110., setup_seconds=100.)
    write(root / 'result.json', result)
    write(root / 'timing.json', timing)
    seal_budget_fixture(root, producer['id'], producer)
    deps = {k: Path(v) for k, v in config['dependencies'].items() if k != old_id}
    deps[producer['id']] = root
    task['needs'][producer['id']] = task['needs'].pop(old_id)
    reference['dependency'] = producer['id']
    req = request(tmp_path / 'early-slot-estimate', task, deps, config['approvals'])
    result = campaign_estimate.run_stage(req)
    assert result.status == 'blocked' and 'terminal bootstrap' in result.message
    assert not result.artifacts and not list(Path(req.output_dir).iterdir())


@pytest.mark.parametrize('screening_rates', [(10000.,) * 4], indirect=True)
def test_month1_estimate_refuses_shortened_wall_that_reduces_maximal_count(screening_request):
    estimate = campaign_estimate.estimate_budget(screening_request)
    assert all(1 < r['repetitions'] <= 25 and r['leaves'] <= 40
               for r in estimate['per_scenario'].values())
    req = update_task(screening_request, lambda p: p.update(wall_seconds=3600))
    result = campaign_estimate.run_stage(req)
    assert result.status == 'blocked' and 'owner-maximal' in result.message
    assert not result.artifacts and not list(Path(req.output_dir).iterdir())


def test_month1_emitted_profile_chain_mode_is_accepted(screening_request, tmp_path, monkeypatch):
    import torch
    from oxyformer.training import fit
    from test_coverage import chain_request
    config = coverage.read_json(screening_request.config_path)
    estimate_task = coverage.read_json(screening_request.task_path)
    endpoint, frame = smoke_inputs.build_inputs()
    tasks = real_frame.screening_profile_tasks(endpoint, frame, config['approvals'])['tasks'][:40]
    deps = {k: Path(v) for k, v in config['dependencies'].items()}
    # Exercise real continuation, publication and receipt admission on CPU;
    # only fitting and hardware metadata are synthetic.
    monkeypatch.setattr(fit, 'resolve_device', lambda *a: torch.device('cuda:0'))
    monkeypatch.setattr(fit, 'fit_environment', lambda *a: {})
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: None)
    calls = []
    def execute(draw, *args, **kwargs):
        calls.append(draw)
        _, records = successful_records(1)
        return {**records[0], 'draw': draw, 'wall_seconds': .001}
    monkeypatch.setattr(coverage, 'execute_draw', execute)
    for task in tasks:
        req = chain_request(tmp_path / ('executed-' + task['id']), task, deps)
        cfg = coverage.read_json(req.config_path)
        cfg['approvals'] = config['approvals']
        write(Path(req.config_path), cfg)
        req = replace(req, config_hash=file_hash(req.config_path))
        completed = run_coverage(req)
        assert completed.status == 'pass', completed.message
        deps[task['id']] = Path(req.output_dir)
    assert len(calls) == 1
    terminal = coverage.read_json(deps[tasks[-1]['id']] / 'result.json')
    assert terminal['mode'] == 'profile' and terminal['campaign_mode'] == 'screening'
    assert tasks[-1]['parameters']['recipe']['campaign_mode'] == 'screening'
    req = request(tmp_path / 'executed-profile-estimate', estimate_task, deps, config['approvals'])
    result = campaign_estimate.run_stage(req)
    assert result.status == 'pass', result.message
    report = coverage.read_json(Path(req.output_dir) / 'budget_estimate.json')
    assert report['per_scenario']['null_effect']['repetitions'] == 25


@pytest.mark.parametrize('screening_rates', [(100.,) * 4], indirect=True)
def test_month1_shorter_wall_preserves_owner_maximal_count(screening_request, tmp_path):
    req = update_task(screening_request, lambda p: p.update(wall_seconds=3600))
    lock_req = screening_lock_request(req, tmp_path)
    report = coverage.read_json(Path(req.output_dir) / 'budget_estimate.json')
    assert report['gpu_hours'] == pytest.approx(sum(r['leaves'] * 3900 / 3600 for r in report['per_scenario'].values()))
    assert all(row['repetitions'] == 25 and row['gpu_hours'] == row['leaves'] * 3900 / 3600
               for row in report['per_scenario'].values())
    locked = campaign.run_stage(lock_req)
    assert locked.status == 'pass', locked.message
    lock = coverage.read_json(Path(lock_req.output_dir) / 'recipe_lock.json')
    assert lock['budget']['per_scenario'] == report['per_scenario']
    assert all(b['wall_seconds'] == 3600 and b['slices'] == report['per_scenario'][b['scenario']['name']]['leaves']
               for b in lock['batches'])


@pytest.mark.parametrize('factor', [1, 1.25, 2.0])
@pytest.mark.parametrize('decided_on', ['2026-10-17', date(2026, 10, 17)])
def test_safety_factor_reader_is_strict_read_only_and_json_safe(factor, decided_on):
    approvals = read_mapping(ROOT / 'configs/approvals.yaml')
    block = approvals['owner_decisions']['screening_month1_safety_factor']
    block.update(profile_safety_factor=factor, decided_on=decided_on)
    before = deepcopy(approvals)
    decision = campaign.screening_safety_factor(approvals)
    assert decision == {**block, 'profile_safety_factor': float(factor), 'decided_on': '2026-10-17'}
    assert type(decision['profile_safety_factor']) is float
    assert json.loads(json.dumps(decision)) == decision
    decision['applies_to'].append('changed-copy')
    assert approvals == before


@pytest.mark.parametrize('approvals', [None, [], {}, {'owner_decisions': None},
    {'owner_decisions': {}}, {'owner_decisions': {'screening_month1_safety_factor': []}}])
def test_safety_factor_reader_refuses_absence(approvals):
    with pytest.raises(ContractError, match='screening_month1_safety_factor'):
        campaign.screening_safety_factor(approvals)


@pytest.mark.parametrize('field', ['profile_safety_factor', 'decided_on', 'applies_to',
    'basis', 'accepted_risk', 'deferred', 'extra'])
def test_safety_factor_reader_requires_exact_keys(field):
    approvals = read_mapping(ROOT / 'configs/approvals.yaml')
    block = approvals['owner_decisions']['screening_month1_safety_factor']
    if field == 'extra':
        block[field] = 'unexpected'
    else:
        block.pop(field)
    with pytest.raises(ContractError, match='screening_month1_safety_factor'):
        campaign.screening_safety_factor(approvals)


@pytest.mark.parametrize('field,value', [
    *[('profile_safety_factor', v) for v in (True, False, '1.25', None, [],
        float('nan'), float('inf'), float('-inf'), .999, 2.001, 10 ** 1000)],
    ('decided_on', ''), ('decided_on', '2026-02-30'), ('decided_on', '17 October 2026'),
    ('decided_on', '2026-10-17T12:00:00'), ('decided_on', datetime(2026, 10, 17)),
    ('decided_on', 20261017), ('applies_to', ['campaign-lock', 'campaign-estimate']),
    ('applies_to', ['campaign-estimate']), ('applies_to', 'campaign-estimate'),
    ('basis', ''), ('basis', '  '), ('accepted_risk', None), ('deferred', 123),
])
def test_safety_factor_reader_refuses_malformed_fields(field, value):
    approvals = read_mapping(ROOT / 'configs/approvals.yaml')
    approvals['owner_decisions']['screening_month1_safety_factor'][field] = value
    with pytest.raises(ContractError, match='screening_month1_safety_factor'):
        campaign.screening_safety_factor(approvals)


@pytest.mark.parametrize('factor', [1., 1.25, 2.])
def test_screening_emitter_uses_dispatched_factor_and_preserves_previous_chain_bytes(factor):
    approvals = read_mapping(ROOT / 'configs/approvals.yaml')
    approvals['owner_decisions']['screening_month1_safety_factor']['profile_safety_factor'] = factor
    tasks = real_frame.screening_profile_tasks(*smoke_inputs.build_inputs(), approvals)['tasks']
    assert tasks[-1]['parameters']['profile_safety_factor'] == factor
    assert all('profile_safety_factor' not in task['parameters'] for task in tasks[:-1])
    # SHA256 of canonical task bytes from the previous emitter at launch base
    # 1cf40a54b2c47a8449fc450a698fa3f23dfb534c on this fixed smoke endpoint/frame.
    # Covers all 160 task IDs, draws, chain owners, parameters and receipt declarations.
    assert coverage.digest(tasks[:-1]) == '51a1a989a1657afb258908acff11ee0a59d25136278d8e75d7c7c876639a7c8d'


@pytest.mark.parametrize('factor', [1., 1.25, 2.])
def test_legacy_emitter_uses_repository_factor(factor, tmp_path, monkeypatch):
    approvals = read_mapping(ROOT / 'configs/approvals.yaml')
    approvals['owner_decisions']['screening_month1_safety_factor']['profile_safety_factor'] = factor
    import yaml
    path = tmp_path / 'approvals.yaml'
    path.write_text(yaml.safe_dump(approvals))
    monkeypatch.setattr(campaign, 'OWNER_APPROVALS', path)
    task = real_frame.profile_tasks(*smoke_inputs.build_inputs())['tasks'][4]
    assert task['parameters']['profile_safety_factor'] == factor


@pytest.mark.parametrize('stage', ['estimate', 'lock'])
@pytest.mark.parametrize('factor', [1., 2.])
def test_screening_consumers_refuse_task_factor_drift(screening_request, tmp_path, stage, factor):
    req = screening_lock_request(screening_request, tmp_path) if stage == 'lock' else screening_request
    req = update_task(req, lambda p: p.update(profile_safety_factor=factor))
    result = (campaign.run_stage if stage == 'lock' else campaign_estimate.run_stage)(req)
    assert result.status == 'blocked' and 'profile_safety_factor drift' in result.message
    assert not result.artifacts and not list(Path(req.output_dir).iterdir())


@pytest.mark.parametrize('field,value', [('profile_safety_factor', 2.), ('decided_on', '2026-10-18'),
    ('basis', 'changed basis'), ('accepted_risk', 'changed risk'), ('deferred', 'changed deferral')])
def test_screening_validation_checks_entire_dispatched_safety_decision(field, value):
    approvals = read_mapping(ROOT / 'configs/approvals.yaml')
    approvals['owner_decisions']['screening_month1_safety_factor'][field] = value
    recipe = {'study': campaign.screening_decision()['recipe']}
    with pytest.raises(ContractError, match='screening_month1_safety_factor approval differs from repository'):
        campaign.validate_screening(recipe, approvals)


def test_legacy_estimate_refuses_task_factor_drift(estimate_request):
    req = update_task(estimate_request, lambda p: p.update(profile_safety_factor=1.25))
    result = campaign_estimate.run_stage(req)
    assert result.status == 'blocked' and 'profile_safety_factor drift' in result.message
    assert not result.artifacts and not list(Path(req.output_dir).iterdir())


@pytest.mark.parametrize('payload', [None, 'owner_decisions: [',
    'owner_decisions: {screening_month1_safety_factor: {decided_on: 2026-02-30}}'])
def test_safety_factor_reader_wraps_missing_or_malformed_yaml(tmp_path, monkeypatch, payload):
    path = tmp_path / 'approvals.yaml'
    if payload is not None:
        path.write_text(payload)
    monkeypatch.setattr(campaign, 'OWNER_APPROVALS', path)
    with pytest.raises(ContractError, match='owner_decisions.screening_month1_safety_factor'):
        campaign.screening_safety_factor()


@pytest.mark.parametrize('text,expected', [
    ('2026-290', '2026-10-17'), ('2026290', '2026-10-17'),
    ('2024-366', '2024-12-31'), ('2024366', '2024-12-31'),
    ('2026-001', '2026-01-01'), ('20261017', '2026-10-17'),
    ('2026-W42-6', '2026-10-17'), ('2026W426', '2026-10-17'),
])
def test_safety_factor_reader_normalizes_complete_iso_date_formats(text, expected):
    approvals = read_mapping(ROOT / 'configs/approvals.yaml')
    approvals['owner_decisions']['screening_month1_safety_factor']['decided_on'] = text
    decision = campaign.screening_safety_factor(approvals)
    assert decision['decided_on'] == expected
    assert approvals['owner_decisions']['screening_month1_safety_factor']['decided_on'] == text


@pytest.mark.parametrize('text', ['2026-000', '2026000', '2026-366', '2026366',
    '2024-367', '2024367', '0000-001', '9999-366'])
def test_safety_factor_reader_refuses_invalid_ordinal_dates(text):
    approvals = read_mapping(ROOT / 'configs/approvals.yaml')
    approvals['owner_decisions']['screening_month1_safety_factor']['decided_on'] = text
    with pytest.raises(ContractError, match='screening_month1_safety_factor'):
        campaign.screening_safety_factor(approvals)


def test_screening_emitter_accepts_equivalent_ordinal_decision_date():
    approvals = read_mapping(ROOT / 'configs/approvals.yaml')
    endpoint, frame = smoke_inputs.build_inputs()
    expected = real_frame.screening_profile_tasks(endpoint, frame, approvals)
    approvals['owner_decisions']['screening_month1_safety_factor']['decided_on'] = '2026-290'
    assert real_frame.screening_profile_tasks(endpoint, frame, approvals) == expected
    campaign.validate_screening(expected['tasks'][-1]['parameters']['recipe'], approvals)


def sealed_json_fixture(root, unit, stage, values):
    """Seal small synthetic probe/admission outputs through publication authority."""
    from oxyformer.execution.integrity import publish_result
    from oxyformer.contracts import StageRequest
    root.mkdir(parents=True, exist_ok=True)
    cfg = write(root / '_execution/config.json', {})
    task_path = write(root / '_execution/task.json', {'id': unit, 'stage': stage, 'parameters': {}})
    req = StageRequest(stage=stage, config_path=str(cfg), config_hash=file_hash(cfg),
        task_path=str(task_path), task_hash=file_hash(task_path), dependency_paths=(), dependency_hashes=(),
        output_dir=str(root), code_identity='c' * 40)
    (root / '_execution/request.json').write_text(req.to_json())
    result = publish_result(root, coverage.publish(req, values, status='pass', message='synthetic receipt'))
    assert result.status == 'pass', result.message
    return root


def test_full_window_factor_only_changes_provenance_and_demand():
    import math
    overheads = dict.fromkeys(coverage.OVERHEAD_PHASES, 100.)
    a = coverage.leaf_budget(14100, 1., overheads)
    b = coverage.leaf_budget(14100, 2., overheads)
    assert a['compute_seconds'] == b['compute_seconds'] == 13400
    assert a['formula'] == b['formula'] == 'arc-2612-window'
    profile = {'seconds_per_repetition': 30000., 'publication_verification_seconds': 4000.,
               'leaf_overheads': overheads}
    capacity = 13800 - 400 - 690 - 330 - coverage.RESUME_OVERHEAD_FLOOR_SECONDS
    assert campaign.slice_count(profile, 2, 14100, 1.25) == math.ceil(1.25 * 64000 / capacity) + 1
    with pytest.raises(ContractError, match='phase unit and checkpoint'):
        coverage.usable_leaf_compute(14100, {**overheads, 'setup_seconds': 12000})


@pytest.mark.parametrize('factor,compute,accepted', [(1., 13800., True), (2., 6900., False), (2., 13800., False), (1., 6900., False)])
def test_budget_generation_exception_is_narrow(factor, compute, accepted):
    old = coverage.leaf_budget(14100, factor, dict.fromkeys(coverage.OVERHEAD_PHASES, 0.))
    old.pop('formula')
    old['compute_seconds'] = compute
    if accepted:
        assert coverage.validate_leaf_budget(old) == old
    else:
        with pytest.raises(ContractError, match='pre-ARC-2612 window formula'):
            coverage.validate_leaf_budget(old)


def test_calibration_uses_own_leaf_percentile_and_max_completed_resume():
    profiles = {n: {'seconds_per_repetition': 1000.} for n in ('a', 'b', 'c')}
    records = []
    for name, setups, delivered, success in [('a', [10, 100], [1500, 1500], 1),
                                             ('b', [20, 200], [2000, 2000], 1),
                                             ('c', [30, 300], [4000, 4000], 0)]:
        for step in range(2):
            leaves = [{'task_id': name + str(i), 'setup_seconds': setups[i],
                'compute_seconds': delivered[i], 'import_seconds': 1., 'export_seconds': 2.,
                'publication_seconds': 3.} for i in range(step + 1)]
            records.append({'leaves': leaves, 'binding': {'scenario': {'name': name}},
                'counts': {'success': success if step else 0}, 'record_hash': coverage.digest(leaves),
                'budget_generation': 'pre-ARC-2612', 'profile': deepcopy(STAMPS)})
    result = campaign_probe.calibration_statistics(records, profiles)
    assert result['leaf_count'] == 6
    assert result['setup_seconds'] == 312.5  # P90 = 250, no cumulative-history duplication.
    assert result['resume_overhead_seconds'] == 1500  # MAX, not mean or completed-chain minimum.
    assert not result['resume_uncalibrated']
    for r in records:
        r['counts']['success'] = 0
    assert campaign_probe.calibration_statistics(records, profiles)['resume_uncalibrated']
    assert campaign_probe.calibration_statistics(records, profiles)['resume_overhead_seconds'] == coverage.RESUME_OVERHEAD_FLOOR_SECONDS
    with pytest.raises(ContractError, match='duplicate'):
        campaign_probe.calibration_statistics(records + [records[0]], profiles)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('rate,chains,repetitions,publication,expected_resume,expected_count', [
    (430000., [('short', 36, 430000.), ('slow', 39, 488500.)], 1, 100., 1500., 40),
    (1000., [('short', 2, 2000.), ('slow', 3, 9000.)], 11, 0., 8000 / 3, 3),
])
def test_resume_max_includes_distinct_chains_of_same_scenario(reverse, rate, chains,
        repetitions, publication, expected_resume, expected_count):
    overheads = dict.fromkeys(coverage.OVERHEAD_PHASES, 0.)
    profile = {'seconds_per_repetition': rate, 'publication_verification_seconds': publication,
               'leaf_overheads': overheads}
    records = []
    for chain, count, delivered in chains:
        leaves = []
        # The later successful prefix must not replace this chain's first completion.
        for step in range(count + 1):
            leaves.append({**overheads, 'task_id': f'{chain}-{step}',
                           'compute_seconds': delivered / count if step < count else 12000.})
            records.append({'leaves': list(leaves), 'binding': {'scenario': {'name': 'a'}},
                'counts': {'success': int(step >= count - 1)},
                'record_hash': coverage.digest(leaves), 'budget_generation': 'pre-ARC-2612', 'profile': deepcopy(STAMPS)})
    if reverse:
        records.reverse()
    result = campaign_probe.calibration_statistics(records, {'a': profile})
    assert result['resume_overhead_seconds'] == expected_resume
    assert result['resume_source']['completed']['a']['leaves'] == chains[1][1]
    assert campaign.slice_count({**profile, 'resume_overhead_seconds': result['resume_overhead_seconds']},
                                repetitions, 14100, 1.) == expected_count


def probe_request_from_screening(req, tmp_path, *, mode='calibration', lock_root=None):
    original = coverage.read_json(req.task_path)
    config = coverage.read_json(req.config_path)
    deps = {k: Path(v) for k, v in config['dependencies'].items()}
    p = original['parameters']
    if mode == 'calibration':
        needs = campaign_probe.calibration_inputs(p)
        parameters = {k: deepcopy(p[k]) for k in ('profiles', 'endpoint_input', 'frame_input')}
    else:
        deps['campaign-lock'] = lock_root
        plans = coverage.read_json(lock_root / 'expanded_units.json')['plans']
        needs = deepcopy(plans[0]['tasks'][0]['needs'])
        needs.pop('campaign-admit')
        parameters = {'lock_input': {'dependency': 'campaign-lock', 'path': 'recipe_lock.json'}}
    task = {'id': 'probe', 'stage': 'campaign-setup-probe', 'parameters': {**parameters, 'mode': mode}, 'needs': needs}
    return request(tmp_path, task, deps, config['approvals'])


def test_probe_modes_measure_exact_shapes_and_bind_lock(screening_request, tmp_path):
    pre = probe_request_from_screening(screening_request, tmp_path / 'pre')
    value = campaign_probe.probe(pre, coverage.read_json(pre.config_path), coverage.read_json(pre.task_path), measurer=lambda: 123.)
    assert value['setup_seconds'] == 123 and value['lock_hash'] is None
    req = screening_lock_request(screening_request, tmp_path)
    result = campaign.run_stage(req)
    assert result.status == 'pass', result.message
    post = probe_request_from_screening(screening_request, tmp_path / 'post', mode='admission', lock_root=Path(req.output_dir))
    task = coverage.read_json(post.task_path)
    result = campaign_probe.probe(post, coverage.read_json(post.config_path), task, measurer=lambda: 234.)
    assert result['lock_hash'] == file_hash(Path(req.output_dir) / 'recipe_lock.json')
    assert result['setup_seconds'] == 234
    task['needs'].pop('campaign-lock')
    with pytest.raises(ContractError, match='input shape'):
        campaign_probe.probe(post, coverage.read_json(post.config_path), task, measurer=lambda: 1.)


def test_prelock_two_draws_smaller_selected_and_inputs_bound(screening_request, tmp_path):
    req = screening_request
    config, task = coverage.read_json(req.config_path), coverage.read_json(req.task_path)
    p = task['parameters']
    profiles = campaign.screening_estimate(req, config, p)['profiles']
    p.pop('setup_calibration')
    pre = probe_request_from_screening(req, tmp_path / 'pre')
    digests = campaign_probe.probe(pre, coverage.read_json(pre.config_path), coverage.read_json(pre.task_path), measurer=lambda: 10.)['input_digests']
    deps = {k: Path(v) for k, v in config['dependencies'].items()}
    p['calibration_probes'] = []
    for i, seconds in enumerate((500., 100.)):
        unit = 'calibration-' + str(i)
        deps[unit] = sealed_json_fixture(tmp_path / unit, unit, 'campaign-setup-probe',
            {'setup_probe.json': {'mode': 'calibration', 'setup_seconds': seconds, 'input_digests': digests}})
        p['calibration_probes'].append({'dependency': unit, 'path': 'setup_probe.json'})
    revised = request(tmp_path / 'two-draw-estimate', task, deps, config['approvals'])
    result, calibration = campaign_probe.calibrate(revised, coverage.read_json(revised.config_path), p, profiles)
    assert calibration['chosen_seconds'] == 100
    assert calibration['setup_seconds'] == 240
    assert len(calibration['draws']) == 2 and calibration['resume_uncalibrated']
    assert all(v['leaf_overheads']['setup_seconds'] == 240 for v in result.values())
    p['calibration_probes'].append(p['calibration_probes'][0])
    with pytest.raises(ContractError, match='one or two'):
        campaign_probe.calibrate(revised, coverage.read_json(revised.config_path), p, profiles)
    p['calibration_probes'].pop()
    p['frame_input']['path'] = 'endpoint.json'
    with pytest.raises(ContractError, match='different inputs'):
        campaign_probe.calibrate(revised, coverage.read_json(revised.config_path), p, profiles)


def seal_stage_fixture(req, result):
    from oxyformer.execution.integrity import publish_result
    root = Path(req.output_dir)
    write(root / '_execution/task.json', coverage.read_json(req.task_path))
    write(root / '_execution/config.json', coverage.read_json(req.config_path))
    (root / '_execution/request.json').write_text(req.to_json())
    sealed = publish_result(root, result)
    assert sealed.status == 'pass', sealed.message
    return root


def admission_fixture(screening_request, tmp_path, *, setup=0., probe_lock_hash=None, setup_draws=None, mutate_receipt=None):
    req = screening_lock_request(screening_request, tmp_path)
    # The small-output bound reads publication manifests, as the real runner does.
    estimate_root = Path(screening_request.output_dir)
    from oxyformer.contracts import StageResult
    from oxyformer.provenance import ArtifactRecord, ArtifactLineage
    from oxyformer.execution.integrity import publish_result
    lineage = ArtifactLineage(source_hashes=(screening_request.config_hash,), unit_ids=(screening_request.content_hash,),
        parent_hashes=(), split_hash=None, config_hash=screening_request.config_hash,
        model_hash=None, environment=(("fixture", "synthetic"),), seed=None, parameter_count=None)
    estimate_result = StageResult(request_hash=screening_request.content_hash, status='pass', message='synthetic estimate',
        artifacts=tuple(ArtifactRecord(path=f.name, sha256=file_hash(f), kind='fixture', lineage=lineage)
                        for f in sorted(estimate_root.glob('*.json'))))
    seal_stage_fixture(screening_request, estimate_result)
    # Refresh request hashes after sealing adds execution records to the producer.
    cfg = coverage.read_json(req.config_path)
    deps = {k: Path(v) for k, v in cfg['dependencies'].items()}
    lock_root = seal_stage_fixture(req, campaign.run_stage(req))
    deps['campaign-lock'] = lock_root
    post = probe_request_from_screening(req, tmp_path / 'post-probe', mode='admission', lock_root=lock_root)
    probe_value = campaign_probe.probe(post, coverage.read_json(post.config_path), coverage.read_json(post.task_path), measurer=lambda: setup)
    lock = coverage.read_json(lock_root / 'recipe_lock.json')
    refs = {}
    for key, depth in (('fresh', 0), ('resumed_1', 1), ('resumed_2', 2)):
        value = deepcopy(probe_value)
        values = setup_draws[key] if setup_draws is not None else [setup, setup]
        value.update(mode='admission' if depth == 0 else 'admission-resumed', depth=depth,
            setup_draws=values, setup_seconds=min(values))
        if probe_lock_hash is not None:
            value['lock_hash'] = probe_lock_hash
        if depth:
            shape = lock['budget']['expected_predecessor_shapes'][value['scenario']]
            value.update(verification={'reuse_miss': False, 'ancestors_reused': 10},
                predecessor_shape=campaign_probe.compare_publication_shape(deepcopy(shape), deepcopy(shape), synthetic=True))
        if mutate_receipt is not None:
            mutate_receipt(key, value)
        unit = 'post-probe-' + key
        deps[unit] = sealed_json_fixture(tmp_path / unit, unit, 'campaign-setup-probe', {'setup_probe.json': value})
        refs[key] = {'dependency': unit, 'path': 'setup_probe.json'}
    task = {'id': 'campaign-admit', 'stage': 'campaign-admit', 'parameters': {
        'lock_input': {'dependency': 'campaign-lock', 'path': 'recipe_lock.json'},
        'probe_inputs': refs}}
    admitted = request(tmp_path / 'admit', task, deps, cfg['approvals'])
    return admitted, lock_root


def test_admission_is_only_authority_and_never_mutates_lock(screening_request, tmp_path):
    req, root = admission_fixture(screening_request, tmp_path)
    before = {p.name: file_hash(p) for p in root.glob('*.json')}
    result = campaign_probe.run_stage(req)
    assert result.status == 'pass', result.message
    result.verify(req)
    receipt = coverage.read_json(Path(req.output_dir) / 'admission.json')
    assert receipt['admitted'] is True
    assert receipt['lock_hash'] == file_hash(root / 'recipe_lock.json')
    assert coverage.read_json(root / 'gate.json')['admitted'] is False
    assert before == {p.name: file_hash(p) for p in root.glob('*.json')}
    plans = coverage.read_json(root / 'expanded_units.json')['plans']
    assert all(plan['spec']['inputs'] == plans[0]['spec']['inputs'] for plan in plans)
    for plan in plans:
        for task in plan['tasks']:
            assert task['admission_receipt'] == {'dependency': 'campaign-admit', 'path': 'admission.json'}
            assert task['needs']['campaign-admit'] == ['admission.json']
    # Resume predecessor dependencies differ, while the admitted spec inputs agree.
    assert plans[0]['tasks'][0]['needs'] != plans[0]['tasks'][1]['needs']


@pytest.mark.parametrize('slow_draw', [0, 1])
def test_admission_uses_three_minima_and_ignores_one_slow_repeat(screening_request, tmp_path, slow_draw):
    draws = {key: [value, value] for key, value in [('fresh', 20.), ('resumed_1', 30.), ('resumed_2', 40.)]}
    for values in draws.values():
        values[slow_draw] = 20000.
    req, lock_root = admission_fixture(screening_request, tmp_path, setup_draws=draws)
    result = campaign_probe.run_stage(req)
    assert result.status == 'pass', result.message
    receipt = coverage.read_json(Path(req.output_dir) / 'admission.json')
    measured = receipt['admission_measurements']
    assert measured['setup_draws'] == draws
    assert measured['setup_minima'] == {'fresh': 20., 'resumed_1': 30., 'resumed_2': 40.}
    assert measured['allowance_seconds'] == 100.
    assert measured['projection']['horizon'] == 0 and measured['projection']['growth'] == 10.


@pytest.mark.parametrize('fault', ['one_draw', 'wrong_depth', 'wrong_estimate', 'reuse_miss', 'shape_reference', 'large_shape'])
def test_admission_refuses_incomplete_or_mismatched_resumed_receipts(screening_request, tmp_path, fault):
    def mutate(key, value):
        if key != 'resumed_2':
            return
        if fault == 'one_draw':
            value['setup_draws'].pop()
        elif fault == 'wrong_depth':
            value['depth'] = 1
        elif fault == 'wrong_estimate':
            value['estimate_hash'] = 'f' * 64
        elif fault == 'reuse_miss':
            value['verification']['reuse_miss'] = True
        elif fault == 'shape_reference':
            value['predecessor_shape']['expected']['archive_bytes'] += 1
        else:
            value['predecessor_shape']['observed']['archive_bytes'] *= 2
    req, _ = admission_fixture(screening_request, tmp_path, mutate_receipt=mutate)
    result = campaign_probe.run_stage(req)
    assert result.status == 'blocked' and not result.artifacts
    assert not (Path(req.output_dir) / 'admission.json').exists()


def test_admission_requires_all_three_receipt_inputs(screening_request, tmp_path):
    req, _ = admission_fixture(screening_request, tmp_path)
    req = update_task(req, lambda p: p['probe_inputs'].pop('resumed_2'))
    result = campaign_probe.run_stage(req)
    assert result.status == 'blocked' and 'three admission probe receipts' in result.message


@pytest.mark.parametrize('observed,status', [(110, 'matched'), (111, 'admission-shape-exceeded'),
    (150, 'admission-shape-exceeded'), (151, 'admission-shape-mismatch'), (40, 'matched')])
def test_runtime_shape_thresholds_use_one_locked_reference(observed, status):
    expected = {'archive_bytes': 100, 'record_count': 100}
    for field in expected:
        actual = {**expected, field: observed}
        assert campaign_probe.compare_publication_shape(expected, actual)['status'] == status
        if observed > 110:
            assert campaign_probe.compare_publication_shape(expected, actual, synthetic=True)['status'] == 'admission-shape-mismatch'


@pytest.mark.parametrize('screening_rates', [(200000.,) * 4], indirect=True)
@pytest.mark.parametrize('offsets,stop', [([300., 300.], True),
    ([300., 0., 300., 0., 300.], True), ([200.] * 5, False),
    ([300., 0., 0., 0., 0.], False)])
def test_runtime_setup_safety_stops_before_import_and_collector_refuses(
        screening_request, tmp_path, monkeypatch, offsets, stop):
    """Measured setups enter real sealed budgets; successors authenticate them."""
    from oxyformer.execution.integrity import FINGERPRINT, RESULT
    from test_coverage import chain_request
    req = screening_lock_request(screening_request, tmp_path)
    assert campaign.run_stage(req).status == 'pass'
    root = Path(req.output_dir)
    lock = coverage.read_json(root / 'recipe_lock.json')
    plan = next(p for p in coverage.read_json(root / 'expanded_units.json')['plans']
        if p['spec']['work'][0]['parameters']['scenario']['name'] == 'null_effect')
    cfg = coverage.read_json(req.config_path)
    deps = {k: Path(v) for k, v in cfg['dependencies'].items()}
    deps['campaign-lock'] = root
    allowance = 3000.
    deps['campaign-admit'] = sealed_json_fixture(tmp_path / 'admitted', 'campaign-admit', 'campaign-admit',
        {'admission.json': {'admitted': True, 'lock_hash': coverage.digest(lock), 'estimate_hash': lock['estimate_hash'],
            'setup_policy': 'three-receipt-minima-v1', 'admission_measurements': {
                'setup_minima': dict.fromkeys(('fresh', 'resumed_1', 'resumed_2'), allowance - 60),
                'allowance_seconds': allowance}}})
    clock = [0.]
    monkeypatch.setattr(coverage.time, 'monotonic', lambda: clock[0])
    original_prepare = coverage.prepare_batch
    setups = [2811., *[allowance + offset for offset in offsets], allowance]
    def prepare(request):
        prepared = original_prepare(request)
        clock[0] += setups[prepared[0]['continuation']['step']]
        return prepared
    monkeypatch.setattr(coverage, 'prepare_batch', prepare)
    def execute(draw, *args, **kwargs):
        clock[0] += 1.
        return {**successful_records(1)[1][0], 'draw': draw, 'wall_seconds': 1.}
    monkeypatch.setattr(coverage, 'execute_draw', execute)
    original_read = coverage._read_continuation
    imports = []
    def read(*args, **kwargs):
        imports.append(1)
        return original_read(*args, **kwargs)
    monkeypatch.setattr(coverage, '_read_continuation', read)
    for step, task in enumerate(plan['tasks'][:len(setups)]):
        leaf = chain_request(tmp_path / task['id'], task, deps)
        config = coverage.read_json(leaf.config_path)
        config['approvals'] = cfg['approvals']
        write(Path(leaf.config_path), config)
        leaf = replace(leaf, config_hash=file_hash(leaf.config_path))
        prior_imports = len(imports)
        result = run_coverage(leaf)
        leaf_root = Path(leaf.output_dir)
        if stop and step == len(setups) - 1:
            assert result.status == 'blocked' and 'allowance-exceeded' in result.message
            assert len(imports) == prior_imports and not (leaf_root / 'repetitions').exists()
            diagnostic = coverage.read_json(leaf_root / coverage.BUDGET_RECORD)
            assert diagnostic['diagnostic_only'] and diagnostic['status'] == 'allowance-exceeded'
            assert not (leaf_root / FINGERPRINT).exists()
            # Failed worker controls are diagnostic, as in the normal runner.
            write(leaf_root / '_execution/task.json', task)
            (leaf_root / '_execution/request.json').write_text(leaf.to_json())
            (leaf_root / RESULT).write_text(result.to_json())
            deps[task['id']] = leaf_root
            collector = request(tmp_path / 'stopped-collector', plan['tasks'][-1], deps, cfg['approvals'])
            outcome = campaign.run_stage(collector)
            assert outcome.status == 'blocked' and 'chain stopped: allowance-exceeded' in outcome.message
            assert not (Path(collector.output_dir) / 'gate.json').exists()
            assert not (Path(collector.output_dir) / FINGERPRINT).exists()
        else:
            assert result.status == 'pass', result.message
            budget = coverage.read_json(leaf_root / coverage.BUDGET_RECORD)
            assert budget['leaves'][-1]['setup_seconds'] == setups[step]
            assert budget['leaves'][-1]['setup_safety']['measured']['overrun'] == (step > 0 and setups[step] > allowance + 281.1)
            deps[task['id']] = leaf_root


@pytest.mark.parametrize('fault', ['slow', 'wrong_lock', 'oversized', 'near_window', 'unequal_specs'])
def test_admission_refusals_leave_sized_lock_reusable(screening_request, tmp_path, monkeypatch, fault):
    setup = 5000. if fault == 'slow' else 13000. if fault == 'near_window' else 0.
    req, root = admission_fixture(screening_request, tmp_path, setup=setup,
        probe_lock_hash='e' * 64 if fault == 'wrong_lock' else None)
    before = {p.name: file_hash(p) for p in root.glob('*.json')}
    if fault == 'oversized':
        monkeypatch.setattr(campaign_probe, 'SMALL_OUTPUT_BYTES', 1)
    if fault == 'unequal_specs':
        original = campaign_probe.read_json
        def read(path):
            value = original(path)
            if Path(path).name == 'expanded_units.json':
                value['plans'][1]['spec']['inputs']['different'] = ['result.json']
            return value
        monkeypatch.setattr(campaign_probe, 'read_json', read)
    result = campaign_probe.run_stage(req)
    assert result.status == 'blocked', result.message
    assert not result.artifacts and not list(Path(req.output_dir).iterdir())
    assert before == {p.name: file_hash(p) for p in root.glob('*.json')}
    if fault in ('slow', 'near_window', 'oversized'):
        assert 'measured setup' in result.message and 'counts' in result.message


@pytest.mark.parametrize('fault', ['absent', 'false', 'wrong_lock'])
def test_screening_consumer_requires_matching_admission(screening_request, tmp_path, fault):
    req = screening_lock_request(screening_request, tmp_path)
    result = campaign.run_stage(req)
    assert result.status == 'pass'
    root = Path(req.output_dir)
    lock = coverage.read_json(root / 'recipe_lock.json')
    config = coverage.read_json(req.config_path)
    deps = {k: Path(v) for k, v in config['dependencies'].items()}
    task = {'id': 'leaf', 'stage': 'coverage'}
    if fault != 'absent':
        task['admission_receipt'] = lock['admission_receipt']
        deps['campaign-admit'] = sealed_json_fixture(tmp_path / 'receipt', 'campaign-admit', 'campaign-admit',
            {'admission.json': {'admitted': fault != 'false', 'lock_hash': 'f' * 64 if fault == 'wrong_lock' else coverage.digest(lock),
                                'estimate_hash': lock['estimate_hash']}})
    consumer = request(tmp_path / 'consumer', task, deps, config['approvals'])
    with pytest.raises(ContractError, match='admission receipt'):
        campaign_probe.require_admission(consumer, coverage.read_json(consumer.config_path), task, lock)


def sizing_replay():
    """Recompute all claims from hashed campaign budget fixtures, not snapshots."""
    import gzip
    import math
    import numpy as np
    from oxyformer.training.fit import PHASE_UNIT_SECONDS
    fixture_path = ROOT / 'tests/fixtures/sizing/replay.json.gz'
    fixture = json.loads(gzip.decompress(fixture_path.read_bytes()))
    budget = fixture['budget']
    assert coverage.digest(budget) == fixture['budget_sha256']
    profiles = budget['profiles']
    calibration = []
    for row in fixture['calibration']:
        selected = fixture['scenarios'][row['scenario']]['record']
        prefix = selected['leaves'][:row['own_leaf']['step'] + 1]
        assert prefix[-1] == row['own_leaf']
        calibration.append({**selected, 'leaves': prefix, 'counts': {'success': row['success']},
            'record_hash': row['sha256'], 'budget_generation': row['own_leaf']['leaf_budget'].get('formula', 'pre-ARC-2612')})
    stats = campaign_probe.calibration_statistics(calibration, profiles)
    for row in fixture['profile_records']:
        assert coverage.digest(row['record']) == row['sha256']
        for leaf in row['record']['leaves']:
            contract = leaf['leaf_budget']
            assert 'formula' not in contract and contract['safety_factor'] == 1.0
            assert coverage.validate_leaf_budget(contract) == contract
    assert sum(len(r['record']['leaves']) for r in fixture['profile_records']) == 160
    result = {'captured_at': fixture['captured_at'], 'fixture_sha256': file_hash(fixture_path),
        'calibration': stats, 'scenarios': {}, 'baseline': {}, 'full_window_seconds': 14100,
        'profile_exception_leaves': 160}
    for name, selected in fixture['scenarios'].items():
        assert coverage.digest(selected['record']) == selected['sha256']
        record = selected['record']
        leaves = record['leaves']
        n = len(leaves)
        completed = record['counts']['success'] >= 1
        assert fixture['choices'][name]['rule'] == ('first_sealed_repetition' if completed else 'last_sealed_leaf')
        profile = deepcopy(profiles[name])
        demand = profile['seconds_per_repetition'] + profile['publication_verification_seconds']
        delivered = math.fsum(v['compute_seconds'] for v in leaves)
        gross = delivered / n
        net = demand / n if completed else None
        contract = leaves[-1]['leaf_budget']
        capacity = contract['compute_seconds']
        old_usable = capacity - max(0., stats['setup_seconds'] - contract['overheads']['setup_seconds']) - campaign.checkpoint_margin_seconds() - max(PHASE_UNIT_SECONDS.values()) - stats['resume_overhead_seconds']
        assert old_usable > 0
        if completed:
            assert old_usable <= net
        old_counts = {str(f): math.ceil(f * demand / old_usable) + 1 for f in (1., 1.25)}
        required = n + 1 if completed else math.ceil(demand / gross) + 1
        assert old_counts['1.0'] >= required
        assert old_counts['1.25'] >= old_counts['1.0']
        # Exact pre-ARC-2612 implementation, using each scenario's own locked overheads.
        old_capacity = (14100 - 300) / budget['profile_safety_factor'] - math.fsum(profile['leaf_overheads'].values())
        baseline_repetitions = budget['per_scenario'][name]['repetitions']
        baseline = math.ceil(profile['seconds_per_repetition'] * baseline_repetitions /
                             (old_capacity - campaign.checkpoint_margin_seconds())) + 1
        assert baseline == budget['per_scenario'][name]['leaves']
        result['baseline'][name] = baseline
        profile['leaf_overheads']['setup_seconds'] = stats['setup_seconds']
        for phase in coverage.OVERHEAD_PHASES[1:]:
            profile['leaf_overheads'][phase] = max(profile['leaf_overheads'][phase], stats['statistics'][phase]['max'])
        profile['resume_overhead_seconds'] = stats['resume_overhead_seconds']
        full_capacity = coverage.usable_leaf_compute(14100, profile['leaf_overheads'],
            resume_overhead_seconds=profile['resume_overhead_seconds']) - profile['resume_overhead_seconds']
        fits = {}
        for factor, minimum in ((1.25, 2), (1., 3)):
            assert campaign.slice_count(profile, minimum, 14100, factor) <= 40
            maximal = max(n for n in range(1, 26) if campaign.slice_count(profile, n, 14100, factor, enforce_limit=False) <= 40)
            fits[str(factor)] = {'repetitions': maximal, 'leaves': campaign.slice_count(profile, maximal, 14100, factor)}
        excess = (delivered - profile['seconds_per_repetition']) / n
        if name == 'null_effect':
            assert coverage.RESUME_OVERHEAD_FLOOR_SECONDS >= excess
        result['scenarios'][name] = {'fixture': fixture['choices'][name],
            'DELIVERED': delivered, 'CAPACITY': capacity, 'DEMAND': demand,
            'REALIZED_GROSS': gross, 'REALIZED_NET': net, 'per_leaf_excess': excess,
            'usable_old': old_usable, 'old_window_counts': old_counts,
            'full_window_usable_net': full_capacity, 'maximal_month1_counts': fits}
    assert result['baseline'] == {'linear_reference': 22, 'sign_changing': 24, 'null_effect': 39, 'nonlinear': 40}
    return result


def test_sealed_campaign_replay():
    result = sizing_replay()
    print(json.dumps(result, indent=2))


@pytest.mark.parametrize('factor', [1., 2.])
def test_measurement_reader_accepts_old_generation_without_budget_authority(tmp_path, monkeypatch, factor):
    from test_campaign import setup_chain_lock, profile_directory, request
    req, _ = setup_chain_lock(tmp_path)
    import shutil
    root = tmp_path / 'old-producer'
    shutil.copytree(profile_directory(req), root)
    task = coverage.read_json(root / '_execution/task.json')
    contract = coverage.leaf_budget(task['parameters']['wall_seconds'], factor, dict.fromkeys(coverage.OVERHEAD_PHASES, 0.))
    contract.pop('formula')
    contract['compute_seconds'] /= factor
    task['parameters']['leaf_budget'] = contract
    # Model an honestly sealed pre-change producer; restore the current loader
    # before consuming its publication.
    with monkeypatch.context() as old:
        old.setattr(coverage, 'task_leaf_budget', lambda task: contract)
        timing = coverage.read_json(root / 'timing.json')
        for leaf in timing['leaves']:
            leaf['leaf_budget'] = contract
        write(root / 'timing.json', timing)
        seal_budget_fixture(root, task['id'], task)
    cfg = coverage.read_json(req.config_path)
    consumer = request(tmp_path / 'generation-consumer', coverage.read_json(req.task_path),
        {k: root if k == task['id'] else Path(v) for k,v in cfg['dependencies'].items()}, cfg['approvals'])
    measured = coverage.read_leaf_measurements(root / coverage.BUDGET_RECORD)
    assert measured['budget_generation'] == 'pre-ARC-2612'
    if factor == 1.:
        assert coverage.load_budget(consumer, coverage.read_json(consumer.config_path), task['id'])['all_successful']
    else:
        with pytest.raises(ContractError, match='pre-ARC-2612 window formula'):
            coverage.load_budget(consumer, coverage.read_json(consumer.config_path), task['id'])
    (root / coverage.BUDGET_RECORD).write_text('{}')
    with pytest.raises(ContractError, match='record hash mismatch'):
        coverage.read_leaf_measurements(root / coverage.BUDGET_RECORD)


@pytest.mark.parametrize('screening_rates', [(340000.,) * 4], indirect=True)
def test_fallback_headroom_limit_names_unpadded_count_and_bounded_recovery(screening_request, tmp_path):
    base = screening_request
    config, task = coverage.read_json(base.config_path), coverage.read_json(base.task_path)
    p = task['parameters']
    p.pop('setup_calibration')
    pre = probe_request_from_screening(base, tmp_path / 'pre')
    digests = campaign_probe.probe(pre, coverage.read_json(pre.config_path), coverage.read_json(pre.task_path), measurer=lambda: 0.)['input_digests']
    deps = {k: Path(v) for k,v in config['dependencies'].items()}
    p['calibration_probes'] = []
    for i, seconds in enumerate((940., 100.)):
        unit = 'bounded-calibration-' + str(i)
        deps[unit] = sealed_json_fixture(tmp_path / unit, unit, 'campaign-setup-probe',
            {'setup_probe.json': {'mode': 'calibration', 'setup_seconds': seconds, 'input_digests': digests}})
        p['calibration_probes'].append({'dependency': unit, 'path': 'setup_probe.json'})
        req = request(tmp_path / ('estimate-' + str(i)), task, deps, config['approvals'])
        result = campaign_estimate.run_stage(req)
        if i == 0:
            assert result.status == 'blocked' and not result.artifacts
            assert '1.5x calibration headroom' in result.message
            assert 'count without headroom: 40' in result.message
            assert 'second calibration probe' in result.message
        else:
            assert result.status == 'pass', result.message
            estimate = coverage.read_json(Path(req.output_dir) / 'budget_estimate.json')
            assert estimate['calibration']['chosen_seconds'] == 100
            assert len(estimate['calibration']['draws']) == 2
            assert not estimate['calibration']['second_probe_recommended']
            assert all(r['repetitions'] == 1 and r['leaves'] <= 40 for r in estimate['per_scenario'].values())


def test_estimate_and_lock_record_historical_calibration_provenance(screening_request, tmp_path):
    """Sealed historical stamps are copied exactly, independently of profiles."""
    import shutil
    base = screening_request
    config, task = coverage.read_json(base.config_path), coverage.read_json(base.task_path)
    deps = {k: Path(v) for k, v in config['dependencies'].items()}
    expected = []
    refs = []
    for i, ref in enumerate(task['parameters']['setup_calibration']):
        unit = f'historical-calibration-{i}'
        root = tmp_path / unit
        shutil.copytree(deps[ref['dependency']], root)
        timing = coverage.read_json(root / 'timing.json')
        timing.update(scientific_fingerprint={'algorithm': 'tracked-science-v2', 'sha256': str(i) * 64},
                      environment_hash=str(i + 4) * 64)
        write(root / 'timing.json', timing)
        producer_task = coverage.read_json(root / '_execution/task.json')
        producer_task['id'] = unit
        seal_budget_fixture(root, unit, producer_task)
        record = coverage.read_leaf_measurements(root / coverage.BUDGET_RECORD)
        assert record['profile']['scientific_fingerprint'] != STAMPS['scientific_fingerprint']
        assert record['profile']['environment_hash'] != STAMPS['environment_hash']
        expected.append({'record_hash': file_hash(root / coverage.BUDGET_RECORD), 'task_id': unit,
            'budget_generation': record['budget_generation'],
            'scientific_fingerprint': record['profile']['scientific_fingerprint'],
            'environment_hash': record['profile']['environment_hash']})
        deps[unit] = root
        refs.append({'dependency': unit, 'path': coverage.BUDGET_RECORD})
    task['parameters']['setup_calibration'] = refs
    estimate_req = request(tmp_path / 'provenance-estimate', task, deps, config['approvals'])
    lock_req = screening_lock_request(estimate_req, tmp_path / 'provenance-lock')
    outcome = campaign.run_stage(lock_req)
    assert outcome.status == 'pass', outcome.message
    outcome.verify(lock_req)
    estimate = coverage.read_json(Path(estimate_req.output_dir) / 'budget_estimate.json')
    budget = coverage.read_json(Path(lock_req.output_dir) / 'budget.json')
    assert estimate['calibration']['records'] == budget['calibration']['records'] == expected


@pytest.mark.parametrize('screening_rates', [(20000.,) * 4], indirect=True)
@pytest.mark.parametrize('values,measured,refused', [([5599., 5792.], False, False),
    ([5599., 5792., 6320.], True, True), ([3500.] * 3, True, False)])
def test_estimate_and_lock_apply_same_lineage_projection(screening_request, tmp_path, values, measured, refused):
    """Canonical sealed calibration records reach both public sizing paths."""
    base = screening_request
    config, task = coverage.read_json(base.config_path), coverage.read_json(base.task_path)
    deps = {k: Path(v) for k, v in config['dependencies'].items()}
    source = deps[task['parameters']['profiles']['null_effect']['dependency']]
    timing = coverage.read_json(source / 'timing.json')
    source_task = coverage.read_json(source / '_execution/task.json')
    history = []
    refs = []
    for step, setup in enumerate([2811., *values]):
        unit = f'calibrated-chain-{step}'
        history.append({'task_id': unit, 'step': step, 'setup_seconds': setup, 'compute_seconds': 1.,
            'import_seconds': 0., 'export_seconds': 0., 'publication_seconds': 1., 'wall_seconds': setup + 2.})
        root = tmp_path / unit
        result = coverage.read_json(source / 'result.json')
        result['records'] = []
        write(root / 'result.json', result)
        write(root / 'timing.json', {**timing, 'leaves': deepcopy(history), 'complete': False,
            'all_successful': False, 'complete_repetition_seconds': [],
            'wall_seconds': sum(row['wall_seconds'] for row in history),
            'setup_seconds': sum(row['setup_seconds'] for row in history)})
        producer = deepcopy(source_task)
        producer.update(id=unit, continuation={'owner': 'calibrated-chain', 'step': step,
            'predecessor': f'calibrated-chain-{step-1}' if step else None})
        seal_budget_fixture(root, unit, producer)
        deps[unit] = root
        refs.append({'dependency': unit, 'path': coverage.BUDGET_RECORD})
    task['parameters']['setup_calibration'] = refs
    req = request(tmp_path / 'projected-estimate', task, deps, config['approvals'])
    if refused:
        result = campaign_estimate.run_stage(req)
        assert result.status == 'blocked' and 'setup allowance does not fit' in result.message
        for field in ('fresh', 'points', 'growth', 'projected', 'longest_chain', 'ceiling'):
            assert field in result.message
        assert '360.5' in result.message
        lock_task = deepcopy(task)
        lock_task['stage'] = 'campaign-lock'
        with pytest.raises(ContractError, match='setup allowance does not fit'):
            campaign.build_screening_lock(req, coverage.read_json(req.config_path), lock_task)
        assert not list(Path(req.output_dir).iterdir())
    else:
        lock_req = screening_lock_request(req, tmp_path / 'projected')
        result = campaign.run_stage(lock_req)
        assert result.status == 'pass', result.message
        estimate = coverage.read_json(Path(req.output_dir) / 'budget_estimate.json')
        budget = coverage.read_json(Path(lock_req.output_dir) / 'budget.json')
        projection = estimate['calibration']['setup_projection']
        assert projection == budget['calibration']['setup_projection']
        assert projection['status'] == ('measured' if measured else 'unmeasured')
        assert projection['chains'][0]['resumed'] == values
        if measured:
            assert projection['chains'][0]['growth'] == 0.
            assert projection['chains'][0]['allowance'] == 3560.


# Exact r8 profile / r5 lock dimensions; all bytes below are synthetic.
ADMISSION_SHAPE_CASES = [
    (66058240, 0, False, 'matched'),
    (66058240, 1, False, 'matched'),
    (66058240, 2, False, 'admission-shape-mismatch'),
    (72664064, 0, False, 'matched'),  # exactly 10% is accepted
    (72664065, 0, False, 'admission-shape-mismatch'),
    (66058240, 0, True, 'matched'),  # additional declared builder receipt
]


@pytest.fixture
def admission_shape_fixture(tmp_path, request):
    from oxyformer.execution.integrity import FINGERPRINT, RESULT, fingerprint_tree
    archive_bytes, extra_payloads, extra_receipt, status = request.param
    fixture = coverage.read_json(ROOT / 'tests/fixtures/admission_shape_r5.json')
    profile = {'setup_profile_sources': {'1': fixture['profile_source']},
        'assumed_usable_per_leaf': 100., 'seconds_per_repetition': 100.}
    expected = campaign_probe.expected_predecessor_shape(profile, fixture['campaign_outputs'], draw_count=1)
    assert expected == fixture['expected'] == {'archive_bytes': 66058240, 'record_count': 18}
    predecessor = tmp_path / 'predecessor'
    for name in fixture['builder_records']:
        write(predecessor / name, {})
    with (predecessor / 'continuation.tar').open('wb') as stream:
        stream.truncate(archive_bytes)
    for i in range(extra_payloads):
        write(predecessor / f'synthetic-records/extra-{i}.json', {'payload': i})
    outputs = list(fixture['builder_outputs'])
    if extra_receipt:
        outputs.append('builder-diagnostic.json')
        write(predecessor / outputs[-1], {'receipt': True})
    write(predecessor / '_execution/task.json', {'stage': 'campaign-setup-probe',
        'parameters': {'mode': 'admission-predecessor', 'depth': 1}, 'outputs': outputs})
    estimate = tmp_path / 'estimate'
    write(estimate / 'budget_estimate.json', {})
    lock_root = tmp_path / 'lock'
    lock = {'estimate_input': {'dependency': 'estimate', 'path': 'budget_estimate.json'},
        'estimate_hash': file_hash(estimate / 'budget_estimate.json'),
        'budget': {'expected_predecessor_shapes': {'null_effect': expected}}}
    lock_path = write(lock_root / 'recipe_lock.json', lock)
    needs = {'lock': ['recipe_lock.json', 'expanded_units.json'], 'estimate': ['budget_estimate.json']}
    write(lock_root / 'expanded_units.json', {'lock_hash': file_hash(lock_path),
        'plans': [{'tasks': [{'needs': needs, 'parameters': {'scenario': {'name': 'null_effect'}}}]}]})
    receipt = {'mode': 'admission-predecessor', 'depth': 1, 'scenario': 'null_effect',
        'lock_hash': file_hash(lock_path), 'estimate_hash': lock['estimate_hash'],
        'dependency_receipt': {'snapshot': {}}}
    write(predecessor / 'setup_probe.json', receipt)
    tree = fingerprint_tree(predecessor)
    assert campaign_probe.publication_shape(tree) == {
        'archive_bytes': archive_bytes, 'record_count': 20 + extra_payloads + int(extra_receipt)}
    write(predecessor / FINGERPRINT, {'entries': {k: v for k, v in tree.items() if k not in (FINGERPRINT, RESULT)}})
    task = {'id': 'resumed', 'stage': 'campaign-setup-probe',
        'needs': {**needs, 'predecessor': ['setup_probe.json', *campaign_probe.PROFILE_PAYLOADS]},
        'recipe_lock': {'sha256': file_hash(lock_path)},
        'parameters': {'mode': 'admission-resumed', 'depth': 1, 'scenario': 'null_effect',
            'lock_input': {'dependency': 'lock', 'path': 'recipe_lock.json'},
            'predecessor_input': {'dependency': 'predecessor', 'path': 'setup_probe.json'}}}
    deps = {'lock': lock_root, 'estimate': estimate, 'predecessor': predecessor}
    # The focused fixture supplies publication metadata; full runner authentication
    # remains covered by test_two_depth_probe_pipeline_uses_fresh_and_grand_predecessor_receipts.
    from test_campaign import request as make_request
    req = make_request(tmp_path / 'probe', task, deps)
    archive = str(predecessor / 'continuation.tar')
    req = replace(req, dependency_paths=(*req.dependency_paths, archive),
        dependency_hashes=(*req.dependency_hashes, file_hash(archive)))
    config = coverage.read_json(req.config_path)
    config['verification_summary'] = {'ancestors_reused': 1, 'reuse_miss': False}
    return req, config, task, deps, tree, receipt, status


@pytest.mark.parametrize('admission_shape_fixture', ADMISSION_SHAPE_CASES, indirect=True)
def test_admit_side_synthetic_shape_matches_r5_lock(admission_shape_fixture, monkeypatch):
    req, config, task, deps, tree, receipt, status = admission_shape_fixture
    # Isolate receipt authentication, not the shape extraction or comparison.
    monkeypatch.setattr(campaign_probe, 'read_receipt', lambda *args: receipt)
    if status == 'admission-shape-mismatch':
        with pytest.raises(ContractError, match=status):
            campaign_probe.probe(req, config, task, measurer=lambda: 0.)
        return
    shape = campaign_probe.probe(req, config, task, measurer=lambda: 0.)['predecessor_shape']
    assert shape['status'] == status
    assert shape['expected'] == {'archive_bytes': 66058240, 'record_count': 18}
    observed_records = sum(name.startswith('synthetic-records/extra-') for name in tree) + 18
    assert shape['observed'] == {'archive_bytes': tree['continuation.tar']['size'], 'record_count': observed_records}
    assert shape['relative_difference'] == {
        'archive_bytes': (tree['continuation.tar']['size'] - 66058240) / 66058240,
        'record_count': (observed_records - 18) / 18}


def test_month2_decision_json_and_key_refusals():
    approvals = read_mapping(ROOT / 'configs/approvals.yaml')
    value = campaign.month2_decision(approvals)
    assert value['gpu_hours_ceiling'] is None
    assert json.loads(json.dumps(value)) == value
    assert value['diagnostic_campaign']['repetitions'] == 20
    for key in value:
        changed = deepcopy(approvals)
        del changed['owner_decisions']['month2_followon'][key]
        with pytest.raises(ContractError, match='month2_followon.*' + key):
            campaign.month2_decision(changed)
    for key in value['diagnostic_campaign']:
        changed = deepcopy(approvals)
        del changed['owner_decisions']['month2_followon']['diagnostic_campaign'][key]
        with pytest.raises(ContractError, match='month2_followon.*' + key):
            campaign.month2_decision(changed)


@pytest.mark.parametrize('key,value', [('scenarios', []), ('scenarios', ['sign_changing'] * 2),
    ('scenarios', ['unknown']), ('repetitions', True), ('repetitions', 0), ('recipe', {}),
    ('partition', 'idle'), ('gpu_hours_per_leaf_max', 4.01), ('gpu_hours_per_leaf_max', float('nan')),
    ('leaves_per_chain_max', 41), ('parallel_chains', False), ('certifies_release_criteria', True),
    ('purpose', ''), ('draws_continue_from', '')])
def test_month2_refuses_malformed_campaign(key, value):
    approvals = read_mapping(ROOT / 'configs/approvals.yaml')
    approvals['owner_decisions']['month2_followon']['diagnostic_campaign'][key] = value
    with pytest.raises(ContractError, match='month2_followon.*' + key):
        campaign.month2_decision(approvals)


@pytest.mark.parametrize('value', [0, True, 17, 1.5])
def test_month2_concurrency_ceiling(value):
    approvals = read_mapping(ROOT / 'configs/approvals.yaml')
    approvals['owner_decisions']['month2_followon']['gpu_concurrency_max']['preemptible'] = value
    with pytest.raises(ContractError, match='gpu_concurrency_max'):
        campaign.month2_decision(approvals)


def test_diagnostic_arithmetic_keeps_twenty_independent_chains_and_no_total_cap():
    profile = {'seconds_per_repetition': 100000., 'publication_verification_seconds': 400.,
        'leaf_overheads': {'setup_seconds': 100., 'import_seconds': 2., 'export_seconds': 3., 'publication_seconds': 5.}}
    decision = campaign.month2_decision()
    row = campaign.diagnostic_sizing({'sign_changing': profile}, decision, 14100)['sign_changing']
    assert row['repetitions'] == row['chains'] == 20
    assert row['per_chain']['repetitions'] == 1
    assert row['leaves'] == 20 * row['per_chain']['leaves']
    assert row['gpu_hours'] > 640
    assert row['gpu_hours'] == 20 * row['per_chain']['gpu_hours']
    profile['seconds_per_repetition'] = 1e8
    with pytest.raises(ContractError, match='chain requires .* leaves > 40'):
        campaign.diagnostic_sizing({'sign_changing': profile}, decision, 14100)


def test_diagnostic_emitter_subset_dependencies_and_screening_unchanged(built):
    approvals, (endpoint, frame, _, _) = built
    approvals = approvals['approvals']
    before = json.dumps(real_frame.screening_profile_tasks(endpoint, frame, approvals), sort_keys=True)
    manifest = real_frame.diagnostic_profile_tasks(endpoint, frame, approvals, campaign_id='m2-diagnostic',
        evaluation_namespace='m2-diagnostic-new', prior_locks=[{'dependency': 'old-lock', 'path': 'recipe_lock.json'}],
        cpu_wall_hours_budget=10000)
    assert manifest['gpu_concurrency_max'] == 16
    profiles = manifest['tasks'][:-2]
    assert len(profiles) == 40
    assert {t['parameters']['scenario']['name'] for t in profiles} == {'sign_changing'}
    estimate, lock = manifest['tasks'][-2:]
    assert estimate['parameters']['profile_safety_factor'] == 2.0
    assert estimate['id'] in lock['needs'] and 'old-lock' in lock['needs']
    assert set(estimate['parameters']['profiles']) == {'sign_changing'}
    assert all(t['campaign_mode'] == 'diagnostic' and not t['certifies_release_criteria'] for t in manifest['tasks'])
    assert json.dumps(real_frame.screening_profile_tasks(endpoint, frame, approvals), sort_keys=True) == before


def test_diagnostic_summary_known_bias_is_not_release():
    draws, records = successful_records(20)
    result = campaign.diagnostic_summary(records, draws, null_scenario=True)
    assert result['coverage_pass'] is False and result['coverage_pass_reason'] == 'diagnostic'
    assert result['production_equivalent'] is False
    for method, row in result['diagnostic_summary'].items():
        assert row['n'] == 20 and row['numerical_failures'] == 0
        assert row['bias'] == pytest.approx(row['mean_estimate'] - row['truth'])
        assert row['coverage_lower'] == coverage.binomial_bound(row['coverage_count'], 20, side='lower')
        assert row['rejection_upper'] == coverage.binomial_bound(row['rejections'], 20, side='upper')
        assert row['certifies_release_criteria'] is False
