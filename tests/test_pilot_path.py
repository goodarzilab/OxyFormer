"""Synthetic archive-layout and bounded lifecycle tests; no fitting/network."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tarfile
from unittest.mock import patch

import pytest
import yaml

from oxyformer.provenance import ContractError, canonical_json, file_hash
from oxyformer.validation import campaign, campaign_probe, coverage, pilot, real_frame
from oxyformer.execution.campaign import expand_campaign, validate_plan
from test_attribution_audit import saved_fixture
from test_campaign import request, write
from test_campaign_profiles import (screening_rates, screening_request, diagnostic_request,
    diagnostic_lock_request, real_fixture)
from oxyformer.validation.generators import generate_suite_a

STUDY = {'reference_cap': 32, 'frozen_epochs': 60, 'grid_points': 2, 'seed_count': 1}
ORIGINAL_PROFILE = campaign._profile


@pytest.fixture(autouse=True)
def pilot_publication_store(tmp_path, monkeypatch):
    monkeypatch.setenv('OXYFORMER_PUBLICATION_STORE', str(tmp_path/'.publications'))


@pytest.fixture
def owner(tmp_path, monkeypatch):
    from oxyformer.execution.runner import read_mapping
    approvals = read_mapping(campaign.OWNER_APPROVALS)
    block = {'purpose': 'diagnostic_pilot', 'cells': {'d': STUDY},
        'partitions': ['preemptible', 'standard'], 'repetitions_per_scenario': 5,
        'scenarios': ['sign_changing'], 'certifies_release_criteria': False,
        'gate_se': 'unclustered_influence', 'paired_from': 'm2-diag-A',
        'wandb_waiver': 'synthetic fixture owner waiver; no training'}
    approvals['owner_decisions']['example_pilot'] = deepcopy(block)
    path = tmp_path / 'approvals.yaml'; path.write_text(yaml.safe_dump(approvals))
    monkeypatch.setattr(campaign, 'OWNER_APPROVALS', path)
    declaration = {k: deepcopy(v) for k, v in block.items() if k != 'cells'}
    declaration.update(campaign_mode='example_pilot', cell='d', study=deepcopy(STUDY))
    recipe = {'campaign_mode': 'example_pilot', 'study': deepcopy(STUDY),
        'pilot_declaration': declaration, 'pilot_partition': 'preemptible'}
    return approvals, recipe


@pytest.fixture(scope='module')
def archive_manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp('paired-source')
    entries = []
    # Real archive shape; seed-specific observations and complete fitted parts.
    for index in range(1, 7):
        path = root / ('c' + str(index))
        actual = generate_suite_a
        def generated(*args, **kwargs):
            args = (args[0], replace(args[1], name='sign_changing'), *args[2:])
            return actual(*args, **{**kwargs, 'seed': index + 6})
        with patch('test_attribution_audit.generate_suite_a', generated):
            saved_fixture(path)
        with tarfile.open(path / 'continuation.tar', 'r:') as archive:
            header = json.load(archive.extractfile('coverage.json'))
            states = []
            for state in header['states']:
                state['parts'] = [archive.extractfile('blobs/' + key).read().decode() for key in state['parts']]
                state['sample'] = {name: archive.extractfile('blobs/' + key).read().decode()
                                   for name, key in state['sample'].items()}
                draw = {'repetition_id': coverage.digest(['pair', index]), 'index': index,
                        'seed': index + 6, 'scenario': 'sign_changing'}
                state['record']['draw'] = draw
                states.append(state)
        # The source lock is common across chains, just as in the real manifest.
        (path / 'continuation.tar').unlink()
        coverage._write_continuation(path / 'continuation.tar', states,
            binding=header['binding'], chain={}, task_id='fixture')
        write(path / 'result.json', {'terminal': True, 'campaign_mode': 'diagnostic',
                                    'records': [s['record'] for s in states]})
        entries.append({'chain': 'c' + str(index), 'terminal_leaf_attempt_dir': str(path)})
    manifest = root / 'collected_chains.json'
    write(manifest, {'campaign': 'm2-diag-A', 'tree': '1' * 40,
                     'lock_hash': header['binding']['lock_hash'], 'chains': list(reversed(entries))})
    return manifest


def locked(owner, archive_manifest):
    _, recipe = owner
    pins = pilot.pin_draws(archive_manifest, campaign.pilot_declaration(recipe))
    return {'recipe': recipe, 'paired_draws': pins,
            'batches': [{'draws': [p['draw']]} for p in pins]}


def test_resolver_calls_validator_and_detects_changed_block(owner):
    approvals, recipe = owner
    with patch.object(campaign, 'validate_campaign', wraps=campaign.validate_campaign) as called:
        d = campaign.pilot_declaration(recipe, approvals)
    called.assert_called_once()
    assert d['study'] == STUDY and d['gate_se'] == 'unclustered_influence'
    changed = deepcopy(approvals)
    changed['owner_decisions']['example_pilot']['cells']['d']['frozen_epochs'] = 30
    with pytest.raises(ContractError, match='study'):
        campaign.pilot_declaration(recipe, changed)
    with pytest.raises(ContractError, match='approval'):
        campaign.pilot_declaration(recipe, {'owner_decisions': {}})


@pytest.mark.parametrize('purpose', ['final_coverage', 'anchor', 'refit_audit', 'recipe_ladder'])
def test_nonpilot_stops_and_asks(owner, purpose):
    approvals, recipe = owner
    approvals['owner_decisions']['example_pilot']['purpose'] = purpose
    with pytest.raises(ContractError, match='mandate stop-and-ask'):
        campaign.pilot_declaration(recipe, approvals)


def test_canonical_pins_and_provenance(owner, archive_manifest):
    lock = locked(owner, archive_manifest)
    assert [p['draw_index'] for p in lock['paired_draws']] == [1, 2, 3, 4, 5]
    assert [p['source_chain'] for p in lock['paired_draws']] == ['c1','c2','c3','c4','c5']
    for pin in lock['paired_draws']:
        for key in ('observations_hash','frame_hash','truth_hash','source_lock_hash','source_tree',
                    'seed','scenario','repetition_id','source_chain','draw_index'):
            assert pin[key]
    assert pilot.verify_regeneration(lock)['verified_repetitions'] == [p['repetition_id'] for p in lock['paired_draws']]


@pytest.mark.parametrize('fault', ['missing', 'hash'])
def test_source_chain_failure_never_substitutes(owner, archive_manifest, tmp_path, fault):
    manifest = coverage.read_json(archive_manifest)
    if fault == 'missing':
        next(e for e in manifest['chains'] if e['chain'] == 'c1')['terminal_leaf_attempt_dir'] = str(tmp_path / 'absent')
    else:
        manifest['lock_hash'] = '0' * 64
    path = write(tmp_path / 'manifest.json', manifest)
    with pytest.raises(ContractError, match='chain c1'):
        pilot.pin_draws(path, campaign.pilot_declaration(owner[1]))


@pytest.mark.parametrize('target', ['observations', 'structural_causal_truth'])
def test_regeneration_falsification_precedes_fit(owner, archive_manifest, monkeypatch, target):
    lock = locked(owner, archive_manifest)
    actual = pilot.generate_suite_a
    def tamper(*args, **kwargs):
        sample = actual(*args, **kwargs)
        artifact = getattr(sample, target)
        changed = replace(artifact, y=tuple(None if y is None else y + .01 for y in artifact.y)) if target == 'observations' else replace(artifact, value=artifact.value + .01)
        return replace(sample, **{target: changed})
    monkeypatch.setattr(pilot, 'generate_suite_a', tamper)
    with pytest.raises(ContractError, match='chain c1: regeneration falsification'):
        pilot.verify_regeneration(lock)


def test_tampered_saved_artifact_stops_named_chain(owner, archive_manifest, monkeypatch):
    lock = locked(owner, archive_manifest)
    lock['paired_draws'][0]['truth_hash'] = '0' * 64
    with pytest.raises(ContractError, match='chain c1.*truth mismatch'):
        pilot.verify_regeneration(lock)


@pytest.mark.parametrize('cell', ['d', 'c'])
def test_paired_readout_basis_and_intervals(owner, archive_manifest, cell):
    if cell != 'd':
        approvals, recipe = owner
        approvals['owner_decisions']['example_pilot']['cells'] = {cell:deepcopy(STUDY)}
        recipe['pilot_declaration']['cell'] = cell
        campaign.OWNER_APPROVALS.write_text(yaml.safe_dump(approvals))
    lock = locked(owner, archive_manifest)
    records = []
    for pin in lock['paired_draws']:
        state, _, _, _, _ = pilot.pinned_source(pin)
        record = deepcopy(state['record'])
        for estimate in record['estimates'].values():
            estimate['value'] += .01
            estimate['se_definitions'] = {'unclustered_influence': {'se': .2,
                'variance_target': 'fixed_frame_repetition', 'assumptions': 'synthetic'}}
        records.append(record)
    report, markdown = pilot.paired_readout(lock, records)
    assert len(report['pairs']) == 5
    assert 'exploratory' in report['statement'] and 'n = 5' in markdown
    assert report['certifies_release_criteria'] is False
    for pair in report['pairs']:
        for row in pair['methods'].values():
            assert row['pilot_minus_source']['value'] == pytest.approx(.01)
            assert row['pilot']['basis'].startswith('cell ' + cell + ';')
            assert row['pilot_minus_source']['basis'].startswith('paired cell ' + cell + ' minus cell e;')
            assert set(row['oracle_cells']) == {'oracle','oracle_Q','oracle_g'}
    assert report['summary']['one_step']['pilot_minus_source']['n']['value'] == 5
    def basis(v):
        if isinstance(v, dict):
            for key, child in v.items():
                if type(child) in (int, float):
                    assert key == 'value' and v['basis']
                else: basis(child)
        elif isinstance(v, list):
            for child in v: basis(child)
    basis(report)


@pytest.mark.parametrize('screening_rates', [(100.,) * 4], indirect=True)
def test_pilot_estimate_lock_expansion_and_admission_dispatch(diagnostic_request, owner, archive_manifest, tmp_path, monkeypatch):
    approvals, minimal = owner
    task = coverage.read_json(diagnostic_request.task_path)
    cfg = coverage.read_json(diagnostic_request.config_path)
    task['parameters']['recipe'].update(minimal)
    task['parameters'].update(campaign.screening_metadata(task['parameters']['recipe']), paired_manifest=str(archive_manifest))
    task.update(campaign.screening_metadata(task['parameters']['recipe']))
    req = request(tmp_path / 'pilot-estimate', task, {k: Path(v) for k,v in cfg['dependencies'].items()}, approvals)
    req = diagnostic_lock_request(req, tmp_path)
    outcome = campaign.run_stage(req)
    assert outcome.status == 'pass', outcome.message
    root = Path(req.output_dir)
    lock = coverage.read_json(root / 'recipe_lock.json')
    expansion = coverage.read_json(root / 'expanded_units.json')
    assert len(lock['batches']) == len(expansion['plans']) == 5
    assert [b['draws'][0] for b in lock['batches']] == [p['draw'] for p in lock['paired_draws']]
    assert campaign.instance_batches(lock, 'sign_changing-c0') == lock['batches'][:1]
    assert lock['study'] == STUDY and lock['recipe']['study'] == STUDY
    for plan in expansion['plans']:
        validate_plan(plan, approvals)
        for unit in plan['units'][:-1]:
            assert '--partition=preemptible' in unit['sbatch'] and '--no-requeue' in unit['sbatch']
            assert '--cpus-per-task=8' in unit['sbatch'] and 'OXYFORMER_DEVICE=cuda' in unit['command']
        assert all(t['study'] == STUDY for t in plan['tasks'])
        assert plan['spec']['work'][0]['parameters']['study'] == STUDY
    assert len(expansion['plans'][0]['tasks'][-1]['expected_leaves']) == sum(len(p['expected_leaves']) for p in expansion['plans'])
    assert all(t['study'] == STUDY for t in expansion['admission']['tasks'])
    cfg = coverage.read_json(req.config_path)
    deps = {k: Path(v) for k,v in cfg['dependencies'].items()}
    deps['campaign-lock'] = root
    from test_coverage import successful_records
    _, records = successful_records(5, 'sign_changing')
    budgets = {}
    for plan, batch, record in zip(expansion['plans'], lock['batches'], records):
        record['draw'] = batch['draws'][0]
        for estimate in record['estimates'].values():
            estimate['se_definitions'] = {
                'unclustered_influence': {'se': .5, 'variance_target':'fixed_frame_repetition','assumptions':'synthetic'},
                'spatial_primary': {'se':1.,'variance_target':'geographic_process','assumptions':'synthetic'}}
        for leaf in plan['tasks'][:-1]:
            directory = tmp_path/'collected'/leaf['id']
            value = {'lock_hash':coverage.digest(lock),'batch_id':batch['batch_id'],'mode':'diagnostic',
                     'scenario':batch['scenario'],'draws':batch['draws'],'records':[record]}
            result_path = write(directory/'result.json', value)
            deps[leaf['id']] = directory
            budgets[leaf['id']] = {'payload_hash':file_hash(result_path),
                'binding':{'recipe_hash':coverage.digest(lock['recipe'])},
                'leaves':[{'wall_seconds':100.}], 'complete':True}
    monkeypatch.setattr(campaign, 'load_lock', lambda *args:lock)
    monkeypatch.setattr(campaign, 'load_budget', lambda req, cfg, unit, **kw:budgets[unit])
    collector = request(tmp_path/'collector', expansion['plans'][0]['tasks'][-1], deps, approvals)
    result = campaign.run_stage(collector)
    assert result.status == 'pass', result.message
    readout = coverage.read_json(Path(collector.output_dir)/'paired_readout.json')
    assert len(readout['pairs']) == 5
    assert (Path(collector.output_dir)/'paired_readout.md').read_text().startswith('# diagnostic pilot')
    gate = coverage.read_json(Path(collector.output_dir)/'gate.json')
    assert gate['certifies_release_criteria'] is False
    summary = gate['scenario_summaries']['sign_changing']
    assert summary['gate_se'] == 'unclustered_influence'
    assert summary['methods']['one_step']['mean_se'] == .5
    assert summary['coverage_pass'] is False
    # Registered public admission must run the regeneration verifier before reading receipts.
    with patch.object(pilot, 'verify_regeneration', side_effect=ContractError('falsified before admission')) as verify:
        admit_task = {'parameters': {'lock_input': {'dependency':'lock','path':'recipe_lock.json'}}}
        fake = type('Req', (), {})()
        with patch.object(campaign_probe, 'dependency', return_value=root/'recipe_lock.json'):
            with pytest.raises(ContractError, match='falsified before admission'):
                campaign_probe.admit(fake, {'approvals': approvals}, admit_task)
        verify.assert_called_once()


def test_pilot_profile_builder_cell(owner, tmp_path):
    approvals, recipe = owner
    from oxyformer.validation import smoke_inputs
    endpoint, frame = smoke_inputs.build_inputs()
    built = real_frame.diagnostic_profile_tasks(endpoint, frame, approvals, campaign_id='pilot-d',
        evaluation_namespace='pilot-evaluation', prior_locks=[], cpu_wall_hours_budget=10000,
        campaign_mode='example_pilot', paired_manifest='/synthetic/manifest.json')
    legacy = real_frame.screening_profile_tasks(endpoint, frame, approvals)
    pilot_profiles = [t for t in built['tasks'] if t['stage'] == 'simulation-smoke']
    assert not {t['id'] for t in pilot_profiles}.intersection(t['id'] for t in legacy['tasks'])
    known = {t['id'] for t in pilot_profiles}
    assert all(t['continuation']['predecessor'] is None or t['continuation']['predecessor'] in known for t in pilot_profiles)
    for task in built['tasks']:
        if task['stage'] in ('campaign-estimate','campaign-lock'):
            assert all(ref['dependency'] in known for ref in task['parameters']['profiles'].values())
            assert all(ref['dependency'] in known for refs in task['parameters']['probe_profiles'].values() for ref in refs.values())
        assert task['parameters']['recipe']['study'] == STUDY
        assert task['parameters']['study'] == STUDY
        assert task['parameters']['partition'] == 'preemptible'


@pytest.mark.parametrize('fault', ['43_leaves', 'leaf_hours'])
def test_measured_profile_bound_refusal(owner, fault):
    _, recipe = owner
    decision = campaign.pilot_runtime(recipe)
    wall = coverage.maximum_leaf_stage_seconds()
    profile = {'seconds_per_repetition': 1., 'publication_verification_seconds': 0.,
               'leaf_overheads': {k: 0. for k in coverage.OVERHEAD_PHASES}}
    if fault == '43_leaves':
        capacity = coverage.usable_leaf_compute(wall, profile['leaf_overheads']) - coverage.RESUME_OVERHEAD_FLOOR_SECONDS
        profile['seconds_per_repetition'] = capacity * 41.5 / 1.25
        assert campaign.slice_count(profile, 1, wall, 1.25, enforce_limit=False) == 43
        match = 'pilot cell d:.*43 leaves'
    else:
        wall = 14401
        match = 'pilot cell d:.*GPU-hours'
    with pytest.raises(ContractError, match=match):
        campaign.pilot_sizing(recipe, {'sign_changing': profile}, decision, wall, 1.25)


def test_first_leaf_checks_real_generation_inputs(owner, archive_manifest, tmp_path, monkeypatch):
    lock = locked(owner, archive_manifest)
    pin = lock['paired_draws'][0]
    _, obs, truth, base, source_recipe = pilot.pinned_source(pin)
    recipe = {**source_recipe, **lock['recipe'], 'endpoint_hash': base.content_hash, 'frame_hash': obs.frame.content_hash}
    lock['recipe'] = recipe
    scenario = json.loads(truth.generator_configuration)['payload']
    lock['scenario_family'] = [scenario]
    from test_campaign import STAMPS
    lock.update(deepcopy(STAMPS))
    task = {'id': 'pilot-leaf', 'stage': 'coverage', **campaign.screening_metadata(recipe),
        'recipe_lock': {'dependency':'lock','path':'recipe_lock.json','sha256':'a'*64},
        'continuation': {'owner':'x','step':0,'predecessor':None},
        'outputs':['continuation.tar'], 'parameters': {**campaign.screening_metadata(recipe),
            'mode':'diagnostic', 'batch_id':'batch', 'draws':[pin['draw']], 'scenario':scenario,
            'study':STUDY, 'partition':'preemptible', 'wall_seconds':14100,
            'endpoint_input':{'dependency':'inputs','path':'endpoint.json'},
            'frame_input':{'dependency':'inputs','path':'frame.json'}}}
    lock['batches'][0].update(batch_id='batch', mode='diagnostic', scenario=scenario, wall_seconds=14100)
    inputs = tmp_path / 'inputs'
    write(inputs/'endpoint.json', base.to_dict()); write(inputs/'frame.json', obs.frame.to_dict())
    req = request(tmp_path/'leaf', task, {'inputs':inputs}, owner[0])
    monkeypatch.setattr(campaign, 'load_lock', lambda *args: lock)
    monkeypatch.setattr(campaign, 'validate_leaf_task', lambda *args: None)
    monkeypatch.setattr(campaign, 'validate_recipe', lambda *args, **kw: None)
    with patch.object(pilot, 'verify_regeneration', wraps=pilot.verify_regeneration) as verify:
        coverage.prepare_batch(req)
        assert verify.call_count == 1
        assert verify.call_args.kwargs['scenario'].content_hash == pin['scm_hash']
    lock['paired_draws'][0]['observations_hash'] = '0'*64
    with pytest.raises(ContractError, match='chain c1.*observations/frame/truth mismatch'):
        coverage.prepare_batch(req)


def legacy_spec(mode):
    metadata = campaign.screening_metadata({'campaign_mode':mode, 'study':campaign.SCREENING_CELL})
    partition = 'standard' if mode == 'screening' else campaign.month2_decision()['diagnostic_campaign']['partition']
    return {'schema_version':1,'id':'golden-'+mode,'kind':mode, **metadata,
        'partition':partition,'gpu_concurrency_max':4 if mode == 'screening' else 16,
        'prerequisites':[],'inputs':{'lock':['recipe_lock.json']},
        'recipe_lock':{'dependency':'lock','path':'recipe_lock.json','sha256':'a'*64},
        'work':[{'id':'golden-work','stage':'coverage',
            'parameters':{'wall_seconds':14100,'study':campaign.SCREENING_CELL,
                          'draws':coverage.repetition_plan('golden','sign_changing',1)},
            'outputs':['result.json','continuation.tar'], 'slices':[{'gpus':1,'wall_seconds':14400}]}],
        'collector':{'stage':'campaign-collect','outputs':['result.json'],'wall_seconds':600,'gpus':0}}


@pytest.mark.parametrize('mode', ['screening','diagnostic'])
def test_legacy_lifecycle_golden_bytes(mode):
    from oxyformer.validation import smoke_inputs
    from oxyformer.execution.runner import read_mapping
    expected = coverage.read_json(Path('tests/fixtures/pilot/legacy_goldens.json'))[mode]
    approvals = read_mapping(Path('configs/approvals.yaml'))
    metadata = campaign.screening_metadata({'campaign_mode':mode,'study':campaign.SCREENING_CELL})
    assert canonical_json(metadata) == canonical_json(expected['metadata'])
    if mode == 'screening':
        tasks = real_frame.screening_profile_tasks(*smoke_inputs.build_inputs(), approvals)
    else:
        tasks = real_frame.diagnostic_profile_tasks(*smoke_inputs.build_inputs(), approvals,
            campaign_id='golden-diag', evaluation_namespace='golden', prior_locks=[], cpu_wall_hours_budget=10000)
    assert coverage.digest(tasks) == expected['profile_tasks_hash']
    plan = expand_campaign(legacy_spec(mode), approvals)
    assert coverage.digest(plan) == expected['expansion_hash']
    validate_plan(plan, approvals)


@pytest.mark.parametrize('blob_kind', ['observations', 'structural_causal_truth'])
def test_saved_blob_tamper_is_named_before_regeneration(owner, archive_manifest, tmp_path, blob_kind):
    import shutil
    lock = locked(owner, archive_manifest)
    pin = lock['paired_draws'][0]
    original = Path(pin['source_path'])
    copied = tmp_path/'tampered-source'
    shutil.copytree(original, copied)
    pin['source_path'] = str(copied)
    with tarfile.open(copied/'continuation.tar', 'r:') as archive:
        header = json.load(archive.extractfile('coverage.json'))
        key = header['states'][0]['sample'][blob_kind]
        raw = archive.extractfile('blobs/'+key).read()
    archive = copied/'continuation.tar'
    payload = archive.read_bytes()
    assert raw in payload
    archive.write_bytes(payload.replace(raw, b'X'+raw[1:], 1))
    with pytest.raises(ContractError, match='paired source chain c1: source archive hash mismatch'):
        pilot.verify_regeneration(lock)


def test_single_owner_cell_drives_study_without_a_hardcoded_cell(owner):
    approvals, recipe = owner
    study = {'reference_cap':64,'frozen_epochs':60,'grid_points':4,'seed_count':3}
    block = approvals['owner_decisions']['example_pilot']
    block['cells'] = {'c':study}
    recipe['study'] = deepcopy(study)
    recipe['pilot_declaration'].update(cell='c', study=deepcopy(study))
    declaration = campaign.pilot_declaration(recipe, approvals)
    assert declaration['cell'] == 'c' and declaration['study'] == study


@pytest.mark.parametrize('dimension,value', [('grid_points',3), ('seed_count',2)])
def test_owner_study_must_still_be_supported_by_engine(owner, dimension, value):
    approvals, recipe = owner
    approvals['owner_decisions']['example_pilot']['cells']['d'][dimension] = value
    recipe['study'][dimension] = value
    recipe['pilot_declaration']['study'][dimension] = value
    with pytest.raises(ContractError, match='study'):
        campaign.pilot_declaration(recipe, approvals)


@pytest.mark.parametrize('pilot_named', [False, True])
def test_public_probe_preserves_legacy_shape_and_requires_pilot_recipe(owner, tmp_path, monkeypatch, pilot_named):
    estimate = write(tmp_path/'budget_estimate.json', {})
    lock = {'estimate_input':{'dependency':'estimate','path':'budget_estimate.json'},
            'estimate_hash':file_hash(estimate)}
    if pilot_named:
        lock['campaign_mode'] = owner[1]['campaign_mode']
    path = write(tmp_path/'recipe_lock.json', lock)
    first = {'needs':{},'parameters':{'scenario':{'name':'sign_changing'}}}
    expansion = write(tmp_path/'expanded_units.json', {'lock_hash':file_hash(path),'plans':[{'tasks':[first]}]})
    paths = {p.name:p for p in (estimate,path,expansion)}
    monkeypatch.setattr(campaign_probe, 'dependency', lambda req,cfg,ref:paths[ref['path']])
    task = {'parameters':{'lock_input':{'dependency':'lock','path':'recipe_lock.json'}}}
    if pilot_named:
        with pytest.raises(ContractError, match='pilot lock requires its declared recipe'):
            campaign_probe.locked_probe_inputs(None, {'approvals':owner[0]}, task)
    else:
        resolved = campaign_probe.locked_probe_inputs(None, {}, task)
        assert resolved[0] == lock and resolved[3] == first and resolved[4] == {}


def test_legacy_admission_plan_preserves_recipe_free_probe_shape():
    refs = {str(depth):{'reference':{'dependency':'profile-'+str(depth),'path':'budget_record.json'}} for depth in (1,2)}
    lock = {'campaign_id':'legacy-probe', 'admission_receipt':{'dependency':'admit','path':'admission.json'},
            'budget':{'profiles':{'sign_changing':{'setup_profile_sources':refs}}}}
    first = {'needs':{},'parameters':{'scenario':{'name':'sign_changing'}}}
    plan = campaign_probe.setup_admission_plan(lock, [{'tasks':[first]}], 'lock', {})
    assert plan['lock_hash'] == coverage.digest(lock)
    assert plan['tasks'][-1]['id'] == 'admit'
    assert all('campaign_mode' not in task for task in plan['tasks'])


def test_legacy_admit_still_checks_estimate_without_a_pilot_recipe(tmp_path, monkeypatch):
    estimate = write(tmp_path/'budget_estimate.json', {})
    lock = write(tmp_path/'recipe_lock.json', {'estimate_input':{'dependency':'estimate','path':'budget_estimate.json'},
                                             'estimate_hash':'0'*64})
    monkeypatch.setattr(campaign_probe, 'dependency', lambda req,cfg,ref:lock if ref['dependency']=='lock' else estimate)
    task = {'parameters':{'lock_input':{'dependency':'lock','path':'recipe_lock.json'}}}
    with pytest.raises(ContractError, match='admission estimate_hash mismatch'):
        campaign_probe.admit(None, {}, task)
