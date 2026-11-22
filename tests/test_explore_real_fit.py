"""Synthetic reconstructions only; no acquisition reads or network access."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest
import torch

from oxyformer.explore import real_fit as driver
from oxyformer.provenance import ContractError
from oxyformer.training import nested_cv
from oxyformer.training.fit import RecipeStudy
from oxyformer.validation import real_frame, coverage
from test_campaign_profiles import real_fixture
from test_nested_cv import endpoint as tiny_endpoint, tiny_config, predictions
from oxyformer.validation.leakage import assert_prediction_invariant


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def acquisition(root, rows=None, header=None):
    root.mkdir(parents=True, exist_ok=True)
    rows = rows or [['1001000100', '1', '1', '100', '78.2', '.2', '1'],
                    ['11001000100', '11', '1', '100', '79', '.3', '2'],
                    ['12001000100', '12', '1', '100', '81', '.4', '3']]
    text = io.StringIO()
    import csv
    writer = csv.writer(text)
    writer.writerow(driver.HEADER if header is None else header)
    writer.writerows(rows)
    raw = text.getvalue().encode()
    payload, receipts = root / 'payload.tar', root / 'receipts.json'
    with tarfile.open(payload, 'w') as archive:
        member = tarfile.TarInfo('us/usaleep_data.csv')
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
    receipts.write_text(json.dumps(dict(resources=[dict(id='usaleep_data', destination=member.name,
        bytes=len(raw), sha256=sha256(raw).hexdigest())])))
    return payload, receipts


def test_grammar(tmp_path):
    paths = acquisition(tmp_path / 'good')
    outcomes, _ = driver.read_outcomes(*paths)
    assert list(outcomes) == ['01001000100', '11001000100', '12001000100']
    assert list(outcomes.values()) == [(78.2, 1), (79., 2), (81., 3)]
    with pytest.raises(ContractError, match='header'):
        driver.read_outcomes(*acquisition(tmp_path / 'header', header=['wrong']))
    duplicate = [['1001000100', '', '', '', '78', '', '1'], ['01001000100', '', '', '', '79', '', '3']]
    with pytest.raises(ContractError, match='duplicate'):
        driver.read_outcomes(*acquisition(tmp_path / 'duplicate', duplicate))
    with pytest.raises(ValueError, match='unparseable'):
        driver.read_outcomes(*acquisition(tmp_path / 'bad-y', [['1001000100', '', '', '', 'bad', '', '1']]))
    payload, receipts = paths
    raw = payload.read_bytes()
    payload.write_bytes(raw.replace(b'78.2', b'78.3', 1))
    with pytest.raises(ContractError, match='sha256'):
        driver.read_outcomes(payload, receipts)


def test_unsealed():
    values = real_fixture()
    # Approval fixtures are copied, never written into repository approvals.
    values['approvals'] = deepcopy(values['approvals'])
    outcomes = {r.original_id: (75. + j / 1000, r.outcome_flag) for j, r in enumerate(values['geography'].rows)}
    design = driver.read_mapping(driver.gate.DESIGN_CONFIG)
    endpoint, audit = driver.build_endpoint(values, design, outcomes)
    assert not endpoint.outer.design_ids and not audit['design_ids'] and not audit['sealed_subblocks']
    assert set(endpoint.outer.frame_original_ids) == set(values['data_manifest'].original_ids)
    assert all(endpoint.data.column('y'))
    assert endpoint.data.column('y') == tuple(outcomes[i][0] for i in endpoint.data.manifest.original_ids)
    pool = [r.exposure_mmhg for r in values['atlas'].rows if r.allocation_qualified and r.population > 0]
    assert endpoint.treatment_design.center == pytest.approx(np.mean(pool))
    assert endpoint.treatment_design.scale == pytest.approx(np.std(pool))
    sealed, _, _, _ = real_frame.build_inputs(values, design)
    assert sealed.outer.design_ids
    assert set(sealed.data.column('y')) == {None}
    assert audit['in_frame_by_flag'] == {'1': len(outcomes)}


class FakeLogger:
    def __init__(self):
        self.configs, self.events, self.finished = [], [], 0
        self.directories = []

    def factory(self, mode, config, directory=None):
        self.configs.append(config)
        self.directories.append(directory)
        return self

    def log(self, event):
        self.events.append(event)

    def finish(self):
        self.finished += 1


def fit_fixture(tmp_path, monkeypatch):
    endpoint = tiny_endpoint(tmp_path)
    recipe = dict(study=driver.STUDY, nested_cv={})
    monkeypatch.setattr(driver, 'load_prepared', lambda p: (endpoint, recipe, {}))
    monkeypatch.setattr(driver, 'checkout_identity', lambda: 'synthetic-checkout')
    # Keep the real nested controller but use its registered tiny fixture settings.
    monkeypatch.setattr(nested_cv, '_settings', lambda r, e: dict(ssl_epochs=1,
        settings=nested_cv.NuisanceSettings(frozen_epochs=1)))
    monkeypatch.setattr(driver, 'RecipeStudy', lambda **kw: RecipeStudy())
    return endpoint


def test_resume_and_logging(tmp_path, monkeypatch):
    prepared = fit_fixture(tmp_path, monkeypatch)
    sentinel = 'synthetic-sentinel-do-not-persist'
    monkeypatch.setenv('WANDB_API_KEY', sentinel)
    logger = FakeLogger()
    monkeypatch.setattr(driver, 'logger_for', logger.factory)
    state = tmp_path / 'state'
    record = driver.fit('fixture', 0, 1103, state, .001, 'none', device='cpu')
    assert record['status'] == 'incomplete' and (state / 'progress.json').exists()
    assert record['identity']['run_logging'] == 'none'
    first = record['slices'][0]['output_dir']
    record = driver.fit('fixture', 0, 1103, state, 120, 'none', device='cpu')
    assert record['status'] == 'complete'
    assert len(record['slices']) == 2 and record['slices'][1]['output_dir'] != first
    full = nested_cv.run_fold(tiny_config(prepared, tmp_path / 'uninterrupted'),
        prepared.outer, 1103, geography=prepared.geography)
    part = coverage.OOFNuisances.from_json(record['parts'][0])
    assert_prediction_invariant(predictions(prepared, full), part)
    before = (state / 'progress.json').read_bytes()
    assert driver.fit('fixture', 0, 1103, state, 120, 'none', device='cpu') == record
    assert (state / 'progress.json').read_bytes() == before
    assert logger.directories == [state.resolve(), state.resolve()]
    assert all('epoch' not in e for e in logger.events)
    assert logger.configs and all({'fold', 'checkpoint_position', 'stop_reason', 'wall_seconds'} <= e.keys() for e in logger.events)
    for path in tmp_path.rglob('*'):
        if path.is_file():
            assert sentinel.encode() not in path.read_bytes(), path
    record['identity']['prepared_hash'] = 'f' * 64
    driver.atomic_record(state / 'progress.json', record)
    with pytest.raises(ContractError, match='identity'):
        driver.fit('fixture', 0, 1103, state, 120, 'none', device='cpu')


def test_logging_absent_before_training(tmp_path, monkeypatch):
    fit_fixture(tmp_path, monkeypatch)
    def absent(name):
        raise ImportError('fixture missing package')
    monkeypatch.setattr(driver.importlib, 'import_module', absent)
    monkeypatch.setattr(nested_cv, 'run_fold', lambda *a, **k: pytest.fail('training started'))
    with pytest.raises(ValueError, match='package is absent'):
        driver.fit('fixture', 0, 1103, tmp_path / 'state', 10, 'wandb', device='cpu')
    assert driver.logger_for('none', {}) is None
    with pytest.raises(ContractError, match='explicit'):
        driver.logger_for(None, {})


def test_score(tmp_path, monkeypatch):
    from oxyformer.validation import smoke_inputs
    from oxyformer.validation.scm import SCMConfig
    endpoint, frame = smoke_inputs.build_inputs()
    sample = coverage.generate_suite_a(frame,
        SCMConfig(name='null_effect', active_mechanisms=('null',), effect='null'), endpoint.policy)
    endpoint = coverage.bind_observations(endpoint, sample.observations)
    # Synthetic nuisance predictions use true outcomes plus a deterministic residual;
    # this exercises real scoring/variance functions without training five folds.
    from oxyformer.contracts import OOFNuisances
    all_ids = endpoint.data.manifest.original_ids
    y = dict(zip(all_ids, endpoint.data.column(endpoint.data.manifest.outcome_field)))
    recipe = driver.recipe_for(endpoint, driver.read_mapping(driver.gate.OWNER_APPROVALS))
    audit = {k: 10 for k in ('rows_read', 'frame_rows', 'labelled', 'eligible', 'support_pool', 'support_candidates', 'targets')}
    audit.update(in_frame_by_flag={'1': 10}, exclusions_by_reason={}, layer_counts={})
    monkeypatch.setattr(driver, 'load_prepared', lambda p: (endpoint, recipe, audit))
    states = []
    for fold in range(5):
        ids = endpoint.outer.evaluation_ids(fold)
        part = OOFNuisances(spec=endpoint.data.manifest.spec, original_ids=ids, fold_ids=(fold,) * len(ids),
            seed_ids=(1103,) * len(ids), mu_a=tuple(y[i] - .1 for i in ids), mu_d=tuple(y[i] + .1 for i in ids),
            r_a=(1.,) * len(ids), r_d=(1.,) * len(ids), origin_weights=(1.,) * len(ids),
            lineage=replace(endpoint.data.manifest.lineage, unit_ids=ids, split_hash=endpoint.outer.content_hash))
        root = tmp_path / str(fold)
        driver.atomic_record(root / 'progress.json', dict(status='complete', parts=[part.to_json()],
            identity=dict(prepared_hash=endpoint.content_hash, recipe=recipe, fold=fold, seed=1103,
                          checkout='synthetic-tree', environment={}, run_logging='none')))
        states.append(root)
    logger = FakeLogger()
    monkeypatch.setattr(driver, 'logger_for', logger.factory)
    result = driver.score('fixture', states, tmp_path / 'score', 'none')
    assert result['exploratory'] is True
    assert result['sealing'] == 'none (owner decision 2026-11-19)'
    assert result['effect_release_authorized'] is False and result['certifies_release_criteria'] is False
    assert set(result['estimates']) == {'one_step', 'cv_tmle'}
    for estimate in result['estimates'].values():
        assert {'value', 'se', 'se_definitions'} <= estimate.keys()
        assert {'county_cluster', 'spatial_primary', 'unclustered_influence',
                'residual_frame_conditional', 'spatial_50_km', 'spatial_100_km',
                'spatial_200_km'} <= estimate['se_definitions'].keys()
    assert logger.directories == [tmp_path / 'score']
    assert logger.events[-1]['final_estimates'] == result['estimates']
    assert (tmp_path / 'score' / 'result.md').exists()
    record = json.loads((states[0] / 'progress.json').read_text())
    record['status'] = 'incomplete'
    driver.atomic_record(states[0] / 'progress.json', record)
    with pytest.raises(ContractError, match='complete folds'):
        driver.score('fixture', states, tmp_path / 'bad-score', 'none')


def test_wandb_driver_config(monkeypatch, tmp_path):
    from types import SimpleNamespace
    logger = FakeLogger()
    calls = []
    def initialize(**kwargs):
        calls.append(kwargs)
        return logger
    monkeypatch.setenv('WANDB_API_KEY', 'synthetic-secret')
    monkeypatch.setattr(driver.importlib, 'import_module', lambda name: SimpleNamespace(init=initialize))
    assert driver.logger_for('wandb', {'recipe': driver.STUDY}, tmp_path) is logger
    assert calls == [dict(entity='goodarzilab', project='oxyformer-v2', config={'recipe': driver.STUDY}, dir=str(tmp_path))]
    assert 'synthetic-secret' not in json.dumps(calls)


@pytest.mark.parametrize('operation', ['fit', 'score'])
@pytest.mark.parametrize('through_symlink', [False, True])
def test_unignored_checkout_output_refuses_before_writing(tmp_path, monkeypatch, operation, through_symlink):
    import subprocess
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    subprocess.run(['git', 'init', '-q', str(checkout)], check=True)
    (checkout / '.gitignore').write_text('/ignored-runs/\n')
    monkeypatch.setattr(driver, 'ROOT', checkout)
    monkeypatch.setattr(driver, 'load_prepared', lambda _: pytest.fail('loaded prepared data before path refusal'))
    base = checkout
    if through_symlink:
        base = tmp_path / 'alias'
        base.symlink_to(checkout, target_is_directory=True)
    destination = base / 'review-runs' / 'fold-0'
    assert not destination.exists()
    with pytest.raises(ContractError, match='use a state directory outside the checkout') as failure:
        if operation == 'fit':
            driver.fit('fixture', 0, 1103, destination, .001, 'none', device='cpu')
        else:
            driver.score('fixture', [], destination, 'none')
    assert str(destination.resolve()) in str(failure.value)
    assert not (checkout / 'review-runs').exists()
    assert driver.write_directory(checkout / 'ignored-runs') == checkout / 'ignored-runs'
    assert driver.write_directory(checkout / 'ignored-runs' / 'fold-0') == checkout / 'ignored-runs' / 'fold-0'
    assert driver.write_directory(tmp_path / 'outside') == tmp_path / 'outside'
    assert not (checkout / 'ignored-runs').exists() and not (tmp_path / 'outside').exists()


def test_checkpoint_position_and_train_epoch_are_distinct(monkeypatch):
    from types import SimpleNamespace
    checkpoint = SimpleNamespace(epoch=11, identity='synthetic-identity')
    controller = dict(active=dict(progress=dict(epoch=7)), ssl_pending=None)
    monkeypatch.setattr(driver, 'load_checkpoint', lambda *args: {'controller': controller}, raising=False)
    assert driver.checkpoint_progress(checkpoint) == dict(checkpoint_position=11, train_epoch=7)
    controller['active'] = None
    assert driver.checkpoint_progress(checkpoint) == dict(checkpoint_position=11)


def test_real_fit_requires_wandb_before_state_or_training(tmp_path, monkeypatch):
    values = real_fixture()
    outcomes = {r.original_id: (75., r.outcome_flag) for r in values['geography'].rows}
    endpoint, _ = driver.build_endpoint(values, driver.read_mapping(driver.gate.DESIGN_CONFIG), outcomes)
    recipe = driver.recipe_for(endpoint, values['approvals'])
    monkeypatch.setattr(driver, 'load_prepared', lambda _: (endpoint, recipe, {}))
    monkeypatch.setattr(driver, 'checkout_identity', lambda: 'synthetic-checkout')
    monkeypatch.setattr(nested_cv, 'run_fold', lambda *a, **k: pytest.fail('training reached without required W&B'))
    state = tmp_path / 'real-state'
    with pytest.raises(ContractError, match='W&B logging is required for real endpoints'):
        driver.fit('fixture', 0, 1103, state, .001, 'none', device='cpu')
    assert not state.exists()


def test_cell_d_uses_effective_study_epochs(tmp_path, monkeypatch):
    endpoint = tiny_endpoint(tmp_path)
    recipe = driver.recipe_for(endpoint, driver.read_mapping(driver.gate.OWNER_APPROVALS))
    config = endpoint.configuration(0, tmp_path / 'work', **nested_cv._settings(recipe, endpoint),
                                    study=RecipeStudy(**recipe['study']))
    assert config.settings.frozen_epochs == 150  # Registered base, not the effective override.
    assert config.frozen_epochs == 60
    assert config.study.reference_cap == 32
    assert len(config.grid) == 2 and config.study.seeds == (1103,)
    observed = []
    class ReachedConfiguration(Exception):
        pass
    def inspect_config(actual, *args, **kwargs):
        observed.append(actual)
        raise ReachedConfiguration
    monkeypatch.setattr(driver, 'load_prepared', lambda _: (endpoint, recipe, {}))
    monkeypatch.setattr(driver, 'checkout_identity', lambda: 'synthetic-checkout')
    monkeypatch.setattr(nested_cv, 'run_fold', inspect_config)
    with pytest.raises(ReachedConfiguration):
        driver.fit('fixture', 0, 1103, tmp_path / 'driver-state', .001, 'none', device='cpu')
    assert observed[0].settings.frozen_epochs == 150
    assert observed[0].frozen_epochs == 60
    assert observed[0].study.reference_cap == 32 and len(observed[0].grid) == 2
    assert observed[0].study.seeds == (1103,)
    print('base_frozen_epochs=150 effective_frozen_epochs=60 driver_frozen_epochs=60 reference_cap=32 grid_points=2 seed=1103')


@pytest.mark.parametrize('active, expected_epoch', [
    ({'progress': {}}, None),
    ({'progress': {'epoch': 7}}, 7),
    ({}, None),
    ({'progress': None}, None),
    ({'progress': {'epoch': None}}, None),
    ({'progress': {'epoch': '7'}}, None),
    ({'progress': {'epoch': 7.5}}, None),
    ({'progress': {'epoch': True}}, None),
])
def test_optional_epoch_publishes_journal(tmp_path, monkeypatch, active, expected_epoch):
    from types import SimpleNamespace
    fit_fixture(tmp_path, monkeypatch)
    checkpoint = SimpleNamespace(epoch=11, identity='synthetic', reason='slice_limit')
    predecessor = '{"synthetic": "returned predecessor"}'
    artifact = SimpleNamespace(complete=False, checkpoint=checkpoint, to_json=lambda: predecessor)
    monkeypatch.setattr(nested_cv, 'run_fold', lambda *a, **k: artifact)
    monkeypatch.setattr(driver, 'load_checkpoint', lambda *a: {
        'controller': {'active': active, 'ssl_pending': None}})
    state = tmp_path / 'state'
    returned = driver.fit('fixture', 0, 1103, state, 10, 'none', device='cpu')
    # Inspect the published replacement, not just the metadata helper's return.
    published = json.loads((state / 'progress.json').read_text())
    assert published == returned
    assert published['predecessor'] == predecessor
    assert published['status'] == 'incomplete' and published['parts'] == []
    assert len(published['slices']) == 1
    event = published['slices'][0]
    assert event['checkpoint_position'] == 11
    assert event['stop_reason'] == 'slice_limit' and event['complete'] is False
    assert event['fold'] == 0 and event['wall_seconds'] >= 0
    assert Path(event['output_dir']).parent == state
    if expected_epoch is None:
        assert 'train_epoch' not in event
    else:
        assert event['train_epoch'] == expected_epoch


@pytest.mark.parametrize('drift', ['registered', 'extra', 'missing', 'changed'])
def test_cell_d_approval_boundaries(tmp_path, monkeypatch, drift):
    from types import SimpleNamespace
    endpoint = tiny_endpoint(tmp_path)
    approvals = deepcopy(driver.read_mapping(driver.gate.OWNER_APPROVALS))
    approval_path = tmp_path / 'approvals.json'
    approval_path.write_text(json.dumps(approvals))
    monkeypatch.setattr(driver.gate, 'OWNER_APPROVALS', approval_path)
    inputs, atlas = tmp_path / 'inputs', tmp_path / 'atlas'
    (inputs / '_execution').mkdir(parents=True)
    atlas.mkdir()
    (inputs / '_execution/result.json').write_text('{}')
    roles = ['data_manifest', 'covariates', 'geography', 'entity_graph']
    for name in roles:
        (inputs / (name + '.json')).write_text('{}')
    for name in ('atlas.parquet', 'quality.json', 'artifact_manifest.json'):
        (atlas / name).write_text('{}')
    publication = SimpleNamespace(status='pass', artifacts=[
        SimpleNamespace(path=name + '.json', sha256='fixture') for name in roles])
    monkeypatch.setattr(driver, 'StageResult', SimpleNamespace(from_json=lambda _: publication))
    monkeypatch.setattr(driver.gate, 'collected_atlas_inputs', lambda *a: {})
    monkeypatch.setattr(driver, 'build_endpoint', lambda *a: (endpoint, {}))
    # The manifest is only needed to forward IDs to the mocked atlas loader.
    manifest = SimpleNamespace(original_ids=endpoint.data.manifest.original_ids)
    monkeypatch.setattr(driver, 'read_artifact',
                        lambda path, *args: manifest if Path(path).name == 'data_manifest.json' else endpoint)
    payload, receipts = acquisition(tmp_path / 'acquisition')
    prepared = tmp_path / 'prepared'
    driver.prepare(inputs, atlas, payload, receipts, prepared)
    assert driver.load_prepared(prepared)[1]['study'] == driver.STUDY

    study = approvals['owner_decisions']['m2_pilot_d']['cells']['d']
    if drift == 'extra':
        study['rationale'] = 'synthetic metadata'
    elif drift == 'missing':
        del study['seed_count']
    elif drift == 'changed':
        study['frozen_epochs'] = 61
    approval_path.write_text(json.dumps(approvals))
    monkeypatch.setattr(driver, 'checkout_identity', lambda: 'synthetic-checkout')
    observed = []
    def make_study(**fields):
        observed.append(fields)
        return RecipeStudy(**fields)
    monkeypatch.setattr(driver, 'RecipeStudy', make_study)
    class Configured(Exception):
        pass
    def inspect_config(config, *args, **kwargs):
        assert config.frozen_epochs == 60
        raise Configured
    monkeypatch.setattr(nested_cv, 'run_fold', inspect_config)
    state = tmp_path / 'state'
    if drift == 'registered':
        with pytest.raises(Configured):
            driver.fit(prepared, 0, 1103, state, 10, 'none', device='cpu')
        assert observed == [dict(reference_cap=32, frozen_epochs=60, grid_points=2, seed_count=1)]
    else:
        with pytest.raises(ContractError, match='cell d recipe drift'):
            driver.prepare(inputs, atlas, payload, receipts, tmp_path / 'refused')
        assert not (tmp_path / 'refused').exists()
        with pytest.raises(ContractError, match='cell d recipe drift'):
            driver.load_prepared(prepared)
        with pytest.raises(ContractError, match='cell d recipe drift'):
            driver.fit(prepared, 0, 1103, state, 10, 'none', device='cpu')
        assert observed == [] and not state.exists()


def sensitivity_approvals():
    return {'owner_decisions': {'exploratory_unsealed': {'policy_support_sensitivity': {
        'delta_mmhg_allowed': [1., 2.], 'raw_x_radius_iqr_max': 3.,
        'unchanged_support_settings': {'bin_width_mmhg': 1., 'min_bin_tracts': 2}}}}}


@pytest.mark.parametrize('delta,caliper,bound', [
    (1., 3., None), (2., 2., None), (1.5, 2., 'delta_mmhg_allowed'),
    (2., 3.5, 'raw_x_radius_iqr_max'), (2., 1.5, 'approved raw_x_radius'),
    (float('nan'), 2., 'delta_mmhg_allowed'), (2., float('inf'), 'support_caliper_iqr')])
def test_sensitivity_bounds(delta, caliper, bound):
    design = {'support': dict(bin_width_mmhg=1., min_bin_tracts=2, raw_x_radius=2.)}
    before = deepcopy(design)
    if bound:
        with pytest.raises(ContractError, match=bound):
            driver.support_settings(design, sensitivity_approvals(), delta, caliper)
    else:
        assert driver.support_settings(design, sensitivity_approvals(), delta, caliper).raw_x_radius == caliper
    assert design == before


@pytest.mark.parametrize('delta,caliper', [(1., 2.), (2., 3.)])
def test_sensitivity_requires_owner_block(delta, caliper, tmp_path):
    design = {'support': dict(bin_width_mmhg=1., min_bin_tracts=2, raw_x_radius=2.)}
    assert driver.support_settings(design, {}, 2., None).raw_x_radius == 2.
    with pytest.raises(ContractError, match='policy_support_sensitivity'):
        driver.support_settings(design, {}, delta, caliper)


@pytest.mark.parametrize('name,value', [('bin_width_mmhg', .5), ('min_bin_tracts', 3)])
def test_sensitivity_fixed_settings(name, value):
    design = {'support': dict(bin_width_mmhg=1., min_bin_tracts=2, raw_x_radius=2.)}
    design['support'][name] = value
    with pytest.raises(ContractError, match=name):
        driver.support_settings(design, sensitivity_approvals(), 1., 3.)


def test_invalid_override_precedes_input_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'read_outcomes', lambda *a: pytest.fail('outcomes read'))
    with pytest.raises(ContractError, match='delta_mmhg_allowed'):
        driver.prepare('missing', 'missing', 'missing', 'missing', tmp_path / 'out', delta_mmhg=1.5)
    assert not (tmp_path / 'out').exists()


def test_default_identity_and_variant_audit(monkeypatch, tmp_path):
    values = real_fixture()
    outcomes = {r.original_id: (75., r.outcome_flag) for r in values['geography'].rows}
    design = driver.read_mapping(driver.gate.DESIGN_CONFIG)
    freeze = driver.freeze_support
    baseline_support = []
    def legacy(*args, recipe, **kwargs):
        frozen = freeze(*args, recipe=recipe)  # Original path: no delta keyword.
        baseline_support.append(frozen)
        return frozen
    monkeypatch.setattr(driver, 'freeze_support', legacy)
    baseline, _ = driver.build_endpoint(values, design, outcomes)
    monkeypatch.setattr(driver, 'freeze_support', freeze)
    actual, audit = driver.build_endpoint(values, design, outcomes)
    assert actual.content_hash == baseline.content_hash
    assert actual.to_json() == baseline.to_json()
    assert actual.policy.policy_id == baseline_support[0].policy.policy_id
    assert actual.policy.support_design_hash == baseline_support[0].policy.support_design_hash
    assert actual.treatment_design.design_hash == baseline_support[0].content_hash
    variant, changed = driver.build_endpoint(values, design, outcomes, delta_mmhg=1., support_caliper_iqr=3.)
    assert variant.policy.delta_mmhg == changed['delta_mmhg'] == 1.
    assert changed['support_caliper_iqr'] == 3.
    assert variant.content_hash != actual.content_hash
    for e, a in [(actual, audit), (variant, changed)]:
        assert a['n_targets'] == len(e.outer.original_ids)
        assert a['n_moved_targets'] == sum(driver.target_policy(e).moved)
        assert a['achieved_mean_shift_mmhg'] == pytest.approx(a['shifted_fraction'] * a['delta_mmhg'], abs=1e-12)
    # Legacy prepared directories contain no sensitivity fields in their audit.
    prepared = tmp_path / 'legacy'
    prepared.mkdir()
    (prepared / 'endpoint.json').write_text(baseline.to_json())
    driver.atomic_record(prepared / 'prepare_audit.json', {'prepared_hash': baseline.content_hash})
    driver.atomic_record(prepared / 'recipe.json', driver.recipe_for(baseline, values['approvals']))
    assert driver.load_prepared(prepared)[0].content_hash == baseline.content_hash


@pytest.mark.parametrize('zero', [False, True])
def test_policy_readout_diagnostics(tmp_path, zero):
    from types import SimpleNamespace
    endpoint = tiny_endpoint(tmp_path)
    endpoint = replace(endpoint, policy=replace(endpoint.policy, components_by_key=(('s', ((0., 6.),)),)))
    if zero:
        endpoint = replace(endpoint, policy=replace(endpoint.policy, delta_mmhg=0.))
    readout = driver.policy_readout(endpoint, 2.)
    ids = endpoint.outer.original_ids
    ratios = tuple(float(j + 1) for j in range(len(ids)))
    nuisances = SimpleNamespace(original_ids=ids[::-1], r_d=ratios[::-1])
    estimates = {'one_step': dict(value=4., se=2.), 'cv_tmle': dict(value=3., se=1.)}
    d = driver.policy_diagnostics(endpoint, readout, estimates, nuisances)
    assert readout['achieved_mean_shift_mmhg'] == readout['shifted_fraction'] * endpoint.policy.delta_mmhg
    if zero:
        assert readout['n_moved_targets'] == 0
        assert all(v is None for v in d.values())
    else:
        assert 0 < readout['shifted_fraction'] < 1
        moved_ratios = np.asarray([r for r, m in zip(ratios, driver.target_policy(endpoint).moved) if m])
        assert d['mover_r_d_kish_ess'] == pytest.approx(moved_ratios.sum() ** 2 / (moved_ratios ** 2).sum())
        assert d['mover_r_d_max_share'] == pytest.approx(moved_ratios.max() / moved_ratios.sum())
        assert d['mover_r_d_p99'] == pytest.approx(np.quantile(moved_ratios, .99))
        for name, v in estimates.items():
            assert d['per_mover'][name]['value'] == v['value'] / readout['shifted_fraction']
            assert d['per_mover'][name]['se'] == v['se'] / readout['shifted_fraction']
            assert d['per_achieved_mmhg'][name]['value'] == v['value'] / readout['achieved_mean_shift_mmhg']
            assert d['per_achieved_mmhg'][name]['se'] == v['se'] / readout['achieved_mean_shift_mmhg']


def test_scored_policy_reporting(tmp_path, monkeypatch):
    original = driver.score
    def checked(*args, **kwargs):
        result = original(*args, **kwargs)
        saved = json.loads((tmp_path / 'score/result.json').read_text())
        text = (tmp_path / 'score/result.md').read_text()
        for name in ('delta_mmhg', 'support_caliper_iqr', 'n_targets', 'n_moved_targets',
                     'shifted_fraction', 'achieved_mean_shift_mmhg', 'policy_diagnostics'):
            assert saved[name] == result[name]
            assert name in text
        assert result['achieved_mean_shift_mmhg'] == result['shifted_fraction'] * result['delta_mmhg']
        return result
    monkeypatch.setattr(driver, 'score', checked)
    test_score(tmp_path, monkeypatch)


def test_sensitivity_cli_forwards_only_prepare_flags(monkeypatch):
    observed = []
    monkeypatch.setattr(driver, 'prepare', lambda **kwargs: observed.append(kwargs))
    driver.main(['prepare', '--tract-inputs', 'inputs', '--atlas', 'atlas',
                 '--usaleep-tar', 'payload', '--usaleep-receipts', 'receipts', '--out', 'prepared',
                 '--delta-mmhg', '1', '--support-caliper-iqr', '3'])
    assert observed[0]['delta_mmhg'] == 1.
    assert observed[0]['support_caliper_iqr'] == 3.
    driver.main(['prepare', '--tract-inputs', 'inputs', '--atlas', 'atlas',
                 '--usaleep-tar', 'payload', '--usaleep-receipts', 'receipts', '--out', 'prepared'])
    assert observed[1]['delta_mmhg'] == 2.
    assert observed[1]['support_caliper_iqr'] is None
