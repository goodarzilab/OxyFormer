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

    def factory(self, mode, config):
        self.configs.append(config)
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
    assert logger.configs and all({'fold', 'epoch', 'stop_reason', 'wall_seconds'} <= e.keys() for e in logger.events)
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
    assert logger.events[-1]['final_estimates'] == result['estimates']
    assert (tmp_path / 'score' / 'result.md').exists()
    record = json.loads((states[0] / 'progress.json').read_text())
    record['status'] = 'incomplete'
    driver.atomic_record(states[0] / 'progress.json', record)
    with pytest.raises(ContractError, match='complete folds'):
        driver.score('fixture', states, tmp_path / 'bad-score', 'none')


def test_wandb_driver_config(monkeypatch):
    from types import SimpleNamespace
    logger = FakeLogger()
    calls = []
    def initialize(**kwargs):
        calls.append(kwargs)
        return logger
    monkeypatch.setenv('WANDB_API_KEY', 'synthetic-secret')
    monkeypatch.setattr(driver.importlib, 'import_module', lambda name: SimpleNamespace(init=initialize))
    assert driver.logger_for('wandb', {'recipe': driver.STUDY}) is logger
    assert calls == [dict(entity='goodarzilab', project='oxyformer-v2', config={'recipe': driver.STUDY})]
    assert 'synthetic-secret' not in json.dumps(calls)
