"""Offline synthetic acceptance tests for outcome-only exploratory fitting."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path

import pytest
import torch

from test_nested_cv import endpoint, tiny_config, state, predictions
from test_attribution_audit import saved_fixture
from oxyformer.design.policies import ShiftOrStayPolicy
from oxyformer.explore import q_refit as driver
from oxyformer.provenance import ContractError, file_hash
from oxyformer.training import nested_cv as n
from oxyformer.training.fit import subset


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def q_predict(p, a):
    return n.predict_q(a, subset(p.data.covariates(('x',)), a.prediction_inputs.original_ids), p.policy)


@pytest.fixture(scope='module')
def q_full(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp('q-full')
    p = endpoint(root)
    a = n.run_fold(tiny_config(p, root / 'q'), p.outer, 1103, geography=p.geography, q_only=True)
    return p, a


def test_exact_a0_equivalence(q_full, tmp_path):
    p, a = q_full
    full = n.run_fold(tiny_config(p, tmp_path / 'full'), p.outer, 1103, geography=p.geography)
    expected = predictions(p, full)
    actual = q_predict(p, a)
    assert actual == dict(original_ids=list(expected.original_ids),
                         mu_a=list(expected.mu_a), mu_d=list(expected.mu_d))
    assert state(a)['selection']['outcome'] == state(full)['selection']['outcome']
    # The policy leaves non-movers' contrasts exactly zero.
    moved = p.policy.shift_mask(a.prediction_inputs.a_mmhg, a.prediction_inputs.policy_covariates.support_keys)
    assert all(x == y for x, y, m in zip(actual['mu_a'], actual['mu_d'], moved) if not m)


@pytest.mark.parametrize('variant,head', [('A0', 'OutcomeTransformer'), ('A0-SW', 'OutcomeTransformer'),
    ('A3-Q', 'EarlyFusionOutcome'), ('A4-Q', 'VaryingCoefficientOutcome')])
def test_no_origin_and_registered_heads(tmp_path, monkeypatch, variant, head):
    from oxyformer.models.origin import OriginTransformer
    def refuse(*args, **kwargs):
        pytest.fail('Q-only constructed an origin model or fitted calibration')
    monkeypatch.setattr(OriginTransformer, '__init__', refuse)
    monkeypatch.setattr(n, 'fit_affine', refuse)
    p = endpoint(tmp_path)
    a = n.run_fold(tiny_config(p, tmp_path / 'fit'), p.outer, 1103,
                   geography=p.geography, variant=variant, q_only=True)
    controller = state(a)
    assert {r['kind'] for r in controller['results']} == {'outcome'}
    assert set(controller['final']) == {'outcome'}
    assert controller['calibration'] is None
    assert controller['counts']['nuisance_fits'] == 13
    model = n._build(controller['final']['outcome'])
    assert type(model).__name__ == head
    assert sum(p.numel() for p in model.parameters()) <= 1_000_000
    assert all(not key.startswith('origin') for key in controller['phase_seconds'])
    assert len(q_predict(p, a)['mu_a']) == len(a.prediction_inputs.original_ids)
    if variant == 'A0-SW':
        config = tiny_config(p, tmp_path / 'unused')
        for r in controller['results']:
            pred = torch.tensor([v[0] for v in r['predictions']], dtype=torch.float32)
            # Independent hand-weighted squared error and effective pooled mass.
            y = torch.tensor(n._values(config, 'y', r['ids']), dtype=torch.float32).double()
            weights = torch.tensor(n._inputs(config, r['ids']).origin_weights).double()
            weights *= torch.tensor(n._q_weights(config, r['ids'])).double()
            assert r['metrics'][0] == pytest.approx(float(((pred.double()-y)**2*weights).sum()/weights.sum()), rel=1e-14)
            assert Fraction(*r['mass_ratio']) == sum(map(Fraction, weights.tolist()))


@pytest.mark.parametrize('variant', ['A1', 'A2', 'A3', 'A6-Q', 'unknown'])
def test_refuse_variant(tmp_path, variant):
    p = endpoint(tmp_path)
    with pytest.raises(ContractError, match='variant'):
        n.run_fold(tiny_config(p, tmp_path/'fit'), p.outer, 1103,
                   geography=p.geography, q_only=True, variant=variant)


def test_support_weights_active_delta():
    policy = ShiftOrStayPolicy(support_design_hash='1'*64,
        components_by_key=(('support', ((10., 14.),)), ('empty', ())))
    a = (10., 11.5, 12., 13.5, 20.)
    keys = ('support',)*4 + ('empty',)
    assert n.support_weights(policy, a, keys) == (2, 2, 3, 2, 1)
    assert n.support_weights(replace(policy, delta_mmhg=1.), a, keys) == (2, 3, 3, 2, 1)


@pytest.mark.parametrize('batches', [1, 6, 12])
def test_portable_q_resume(q_full, tmp_path, batches):
    p, expected = q_full
    a = n.run_fold(tiny_config(p, tmp_path/'first', max_batches=batches), p.outer, 1103,
                   geography=p.geography, q_only=True)
    assert not a.complete
    archive = tmp_path/'portable.tar'
    n.export_continuation(a, archive, binding={'q': True}, task_id='first',
        chain={'owner':'synthetic','step':0,'predecessor':None}, allowed_root=tmp_path)
    dest = tmp_path/'restored'
    dest.mkdir()
    restored = n.import_continuation(archive, dest, binding={'q':True},
        chain={'owner':'synthetic','step':1,'predecessor':'first'})
    actual = n.run_fold(tiny_config(p, dest/'next', predecessor=restored), p.outer, 1103,
                        geography=p.geography, q_only=True)
    assert q_predict(p, actual) == q_predict(p, expected)
    assert state(actual)['counts'] == state(expected)['counts']


@pytest.mark.parametrize('variant', ['A0', 'A0-SW', 'A3-Q', 'A4-Q'])
def test_outer_outcome_leakage(tmp_path, variant):
    p = endpoint(tmp_path)
    def fit(p, name):
        a = n.run_fold(tiny_config(p, tmp_path/name), p.outer, 1103,
                       geography=p.geography, q_only=True, variant=variant)
        return q_predict(p, a)
    expected = fit(p, 'before')
    held = set(p.outer.evaluation_ids(0))
    names = [c.name for c in p.data.manifest.schema]
    y = names.index('y')
    rows = tuple(tuple(v + 1234. if j == y and oid in held else v for j,v in enumerate(row))
                 for oid,row in zip(p.data.manifest.original_ids, p.data.rows))
    changed = replace(p, data=replace(p.data, rows=rows))
    assert fit(changed, 'after') == expected


@pytest.mark.parametrize('mode', ['diagnostic', 'm2_pilot_d'])
def test_bind_authenticated_archive(tmp_path, mode):
    chain = tmp_path/'chain'
    saved_fixture(chain)
    result = json.loads((chain/'result.json').read_text())
    result['campaign_mode'] = mode
    (chain/'result.json').write_text(json.dumps(result))
    before = {str(p.relative_to(chain)): file_hash(p) for p in chain.rglob('*') if p.is_file()}
    bound = driver.bind(chain, tmp_path/'bound')
    p, _, info = driver.load_bound(tmp_path/'bound')
    assert bound == info and info['archive_hash'] == file_hash(chain/'continuation.tar')
    assert sum(len(v['original_ids']) for v in info['expected_parts']) == len(p.outer.original_ids)
    assert before == {str(p.relative_to(chain)): file_hash(p) for p in chain.rglob('*') if p.is_file()}
    # A changed blob digest refuses, and an unregistered third mode refuses.
    raw = (chain/'continuation.tar').read_bytes()
    (chain/'continuation.tar').write_bytes(raw.replace(b'synthetic', b'synthetiX', 1))
    with pytest.raises((ContractError, ValueError)):
        driver.bind(chain, tmp_path/'bad-digest')
    (chain/'continuation.tar').write_bytes(raw)
    result['campaign_mode'] = 'coverage'
    (chain/'result.json').write_text(json.dumps(result))
    with pytest.raises(ContractError, match='terminal'):
        driver.bind(chain, tmp_path/'bad-mode')


def prepared_tiny(root):
    root.mkdir()
    p = endpoint(root)
    (root/'endpoint.json').write_text(p.to_json())
    recipe = dict(nested_cv=dict(ssl_epochs=1, frozen_epochs=1, synthetic=True))
    driver.atomic_record(root/'recipe.json', recipe)
    binding = dict(chain_dir=str(root/'synthetic-chain'), frame_hash='2'*64, archive_hash='3'*64,
        endpoint_hash=p.content_hash, recipe_hash=driver.coverage.digest(recipe))
    driver.atomic_record(root/'binding.json', binding)
    return p, binding


def test_fit_slice_journal_resume(tmp_path, monkeypatch):
    p, _ = prepared_tiny(tmp_path/'bound')
    prepared_tiny(tmp_path/'uninterrupted')
    # Synthetic unit-fit schedules only; production STUDIES remain the registered cells.
    monkeypatch.setattr(driver, 'STUDIES', {'cell-e':dict(reference_cap=32, frozen_epochs=None,
                                                       grid_points=2, seed_count=1)})
    monkeypatch.setattr(driver, 'checkout_identity', lambda: '4'*40)
    actual_run = n.run_fold
    saved_counts = []
    calls = []
    def bounded(config, *args, **kwargs):
        if not calls:
            config = replace(config, max_batches=7)
        if config.predecessor:
            saved_counts.append(state(config.predecessor)['counts'])
        result = actual_run(config, *args, **kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(n, 'run_fold', bounded)
    args = dict(fold=0, variant='A0', study='cell-e', slice_seconds=600., run_logging='none', device='cpu')
    first = driver.fit(tmp_path/'bound', **args)
    assert first['status'] == 'incomplete' and first['predecessor']
    second = driver.fit(tmp_path/'bound', **args)
    assert second['status'] == 'complete'
    assert saved_counts == [first['counts']]
    assert len(second['slices']) == 2
    complete = driver.fit(tmp_path/'uninterrupted', **args)
    assert second['parts'] == complete['parts']
    assert second['counts'] == complete['counts']
    assert set(second['phase_seconds']) >= {'ssl/0/None', 'outcome/0/0', 'outcome/3/0', 'prediction'}
    # Once complete, another invocation performs no work.
    count = len(calls)
    assert driver.fit(tmp_path/'bound', **args) == second
    assert len(calls) == count
    monkeypatch.setattr(driver, 'fit', lambda **kwargs: {'status':'incomplete'})
    assert driver.main(['fit','unused','--fold','0','--variant','A0','--study','cell-e',
                        '--slice-seconds','600','--run-logging','none']) == 75


def test_collect_exact_schema_and_order(tmp_path):
    root = tmp_path/'bound'
    p, binding = prepared_tiny(root)
    identity = dict(endpoint_hash=p.content_hash, frame_hash=binding['frame_hash'],
        chain_dir=binding['chain_dir'], variant='A4-Q', study=driver.STUDIES['cell-d'], code_commit='4'*40)
    driver.atomic_record(root/'candidate.json', identity)
    for fold in range(5):
        ids = list(p.outer.evaluation_ids(fold))
        part = dict(fold=fold, seed=1103, original_ids=ids, mu_a=[1.]*len(ids), mu_d=[2.]*len(ids))
        driver.atomic_record(root/f'fold-{fold}'/'progress.json',
            dict(identity=identity, status='complete', fold=fold, seed=1103, parts=[part]))
    out = driver.collect(root)
    assert set(out) == {'schema','chain_dir','frame_hash','variant','study','code_commit','parts'}
    assert out['schema'] == 'oxyformer.q_parts.v1'
    assert all(set(p) == driver.PART_KEYS for p in out['parts'])
    path = root/'fold-0'/'progress.json'
    good = json.loads(path.read_text())
    for bad in [dict(good['parts'][0], fold=1), dict(good['parts'][0], r_a=[1.]*6),
                dict(good['parts'][0], mu_a=[float('nan')]*6),
                dict(good['parts'][0], original_ids=list(reversed(good['parts'][0]['original_ids'])))]:
        driver.atomic_record(path, dict(good, parts=[bad]))
        with pytest.raises(ContractError):
            driver.collect(root)


@pytest.mark.parametrize('seconds', [0., -1., 10800.1, float('nan')])
def test_slice_bounds(seconds):
    with pytest.raises(ContractError, match='slice'):
        driver.fit('unused', 0, 'A0', 'cell-d', seconds, 'none')
