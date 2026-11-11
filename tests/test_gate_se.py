"""OD-5: labelled uncertainty, explicit selection, sealed legacy behavior."""
import ast
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oxyformer.provenance import ContractError, canonical_json
from oxyformer.validation import coverage, campaign, attribution_audit as audit
from oxyformer.estimation.mtp import one_step
from oxyformer.estimation.targeting import cv_tmle
from test_scores import make_fixture
from test_campaign import owner_cell

FIXTURES = Path(__file__).parent / 'fixtures/gate_se'


def fixture(name):
    return json.loads((FIXTURES / (name + '.json')).read_text())


def labelled_records(name='diagnostic'):
    f = fixture(name)
    for row in f['records']:
        if row['status'] == 'success':
            for estimate in row['estimates'].values():
                estimate['se_definitions'] = {
                    'unclustered_influence': {'se': .3, 'variance_target': 'fixed_frame_repetition'},
                    'spatial_primary': {'se': 2., 'variance_target': 'geographic_process'}}
    return f


@pytest.mark.parametrize('name', ['screening', 'diagnostic'])
def test_legacy_golden(name):
    f = labelled_records(name)
    assert canonical_json(coverage.summarize(f['records'], f['draws'], gate_se=None, **f['args'])) == canonical_json(f['summary'])


def test_diagnostic_collector_golden():
    f = labelled_records()
    # This is the function the diagnostic collector uses to populate its gate.
    assert canonical_json(campaign.diagnostic_summary(f['records'], f['draws'], null_scenario=True)) == canonical_json(f['diagnostic_gate_summary'])


@pytest.mark.parametrize('gate,se,target', [('unclustered_influence', .3, 'fixed_frame_repetition'),
                                          ('spatial_primary', 2., 'geographic_process')])
def test_selected_gate_numbers(gate, se, target):
    f = labelled_records()
    selected = coverage.summarize(f['records'], f['draws'], gate_se=gate, **f['args'])
    # An independent legacy run with the chosen SE substituted gives every gate number.
    substituted = deepcopy(f['records'])
    for row in substituted:
        if row['status'] == 'success':
            for estimate in row['estimates'].values():
                estimate['se'] = se
    expected = coverage.summarize(substituted, f['draws'], **f['args'])
    assert selected.pop('gate_se') == gate
    assert selected.pop('gate_variance_target') == target
    for method in coverage.METHODS:
        actual = selected['methods'][method]
        assert actual.pop('reported_se_mean') == 1.
        assert actual.pop('reported_se_variance_target') == 'geographic_process'
        assert actual['coverage_count'] != f['summary']['methods'][method]['coverage_count']
        assert actual['null_rejection_upper'] != f['summary']['methods'][method]['null_rejection_upper']
    assert selected == expected


@pytest.mark.parametrize('method', coverage.METHODS)
@pytest.mark.parametrize('gate', ['unclustered_influence', 'spatial_primary'])
@pytest.mark.parametrize('bad', ['missing', None, -1., float('nan'), float('inf'), True, '1'])
def test_declared_definition_fails_closed(method, gate, bad):
    f = labelled_records()
    entry = f['records'][0]['estimates'][method]['se_definitions']
    if bad == 'missing':
        del entry[gate]
    else:
        entry[gate]['se'] = bad
    with pytest.raises(ContractError, match=method + '.*' + gate):
        coverage.summarize(f['records'], f['draws'], gate_se=gate, **f['args'])


def test_unknown_selection():
    f = fixture('diagnostic')
    with pytest.raises(ContractError, match='gate_se'):
        coverage.summarize(f['records'], f['draws'], gate_se='iid', **f['args'])


@pytest.mark.parametrize('primary', [50., 100., 200.])
def test_definitions_match_attribution(primary):
    policy, data, split, n, weights = make_fixture()
    targeted = cv_tmle(n, data, weights, 'identity', n.spec, split=split, policy=policy)
    estimates = {'one_step': one_step(n, data, weights, n.spec, split=split, policy=policy),
                 'cv_tmle': targeted.estimate}
    groups = dict(zip(split.original_ids, ('a', 'a', 'b', 'b')))
    locations = {'a': (0., 0.), 'b': (1., 1.)}
    endpoint = SimpleNamespace(data=data, outer=split, policy=policy,
        geography=SimpleNamespace(rows=[SimpleNamespace(original_id=i, county=groups[i]) for i in split.original_ids]))
    # Audit merges parts in the supplied order; retain the two seed-specific fits.
    parts = [replace(n, original_ids=n.original_ids[j:j+4], fold_ids=n.fold_ids[j:j+4],
        seed_ids=n.seed_ids[j:j+4], mu_a=n.mu_a[j:j+4], mu_d=n.mu_d[j:j+4],
        r_a=n.r_a[j:j+4], r_d=n.r_d[j:j+4], origin_weights=n.origin_weights[j:j+4]) for j in (0, 4)]
    oracle = SimpleNamespace(nuisances=lambda ids: n)
    expected = audit.replay(endpoint, parts, oracle, locations)['fitted']
    result = coverage.labelled_estimates(estimates, n, data, split, weights, targeted, groups, locations, primary)
    names = {'spatial_50_km': 'spatial_50_km', 'spatial_100_km': 'spatial_100_km',
             'spatial_200_km': 'spatial_200_km', 'spatial_primary': f'spatial_{primary:g}_km',
             'county_cluster': 'county_0_km', 'unclustered_influence': 'iid',
             'residual_frame_conditional': 'frame_conditional'}
    for method, row in result.items():
        assert row['value'] == estimates[method].value
        assert row['se'] == row['spatial_se'][str(primary)]
        assert set(row['se_definitions']) == set(names)
        for name, audit_name in names.items():
            definition = row['se_definitions'][name]
            assert definition['se'] == pytest.approx(expected[method]['se'][audit_name]['value'], abs=1e-12, rel=0)
            assert definition['variance_target'] == ('fixed_frame_repetition' if name == 'unclustered_influence' else
                'frame_conditional_residual' if name == 'residual_frame_conditional' else 'geographic_process')
            assert definition['assumptions'].endswith('.')
            assert np.isfinite(definition['se'])


def test_leaf_adds_definitions(tmp_path):
    from test_attribution_audit import saved_fixture
    from oxyformer.validation.generators import ObservedRecords
    from oxyformer.training.fit import RecipeStudy
    saved_fixture(tmp_path / 'saved')
    _, states, base, recipe = audit.load(tmp_path / 'saved')
    state = states[0]
    endpoint = coverage.bind_observations(base, ObservedRecords.from_json(state['sample']['observations']))
    root = tmp_path / 'leaf'; root.mkdir()
    result = coverage.estimate_repetition(endpoint, recipe, root, float('inf'),
        study=RecipeStudy(seed_count=1), continuation={'parts': [p.to_json() for p in state['parts']]})
    for method in coverage.METHODS:
        assert len(result[method]['se_definitions']) == 7
        assert result[method]['se'] == result[method]['se_definitions']['spatial_primary']['se']


@pytest.mark.parametrize('gate', ['unclustered_influence', 'spatial_primary'])
def test_pilot_declaration(owner_cell, gate):
    declaration, approvals = owner_cell
    block = approvals['owner_decisions']['example_final']
    declaration.update(purpose='diagnostic_pilot', gate_se=gate)
    block.update(purpose='diagnostic_pilot', gate_se=gate)
    assert campaign.validate_campaign(declaration, approvals) == declaration
    assert declaration['repetitions_per_scenario'] >= coverage.MIN_REPETITIONS
    declaration['repetitions_per_scenario'] = block['repetitions_per_scenario'] = 5
    assert campaign.validate_campaign(declaration, approvals)['certifies_release_criteria'] is False


@pytest.mark.parametrize('purpose', ['final_coverage', 'anchor', 'refit_audit', 'recipe_ladder'])
@pytest.mark.parametrize('where', ['both', 'block', 'declaration'])
def test_gate_refused_elsewhere(owner_cell, purpose, where):
    declaration, approvals = owner_cell
    block = approvals['owner_decisions']['example_final']
    declaration['purpose'] = block['purpose'] = purpose
    if where in ('both', 'block'): block['gate_se'] = 'spatial_primary'
    if where in ('both', 'declaration'): declaration['gate_se'] = 'spatial_primary'
    with pytest.raises(ContractError, match='gate_se'):
        campaign.validate_campaign(declaration, approvals)


@pytest.mark.parametrize('where', ['block', 'declaration', 'both'])
@pytest.mark.parametrize('gate', ['iid', None, [], 1])
def test_unknown_gate_declaration(owner_cell, where, gate):
    declaration, approvals = owner_cell
    block = approvals['owner_decisions']['example_final']
    declaration['purpose'] = block['purpose'] = 'diagnostic_pilot'
    if where in ('block', 'both'): block['gate_se'] = gate
    if where in ('declaration', 'both'): declaration['gate_se'] = gate
    with pytest.raises(ContractError, match='gate_se'):
        campaign.validate_campaign(declaration, approvals)


@pytest.mark.parametrize('where', ['block', 'declaration', 'unequal'])
def test_gate_declaration_equality(owner_cell, where):
    declaration, approvals = owner_cell
    block = approvals['owner_decisions']['example_final']
    declaration['purpose'] = block['purpose'] = 'diagnostic_pilot'
    if where in ('block', 'unequal'): block['gate_se'] = 'spatial_primary'
    if where in ('declaration', 'unequal'): declaration['gate_se'] = 'unclustered_influence'
    with pytest.raises(ContractError, match='gate_se'):
        campaign.validate_campaign(declaration, approvals)


def collector_gate(f, tmp_path, monkeypatch, module=campaign):
    """Synthetic sealed inputs; exercise the real diagnostic collector gate assembly."""
    scenario = {'name': 'null_effect', 'effect': 'null'}
    batch = {'mode': 'diagnostic', 'scenario': scenario, 'draws': f['draws']}
    lock = {'recipe': {}, 'instances': {'diagnostic': 'synthetic'}, 'batches': [batch],
            'scenario_family': [scenario], 'retry_rules': coverage.RETRY_RULES}
    leaf = {'id': 'synthetic-leaf', 'parameters': {**batch, 'batch_id': 'batch-0', 'wall_seconds': 600}}
    task = {'campaign': 'synthetic', 'expected_leaves': [leaf['id']]}
    plan = {'tasks': [leaf, task], 'spec': {}}
    payload = {'lock_hash': coverage.digest(lock), **batch, 'batch_id': 'batch-0', 'records': f['records']}
    path = tmp_path / 'payload.json'; path.write_text(canonical_json(payload))
    budget = {'payload_hash': coverage.file_hash(path), 'binding': {'recipe_hash': coverage.digest(lock['recipe'])},
              'leaves': [{'wall_seconds': 15.}], 'complete': True}
    with monkeypatch.context() as m:
        m.setattr(module, 'load_lock', lambda *a: lock)
        m.setattr(module, '_plan_for_task', lambda *a: plan)
        m.setattr(module, 'load_budget', lambda *a, **k: budget)
        m.setattr(module, 'dependency', lambda *a: path)
        values, passing = module.collect(None, {}, task)
    assert passing is False  # Two retained failed/rejected repetitions.
    return values['gate.json']


def test_full_diagnostic_collector_gate_golden(tmp_path, monkeypatch):
    f = labelled_records()
    assert canonical_json(collector_gate(f, tmp_path, monkeypatch)) == canonical_json(f['collector_gate'])


def test_pinned_source_scoring_calls():
    """Keep the scoring operations visible to the pinned-source cost walker."""
    source = Path(coverage.__file__).read_text()
    tree = ast.parse(source)
    estimate = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == 'estimate_repetition')
    calls = Counter(c.func.id for c in ast.walk(estimate)
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                    and c.func.id in ('one_step', 'cv_tmle', 'spatial_sensitivities'))
    assert calls == Counter(one_step=1, cv_tmle=1, spatial_sensitivities=1)
