"""Synthetic terminal archives model the pinned producer's retained evidence."""
import io
import json
import math
import tarfile

import pytest

from oxyformer.validation import cost_reconcile as cr


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def inputs(tmp_path):
    entries = []
    for index, seconds in enumerate((3600., 7200.)):
        root = tmp_path / str(index)
        root.mkdir()
        draw = {'repetition_id': 'rep-' + str(index), 'scenario': 'sign_changing'}
        binding = {'code_identity': cr.TREE, 'draws': [draw],
                   'scenario': {'name': 'sign_changing'},
                   'study': dict(frozen_epochs=30, grid_points=2, reference_cap=32, seed_count=1)}
        leaves = [{'task_id': f'task-{index}-{step}', 'step': step,
                   'compute_seconds': seconds * fraction,
                   'setup_seconds': 10., 'import_seconds': 2.,
                   'export_seconds': 3., 'publication_seconds': 4.}
                  for step, fraction in enumerate((.75, .25))]
        write(root / 'budget_record.json', {'binding': binding, 'complete': True, 'success': True,
              'profile': {'complete_repetition_seconds': [seconds]}, 'leaves': leaves})
        coverage = {'binding': binding, 'states': [{'binding': {'draw': draw},
                    'compute_seconds': seconds, 'pending': None, 'record': {'status': 'success'},
                    'parts': ['part']}]}
        with tarfile.open(root / 'continuation.tar', 'w') as tar:
            for name, value in [('coverage.json', coverage),
                                ('blobs/part', {'type': 'OOFNuisances', 'payload': {}})]:
                data = json.dumps(value).encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        for leaf in leaves:
            leaf['wall_seconds'] = sum(leaf[k] for k in cr.PHASES)
        write(root / 'budget_record.json', {'binding': binding, 'complete': True, 'success': True,
              'profile': {'complete_repetition_seconds': [seconds]}, 'leaves': leaves})
        entries.append({'chain': 'chain-' + str(index), 'terminal_leaf_attempt_dir': str(index),
                        'terminal_leaf_state': 'DONE'})
    manifest = tmp_path / 'manifest.json'
    write(manifest, {'tree': cr.TREE, 'campaign': 'm2-diag-A', 'chains': entries})
    formula = {cell: {'counts': {t: {'value': 2, 'basis': ['measured']} for t in cr.TERMS},
                     'seconds': {t: {'value': 1800., 'basis': ['measured']} for t in cr.TERMS}}
               for cell in 'abcde'}
    report = tmp_path / 'report.json'
    write(report, {'tree': cr.REPORT_TREE, 'formula': formula,
                  'cells': {c: {'compute_h100_hours_per_repetition': {'value': 7., 'basis': ['measured']}} for c in 'abcde'},
                  'calibration': {'receipt': {'compute_hours': {'value': 1.4, 'basis': ['measured']},
                                            'frame_matches': True, 'sources_match': True}},
                  'overhead': {'ratio': {'value': 2.1, 'basis': ['measured']}}})
    return manifest, report


def test_arithmetic_and_fallback(inputs):
    report = cr.reconcile(*inputs)
    s = report['summary']
    assert s['mean_compute_hours']['value'] == 1.5
    assert s['excess_hours']['value'] == 5.5
    assert s['sample_sd_hours']['value'] == pytest.approx(math.sqrt(.5))
    assert s['mean_standard_error_hours']['value'] == pytest.approx(.5)
    assert s['cell_e_correction_factor']['value'] == pytest.approx(1.5 / 7)
    assert s['factor_observed_min']['value'] == pytest.approx(1 / 7)
    assert s['factor_observed_max']['value'] == pytest.approx(2 / 7)
    assert report['scenario_adjustment']['ratio'] == cr.q(31.8 / 29.)
    assert report['scenario_adjustment']['receipt_adjusted_hours']['value'] == pytest.approx(1.4 * 31.8 / 29.)
    assert report['corrected_projection_hours'] == {'e': cr.q(1.5, 'extrapolated'), 'd': None, 'c': None, 'a': None}
    for term in report['terms'].values():
        assert term['harness_total_hours']['value'] == 1.
        assert term['real_count'] is term['real_seconds'] is term['excess_share'] is None
        assert term['correction_factor'] is None
    for chain, seconds in zip(report['chains'], (3600., 7200.)):
        assert chain['compute_seconds'] == cr.q(seconds, 'measured')
        assert chain['leaf_compute_seconds']['value'] == seconds
        assert chain['overhead_seconds']['setup_seconds']['value'] == 20.
        assert chain['operation_counters'] is None
    assert report['limit'] in cr.markdown(report)
    assert set(report['candidates']) == {'representative-fold differences', 'epoch-slicing reconstruction', 'receipt identity'}


def test_every_numeric_leaf_has_basis(inputs):
    report = cr.reconcile(*inputs)
    def walk(value):
        if isinstance(value, dict):
            if 'value' in value:
                assert set(value) == {'value', 'basis'}
                assert value['basis'] in ('[measured]', '[derived]', '[extrapolated]')
                assert math.isfinite(value['value'])
            else:
                for v in value.values():
                    walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        else:
            assert type(value) not in (int, float)
    walk(report)
    rendered = cr.markdown(report)
    assert '1.5 [derived]' in rendered
    assert '3600 [measured]' in rendered
    assert '1.5 [extrapolated]' in rendered


@pytest.mark.parametrize('mutation', ['incomplete', 'tree', 'sum', 'duplicate', 'formula', 'nan'])
def test_invalid_evidence_refuses(inputs, mutation):
    manifest, report = inputs
    budget_path = manifest.parent / '0/budget_record.json'
    budget = cr.load(budget_path)
    if mutation == 'incomplete':
        budget['complete'] = False
    elif mutation == 'tree':
        budget['binding']['code_identity'] = 'other'
    elif mutation == 'sum':
        budget['leaves'][0]['compute_seconds'] += 10
    elif mutation == 'nan':
        budget['leaves'][0]['setup_seconds'] = float('nan')
    elif mutation == 'duplicate':
        m = cr.load(manifest)
        m['chains'].append(m['chains'][0])
        write(manifest, m)
    elif mutation == 'formula':
        r = cr.load(report)
        r['formula']['e']['counts']['inner_ssl']['value'] = 3
        write(report, r)
    write(budget_path, budget)
    with pytest.raises(ValueError):
        cr.reconcile(*inputs)


def test_cli_outputs(inputs, tmp_path, monkeypatch):
    out = tmp_path / 'out'
    monkeypatch.setattr('sys.argv', ['cost_reconcile', '--manifest', str(inputs[0]),
                                    '--report', str(inputs[1]), '--out', str(out)])
    cr.main()
    result = cr.load(out / 'reconciliation.json')
    assert (out / 'reconciliation.md').read_text() == cr.markdown(result)


def test_leaf_wrapper_compute_is_separate_from_repetition(inputs):
    manifest, _ = inputs
    path = manifest.parent / '0/budget_record.json'
    budget = cr.load(path)
    budget['leaves'][-1]['compute_seconds'] += 3.
    budget['leaves'][-1]['wall_seconds'] += 3.
    write(path, budget)
    result = cr.reconcile(*inputs)['chains'][0]
    assert result['compute_seconds']['value'] == 3600.
    assert result['leaf_compute_seconds']['value'] == 3603.
    assert result['leaf_wrapper_seconds'] == cr.q(3.)


def test_formula_counts_are_derived(inputs):
    result = cr.reconcile(*inputs)
    for term in result['terms'].values():
        assert term['harness_count']['basis'] == '[derived]'
    for cell in result['harness_formula'].values():
        for count in cell['counts'].values():
            assert count['basis'] == '[derived]'
