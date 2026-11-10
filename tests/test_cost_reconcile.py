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
                assert value['basis'] in ('[measured]', '[derived]', '[extrapolated]', '[derived, measured]', ['measured'])
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
        assert term['harness_count']['basis'] == '[derived, measured]'
    for cell in result['harness_formula'].values():
        for count in cell['counts'].values():
            assert count['basis'] == '[derived, measured]'


@pytest.mark.parametrize('path', ['receipt', 'harness_overhead', 'harness_formula'])
def test_source_bases(inputs, path):
    m, p = inputs
    source = cr.load(p)
    quantity = {'value': 2, 'basis': ['measured', 'projected', 'uncalibrated', 'derived']}
    if path == 'harness_formula':
        source['formula']['e']['seconds']['inner_ssl'] = quantity
        source['cells']['e']['compute_h100_hours_per_repetition']['value'] = 6.001111111111111
    else:
        source['calibration']['receipt']['extra'] = quantity
        source['overhead']['extra'] = quantity
    write(p, source)
    result = cr.reconcile(m, p)[path]
    item = result['e']['seconds']['inner_ssl'] if path == 'harness_formula' else result['extra']
    assert item == {'value': 2, 'basis': '[derived, measured, projected, uncalibrated]'}


def test_nested_count_labels():
    result = cr.relabel({'work': {'harness_count': {'value': 2, 'basis': ['measured']},
                                 'counts': {'x': {'value': 3, 'basis': ['projected']}}}})
    assert result['work']['harness_count']['basis'] == '[derived, measured]'
    assert result['work']['counts']['x']['basis'] == '[derived, projected]'


def test_rounded_cell_hours(inputs):
    source = cr.load(inputs[1])
    source['cells']['e']['compute_h100_hours_per_repetition']['value'] = 7.000001
    write(inputs[1], source)
    assert cr.reconcile(*inputs)['summary']['harness_cell_e_hours']['value'] == 7.


@pytest.mark.parametrize('kind', ['extra', 'history'])
def test_archive_schema(inputs, kind):
    archive = inputs[0].parent / '0/continuation.tar'
    with tarfile.open(archive, 'a') as tar:
        name = 'controllerless_marker.json' if kind == 'extra' else 'blobs/part'
        data = json.dumps({'type': 'ControllerTimingHistory'}).encode()
        member = tarfile.TarInfo(name)
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    if kind == 'history':
        with pytest.raises(ValueError, match='ControllerTimingHistory'):
            cr.reconcile(*inputs)
    else:
        assert cr.reconcile(*inputs)['summary']['chain_count']['value'] == 2


def test_calibration_disclosure(inputs):
    source = cr.load(inputs[1])
    source['calibration'].update(passed=False, reasons=['frame mismatch'],
                                ratio={'value': 5., 'basis': ['uncalibrated']})
    source['calibration']['receipt']['tree'] = 'receipt-tree'
    write(inputs[1], source)
    result = cr.reconcile(*inputs)
    assert result['calibration'] == source['calibration']
    text = cr.markdown(result)
    opening = text[:text.index('**Accounting**')]
    for expected in ('False', 'frame mismatch', 'receipt-tree', 'frame_matches',
                     'uncalibrated', '31.8/29.0', 'transfer uncertainty'):
        assert expected in opening


def test_tree_provenance(inputs):
    r = cr.reconcile(*inputs)
    text = cr.markdown(r)
    section = text.split('**harness**')[1].split('**chains**')[0]
    assert '- tree: ' + cr.REPORT_TREE in section
    assert '- chain_tree: ' + cr.TREE in text


def test_input_output_collision(inputs, monkeypatch):
    p = inputs[1].with_name('reconciliation.json')
    p.write_bytes(inputs[1].read_bytes())
    original = p.read_bytes()
    monkeypatch.setattr('sys.argv', ['cost_reconcile', '--manifest', str(inputs[0]),
                                    '--report', str(p), '--out', str(p.parent)])
    with pytest.raises(ValueError, match='input'):
        cr.main()
    assert p.read_bytes() == original


@pytest.mark.parametrize('mode', ['forward', 'reverse', 'hardlink'])
def test_output_alias_refuses(inputs, monkeypatch, mode):
    a, b = (inputs[0].parent / name for name in ('reconciliation.json', 'reconciliation.md'))
    if mode == 'hardlink':
        a.write_text('preserved')
        b.hardlink_to(a)
    elif mode == 'forward':
        a.symlink_to(b)
    else:
        b.symlink_to(a)
    monkeypatch.setattr('sys.argv', ['cost_reconcile', '--manifest', str(inputs[0]),
                                    '--report', str(inputs[1]), '--out', str(a.parent)])
    with pytest.raises(ValueError, match='alias'):
        cr.main()
    if mode == 'hardlink':
        assert a.read_text() == b.read_text() == 'preserved'
    else:
        assert not (b if mode == 'forward' else a).exists()


@pytest.mark.parametrize('mode', ['receipt_symlink', 'receipt_hardlink', 'overhead_symlink',
                                  'dangling', 'existing', 'directory', 'forward', 'reverse'])
@pytest.mark.parametrize('destination', ['reconciliation.json', 'reconciliation.md'])
def test_fresh_outputs_preserve_sources(inputs, monkeypatch, mode, destination):
    import hashlib
    manifest, source = inputs
    receipt, overhead = manifest.parent / 'receipt.json', manifest.parent / 'overhead.json'
    receipt.write_text('sealed receipt')
    overhead.write_text('sealed overhead')
    harness = cr.load(source)
    harness['calibration']['receipt']['path'] = str(receipt)
    harness['overhead']['files'] = [{'path': str(overhead)}]
    write(source, harness)
    paths = [p for p in manifest.parent.rglob('*') if p.is_file()]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    out = manifest.parent / 'out'
    out.mkdir()
    dest = out / destination
    other = out / ('reconciliation.md' if destination.endswith('json') else 'reconciliation.json')
    if mode == 'receipt_symlink':
        dest.symlink_to(receipt)
    elif mode == 'receipt_hardlink':
        dest.hardlink_to(receipt)
    elif mode == 'overhead_symlink':
        dest.symlink_to(overhead)
    elif mode == 'dangling':
        dest.symlink_to(out / 'absent')
    elif mode == 'existing':
        dest.write_text('preserve existing')
    elif mode == 'directory':
        dest.mkdir()
    elif mode == 'forward':
        dest.symlink_to(other)
    else:
        other.symlink_to(dest)
    monkeypatch.setattr('sys.argv', ['cost_reconcile', '--manifest', str(manifest),
                                    '--report', str(source), '--out', str(out)])
    with pytest.raises((ValueError, FileExistsError)):
        cr.main()
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths} == hashes
    if mode == 'existing':
        assert dest.read_text() == 'preserve existing'
    if mode == 'dangling':
        assert not (out / 'absent').exists()
    if mode not in ('forward', 'reverse'):
        assert not other.exists()


def test_extra_study_metadata(inputs):
    path = inputs[0].parent / '0/budget_record.json'
    budget = cr.load(path)
    budget['binding']['study']['note'] = 'screening'
    write(path, budget)
    assert cr.reconcile(*inputs)['chains'][0]['unrecognized_study_keys'] == ['note']
    budget['binding']['study']['seed_count'] = 2
    write(path, budget)
    with pytest.raises(ValueError, match='screening'):
        cr.reconcile(*inputs)


@pytest.mark.parametrize('difference', [1e-5, 1.])
def test_archive_budget_rounding(inputs, difference):
    path = inputs[0].parent / '0/budget_record.json'
    budget = cr.load(path)
    budget['profile']['complete_repetition_seconds'][0] += difference
    budget['leaves'][0]['compute_seconds'] += difference
    budget['leaves'][0]['wall_seconds'] += difference
    write(path, budget)
    if difference == 1.:
        with pytest.raises(ValueError, match=r'archive compute mismatch.*3600.*3601'):
            cr.reconcile(*inputs)
    else:
        chain = cr.reconcile(*inputs)['chains'][0]
        assert chain['archive_budget_discrepancy_seconds'] == cr.q(
            abs(3600. - (3600. + difference)), 'measured')


def test_copied_formula_counts_outside_count_subtrees(inputs):
    source = cr.load(inputs[1])
    # Enumerate semantic counts independently of their position in the JSON.
    fields = [('outer_folds', 5), ('seed_count', 1),
              ('scoring_operations', {'one_step': 1, 'cv_tmle': 1, 'spatial_sensitivities': 1})]
    work = dict(fields)
    source['calibration']['receipt']['work'] = work
    write(inputs[1], source)
    result = cr.reconcile(*inputs)['receipt']['work']
    def check(source_value, copied):
        if isinstance(source_value, dict):
            for key, value in source_value.items():
                check(value, copied[key])
        else:
            assert copied['value'] == source_value
            assert 'derived' in copied['basis']
    for key, value in fields:
        check(value, result[key])


@pytest.mark.parametrize('receipt_hours,verdict', [(1.4, 'consistent'), (100., 'inconsistent')])
def test_candidate_assessment(inputs, receipt_hours, verdict):
    source = cr.load(inputs[1])
    source['calibration']['receipt']['compute_hours']['value'] = receipt_hours
    write(inputs[1], source)
    result = cr.reconcile(*inputs)
    assessment = result['candidates']['receipt identity']
    for name, hours in [('unadjusted', receipt_hours), ('scenario_adjusted', receipt_hours * 31.8 / 29.)]:
        item = assessment[name]
        assert item['verdict'] == verdict
        assert item['distance_sample_sds']['value'] == pytest.approx(abs(hours - 1.5) / math.sqrt(.5))
    assert 'two sample SDs' in assessment['rule']
    assert 'null_effect to sign_changing' in assessment['adjustment_direction']
    for name in ('representative-fold differences', 'epoch-slicing reconstruction'):
        assert result['candidates'][name]['verdict'] == 'not assessable'
        assert result['candidates'][name]['missing_input']
    from pathlib import Path
    assert 'Independent complete repetitions corroborate the receipt scale.' not in Path(cr.__file__).read_text()


def test_exclusive_reservation_failure_writes_no_payload(tmp_path, monkeypatch):
    import os
    a, b = tmp_path / 'a.json', tmp_path / 'b.md'
    real_open = os.open
    def collide(path, flags, mode):
        assert flags & os.O_EXCL and flags & os.O_NOFOLLOW
        if path == b:
            assert a.read_bytes() == b''
            b.write_text('concurrent file')
        return real_open(path, flags, mode)
    monkeypatch.setattr(cr.os, 'open', collide)
    with pytest.raises(FileExistsError):
        cr.write_outputs([a, b], [], ['json payload', 'markdown payload'])
    assert not a.exists()
    assert b.read_text() == 'concurrent file'
