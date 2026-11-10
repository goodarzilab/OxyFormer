from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test_scores import make_fixture
from test_coverage import synthetic_endpoint, smoke_recipe
from test_scm import config
from oxyformer.contracts import OOFNuisances, StageRequest
from oxyformer.estimation.mtp import one_step
from oxyformer.estimation.targeting import cv_tmle
from oxyformer.provenance import canonical_json
from oxyformer.validation import attribution_audit as audit
from oxyformer.validation.coverage import bind_observations, digest, _write_continuation
from oxyformer.validation.generators import generate_suite_a
from oxyformer.validation.oracle import ObservedLawOracle

def pinned_estimates():
    p, d, split, n, weights = make_fixture()
    common = dict(split=split, policy=p)
    return {'one_step': one_step(n, d, weights, n.spec, **common).to_json(),
            'cv_tmle': cv_tmle(n, d, weights, 'identity', n.spec, **common).estimate.to_json()}

def test_existing_estimators_byte_identical():
    assert {k: sha256(v.encode()).hexdigest() for k,v in pinned_estimates().items()} == json.loads(Path('tests/fixtures/oracle/estimates.json').read_text())['cuda' if torch.version.cuda else 'cpu']

@pytest.mark.parametrize('case,expected', [('i', 'branch i'), ('ii', 'branch ii'), ('partial', 'inconclusive (partial recovery)'), ('single', 'inconclusive (partial recovery)')])
def test_decision_rules(case, expected):
    truth = [.0178]*4
    fitted = [.003, .004, .005, .004]
    oracle = [.017, .018, .019, .018] if case == 'i' else [.002, .006, .004, .004]
    if case == 'partial':
        oracle = [.009, .010, .011, .010]
    if case == 'single':
        truth, fitted, oracle = truth[:1], fitted[:1], oracle[:1]
    result = audit.decision(truth, fitted, oracle)
    assert result['verdict'] == expected
    assert result['truth_flagged'] is False
    assert audit.decision([.02]*4, fitted*4 if len(fitted)==1 else fitted, [.02]*4)['truth_flagged']

def saved_fixture(root):
    template, frame = synthetic_endpoint(root)
    sample = generate_suite_a(frame, config('sign_changing'), template.policy, seed=7)
    endpoint = bind_observations(template, sample.observations)
    split = replace(endpoint.outer, seed_ids=(1103,))
    ids = split.original_ids
    weights = dict(zip(frame.original_ids, frame.weights))
    lineage = replace(split.lineage, unit_ids=ids, split_hash=split.content_hash,
        parent_hashes=(endpoint.data.manifest.content_hash,), model_hash=None, seed=None, parameter_count=None)
    n = OOFNuisances(spec=split.spec, original_ids=ids, fold_ids=split.fold_ids, seed_ids=(1103,)*len(ids),
        mu_a=(50.,)*len(ids), mu_d=(50.1,)*len(ids), r_a=(1.,)*len(ids), r_d=(1.,)*len(ids),
        origin_weights=tuple(weights[i] for i in ids), lineage=lineage)
    parts = []
    for fold in range(5):
        selected = [j for j in range(len(ids)) if n.fold_ids[j] == fold]
        values = {name: tuple(getattr(n, name)[j] for j in selected) for name in audit.FIELDS}
        parts.append(replace(n, lineage=replace(lineage, unit_ids=values['original_ids']), **values))
    oracle = ObservedLawOracle(sample.observations, sample.structural_causal_truth, template.policy)
    recipe = smoke_recipe(template, frame)
    with patch.object(audit, 'cv_tmle', wraps=cv_tmle) as calls:
        cells = audit.replay(endpoint, parts, oracle, recipe['inference']['county_locations'])
    for cell, call in zip(audit.CELLS, calls.call_args_list):
        seen = call.args[0]
        truth = oracle.nuisances(seen.original_ids)
        for field in ('mu_a', 'mu_d', 'r_a', 'r_d'):
            use = cell == 'oracle' or cell == ('oracle_Q' if field.startswith('mu') else 'oracle_g')
            assert getattr(seen, field) == (getattr(truth, field) if use else getattr(n, field))

    estimates = {method: {'value': cells['fitted'][method]['estimate']['value'], 'se': cells['fitted'][method]['se']['spatial_100_km']['value'],
        'spatial_se': {str(float(b)): cells['fitted'][method]['se'][f'spatial_{b}_km']['value'] for b in (50,100,200)}} for method in ('cv_tmle','one_step')}
    record = {'draw': {'seed': 7, 'repetition_id': 'synthetic-7'}, 'status': 'success',
              'causal_truth': sample.structural_causal_truth.value, 'estimates': estimates}
    root.mkdir(parents=True, exist_ok=True)
    lock = root/'lock.json'
    lock_raw=canonical_json({'recipe': recipe}); lock.write_text(lock_raw)
    lock_hash=sha256(lock_raw.encode()).hexdigest()
    binding = {'lock_hash': lock_hash, 'frame_hash': frame.content_hash, 'endpoint_hash': template.content_hash, 'recipe_hash': digest(recipe)}
    state = {'binding': binding, 'parts': [p.to_json() for p in parts], 'record': record,
        'pending': None, 'compute_seconds': 0.,
        'sample': {name: getattr(sample, name).to_json() for name in
                   ('observations','structural_causal_truth','observed_law_truth')}}
    _write_continuation(root/'continuation.tar', [state], binding=binding, chain={}, task_id='fixture')
    (root/'result.json').write_text(json.dumps({'terminal': True, 'campaign_mode': 'diagnostic', 'records': [record]}))
    execution=root/'_execution'; execution.mkdir()
    ep=root/'endpoint.json'; ep.write_text(template.to_json())
    task={'parameters': {}}; task_path=execution/'task.json'; task_path.write_text(canonical_json(task))
    request=StageRequest(stage='coverage', task_path=str(task_path), task_hash=digest(task),
        config_path=str(task_path), config_hash=digest(task), output_dir=str(root), code_identity='1'*40,
        dependency_paths=(str(ep), str(lock)), dependency_hashes=(template.content_hash, lock_hash))
    (execution/'request.json').write_text(request.to_json())
    return cells

def test_archive_cli(tmp_path, capsys):
    expected = saved_fixture(tmp_path/'saved')
    output=tmp_path/'audit'
    assert audit.main([str(tmp_path/'saved'), '--output-dir', str(output)]) == 0
    report=json.loads((output/'attribution_audit.json').read_text())
    assert report['repetitions'][0]['cells'] == expected
    assert set(expected) == set(audit.CELLS)
    assert expected['oracle'] != expected['fitted']
    assert set(expected['fitted']['one_step']['se']) == {'spatial_50_km','spatial_100_km','spatial_200_km','county_0_km','iid','frame_conditional'}
    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if type(child) in (int,float):
                    assert key == 'value' and value['basis']
                else:
                    walk(child)
        elif isinstance(value,list):
            for child in value: walk(child)
    walk(report)
    assert report['summary']['cv_tmle']['decision']['verdict'].startswith('inconclusive')
    assert 'paired t interval' in capsys.readouterr().out
    assert (output/'attribution_audit.md').exists()

def test_missing_archive(tmp_path, capsys):
    assert audit.main([str(tmp_path/'missing'), '--output-dir', str(tmp_path/'out')]) == 1
    report=json.loads((tmp_path/'out/attribution_audit.json').read_text())
    assert report['repetitions'] == [] and report['summary'] == {}
    assert report['skipped'][0]['status'].startswith('unreadable:')

def test_paired_t_interval_and_ratios():
    from scipy.stats import t
    result=audit.paired([1.,2.,3.])
    assert result['mean']['value'] == 2
    assert result['lower']['value'] == pytest.approx(2-t.ppf(.975,2)/np.sqrt(3))
