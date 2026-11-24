"""Synthetic archives only; no saved study artifacts."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.stats import beta, t

from test_attribution_audit import saved_fixture
from oxyformer.provenance import ContractError
from oxyformer.validation import attribution_audit as audit
from oxyformer.validation import q_screen as screen
from oxyformer.validation.coverage import bind_observations
from oxyformer.validation.generators import ObservedRecords, TruthArtifact
from oxyformer.validation.oracle import ObservedLawOracle


def payload(chain, state, parts):
    return dict(schema=screen.SCHEMA, chain_dir=str(chain.resolve()),
        frame_hash=state['binding']['frame_hash'], variant='synthetic', study={}, code_commit='a'*40,
        parts=[dict(fold=p.fold_ids[0], seed=p.seed_ids[0], original_ids=list(p.original_ids),
                    mu_a=list(p.mu_a), mu_d=list(p.mu_d)) for p in parts])


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    chain = tmp_path_factory.mktemp('q-screen')/'chain'
    expected, _ = saved_fixture(chain)
    _, states, base, recipe = screen.load(chain)
    state = states[0]
    obs = ObservedRecords.from_json(state['sample']['observations'])
    truth = TruthArtifact.from_json(state['sample']['structural_causal_truth'])
    oracle = ObservedLawOracle(obs, truth, base.policy)
    endpoint = bind_observations(base, obs)
    return chain, state, endpoint, oracle, recipe, expected


def write(chain, value):
    path = chain/'candidate.json'; path.write_text(json.dumps(value)); return path


@pytest.mark.parametrize('oracle_q', [False, True])
def test_identity_and_oracle_splice(fixture, oracle_q):
    chain, state, endpoint, oracle, recipe, expected = fixture
    parts = state['parts']
    source = parts
    if oracle_q:
        source = [replace(p, mu_a=oracle.nuisances(p.original_ids).mu_a,
                          mu_d=oracle.nuisances(p.original_ids).mu_d) for p in parts]
    path = write(chain, payload(chain, state, source))
    new, hashes = screen.splice(path, chain, state['binding']['frame_hash'], endpoint, parts)
    assert hashes['non_q_before'] == hashes['non_q_after']
    for old, candidate in zip(parts, new):
        for field in ('original_ids', 'fold_ids', 'seed_ids', 'r_a', 'r_d', 'origin_weights', 'lineage', 'spec'):
            assert getattr(old, field) == getattr(candidate, field)
    actual = audit.replay(endpoint, new, oracle, recipe['inference']['county_locations'])['fitted']
    reference = expected['oracle_Q' if oracle_q else 'fitted']
    for method in reference:
        assert actual[method]['estimate']['value'] == pytest.approx(reference[method]['estimate']['value'], abs=1e-12)
        assert actual[method]['se'] == reference[method]['se']


@pytest.mark.parametrize('case,key', [('row_order','original_ids'), ('missing','parts'), ('extra','parts'),
    ('frame_hash','frame_hash'), ('chain_dir','chain_dir'), ('nan','mu_a'), ('length','mu_d'),
    ('r_a','r_a'), ('unknown','bogus'), ('missing_key','study'), ('part_missing_key','mu_d'),
    ('fold_type','fold'), ('seed_type','seed'), ('duplicate','parts')])
def test_refusals(fixture, case, key):
    chain, state, endpoint, _, _, _ = fixture
    value = payload(chain, state, state['parts'])
    if case == 'row_order': value['parts'][0]['original_ids'].reverse()
    elif case == 'missing': value['parts'].pop()
    elif case == 'extra': value['parts'].append({**deepcopy(value['parts'][0]), 'fold': 99})
    elif case in ('frame_hash','chain_dir'): value[case] = 'wrong'
    elif case == 'nan': value['parts'][0]['mu_a'][0] = float('nan')
    elif case == 'length': value['parts'][0]['mu_d'].pop()
    elif case == 'r_a': value['parts'][0]['r_a'] = [2.]*len(value['parts'][0]['mu_a'])
    elif case == 'unknown': value['bogus'] = 1
    elif case == 'missing_key': del value['study']
    elif case == 'part_missing_key': del value['parts'][0]['mu_d']
    elif case == 'fold_type': value['parts'][0]['fold'] = True
    elif case == 'seed_type': value['parts'][0]['seed'] = '1103'
    elif case == 'duplicate': value['parts'].append(deepcopy(value['parts'][0]))
    with pytest.raises(ContractError, match=key) as exc:
        screen.splice(write(chain, value), chain, state['binding']['frame_hash'], endpoint, state['parts'])
    assert str(chain.resolve()) in str(exc.value) and 'fold' in str(exc.value)


def test_g_hash_changes(fixture):
    _, state, endpoint, _, _, _ = fixture
    parts = state['parts']
    changed = [replace(p, r_a=(2.,)*len(p.r_a)) for p in parts]
    assert screen.non_q_hash(endpoint, parts) != screen.non_q_hash(endpoint, changed)


def test_nonmover_metrics(fixture):
    _, state, endpoint, oracle, _, _ = fixture
    source = []
    for p in state['parts']:
        true = oracle.nuisances(p.original_ids)
        # Perfect on movers; true contrast is identically zero off the mask.
        source.append(replace(p, mu_a=true.mu_a, mu_d=true.mu_d))
    result = screen.diagnostics(endpoint, source, oracle)
    assert result['contrast']['nonmovers']['rmse'] == 0
    assert result['contrast']['nonmovers']['bias'] == 0
    corrupted = []
    for p in source:
        rows = [oracle.rows[i] for i in p.original_ids]
        mask = endpoint.policy.shift_mask([oracle.observed.a[i] for i in rows], [oracle.frame.support_keys[i] for i in rows])
        corrupted.append(replace(p, mu_d=tuple(d if m else d+10 for d,m in zip(p.mu_d, mask))))
    metrics_with_bad_nonmovers = screen.diagnostics(endpoint, corrupted, oracle)
    assert metrics_with_bad_nonmovers['contrast']['movers']['rmse'] == 0
    assert metrics_with_bad_nonmovers['contrast']['all']['rmse'] > 0
    d = np.array([100., 2., 4., -100.]); d0 = np.array([0., 1., 3., 0.])
    metrics = screen.contrast_metrics(d, d0, np.array([False, True, True, False]))
    assert metrics == dict(n=2, bias=1., rmse=1., correlation=pytest.approx(1.), slope=1., sign_agreement=1.)


def table():
    rows=[]
    for i, delta in enumerate([-.001, 0, .001, 0]):
        cells={}
        for name, estimate, rmse in [('baseline', .01+delta, 2.), ('candidate', .001+delta, 1.),
                                      ('fitted', .01+delta, 2.), ('oracle_Q', delta, .1)]:
            cells[name] = dict(estimates={'cv_tmle':dict(estimate={'value':estimate}, se={'iid':{'value':.01},'spatial_100_km':{'value':.02}},
                                coverage={'iid':True,'spatial':True})},
                                diagnostics={'contrast':{'movers':{'rmse':rmse}}})
        rows.append(dict(chain_dir=str(i), repetition_id=str(i), psi0=0., campaign_mode='diagnostic', cells=cells))
    return rows


def test_statistics_and_direction():
    rows=table(); out=screen.summarize(rows, iid_coverage_min=4)
    c=out['cells']['candidate']
    assert out['B_0'] == pytest.approx(.01) and out['s'] == -1
    assert c['B_c'] == pytest.approx(.001) and c['bias_ratio'] == pytest.approx(.1)
    assert c['d'] == pytest.approx([.009]*4) and c['mean_d'] == pytest.approx(.009)
    assert c['mover_rmse_reduction'] == .5 and c['verdict'] == 'pass'
    assert c['coverage']['iid']['interval'] == pytest.approx([beta.ppf(.025,4,1),1.])
    # A nonconstant difference detects a one-sided quantile.
    rows[0]['cells']['candidate']['estimates']['cv_tmle']['estimate']['value'] += .001
    c=screen.summarize(rows, iid_coverage_min=4)['cells']['candidate']
    d=np.array([.008,.009,.009,.009]); h=t.ppf((1+.9875)/2,3)*d.std(ddof=1)/2
    assert c['paired_interval'] == pytest.approx([d.mean()-h,d.mean()+h])
    assert screen.clopper_pearson(0,4) == pytest.approx([0,beta.ppf(.975,1,4)])
    assert screen.clopper_pearson(2,4) == pytest.approx([beta.ppf(.025,2,3),beta.ppf(.975,3,2)])


@pytest.mark.parametrize('criterion,kwargs', [('bias_ratio', {'bias_ratio':.05}),
    ('paired_interval', {}), ('contrast_reduction', {'contrast_reduction':.75}),
    ('iid_coverage_min', {'iid_coverage_min':5})])
def test_each_failure(criterion, kwargs):
    rows=table()
    if criterion == 'paired_interval':
        for row, change in zip(rows, [-.02,.02,-.02,.02]):
            row['cells']['candidate']['estimates']['cv_tmle']['estimate']['value'] += change
    out=screen.summarize(rows, **{'iid_coverage_min':4, **kwargs})['cells']['candidate']
    assert out['verdict'] == 'fail' and out['failing_criteria'] == [criterion]


def test_floor_and_futility():
    rows=table()
    assert screen.summarize(rows,bias_floor=.02)['cells']['candidate']['verdict'] == 'floor_not_met'
    for row in rows:
        row['cells']['candidate']['estimates']['cv_tmle']['estimate']['value'] = .02
        row['cells']['candidate']['diagnostics']['contrast']['movers']['rmse'] = 3.
    assert screen.summarize(rows)['cells']['candidate']['gross_futility']


def test_modes_and_cli(fixture, tmp_path):
    chain, state, _, _, _, _=fixture
    path=write(chain,payload(chain,state,state['parts']))
    baseline=chain/'baseline.json'; baseline.write_bytes(path.read_bytes())
    row=screen.screen_chain(chain, baseline, {'candidate':path})
    assert row['cells']['baseline']['estimates'] == row['cells']['fitted']['estimates']
    assert row['cells']['baseline']['diagnostics']['decomposition']['one_step']['overall']['plugin'] == pytest.approx(.1)
    result=chain/'result.json'; data=json.loads(result.read_text())
    data['campaign_mode']='m2_pilot_d'; result.write_text(json.dumps(data))
    assert screen.load(chain)[0]['campaign_mode']=='m2_pilot_d'
    with pytest.raises(ContractError): audit.load(chain)
    data['campaign_mode']='unregistered'; result.write_text(json.dumps(data))
    with pytest.raises(ContractError): screen.load(chain)
    # CLI on two independent synthetic archives, with synthetic baseline/candidate files.
    data['campaign_mode']='diagnostic'; result.write_text(json.dumps(data))
    entry=dict(chain_dir=str(chain),baseline=str(baseline),candidates={'candidate':str(path)})
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps([entry,entry]))
    with pytest.raises(ContractError,match='duplicate'):
        screen.main([str(manifest),'--output-dir',str(tmp_path/'out')])


def test_unpaired_refuses():
    rows=table(); rows[0]['chain_dir']=rows[1]['chain_dir']
    with pytest.raises(ContractError,match='duplicate'): screen.summarize(rows)
    rows=table(); del rows[0]['cells']['candidate']
    with pytest.raises(ContractError,match='unpaired'): screen.summarize(rows)


def test_t3_splice_non_q_unchanged(fixture):
    chain, state, endpoint, _, _, _ = fixture
    parts = state['parts']; path=write(chain,payload(chain,state,parts))
    new, hashes=screen.splice(path,chain,state['binding']['frame_hash'],endpoint,parts)
    assert hashes['non_q_before']==hashes['non_q_after']
    assert [p.r_a for p in new]==[p.r_a for p in parts]


def test_cli_writes_exploratory_report(tmp_path, monkeypatch):
    rows=iter(table())
    monkeypatch.setattr(screen,'screen_chain',lambda **kwargs:next(rows))
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps([{}]*4))
    output=tmp_path/'report'
    assert screen.main([str(manifest),'--output-dir',str(output),'--iid-coverage-min','4'])==0
    report=json.loads((output/'q_screen.json').read_text())
    assert report['summary']['cells']['candidate']['verdict']=='pass'
    assert report['summary']['thresholds']=={**screen.DEFAULTS,'iid_coverage_min':4}
    assert 'exploratory' in (output/'q_screen.md').read_text()


@pytest.mark.parametrize('rmse', [0., None])
def test_undefined_contrast_reduction_preserves_floor(rmse):
    rows=table()
    for row in rows:
        for cell in row['cells'].values():
            cell['estimates']['cv_tmle']['estimate']['value'] *= .1
            cell['diagnostics']['contrast']['movers']['rmse']=rmse
    result=screen.summarize(rows,iid_coverage_min=4)['cells']['candidate']
    assert result['verdict']=='floor_not_met'
    assert result['mover_rmse_reduction'] is None
    assert 'contrast_reduction' in result['failing_criteria']
    assert not result['gross_futility']


def test_single_chain_has_undefined_paired_interval():
    result=screen.summarize(table()[:1],iid_coverage_min=1)['cells']['candidate']
    assert result['paired_interval']==[None,None]
    assert result['empirical_sd'] is None
    assert result['verdict']=='fail' and result['failing_criteria']==['paired_interval']
