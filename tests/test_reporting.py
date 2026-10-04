"""Synthetic, offline CPU tests for reporting boundaries and scientific stops."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from oxyformer.contracts import CovariateView, Estimate, EstimandSpec, SourceManifest, StageRequest, StageResult, source_lineage_hash
from oxyformer.data.feature_roles import FeatureRegistry, FeatureRule
from oxyformer.provenance import ArtifactLineage, ArtifactRecord, ContractError, canonical_json, file_hash, write_artifact
from oxyformer.reporting import run_stage
from oxyformer.reporting.diagnostics import concentration, summarize
from oxyformer.reporting.evidence_matrix import STAGE_GATES, evaluate, multiplicity
from oxyformer.reporting.records import CoverageScenario, ExpectedTask, ExpectedTasks, ReportBundle, Sensitivity, TaskReceipt, TaskReceipts
from oxyformer.reporting.render import render_forest, render_html
from oxyformer.reporting import stage
from oxyformer.validation.overlap import overlap_report, weight_diagnostics

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/reporting.yaml'


def owner_parameters():
    # Exact approved values in PR #18; no writes to configs/approvals.yaml.
    return {"owner_decisions": {
        "release_gates": {
            "min_repetitions_per_scenario": 1000,
            "coverage_one_sided_95_lower_bound_min": .925,
            "null_rejection_upper_bound_max": .075,
            "abs_bias_over_empirical_sd_target": .10,
            "abs_bias_over_empirical_sd_investigate_above": .20,
            "mean_se_over_empirical_sd_range": [.90, 1.10],
            "numerical_failure_upper_bound_max": .01,
            "publish_all_failures_and_registered_retry_rules": True,
        },
        "influence_concentration_gate": {
            "applies_to": ["one_step", "cv_tmle"],
            "definitions": "U_g = sum of u_i over county g (unkernelized); D = sum_g U_g^2; s_g = U_g^2 / D; s_max = max_g s_g; G_eff = 1 / sum_g s_g^2",
            "s_max_max": .10, "g_eff_min": 30,
            "require": "D > 0 and every contribution finite",
            "on_failure": "block confirmatory release; publish the estimate as diagnostic-only with s_max, G_eff and ranked county contributions; never trim counties, cap ratios or reweight post hoc",
        },
    }}


def approve(bundle, manifest, receipts):
    approvals = owner_parameters()
    scope = dict(stage=manifest.stage, bundle_hash=bundle.content_hash, manifest_hash=manifest.content_hash,
                 receipts_hash=receipts.content_hash, config_hash=file_hash(CONFIG))
    gates = list(STAGE_GATES[manifest.stage]) + ['expected_manifest', 'tract_release', 'bias_investigation']
    approvals['owner_decisions']['reporting_approvals'] = [
        dict(scope, gate=g, status='approved', reviewer='synthetic reviewer', reference='synthetic-only') for g in gates]
    return approvals


@pytest.fixture
def case(tmp_path):
    n = 40
    ids = tuple(f'id-{i}' for i in range(n))
    source = SourceManifest(source_id='synthetic', version='1', uri='synthetic://only', payload_hash='1'*64,
                            license_hash='2'*64, schema_hash='3'*64, field_mapping=(('raw', 'id'),),
                            mapping_status='reviewed', mapping_review_id='synthetic')
    spec = EstimandSpec(endpoint='usaleep_life_expectancy', target_id='synthetic target', outcome_scale='years',
                        policy_id='synthetic shift2', weight_id='equal tract', adjustment_schema_hash='4'*64,
                        inference_unit='county', source_lineage_hash=source_lineage_hash((source,)))
    lineage = ArtifactLineage(source_hashes=(source.payload_hash,), unit_ids=ids, parent_hashes=(),
                              split_hash='5'*64, config_hash='6'*64, model_hash=None,
                              environment=(('python', 'synthetic'),), seed=None, parameter_count=None)
    seeds = (1103, 2207, 3301)
    u = tuple(.01 if i % 2 else -.01 for i in range(n))
    one = Estimate(spec=spec, method='mtp_one_step', value=.4, standard_error=None, original_ids=ids,
                   scores=tuple(.4+n*v for v in u), influence=u, seed_ids=seeds, lineage=lineage)
    tmle = replace(one, method='cv_tmle_identity', value=.41, scores=tuple(.41+n*v for v in u))
    coverage = CoverageScenario(scenario_id='null', repetitions=1000, independent_repetitions=True,
                                production_tuning_and_stopping=True, coverage_one_sided_95_lower_bound=.94,
                                null_rejection_upper_bound=.06, abs_bias_over_empirical_sd=.05,
                                mean_se_over_empirical_sd=1., numerical_failure_upper_bound=.005,
                                all_failures_published=True, registered_retry_rules=('no retries',))
    bundle = ReportBundle(spec=spec, sources=(source,), original_ids=ids, attrition=(('source', 60), ('eligible', 50), ('target', n)),
                          weights=(1.,)*n, observed_exposure=(1.,)*n, shifted_exposure=(3.,)*n,
                          seed_ids=seeds, ratios=((1.,)*n,)*len(seeds), balance_basis_id='frozen synthetic functions',
                          balance_names=('constant', 'A'), balance_observed=((1., 1.),)*n, balance_shifted=((1., 3.),)*n,
                          estimates=(one, tmle), counties=tuple(f'county-{i}' for i in range(n)),
                          states=tuple(f'state-{i//4}' for i in range(n)),
                          county_locations=tuple((f'county-{i}', (30.+i*.1, -110.)) for i in range(n)),
                          coverage=(coverage,), p_values=(('usaleep_life_expectancy', .01),))
    tasks, receipts = [], []
    for i, gate in enumerate(STAGE_GATES['tract_release'] + ('coverage',)):
        upstream = tmp_path / f'upstream-{i}'
        upstream.mkdir()
        config, task = upstream / 'config.yaml', upstream / 'task.json'
        config.write_text('synthetic: true\n')
        task.write_text('{}')
        output = upstream / 'out'
        output.mkdir()
        result_file = output / 'result.json'
        result_file.write_text('{"synthetic":true}')
        request = StageRequest(stage=gate, config_path=str(config), config_hash=file_hash(config),
                               task_path=str(task), task_hash=file_hash(task), dependency_paths=(), dependency_hashes=(),
                               output_dir=str(output), code_identity='a'*40)
        result = StageResult(request_hash=request.content_hash, status='pass', message='synthetic task passed',
                             artifacts=(ArtifactRecord(path='result.json', sha256=file_hash(result_file), lineage=lineage, kind='synthetic'),))
        task_id = f'{gate}-{i}'
        tasks.append(ExpectedTask(task_id=task_id, gate=gate, request_hash=request.content_hash))
        receipts.append(TaskReceipt(task_id=task_id, request=request, result=result))
    manifest = ExpectedTasks(stage='tract_release', spec=spec, seed_ids=seeds, tasks=tuple(tasks), coverage_scenarios=('null',))
    receipts = TaskReceipts(items=tuple(receipts))
    return bundle, manifest, receipts


def evaluate_case(case, approvals=None):
    b, m, r = case
    return evaluate(b, m, r, approve(b, m, r) if approvals is None else approvals, file_hash(CONFIG))


def test_complete_scoped_release_keeps_both_estimators_and_every_spatial_bandwidth(case):
    report = evaluate_case(case)
    assert report['state'] == 'released'
    assert report['releasable']
    assert [e['method'] for e in report['estimators']] == ['mtp_one_step', 'cv_tmle_identity']
    assert set(report['diagnostics']['spatial_sensitivities']) == {'50.0', '100.0', '200.0'}
    assert len(report['diagnostics']['aligned_influence']['values']) == 40
    assert np.asarray(report['diagnostics']['cluster_covariance']['matrix']).shape == (2, 2)
    assert not report['multiplicity']['tract_lung']['complete']
    assert not report['multiplicity']['adult_mortality']['complete']
    assert report['multiplicity']['tract_lung']['adjusted_p_values'] == {}
    for renderer in (render_html, render_forest):
        rendered = renderer(report)
        assert 'mtp_one_step' in rendered and 'cv_tmle_identity' in rendered
    assert 'Expected anchor signs are never acceptance criteria' in render_html(report)


def test_missing_coverage_approval_blocks_release(case):
    approvals = approve(*case)
    approvals['owner_decisions']['reporting_approvals'] = [a for a in approvals['owner_decisions']['reporting_approvals'] if a['gate'] != 'coverage']
    report = evaluate_case(case, approvals)
    assert report['state'] == 'blocked'
    assert not report['releasable']
    assert report['evidence_label'] == 'diagnostic-only'


@pytest.mark.parametrize('missing', ['release_gates', 'influence_concentration_gate', 'reporting_approvals'])
def test_missing_owner_gate_parameters_never_release(case, missing):
    approvals = approve(*case)
    approvals['owner_decisions'].pop(missing)
    assert not evaluate_case(case, approvals)['releasable']


@pytest.mark.parametrize('index', range(13))
def test_every_external_approval_is_required(case, index):
    approvals = approve(*case)
    records = approvals['owner_decisions']['reporting_approvals']
    if index < len(records)-1:  # last entry is conditional bias investigation
        records.pop(index)
        assert not evaluate_case(case, approvals)['releasable']


def test_partial_fan_in_even_with_all_gate_names_and_approvals_is_missing(case):
    b, m, r = case
    partial = replace(r, items=r.items[:-1])  # second coverage task, first is still present
    report = evaluate_case((b, m, partial))
    assert report['state'] == 'missing'
    assert not report['releasable']
    assert any(g.get('task_id') == m.tasks[-1].task_id and g['status'] == 'missing' for g in report['gates'])


def test_incomplete_expected_manifest_is_missing(case):
    b, m, r = case
    m = replace(m, tasks=tuple(t for t in m.tasks if t.gate != 'birth_anchor'))
    r = replace(r, items=tuple(t for t in r.items if t.task_id in {x.task_id for x in m.tasks}))
    assert evaluate_case((b, m, r))['state'] == 'missing'


@pytest.mark.parametrize('status,expected', [('blocked', 'blocked'), ('fail', 'failed')])
def test_upstream_blocked_and_failed_are_distinct(case, status, expected):
    b, m, r = case
    first = replace(r.items[0], result=replace(r.items[0].result, status=status))
    report = evaluate_case((b, m, replace(r, items=(first,)+r.items[1:])))
    assert report['state'] == expected
    assert len(report['estimators']) == 2


def test_upstream_missing_artifact_and_tampered_artifact(case):
    b, m, r = case
    artifact = Path(r.items[0].request.output_dir) / 'result.json'
    artifact.unlink()
    assert evaluate_case(case)['state'] == 'missing'
    artifact.write_text('changed')
    assert evaluate_case(case)['state'] == 'failed'


@pytest.mark.parametrize('field,value', [('target_id', 'other'), ('policy_id', 'other'), ('weight_id', 'other')])
def test_target_identity_mismatch_keeps_every_estimator_visible(case, field, value):
    b, m, r = case
    altered = replace(b.estimates[1], spec=replace(b.spec, **{field: value}))
    report = evaluate_case((replace(b, estimates=(b.estimates[0], altered)), m, r))
    assert report['state'] == 'failed'
    assert len(report['estimators']) == 2
    assert 'mismatch' in report['gates'][-1]['reason']


def test_seed_misalignment_rejected_and_original_ids_aligned(case):
    b, m, r = case
    wrong = replace(b.estimates[1], seed_ids=(1103,))
    assert evaluate_case((replace(b, estimates=(b.estimates[0], wrong)), m, r))['state'] == 'failed'
    e = b.estimates[1]
    reversed_e = replace(e, original_ids=e.original_ids[::-1], influence=e.influence[::-1], scores=e.scores[::-1])
    aligned = evaluate_case((replace(b, estimates=(b.estimates[0], reversed_e)), m, r))
    assert aligned['state'] == 'released'
    np.testing.assert_allclose(aligned['diagnostics']['cluster_covariance']['matrix'],
                               evaluate_case(case)['diagnostics']['cluster_covariance']['matrix'])


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_scores_and_influence_refused(case, bad):
    b, _, _ = case
    for field in ('scores', 'influence'):
        with pytest.raises(ContractError):
            replace(b.estimates[0], **{field: (bad,)+getattr(b.estimates[0], field)[1:]})


def test_concentration_uses_county_sum_before_squaring_and_seed_count_is_irrelevant():
    result = concentration([1., -1., 2., -2.], ['a', 'a', 'b', 'c'], ['s', 's', 't', 't'])
    assert result['D'] == 8
    assert result['G_eff'] == 2
    assert result['s_max'] == .5
    assert result['ranked_counties'][-1]['share'] == 0
    zero = concentration([1., -1.], ['a', 'a'], ['s', 's'])
    assert not zero['positive_D']


@pytest.mark.parametrize('which', [0, 1])
def test_concentrated_estimator_blocks_and_keeps_diagnostic_estimates(case, which):
    b, m, r = case
    estimates = list(b.estimates)
    estimates[which] = replace(estimates[which], influence=(1., -1.) + (0.,)*38)
    report = evaluate_case((replace(b, estimates=tuple(estimates)), m, r))
    assert report['state'] == 'failed'
    assert report['evidence_label'] == 'diagnostic-only'
    assert len(report['estimators']) == 2
    assert report['diagnostics']['information'][estimates[which].method]['G_eff'] == 2


@pytest.mark.parametrize('field,value', [
    ('repetitions', 999), ('independent_repetitions', False), ('production_tuning_and_stopping', False),
    ('coverage_one_sided_95_lower_bound', .92), ('null_rejection_upper_bound', .08),
    ('mean_se_over_empirical_sd', 1.11), ('numerical_failure_upper_bound', .02),
    ('all_failures_published', False), ('registered_retry_rules', ())])
def test_each_coverage_failure_blocks_even_with_external_approval(case, field, value):
    b, m, r = case
    b = replace(b, coverage=(replace(b.coverage[0], **{field: value}),))
    report = evaluate_case((b, m, r))
    assert report['state'] == 'failed'
    assert report['evidence_label'] == 'diagnostic-only'


def test_bias_is_target_and_large_bias_requires_investigation(case):
    b, m, r = case
    b = replace(b, coverage=(replace(b.coverage[0], abs_bias_over_empirical_sd=.15),))
    assert evaluate_case((b, m, r))['state'] == 'released'
    b = replace(b, coverage=(replace(b.coverage[0], abs_bias_over_empirical_sd=.21),))
    approvals = approve(b, m, r)
    approvals['owner_decisions']['reporting_approvals'] = [a for a in approvals['owner_decisions']['reporting_approvals'] if a['gate'] != 'bias_investigation']
    assert evaluate_case((b, m, r), approvals)['state'] == 'blocked'


def test_overlap_distinguishes_sampling_and_signed_weights_and_affected_stayers():
    report = overlap_report([1, 1, 1, 1], [0, 1, 2, 1], [1, 5, 9, 9], [3, 7, 9, 9], ['A'], [[1], [5], [9], [9]], [[3], [7], [9], [9]])
    assert report['moved_fraction'] == .5
    assert report['affected_fraction'] == .75
    assert report['achieved_shift'] == 1
    assert report['subsets']['all']['signed_correction']['ess'] is None
    assert report['subsets']['all']['target_weights']['ess'] == 4
    assert report['subsets']['all']['ratio_weights']['ess'] == pytest.approx(16/6)
    balance = report['functional_balance'][0]
    assert balance['ratio_expectation'] == 8
    assert balance['shifted_expectation'] == 7
    with pytest.raises(ContractError):
        weight_diagnostics([-1, 2])


def test_ratio_warnings_do_not_automatically_fail_release(case):
    b, m, r = case
    b = replace(b, ratios=((100.,)+(0.,)*39,)*3)
    report = evaluate_case((b, m, r))
    assert report['state'] == 'released'  # overlap review scoped to these diagnostics is present
    assert report['diagnostics']['overlap_by_seed']['1103']['subsets']['all']['warnings']


def test_material_disagreement_requires_review_never_selects_estimator(case):
    b, m, r = case
    b = replace(b, estimates=(b.estimates[0], replace(b.estimates[1], value=100.)))
    approvals = approve(b, m, r)
    approvals['owner_decisions']['reporting_approvals'] = [a for a in approvals['owner_decisions']['reporting_approvals'] if a['gate'] != 'estimator_agreement']
    report = evaluate_case((b, m, r), approvals)
    assert not report['releasable']
    assert [e['value'] for e in report['estimators']] == [.4, 100.]
    assert report['diagnostics']['cv_tmle_minus_one_step']['cv_tmle_identity'] == 99.6


def test_target_changes_are_disclosed_and_never_pooled(case):
    b, m, r = case
    e = replace(b.estimates[0], spec=replace(b.spec, target_id='buffer25'))
    sensitivity = Sensitivity(name='buffer25', estimate=e, target_change='25 km buffer excludes original tracts')
    report = evaluate_case((replace(b, sensitivities=(sensitivity,)), m, r))
    assert report['state'] == 'released'
    assert report['diagnostics']['sensitivities'][0]['changed_spec_fields']['target_id']['sensitivity'] == 'buffer25'
    undisclosed = replace(sensitivity, target_change='unchanged')
    assert evaluate_case((replace(b, sensitivities=(undisclosed,)), m, r))['state'] == 'failed'


def test_multiplicity_registry_never_completes_available_subset():
    report = multiplicity((('usaleep_life_expectancy', .01),), ())
    assert report['tract_lung']['status'] == 'unfinished'
    assert report['tract_lung']['adjusted_p_values'] == {}
    complete = multiplicity((('usaleep_life_expectancy', .01), ('us_lung_incidence', .04)), ())
    assert complete['tract_lung']['adjusted_p_values'] == {'usaleep_life_expectancy': .02, 'us_lung_incidence': .04}


@pytest.mark.parametrize('name', ['anchor_review', 'audit_collection'])
def test_successful_collector_remains_exploratory(case, name):
    b, m, r = case
    m = replace(m, stage=name)
    report = evaluate_case((b, m, r))
    assert report['state'] == 'exploratory'
    assert not report['releasable']


def make_request(tmp_path, case, monkeypatch, approvals=None):
    b, m, r = case
    paths = {}
    for name, record in [('bundle', b), ('manifest', m), ('receipts', r)]:
        path = tmp_path / f'{name}.json'
        write_artifact(path, record)
        paths[name] = str(path)
    approved = tmp_path / 'owner-approvals.yaml'
    approved.write_text(yaml.safe_dump(approve(*case) if approvals is None else approvals))
    monkeypatch.setattr(stage, 'OWNER_APPROVALS', approved)
    paths['approvals'] = str(approved)
    task = tmp_path / 'report-task.json'
    task.write_text(json.dumps(paths))
    return StageRequest(stage=m.stage, config_path=str(CONFIG), config_hash=file_hash(CONFIG),
                        task_path=str(task), task_hash=file_hash(task), dependency_paths=tuple(paths.values()),
                        dependency_hashes=tuple(file_hash(p) for p in paths.values()),
                        output_dir=str(tmp_path / 'report-output'), code_identity='b'*40)


def test_run_stage_writes_only_attempt_outputs_and_verifies_result(case, tmp_path, monkeypatch):
    request = make_request(tmp_path, case, monkeypatch)
    before = {p: Path(p).read_bytes() for p in request.dependency_paths}
    result = run_stage(request)
    assert result.status == 'pass'
    result.verify(request)
    assert {a.path for a in result.artifacts} == {'report.json', 'report.html', 'estimators.svg'}
    assert all(Path(p).read_bytes() == content for p, content in before.items())
    report = json.loads((Path(request.output_dir) / 'report.json').read_text())
    assert report['state'] == 'released'
    assert run_stage(request) == result


def test_run_stage_real_owner_file_at_launch_base_blocks_without_edits(case, tmp_path):
    # This test intentionally does not require future approvals to exist locally.
    approvals = yaml.safe_load((ROOT / 'configs/approvals.yaml').read_text())
    before = file_hash(ROOT / 'configs/approvals.yaml')
    assert not evaluate_case(case, approvals)['releasable']
    assert file_hash(ROOT / 'configs/approvals.yaml') == before


def test_run_stage_rejects_nonfinite_serialized_scores(case, tmp_path, monkeypatch):
    request = make_request(tmp_path, case, monkeypatch)
    bundle_path = Path(request.dependency_paths[0])
    payload = json.loads(bundle_path.read_text())
    payload['payload']['estimates'][0]['scores'][0] = float('nan')
    bundle_path.write_text(json.dumps(payload))
    request = replace(request, dependency_hashes=tuple(file_hash(p) for p in request.dependency_paths))
    result = run_stage(request)
    assert result.status == 'fail'
    report = json.loads((Path(request.output_dir)/'report.json').read_text())
    assert not report['releasable']
    assert any('nonfinite' in g['reason'] for g in report['gates'])


def test_report_isolation_hash_mismatch_and_missing_inputs(case, tmp_path, monkeypatch):
    request = make_request(tmp_path, case, monkeypatch)
    Path(request.dependency_paths[0]).write_text('changed')
    assert run_stage(request).status == 'fail'
    Path(request.dependency_paths[0]).unlink()
    missing = replace(request, output_dir=str(tmp_path/'missing-report'))
    assert run_stage(missing).status == 'blocked'
    report = json.loads((Path(missing.output_dir)/'report.json').read_text())
    assert report['state'] == 'missing'


def load_probes():
    spec = importlib.util.spec_from_file_location('reporting_test_geography_probes', ROOT/'geography_probes.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_diagnostic_geography_and_probe_outputs_cannot_be_nuisance_inputs(case):
    probes = load_probes()
    registry = FeatureRegistry(registry_id='diagnostics', rules=(
        FeatureRule(name='latitude', role='precise_geography', endpoints=('synthetic',), uses=('diagnostic',), approval_id='synthetic'),))
    values = [[0.], [1.], [2.], [3.]]
    view = probes.DiagnosticView(endpoint='synthetic', registry=registry, original_ids=('a','b','c','d'), columns=('latitude',), values=values)
    values[0][0] = 999
    assert view.values[0][0] == 0
    result = probes.exposure_probe(view, {'a':0.,'b':2.,'c':4.,'d':6.}, ('a','b','c'), ('d',))
    assert result.mse == pytest.approx(0, abs=1e-20)
    for item in (view, result):
        with pytest.raises(ContractError, match='cannot become nuisance'):
            item.as_covariates()
    with pytest.raises(ContractError):
        registry.require('latitude', 'synthetic', 'nuisance')
    with pytest.raises(ContractError):
        FeatureRule(name='renamed_probe_output', role='exposure_proxy', endpoints=('synthetic',), uses=('nuisance',), approval_id='synthetic')
    # Training-facing merged contract refuses the actual diagnostic column too.
    b, _, _ = case
    with pytest.raises(ContractError):
        CovariateView(spec=replace(b.spec, endpoint='synthetic', adjustment_schema_hash=registry.content_hash),
                      registry=registry, original_ids=view.original_ids, columns=view.columns, values=view.values, use='nuisance',
                      lineage=replace(b.estimates[0].lineage, unit_ids=view.original_ids))


def test_probe_fitting_does_not_see_heldout_exposure():
    probes = load_probes()
    registry = FeatureRegistry(registry_id='diagnostics', rules=(FeatureRule(name='terrain', role='exposure_proxy', endpoints=('synthetic',), uses=('diagnostic',), approval_id='synthetic'),))
    view = probes.DiagnosticView(endpoint='synthetic', registry=registry, original_ids=('a','b','c'), columns=('terrain',), values=((0.,),(1.,),(2.,)))
    first = probes.exposure_probe(view, {'a':0., 'b':2., 'c':4.}, ('a','b'), ('c',))
    second = probes.exposure_probe(view, {'a':0., 'b':2., 'c':100.}, ('a','b'), ('c',))
    assert first.heldout_predictions == second.heldout_predictions
    assert first.mse != second.mse


@pytest.mark.parametrize('script', ['build_white_paper_report.py', 'build_technical_paper_assets.py', 'plot_phase3_site_forest.py'])
def test_legacy_entrypoints_expose_v2_without_touching_shared_outputs(script, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT/script), '--v2-request', 'unused.json', '--help'],
                            cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(ROOT/'src')), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--v2-request' in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_stale_or_rejected_approval_cannot_release(case):
    approvals = approve(*case)
    approvals['owner_decisions']['reporting_approvals'][0]['bundle_hash'] = '0'*64
    assert not evaluate_case(case, approvals)['releasable']
    approvals = approve(*case)
    approvals['owner_decisions']['reporting_approvals'][0]['status'] = 'rejected'
    assert not evaluate_case(case, approvals)['releasable']


def test_contradictory_external_approvals_stop(case):
    approvals = approve(*case)
    duplicate = dict(approvals['owner_decisions']['reporting_approvals'][0], status='rejected')
    approvals['owner_decisions']['reporting_approvals'].append(duplicate)
    assert evaluate_case(case, approvals)['state'] == 'failed'


def test_mortality_by_requires_all_registered_endpoints_per_country():
    from oxyformer.reporting.evidence_matrix import MORTALITY_ENDPOINTS
    with pytest.raises(ContractError, match='incomplete registered mortality'):
        multiplicity((('MEX:all_cause', .01),), ('MEX:all_cause',))
    family = tuple(f'MEX:{e}' for e in MORTALITY_ENDPOINTS)
    partial = multiplicity((('MEX:all_cause', .01),), family)
    assert not partial['adult_mortality']['complete']
    full = multiplicity(tuple((e, .01) for e in family), family)
    assert full['adult_mortality']['complete']
    assert full['adult_mortality']['adjusted_p_values']['MEX:all_cause'] == pytest.approx(.01*sum(1/i for i in range(1,9)))


def test_zero_information_is_diagnostic_only(case):
    b, m, r = case
    estimates = tuple(replace(e, influence=(0.,)*40) for e in b.estimates)
    report = evaluate_case((replace(b, estimates=estimates), m, r))
    assert report['state'] == 'failed'
    assert all(not metric['positive_D'] for metric in report['diagnostics']['information'].values())


def test_missing_confirmation_does_not_hide_available_one_step(case):
    b, m, r = case
    report = evaluate_case((replace(b, estimates=(b.estimates[0],)), m, r))
    assert report['state'] == 'missing'
    assert report['estimators'][0]['method'] == 'mtp_one_step'


def test_missing_final_scenario_blocks(case):
    b, m, r = case
    m = replace(m, coverage_scenarios=('null', 'nonlinear'))
    assert evaluate_case((b, m, r))['state'] == 'missing'


def test_source_and_attrition_contradictions_stop(case):
    b, m, r = case
    altered = replace(b.sources[0], version='different')
    assert evaluate_case((replace(b, sources=(altered,)), m, r))['state'] == 'failed'
    assert evaluate_case((replace(b, attrition=(('source', 30), ('target',40))), m, r))['state'] == 'failed'


def test_production_approval_path_cannot_be_redirected(case, tmp_path, monkeypatch):
    request = make_request(tmp_path, case, monkeypatch)
    monkeypatch.setattr(stage, 'OWNER_APPROVALS', ROOT/'configs/approvals.yaml')
    result = run_stage(request)
    assert result.status == 'fail'
    report = json.loads((Path(request.output_dir)/'report.json').read_text())
    assert any('not owner registry' in gate['reason'] for gate in report['gates'])


def test_review_tiny_nonzero_county_information_does_not_fail_release(case):
    b, m, r = case
    tiny = tuple(1e-200 if i % 2 else -1e-200 for i in range(40))
    estimates = tuple(replace(e, influence=tiny) for e in b.estimates)
    report = evaluate_case((replace(b, estimates=estimates), m, r))
    assert report['state'] == 'released'
    for metric in report['diagnostics']['information'].values():
        assert metric['positive_D']
        assert metric['G_eff'] == pytest.approx(40)


def test_review_plain_cv_tmle_name_is_paired_and_concentration_checked(case):
    b, m, r = case
    e = replace(b.estimates[1], method='cv_tmle')
    report = evaluate_case((replace(b, estimates=(b.estimates[0], e)), m, r))
    assert report['state'] == 'released'
    assert 'cv_tmle' in report['diagnostics']['cv_tmle_minus_one_step']
    assert any(g['gate'] == 'influence_concentration:cv_tmle' for g in report['gates'])
    e = replace(e, influence=(1., -1.) + (0.,)*38)
    assert evaluate_case((replace(b, estimates=(b.estimates[0], e)), m, r))['state'] == 'failed'


def test_review_nonfinite_diagnostic_values_rejected_at_construction():
    probes = load_probes()
    registry = FeatureRegistry(registry_id='diagnostics', rules=(FeatureRule(name='terrain', role='exposure_proxy', endpoints=('synthetic',), uses=('diagnostic',), approval_id='synthetic'),))
    for value in (float('nan'), float('inf')):
        with pytest.raises(ContractError, match='nonfinite'):
            probes.DiagnosticView(endpoint='synthetic', registry=registry, original_ids=('a', 'b'), columns=('terrain',), values=((value,), (1.,)))


def test_review_identical_rerun_returns_verified_result(case, tmp_path, monkeypatch):
    request = make_request(tmp_path, case, monkeypatch)
    first = run_stage(request)
    second = run_stage(request)
    assert second == first
    second.verify(request)


def test_review_primary_confirmation_requires_same_frozen_oof_split(case):
    b, m, r = case
    other = replace(b.estimates[1], lineage=replace(b.estimates[1].lineage, split_hash='f'*64))
    report = evaluate_case((replace(b, estimates=(b.estimates[0], other)), m, r))
    assert report['state'] == 'failed'  # plan 4.4: same frozen OOF initial models
    assert len(report['estimators']) == 2
    sensitivity = Sensitivity(name='registered alternate split', estimate=other, target_change='unchanged')
    report = evaluate_case((replace(b, sensitivities=(sensitivity,)), m, r))
    assert report['state'] == 'released'
    assert report['diagnostics']['sensitivities'][0]['estimate']['lineage']['split_hash'] == 'f'*64


def test_review_equal_thirty_counties_reach_approved_boundary():
    metric = concentration([(-1.)**i for i in range(30)], [str(i) for i in range(30)], ['state']*30)
    assert metric['G_eff'] >= 30
    assert metric['s_max'] <= .1


def test_review_invalid_output_isolation_returns_failed_result(case, tmp_path, monkeypatch):
    request = make_request(tmp_path, case, monkeypatch)
    request = replace(request, output_dir=str(tmp_path))  # contains input artifacts
    result = run_stage(request)
    assert result.status == 'fail'
    assert 'isolated' in result.message


def test_review_partial_report_publication_resumes(case, tmp_path, monkeypatch):
    request = make_request(tmp_path, case, monkeypatch)
    first = run_stage(request)
    (Path(request.output_dir)/'report.html').unlink()
    second = run_stage(request)
    assert second == first
    second.verify(request)


def test_review_conflicting_report_never_returns_stale_released_artifacts(case, tmp_path, monkeypatch):
    request = make_request(tmp_path, case, monkeypatch)
    assert run_stage(request).status == 'pass'
    path = Path(request.output_dir)/'report.json'
    path.write_text('conflicting prior content')
    result = run_stage(request)
    assert result.status == 'fail'
    assert result.artifacts == ()
    assert 'conflict' in result.message
    assert path.read_text() == 'conflicting prior content'


def test_review_rerun_reverifies_upstream_science(case, tmp_path, monkeypatch):
    request = make_request(tmp_path, case, monkeypatch)
    assert run_stage(request).status == 'pass'
    _, _, receipts = case
    (Path(receipts.items[0].request.output_dir)/'result.json').write_text('changed upstream evidence')
    result = run_stage(request)
    assert result.status == 'fail'
    assert result.artifacts == ()


@pytest.mark.parametrize('role,payload', [('config', ''), ('config', '[]\n'), ('approvals', ''), ('approvals', '[]\n')])
def test_review_malformed_yaml_returns_structured_failure(case, tmp_path, monkeypatch, role, payload):
    request = make_request(tmp_path, case, monkeypatch)
    if role == 'config':
        config = tmp_path/'incomplete-config.yaml'
        config.write_text(payload)
        request = replace(request, config_path=str(config), config_hash=file_hash(config))
    else:
        approved = Path(request.dependency_paths[-1])
        approved.write_text(payload)
        request = replace(request, dependency_hashes=tuple(file_hash(p) for p in request.dependency_paths))
    result = run_stage(request)
    assert result.status in ('fail', 'blocked')
