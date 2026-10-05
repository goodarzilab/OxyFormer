"""Synthetic, offline CPU tests for reporting boundaries and scientific stops."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
import shutil
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
import oxyformer.validation.geography_probes as probes
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
    lineage = ArtifactLineage(source_hashes=(source.payload_hash,), unit_ids=ids, parent_hashes=('7'*64, '8'*64, '5'*64, '9'*64),
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
        if gate == 'coverage':
            write_artifact(result_file, coverage)
        else:
            result_file.write_text('{"synthetic":true}')
        request = StageRequest(stage=gate, config_path=str(config), config_hash=file_hash(config),
                               task_path=str(task), task_hash=file_hash(task), dependency_paths=(), dependency_hashes=(),
                               output_dir=str(output), code_identity='a'*40)
        result = StageResult(request_hash=request.content_hash, status='pass', message='synthetic task passed',
                             artifacts=(ArtifactRecord(path='result.json', sha256=file_hash(result_file), lineage=lineage,
                                                       kind='coverage_scenario' if gate == 'coverage' else 'synthetic'),))
        task_id = f'{gate}-{i}'
        tasks.append(ExpectedTask(task_id=task_id, gate=gate, request_hash=request.content_hash))
        receipts.append(TaskReceipt(task_id=task_id, request=request, result=result))
    manifest = ExpectedTasks(stage='tract_release', spec=spec, seed_ids=seeds, tasks=tuple(tasks), coverage_scenarios=('null',))
    receipts = TaskReceipts(items=tuple(receipts))
    return bundle, manifest, receipts


def evaluate_case(case, approvals=None):
    b, m, r = case
    return evaluate(b, m, r, approve(b, m, r) if approvals is None else approvals, file_hash(CONFIG))


def publish_coverage(receipts, scenario, *, only_task_id=None):
    """Model an upstream summary publication; never silently change bundle metrics."""
    items = []
    for item in receipts.items:
        if item.request.stage == 'coverage' and (only_task_id is None or item.task_id == only_task_id):
            artifact = item.result.artifacts[0]
            path = Path(item.request.output_dir) / artifact.path
            path.unlink()
            digest = write_artifact(path, scenario)
            item = replace(item, result=replace(item.result, artifacts=(
                replace(artifact, sha256=digest, kind='coverage_scenario'),)))
        items.append(item)
    return replace(receipts, items=tuple(items))


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
    r = publish_coverage(r, b.coverage[0])
    report = evaluate_case((b, m, r))
    assert report['state'] == 'failed'
    assert report['evidence_label'] == 'diagnostic-only'
    assert any(g['gate'] == 'coverage:null' and g['status'] == 'failed' for g in report['gates'])


def test_bias_is_target_and_large_bias_requires_investigation(case):
    b, m, r = case
    b = replace(b, coverage=(replace(b.coverage[0], abs_bias_over_empirical_sd=.15),))
    r = publish_coverage(r, b.coverage[0])
    assert evaluate_case((b, m, r))['state'] == 'released'
    b = replace(b, coverage=(replace(b.coverage[0], abs_bias_over_empirical_sd=.21),))
    r = publish_coverage(r, b.coverage[0])
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


def test_diagnostic_geography_and_probe_outputs_cannot_be_nuisance_inputs(case):
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


def assert_structured_input_failure(request, name):
    result = run_stage(request)
    assert result.status == 'fail'
    result.verify(request)
    report = json.loads((Path(request.output_dir)/'report.json').read_text())
    assert report['state'] == 'failed'
    assert report['evidence_label'] == 'diagnostic-only'
    assert not report['releasable']
    assert any(name in gate['reason'] for gate in report['gates'])


@pytest.mark.parametrize('role,payload', [('config','null\n'), ('config','42\n'), ('approvals','null\n'), ('approvals','42\n')])
def test_recovery_null_scalar_yaml_roots_are_structured_failures(case, tmp_path, monkeypatch, role, payload):
    request = make_request(tmp_path, case, monkeypatch)
    if role == 'config':
        path = tmp_path/'invalid-config.yaml'
        path.write_text(payload)
        request = replace(request, config_path=str(path), config_hash=file_hash(path))
    else:
        path = Path(request.dependency_paths[-1])
        path.write_text(payload)
        request = replace(request, dependency_hashes=tuple(file_hash(p) for p in request.dependency_paths))
    assert_structured_input_failure(request, 'config' if role == 'config' else 'approvals')


@pytest.mark.parametrize('field,value', [
    ('owner_decisions', None), ('owner_decisions', []), ('owner_decisions', 7),
    ('reporting_approvals', None), ('reporting_approvals', {}), ('reporting_approvals', 7),
    ('approval_record', None), ('approval_record', []), ('approval_record', 'invalid'),
])
def test_recovery_approval_container_shapes(case, tmp_path, monkeypatch, field, value):
    approvals = approve(*case)
    if field == 'owner_decisions':
        approvals[field] = value
    elif field == 'reporting_approvals':
        approvals['owner_decisions'][field] = value
    else:
        approvals['owner_decisions']['reporting_approvals'] = [value]
    request = make_request(tmp_path, case, monkeypatch, approvals)
    assert_structured_input_failure(request, 'owner_decisions' if field == 'owner_decisions' else 'reporting_approvals')
    # Independently callable consumers use the same controlled shape boundary.
    from oxyformer.reporting.evidence_matrix import external_approval
    assert evaluate_case(case, approvals)['state'] == 'failed'
    with pytest.raises(ContractError):
        external_approval(approvals, 'coverage', {})


@pytest.mark.parametrize('value', [None, [], 'invalid', 4])
def test_recovery_direct_approval_consumers_reject_nonmappings(case, value):
    from oxyformer.reporting.evidence_matrix import external_approval
    # evaluate_case's None means its default fixture, so call evaluate directly.
    b, m, r = case
    assert evaluate(b, m, r, value, file_hash(CONFIG))['state'] == 'failed'
    with pytest.raises(ContractError):
        external_approval(value, 'coverage', {})


@pytest.mark.parametrize('payload', ['null', '[]', '42', '["bundle", "manifest", "receipts", "approvals"]'])
def test_recovery_task_json_requires_object(case, tmp_path, monkeypatch, payload):
    request = make_request(tmp_path, case, monkeypatch)
    path = Path(request.task_path)
    path.write_text(payload)
    request = replace(request, task_hash=file_hash(path))
    assert_structured_input_failure(request, 'reporting task')


@pytest.mark.parametrize('approvals', [{}, {'owner_decisions': {}}])
def test_recovery_absent_optional_approval_containers_remain_blocked(case, approvals):
    assert evaluate_case(case, approvals)['state'] == 'blocked'


def test_review_stale_passing_coverage_cannot_override_verified_bad_metrics(case):
    b, m, r = case
    bad = replace(b.coverage[0], repetitions=850, coverage_one_sided_95_lower_bound=.8)
    r = publish_coverage(r, bad)
    for receipt in r.items:
        receipt.result.verify(receipt.request)
    report = evaluate_case((b, m, r))  # fresh scoped approvals cannot waive a contradiction
    assert report['state'] == 'failed' and not report['releasable']
    assert len(report['estimators']) == 2
    assert any('coverage metrics differ' in g['reason'] for g in report['gates'])


def test_review_coverage_tasks_cannot_publish_conflicting_summaries(case):
    b, m, r = case
    other = replace(b.coverage[0], repetitions=850)
    r = publish_coverage(r, other, only_task_id=r.items[-1].task_id)
    report = evaluate_case((b, m, r))
    assert report['state'] == 'failed'
    assert any('contradictory coverage artifacts' in g['reason'] for g in report['gates'])


def test_review_each_coverage_task_requires_summary_artifacts(case):
    b, m, r = case
    last = r.items[-1]
    last = replace(last, result=replace(last.result, artifacts=(replace(last.result.artifacts[0], kind='unrelated'),)))
    report = evaluate_case((b, m, replace(r, items=r.items[:-1] + (last,))))
    assert report['state'] == 'missing' and not report['releasable']
    assert any(g['gate'] == 'coverage_evidence' and g['task_id'] == last.task_id for g in report['gates'])


def test_review_coverage_report_discloses_verified_task_and_artifact_identities(case):
    report = evaluate_case(case)
    assert report['state'] == 'released'
    _, _, receipts = case
    coverage = [r for r in receipts.items if r.request.stage == 'coverage']
    assert {x['task_id'] for x in report['coverage_evidence']} == {r.task_id for r in coverage}
    assert {x['artifact_hash'] for x in report['coverage_evidence']} == {case[0].coverage[0].content_hash}
    assert {x['request_hash'] for x in report['coverage_evidence']} == {r.request.content_hash for r in coverage}


def test_review_state_qualified_county_identities_remain_distinct(case):
    with pytest.raises(ContractError, match='county crosses states'):
        concentration([.1, -.1], ['Benton', 'Benton'], ['AR', 'MO'])
    # The merged covariance API groups by county key, also used for its centroid.
    # Distinct physical counties need distinct keys even with the same display name.
    b, m, r = case
    counties = ('AR:Benton', 'MO:Benton') + b.counties[2:]
    locations = tuple((key, xy) for key, (_, xy) in zip(counties, b.county_locations))
    b = replace(b, counties=counties, states=('AR', 'MO') + b.states[2:], county_locations=locations)
    report = evaluate_case((b, m, r))
    assert report['state'] == 'released'
    assert report['diagnostics']['cluster_covariance']['dependence_units'] == 40


@pytest.fixture
def produced_primary_case(case):
    """Actual merged producers, forty counties and two distinct valid OOF fits."""
    from test_scores import make_fixture
    from oxyformer.contracts import OOFNuisances
    from oxyformer.data.loaders import load_records
    from oxyformer.estimation.mtp import one_step
    from oxyformer.estimation.targeting import cv_tmle

    b, m, receipts = case
    policy, original_data, original_split, _, _ = make_fixture()
    ids = b.original_ids
    n = len(ids)
    lineage = replace(original_data.manifest.lineage, unit_ids=ids)
    data_manifest = replace(original_data.manifest, original_ids=ids, lineage=lineage)
    data = load_records([dict(id=oid, y=float(2*(i % 2)), a=1., w=1.) for i, oid in enumerate(ids)],
                        data_manifest, data_manifest.spec, data_manifest.schema_hash)
    split = replace(original_split, original_ids=ids, fold_ids=tuple(i % 2 for i in range(n)),
                    lineage=replace(lineage, parent_hashes=(data_manifest.content_hash,)))
    nuisances = OOFNuisances(spec=data_manifest.spec, original_ids=ids*2, fold_ids=split.fold_ids*2,
                            seed_ids=(1103,)*n+(2207,)*n, mu_a=(1.,)*(2*n), mu_d=(2.,)*(2*n),
                            r_a=(1.5,)*(2*n), r_d=(1.5,)*(2*n), origin_weights=(1.,)*(2*n),
                            lineage=replace(lineage, parent_hashes=(data_manifest.content_hash,),
                                            split_hash=split.content_hash))
    weights = dict.fromkeys(ids, 1.)
    one = one_step(nuisances, data, weights, data_manifest.spec, split=split, policy=policy)
    tmle = cv_tmle(nuisances, data, weights, 'identity', data_manifest.spec, split=split, policy=policy).estimate
    other = replace(nuisances, mu_a=(.8,)*(2*n))
    different_fit = cv_tmle(other, data, weights, 'identity', data_manifest.spec, split=split, policy=policy).estimate
    b = replace(b, spec=data_manifest.spec, sources=data_manifest.sources,
                estimates=(one, tmle), seed_ids=split.seed_ids, ratios=((1.5,)*n,)*2)
    m = replace(m, spec=data_manifest.spec, seed_ids=split.seed_ids)
    return (b, m, receipts), different_fit


def assert_primary_input_failure(report, methods):
    assert report['state'] == 'failed'
    assert not report['releasable']
    assert report['evidence_label'] == 'diagnostic-only'
    gate = next(g for g in report['gates'] if g['gate'] == 'primary_input_consistency')
    assert gate['status'] == 'failed'
    assert [e['method'] for e in report['estimators']] == methods
    assert set(report['diagnostics']['information']) == set(methods)
    assert set(report['diagnostics']['spatial_sensitivities']) == {'50.0', '100.0', '200.0'}
    assert len(report['diagnostics']['aligned_influence']['values']) == 40
    assert 'overlap_by_seed' in report['diagnostics']
    assert 'multiplicity' in report and 'coverage_evidence' in report
    for render in (render_html, render_forest):
        assert all(method in render(report) for method in methods)
    return gate


def test_primary_input_different_initial_oof_from_actual_producers_never_releases(produced_primary_case):
    same_case, different_fit = produced_primary_case
    b, m, receipts = same_case
    assert evaluate_case(same_case)['state'] == 'released'
    one = b.estimates[0]
    assert one.lineage.split_hash == different_fit.lineage.split_hash
    assert one.lineage.parent_hashes[0] != different_fit.lineage.parent_hashes[0]
    assert one.lineage.parent_hashes[1:] == different_fit.lineage.parent_hashes[1:]
    report = evaluate_case((replace(b, estimates=(one, different_fit)), m, receipts))
    assert_primary_input_failure(report, ['mtp_one_step', 'cv_tmle_identity'])
    assert all(g['status'] == 'pass' for g in report['gates']
               if g['gate'].startswith('influence_concentration:'))


@pytest.mark.parametrize('method', ['cv_tmle', 'cv_tmle_identity', 'cv_tmle_logistic', 'cv_tmle_poisson'])
@pytest.mark.parametrize('slot', range(4))
def test_primary_input_every_confirmation_and_parent_role_checked(case, method, slot):
    b, m, r = case
    parents = list(b.estimates[1].lineage.parent_hashes)
    parents[slot] = 'f'*64
    # Keep valid confirmations before the invalid one: every primary is checked.
    confirmations = tuple(replace(b.estimates[1], method=name) for name in
                          ('cv_tmle', 'cv_tmle_identity', 'cv_tmle_logistic', 'cv_tmle_poisson') if name != method)
    altered = replace(b.estimates[1], method=method,
                      lineage=replace(b.estimates[1].lineage, parent_hashes=tuple(parents)))
    estimates = (b.estimates[0],) + confirmations + (altered,)
    report = evaluate_case((replace(b, estimates=estimates), m, r))
    assert_primary_input_failure(report, [e.method for e in estimates])


@pytest.mark.parametrize('method', ['mtp_one_step', 'cv_tmle', 'cv_tmle_identity', 'cv_tmle_logistic', 'cv_tmle_poisson'])
@pytest.mark.parametrize('length', range(4))
def test_primary_input_requires_four_parents(case, method, length):
    b, m, r = case
    one, tmle = b.estimates
    estimate = one if method == 'mtp_one_step' else replace(tmle, method=method)
    altered = replace(estimate, lineage=replace(estimate.lineage, parent_hashes=estimate.lineage.parent_hashes[:length]))
    estimates = (altered, tmle) if method == 'mtp_one_step' else (one, altered)
    report = evaluate_case((replace(b, estimates=estimates), m, r))
    assert_primary_input_failure(report, [e.method for e in estimates])


@pytest.mark.parametrize('split_hash', ['f'*64, None])
def test_primary_input_common_prefix_must_bind_split_hash(case, split_hash):
    b, m, r = case
    estimates = tuple(replace(e, lineage=replace(e.lineage, split_hash=split_hash)) for e in b.estimates)
    assert estimates[0].lineage.parent_hashes == estimates[1].lineage.parent_hashes
    report = evaluate_case((replace(b, estimates=estimates), m, r))
    assert_primary_input_failure(report, [e.method for e in estimates])


def test_primary_input_prefix_order_matters(case):
    b, m, r = case
    parents = b.estimates[1].lineage.parent_hashes
    changed = replace(b.estimates[1], lineage=replace(b.estimates[1].lineage,
                      parent_hashes=(parents[1], parents[0], parents[2], parents[3])))
    report = evaluate_case((replace(b, estimates=(b.estimates[0], changed)), m, r))
    assert_primary_input_failure(report, ['mtp_one_step', 'cv_tmle_identity'])


def test_primary_input_suffix_provenance_and_nonprimary_inputs_remain_visible(case):
    b, m, r = case
    one, tmle = (replace(e, lineage=replace(e.lineage, parent_hashes=e.lineage.parent_hashes+(suffix,)))
                 for e, suffix in zip(b.estimates, ('a'*64, 'b'*64)))
    comparator = replace(one, method='riesz_comparator', lineage=replace(one.lineage, parent_hashes=()))
    sensitivity = Sensitivity(name='alternate OOF', estimate=replace(tmle, lineage=replace(tmle.lineage,
                              parent_hashes=('c'*64,))), target_change='unchanged')
    report = evaluate_case((replace(b, estimates=(one, tmle, comparator), sensitivities=(sensitivity,)), m, r))
    assert report['state'] == 'released'
    gate = next(g for g in report['gates'] if g['gate'] == 'primary_input_consistency')
    assert gate['status'] == 'pass'
    assert set(gate['inputs']) == {'mtp_one_step', 'cv_tmle_identity'}
    for estimate, row in zip((one, tmle, comparator), report['estimators']):
        assert tuple(row['lineage']['parent_hashes']) == estimate.lineage.parent_hashes
    assert report['diagnostics']['sensitivities'][0]['estimate']['lineage']['parent_hashes'] == ('c'*64,)


def test_fresh_round1_county_cancellation_preserves_information():
    influence = [value for sign in [1.]*15 + [-1.]*15 for value in (sign*1e16, sign, -sign*1e16)]
    counties = [str(i) for i in range(30) for _ in range(3)]
    metric = concentration(influence, counties, ['state']*90)
    assert metric['positive_D']
    assert metric['D'] == pytest.approx(30.)
    assert metric['s_max'] == pytest.approx(1/30)
    assert metric['G_eff'] >= 30
    assert {row['U_g'] for row in metric['ranked_counties']} == {-1., 1.}


def test_fresh_round1_county_cancellation_keeps_covariance_consistent(case):
    b, m, r = case
    ids = tuple(f'cancel-{i}' for i in range(90))
    influence = tuple(value for sign in [1.]*15 + [-1.]*15 for value in (sign*1e16, sign, -sign*1e16))
    estimates = tuple(replace(e, original_ids=ids, influence=influence,
                             scores=tuple(e.value+90*v for v in influence),
                             lineage=replace(e.lineage, unit_ids=ids)) for e in b.estimates)
    b = replace(b, original_ids=ids, estimates=estimates, attrition=(('source', 90), ('target', 90)),
                weights=(1.,)*90, observed_exposure=(1.,)*90, shifted_exposure=(3.,)*90,
                ratios=((1.,)*90,)*3, balance_observed=((1., 1.),)*90, balance_shifted=((1., 3.),)*90,
                counties=tuple(f'county-{i}' for i in range(30) for _ in range(3)), states=('state',)*90,
                county_locations=b.county_locations[:30])
    report = evaluate_case((b, m, r))
    # Exact county totals are +/-1. Both covariance and gate use these totals.
    assert np.asarray(report['diagnostics']['cluster_covariance']['matrix']) == pytest.approx(np.full((2, 2), 30*30/29))
    assert report['state'] == 'released'
    assert report['diagnostics']['aligned_influence']['values'][0] == (1e16, 1e16)


def test_fresh_round1_bad_sensitivity_disclosure_retains_every_diagnostic(case, tmp_path, monkeypatch):
    b, m, r = case
    alternate = replace(b.estimates[1], spec=replace(b.spec, policy_id='shift3'))
    sensitivities = (Sensitivity(name='incorrect disclosure', estimate=alternate, target_change='unchanged'),
                     Sensitivity(name='correct disclosure', estimate=alternate, target_change='three mmHg policy'))
    b = replace(b, sensitivities=sensitivities)
    request = make_request(tmp_path, (b, m, r), monkeypatch)
    result = run_stage(request)
    assert result.status == 'fail'
    result.verify(request)
    report = json.loads((Path(request.output_dir)/'report.json').read_text())
    assert report['state'] == 'failed' and not report['releasable']
    assert len(report['diagnostics']['sensitivities']) == 2
    assert len(report['diagnostics']['aligned_influence']['values']) == 40
    assert set(report['diagnostics']['spatial_sensitivities']) == {'50.0', '100.0', '200.0'}
    assert 'overlap_by_seed' in report['diagnostics']
    assert len(report['estimators']) == 2
    assert any(g['status'] == 'failed' and 'undisclosed target change' in g['reason'] for g in report['gates'])
    assert len(report['coverage_evidence']) == 2
    html = (Path(request.output_dir)/'report.html').read_text()
    assert all(s.name in html for s in sensitivities)
    assert 'shift3' in html


@pytest.mark.parametrize('direction', [-1, 0, 1])
def test_fresh_round2_exact_county_share_boundary_and_neighbors(case, direction):
    from math import fsum
    b, m, r = case
    ids = tuple(f'boundary-{i}' for i in range(512))
    u = [7/256 if i == 215 else -1/256 if i <= 224 else 1/256 if i <= 441 else 0. for i in range(512)]
    if direction:
        shifted = float(np.nextafter(u[215], np.inf if direction > 0 else -np.inf))
        delta = shifted-u[215]
        u[215] = shifted
        u[225] -= delta
    assert fsum(u) == 0.
    estimates = tuple(replace(e, value=.5, original_ids=ids, influence=tuple(u),
                             scores=tuple(.5+512*v for v in u),
                             lineage=replace(e.lineage, unit_ids=ids)) for e in b.estimates)
    b = replace(b, original_ids=ids, estimates=estimates, attrition=(('source', 512), ('target', 512)),
                weights=(1.,)*512, observed_exposure=(1.,)*512, shifted_exposure=(3.,)*512,
                ratios=((1.,)*512,)*3, balance_observed=((1., 1.),)*512, balance_shifted=((1., 3.),)*512,
                counties=tuple(f'c{i}' for i in range(442))+('c0',)*70, states=('state',)*512,
                county_locations=tuple((f'c{i}', (30.+i*.01, -110.)) for i in range(442)))
    report = evaluate_case((b, m, r))
    assert report['state'] == ('failed' if direction > 0 else 'released')
    gates = [g for g in report['gates'] if g['gate'].startswith('influence_concentration:')]
    assert len(gates) == 2
    assert all(g['status'] == ('failed' if direction > 0 else 'pass') for g in gates)
    if direction == 0:
        metric = report['diagnostics']['information']['mtp_one_step']
        assert metric['D'] == 490/65536
        assert metric['s_max'] == .1
        assert metric['G_eff'] == pytest.approx(2450/29)


@pytest.mark.parametrize('mode', ['shared', 'mismatched', 'alternate_approval', 'protected_output',
    'configs_alias_shared', 'configs_alias_protected', 'renamed_configs_shared',
    'renamed_configs_protected', 'config_file_alias_shared'])
def test_installed_reporting_uses_repository_config_binding(produced_primary_case, tmp_path, monkeypatch, mode):
    same_case, different_fit = produced_primary_case
    b, m, receipts = same_case
    if mode == 'mismatched':
        b = replace(b, estimates=(b.estimates[0], different_fit))
    request = make_request(tmp_path, (b, m, receipts), monkeypatch)
    # Model an ordinary non-editable installation: package code is separate from
    # the repository containing its externally frozen config and owner registry.
    installed = tmp_path / 'venv/lib/python3.11/site-packages'
    shutil.copytree(ROOT / 'src/oxyformer', installed / 'oxyformer',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    repository = tmp_path / 'project'
    config = repository / 'configs/reporting.yaml'
    if mode.startswith(('configs_alias_', 'renamed_configs_')):
        storage = tmp_path / 'config-storage' / ('configs' if mode.startswith('configs_alias_') else 'settings')
        storage.mkdir(parents=True)
        repository.mkdir()
        config.parent.symlink_to(storage, target_is_directory=True)
    else:
        config.parent.mkdir(parents=True)
    if mode == 'config_file_alias_shared':
        stored_config = tmp_path / 'stored-reporting-configuration.yaml'
        stored_config.write_bytes(CONFIG.read_bytes())
        config.symlink_to(stored_config)
    else:
        config.write_bytes(CONFIG.read_bytes())
    task = json.loads(Path(request.task_path).read_text())
    approved = config.with_name('approvals.yaml')
    approved.write_bytes(Path(task['approvals']).read_bytes())
    if mode == 'alternate_approval':
        copied = tmp_path / 'copied-approvals.yaml'
        copied.write_bytes(approved.read_bytes())
        task['approvals'] = str(copied)
    else:
        task['approvals'] = str(approved)
    Path(request.task_path).write_text(json.dumps(task))
    request = replace(request, config_path=str(config), config_hash=file_hash(config),
                      task_hash=file_hash(request.task_path), dependency_paths=tuple(task.values()),
                      dependency_hashes=tuple(file_hash(p) for p in task.values()))
    if mode == 'protected_output' or mode.endswith('_protected'):
        request = replace(request, output_dir=str(repository / 'src/report-output'))
    request_path = tmp_path / 'installed-request.json'
    request_path.write_text(request.to_json())
    script = """
import sys
from pathlib import Path
from oxyformer.contracts import StageRequest
from oxyformer.reporting import stage
assert Path(stage.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())
request = StageRequest.from_json(Path(sys.argv[1]).read_text())
print(stage.run_stage(request).to_json())
"""
    completed = subprocess.run([sys.executable, '-c', script, str(request_path), str(installed)],
                               cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(installed),
                                                 'CUDA_VISIBLE_DEVICES': ''},
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    result = StageResult.from_json(completed.stdout)
    if mode == 'protected_output' or mode.endswith('_protected'):
        assert result.status == 'fail'
        assert not result.artifacts
        assert not Path(request.output_dir).exists()
        return
    result.verify(request)
    report = json.loads((Path(request.output_dir) / 'report.json').read_text())
    if mode == 'alternate_approval':
        assert result.status == 'fail'
        assert not report['releasable']
        assert any('approval path is not owner registry' in g['reason'] for g in report['gates'])
        return
    if mode == 'shared' or mode.endswith('_shared'):
        assert result.status == 'pass', report['gates']
        assert report['state'] == 'released'
    else:
        assert result.status == 'fail'
        assert_primary_input_failure(report, ['mtp_one_step', 'cv_tmle_identity'])
    assert [e['method'] for e in report['estimators']] == ['mtp_one_step', 'cv_tmle_identity']
    assert 'aligned_influence' in report['diagnostics']
    assert 'overlap_by_seed' in report['diagnostics']


def test_report_publication_is_stable_with_windows_text_newlines(produced_primary_case, tmp_path, monkeypatch):
    same_case, _ = produced_primary_case
    request = make_request(tmp_path, same_case, monkeypatch)
    real_temporary_file = stage.tempfile.NamedTemporaryFile

    def windows_temporary_file(*args, **kwargs):
        # Exercise real TextIOWrapper CRLF translation on the CPU/Linux host.
        # Binary streams are platform-independent and receive no newline option.
        if 'b' not in kwargs.get('mode', 'w+b') and kwargs.get('newline') is None:
            kwargs['newline'] = '\r\n'
        return real_temporary_file(*args, **kwargs)

    monkeypatch.setattr(stage.tempfile, 'NamedTemporaryFile', windows_temporary_file)
    first = run_stage(request)
    assert first.status == 'pass'
    first.verify(request)
    second = run_stage(request)
    assert second == first
    second.verify(request)
    root = Path(request.output_dir)
    report = json.loads((root / 'report.json').read_text())
    assert report['state'] == 'released'
    for name in ('report.html', 'estimators.svg'):
        content = (root / name).read_bytes()
        assert b'\n' in content
        assert b'\r\n' not in content


@pytest.mark.parametrize('rules', [('',), (' \t\n',), ('no retries', ''), (' ', 'no retries')])
def test_publication_retry_rules_require_every_entry(case, rules):
    assert evaluate_case(case)['releasable']
    b, m, r = case
    scenario = replace(b.coverage[0], registered_retry_rules=rules)
    b = replace(b, coverage=(scenario,))
    r = publish_coverage(r, scenario)
    for receipt in r.items:
        receipt.result.verify(receipt.request)
    report = evaluate_case((b, m, r))
    assert not report['releasable']
    assert any(g['gate'] == 'coverage:null' and g['status'] == 'failed' for g in report['gates'])
    assert len(report['estimators']) == 2


@pytest.mark.parametrize('field', ['attrition', 'balance_names', 'counties', 'states'])
@pytest.mark.parametrize('blank', ['', ' \t'])
def test_publication_diagnostic_labels_require_content(case, field, blank):
    b, m, r = case
    values = list(getattr(b, field))
    values[0] = (blank, values[0][1]) if field == 'attrition' else blank
    changes = {field: tuple(values)}
    if field == 'counties':
        changes['county_locations'] = ((blank, b.county_locations[0][1]),) + b.county_locations[1:]
    report = evaluate_case((replace(b, **changes), m, r))
    assert not report['releasable']
    assert len(report['estimators']) == 2


@pytest.mark.parametrize('field', ['reviewer', 'reference', 'definitions', 'require', 'on_failure'])
@pytest.mark.parametrize('invalid', ['', ' \t', ['text'], {'text': 'value'}, 1])
def test_publication_approval_text_requires_content(case, field, invalid):
    approvals = approve(*case)
    owner = approvals['owner_decisions']
    if field in ('reviewer', 'reference'):
        owner['reporting_approvals'][0][field] = invalid
    else:
        owner['influence_concentration_gate'][field] = invalid
    report = evaluate_case(case, approvals)
    assert not report['releasable']
    assert len(report['estimators']) == 2


@pytest.mark.parametrize('role', ['receipts', 'manifest', 'approvals', 'config'])
@pytest.mark.parametrize('damage', ['missing', 'hash_mismatch'])
def test_refusal_retains_authenticated_estimates(produced_primary_case, tmp_path, monkeypatch, role, damage):
    same_case, _ = produced_primary_case
    b, m, r = same_case
    request = make_request(tmp_path, same_case, monkeypatch)
    task = json.loads(Path(request.task_path).read_text())
    if role == 'config':
        path = tmp_path / 'reporting.yaml'
        path.write_bytes(CONFIG.read_bytes())
        request = replace(request, config_path=str(path))
    else:
        path = Path(task[role])
    if damage == 'missing':
        path.unlink()
    else:
        path.write_text('changed')
    result = run_stage(request)
    root = Path(request.output_dir)
    report = json.loads((root / 'report.json').read_text())
    assert result.status == ('blocked' if damage == 'missing' else 'fail')
    assert report['state'] == ('missing' if damage == 'missing' else 'failed')
    assert not report['releasable'] and report['evidence_label'] == 'diagnostic-only'
    assert report['estimators'] == [e.to_dict()['payload'] for e in b.estimates]
    assert report['bundle_hash'] == b.content_hash
    if role != 'manifest':
        assert report['diagnostics'] == json.loads(canonical_json(summarize(b, m)))
    for name in ('report.html', 'estimators.svg'):
        content = (root / name).read_text()
        assert 'diagnostic-only' in content
        assert all(e.method in content for e in b.estimates)


@pytest.mark.parametrize('role', ['bundle', 'task'])
@pytest.mark.parametrize('damage', ['missing', 'hash_mismatch'])
def test_refusal_never_displays_unauthenticated_estimates(case, tmp_path, monkeypatch, role, damage):
    request = make_request(tmp_path, case, monkeypatch)
    task = json.loads(Path(request.task_path).read_text())
    path = Path(request.task_path if role == 'task' else task['bundle'])
    if damage == 'missing':
        path.unlink()
    else:
        path.write_text('changed')
    result = run_stage(request)
    report = json.loads((Path(request.output_dir) / 'report.json').read_text())
    assert result.status != 'pass'
    assert not report['releasable'] and report['estimators'] == []


def overflowing_balance_case(case, mismatch=False):
    b, m, r = case
    b = replace(b, balance_basis_id='f(A)=1e308*(A-2)', balance_names=('large signed exposure',),
                balance_observed=((-1e308,),) * len(b.original_ids),
                balance_shifted=((1e308,),) * len(b.original_ids))
    if mismatch:
        one, confirmation = b.estimates
        parents = ('0' * 64,) + confirmation.lineage.parent_hashes[1:]
        b = replace(b, estimates=(one, replace(confirmation,
                    lineage=replace(confirmation.lineage, parent_hashes=parents))))
    return b, m, r


def test_nonfinite_balance_difference_cannot_release(case):
    b, m, r = overflowing_balance_case(case)
    report = evaluate_case((b, m, r))
    assert not report['releasable']
    assert report['state'] == 'failed'
    assert json.loads(canonical_json(report['estimators'])) == [e.to_dict()['payload'] for e in b.estimates]
    canonical_json(report)


@pytest.mark.parametrize('mismatch', [False, True])
def test_nonfinite_balance_difference_keeps_authenticated_estimates(case, tmp_path, monkeypatch, mismatch):
    b, m, r = overflowing_balance_case(case, mismatch)
    request = make_request(tmp_path, (b, m, r), monkeypatch)
    result = run_stage(request)
    assert result.status == 'fail'
    assert {a.path for a in result.artifacts} == {'report.json', 'report.html', 'estimators.svg'}
    result.verify(request)
    root = Path(request.output_dir)
    report = json.loads((root / 'report.json').read_text())
    assert report['state'] == 'failed' and not report['releasable']
    assert report['evidence_label'] == 'diagnostic-only'
    assert report['estimators'] == [e.to_dict()['payload'] for e in b.estimates]
    assert any('nonfinite' in gate['reason'] for gate in report['gates'])
    for name in ('report.html', 'estimators.svg'):
        assert all(e.method in (root / name).read_text() for e in b.estimates)


def test_finite_balanced_products_must_not_overflow_before_cancellation(case, tmp_path, monkeypatch):
    b, m, r = case
    n = len(b.original_ids)
    b = replace(b, balance_basis_id='f(A,X)=1e308*X*1[A=3]', balance_names=('scaled interaction',),
                observed_exposure=(3., 3.) + (1.,) * (n - 2),
                shifted_exposure=(5., 5.) + (3.,) * (n - 2),
                ratios=((100., 100.) + (0.,) * (n - 2),) * len(b.seed_ids),
                balance_observed=((1e308,), (-1e308,)) + ((0.,),) * (n - 2),
                balance_shifted=((0.,),) * n)
    report = evaluate_case((b, m, r))
    assert report['state'] == 'released'
    for overlap in report['diagnostics']['overlap_by_seed'].values():
        assert overlap['functional_balance'] == [dict(function='scaled interaction',
                ratio_expectation=0., shifted_expectation=0., difference=0.)]
        # The unchanged binary64 weighted mean may round by a few ulps.
        assert overlap['achieved_shift'] == pytest.approx(2., rel=0., abs=4 * np.spacing(2.))
        assert 'ratio p99 > 10' in overlap['subsets']['all']['warnings']
    request = make_request(tmp_path, (b, m, r), monkeypatch)
    result = run_stage(request)
    assert result.status == 'pass'
    result.verify(request)
    saved = json.loads((Path(request.output_dir) / 'report.json').read_text())
    assert saved['releasable']


def test_forest_coordinates_remain_finite_for_finite_estimates(case):
    import xml.etree.ElementTree as ET
    b, _, _ = case
    estimates = [replace(e, value=value).to_dict()['payload']
                 for e, value in zip(b.estimates, (-1e306, 1e306))]
    report = dict(state='failed', evidence_label='diagnostic-only', estimators=estimates)
    root = ET.fromstring(render_forest(report))
    coordinates = [float(node.attrib['cx']) for node in root.findall('{http://www.w3.org/2000/svg}circle')]
    assert coordinates == [400., 900.]


def test_invalid_approval_metadata_keeps_derived_diagnostics(case, tmp_path, monkeypatch):
    request = make_request(tmp_path, case, monkeypatch, {'owner_decisions': []})
    result = run_stage(request)
    assert result.status == 'fail'
    result.verify(request)
    report = json.loads((Path(request.output_dir) / 'report.json').read_text())
    assert not report['releasable']
    assert report['diagnostics'] == json.loads(canonical_json(summarize(case[0], case[1])))
    assert len(report['estimators']) == 2


def repository_request(case, tmp_path, monkeypatch, repository):
    """Bind synthetic owner inputs to a repository independently of its storage."""
    request = make_request(tmp_path, case, monkeypatch)
    task = json.loads(Path(request.task_path).read_text())
    registry = repository / 'configs/approvals.yaml'
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_bytes(Path(task['approvals']).read_bytes())
    config = registry.with_name('reporting.yaml')
    config.write_bytes(CONFIG.read_bytes())
    monkeypatch.setattr(stage, 'OWNER_APPROVALS', registry)
    task['approvals'] = str(registry)
    Path(request.task_path).write_text(json.dumps(task))
    return replace(request, config_path=str(config), config_hash=file_hash(config),
                   task_hash=file_hash(request.task_path), dependency_paths=tuple(task.values()),
                   dependency_hashes=tuple(file_hash(p) for p in task.values()))


@pytest.mark.parametrize('protected', ['outputs', 'report', 'src', 'configs'])
@pytest.mark.parametrize('route', ['protected_alias', 'direct_target', 'alternate_alias',
                                  'chain', 'missing_target', 'equal_target'])
def test_protected_storage_alias_refuses_before_creation(case, tmp_path, monkeypatch, protected, route):
    repository = tmp_path / 'synthetic-repository'
    repository.mkdir()
    storage = tmp_path / 'protected-storage'
    if route != 'missing_target' or protected == 'configs':
        storage.mkdir()
    leaf = repository / protected
    if route == 'chain':
        intermediate = tmp_path / 'intermediate'
        intermediate.symlink_to(storage, target_is_directory=True)
        leaf.symlink_to(intermediate, target_is_directory=True)
    else:
        leaf.symlink_to(storage, target_is_directory=True)
    request = repository_request(case, tmp_path, monkeypatch, repository)
    if route == 'alternate_alias':
        alias = tmp_path / 'other-alias'
        alias.symlink_to(storage, target_is_directory=True)
        output = alias / 'new-report'
    elif route == 'direct_target':
        output = storage / 'new-report'
    elif route == 'equal_target':
        output = storage
    else:
        output = leaf / 'missing-suffix/new-report'
    request = replace(request, output_dir=str(output))
    before = {str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*')}
    result = run_stage(request)
    assert result.status == 'fail'
    assert result.artifacts == ()
    assert result.request_hash == request.content_hash
    assert {str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*')} == before


@pytest.mark.parametrize('route', ['new_directory', 'isolated_alias', 'alias_missing_suffix', 'lookalike_prefix'])
def test_isolated_output_alias_remains_releasable(case, tmp_path, monkeypatch, route):
    repository = tmp_path / 'synthetic-repository'
    request = repository_request(case, tmp_path, monkeypatch, repository)
    if route == 'lookalike_prefix':
        output = repository / 'outputs-independent/new-report'
    elif route == 'new_directory':
        output = tmp_path / 'new-parent/new-report'
    else:
        storage = tmp_path / 'isolated-storage'
        if route == 'isolated_alias':
            storage.mkdir()
        alias = tmp_path / 'isolated-alias'
        alias.symlink_to(storage, target_is_directory=True)
        output = alias / 'new-report'
    request = replace(request, output_dir=str(output))
    result = run_stage(request)
    assert result.status == 'pass', result.message
    result.verify(request)
    assert json.loads((output / 'report.json').read_text())['state'] == 'released'
    assert run_stage(request) == result
