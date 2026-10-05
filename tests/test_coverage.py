"""Offline synthetic accounting and complete-repetition controller regressions."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

from oxyformer.contracts import StageRequest
from oxyformer.provenance import ContractError, canonical_json, file_hash
from oxyformer.validation import coverage
from oxyformer.validation import campaign
from oxyformer.validation.scm import SCMConfig, CovariateFrame


def successful_records(n=1000, scenario="null_effect"):
    draws = coverage.repetition_plan("independent-test", scenario, n)
    # Symmetric quantiles give deterministic calibrated sampling variation.
    values = coverage.norm.ppf((np.arange(n) + .5) / n)
    records = [{"draw": d, "status": "success", "truth": 0., "causal_truth": 0.,
                "truth_integration_error": 0., "estimates": {
                    method: {"value": float(v), "se": 1.} for method in coverage.METHODS}}
               for d, v in zip(draws, values)]
    return draws, records


def summarize(draws, records, **kwargs):
    return coverage.summarize(records, draws, production_equivalent=True, null_scenario=True, **kwargs)


@pytest.mark.parametrize("n,k", [(1000, 950), (1000, 0), (1000, 50), (1000, 1000), (1, 0), (1, 1)])
def test_exact_one_sided_bounds(n, k):
    lower = coverage.binomial_bound(k, n, side="lower")
    upper = coverage.binomial_bound(k, n, side="upper")
    assert 0 <= lower <= upper <= 1
    if k:
        assert binom.sf(k - 1, n, lower) == pytest.approx(.05, abs=2e-14)
    else:
        assert lower == 0
    if k < n:
        assert binom.cdf(k, n, upper) == pytest.approx(.05, abs=2e-14)
    else:
        assert upper == 1


def test_empty_sample_has_uninformative_bounds():
    assert coverage.binomial_bound(0, 0, side="lower") == 0
    assert coverage.binomial_bound(0, 0, side="upper") == 1


def test_calibrated_final_scenario_passes_and_reports_all_metrics():
    draws, records = successful_records()
    result = summarize(draws, records)
    assert result["coverage_pass"]
    assert result["declared_repetitions"] == 1000
    assert result["numerical_failure_upper"] == pytest.approx(1 - .05**.001)
    metrics = result["methods"]["one_step"]
    assert metrics["coverage_count"] == 950
    assert metrics["abs_bias_over_sd"] < 1e-14
    assert .99 < metrics["mean_se_over_sd"] < 1.01
    assert result["truth_integration_error_max"] == 0


def test_numerical_failures_remain_in_failure_bound_and_denominators():
    draws, records = successful_records()
    for record in records[:12]:
        record.update(status="numerical_failure", estimates={})
    result = summarize(draws, records)
    assert result["counts"]["numerical_failure"] == 12
    assert result["numerical_failure_upper"] == pytest.approx(coverage.binomial_bound(12, 1000, side="upper"))
    assert result["numerical_failure_upper"] > .01
    assert result["declared_repetitions"] == 1000
    assert result["conditional_summary_count"] == 988
    assert not result["coverage_pass"]


@pytest.mark.parametrize("kind", ["missing", "incomplete", "insufficient", "smoke"])
def test_incomplete_or_small_or_smoke_cannot_certify(kind):
    draws, records = successful_records(999 if kind == "insufficient" else 1000)
    if kind == "missing":
        records.pop()
    if kind == "incomplete":
        records[-1]["status"] = "incomplete"
    result = coverage.summarize(records, draws, production_equivalent=kind != "smoke", null_scenario=True)
    assert not result["coverage_pass"]


def test_expected_eligibility_is_separate_from_numerical_failure():
    draws, records = successful_records()
    for row in records:
        row["status"] = "eligibility_rejection"
    result = summarize(draws, records, expected_rejection=True)
    assert result["eligibility_test_pass"]
    assert result["counts"]["numerical_failure"] == 0
    assert result["methods"]["one_step"]["coverage_lower"] == 0
    assert not result["coverage_pass"]


@pytest.mark.parametrize("kind", ["result", "declaration", "seed", "changed_seed"])
def test_duplicate_and_changed_draws_are_rejected(kind):
    draws, records = successful_records()
    if kind == "result":
        records.append(deepcopy(records[0]))
    elif kind == "declaration":
        draws.append(deepcopy(draws[0]))
    elif kind == "seed":
        draws[1]["seed"] = draws[0]["seed"]
    else:
        records[0] = deepcopy(records[0])
        records[0]["draw"]["seed"] += 1
    with pytest.raises(ContractError, match="duplicate|changed"):
        summarize(draws, records)


def test_seeds_are_deterministic_and_independent_of_batching():
    rows = coverage.repetition_plan("lock-one", "null", 1000)
    batched = [rows[start:start + 25] for start in range(0, 1000, 25)]
    assert rows == [row for batch in batched for row in batch]
    assert rows == coverage.repetition_plan("lock-one", "null", 1000)
    new = coverage.repetition_plan("lock-two", "null", 1000)
    assert not {r["seed"] for r in rows} & {r["seed"] for r in new}
    assert len(rows) == 1000  # Three neural seeds never multiply this count.


def test_integration_error_is_reported_and_affects_coverage():
    draws, records = successful_records()
    for row in records:
        row["truth_integration_error"] = 1.
    result = summarize(draws, records)
    assert result["truth_integration_error_max"] == 1
    assert not result["coverage_pass"]


def synthetic_endpoint(tmp_path):
    from test_nested_cv import endpoint
    from oxyformer.design.splits import InnerSplit
    from oxyformer.data.entity_graph import EntityGraph, EntityLink
    from oxyformer.design.eligibility import GeographyTable
    prepared = endpoint(tmp_path)
    ids = tuple(f"o{i:02d}" for i in range(60)) + ("unlabeled-acs",)
    names = tuple(c.name for c in prepared.data.manifest.schema)
    records = []
    geography_rows = []
    for index, oid in enumerate(ids[:-1]):
        row = dict(zip(names, prepared.data.rows[index % 30]))
        row.update(id=oid, county="c" if index < 30 else "d")
        records.append(tuple(row[n] for n in names))
        geography_rows.append(replace(prepared.geography.rows[index % 30], original_id=oid,
            tract_id=oid, county=row["county"], state="s" if index < 30 else "t",
            subblock=str(index // 2), assignment_geography=str(index // 2), latitude=float(index // 2)))
    records.append(prepared.data.rows[-1])
    geography_rows.append(prepared.geography.rows[-1])
    graph = EntityGraph(original_ids=ids, links=tuple(EntityLink(observation_id=oid,
        relation="repeated_geography", namespace="fixture", entity_id=str(i // 2)) for i, oid in enumerate(ids[:-1])))
    lineage = replace(prepared.data.manifest.lineage, unit_ids=ids)
    manifest = replace(prepared.data.manifest, original_ids=ids, entity_graph_hash=graph.content_hash, lineage=lineage)
    data = replace(prepared.data, manifest=manifest, rows=tuple(records))
    outer = replace(prepared.outer, original_ids=ids[:-1], fold_ids=tuple((i % 30) // 6 for i in range(60)),
        entity_graph_hash=graph.content_hash, lineage=replace(lineage, parent_hashes=(manifest.content_hash,)))
    geography = replace(prepared.geography, rows=tuple(geography_rows), data_manifest_hash=manifest.content_hash)
    covariates = replace(prepared.policy_covariates, original_ids=ids,
        geography_ids=tuple(str(i // 2) for i in range(len(ids))), support_keys=("s",) * len(ids))
    prepared = replace(prepared, data=data, entity_graph=graph, outer=outer, geography=geography,
                       policy_covariates=covariates)
    # The existing fixture has only fold zero's inner split. Complete it using
    # the same two-record geographic components in all five parent partitions.
    inners = []
    for fold in range(5):
        ids = prepared.outer.training_ids(fold)
        graph = EntityGraph(original_ids=ids, links=tuple(l for l in prepared.entity_graph.links if l.observation_id in ids))
        manifest = replace(prepared.data.manifest, original_ids=ids, entity_graph_hash=graph.content_hash,
            lineage=replace(prepared.data.manifest.lineage, unit_ids=ids, split_hash=prepared.outer.content_hash))
        split = replace(prepared.inner[0].split, original_ids=ids,
            fold_ids=tuple((i // 2) % 3 for i in range(len(ids))), entity_graph_hash=graph.content_hash,
            lineage=replace(manifest.lineage, parent_hashes=(manifest.content_hash,)))
        inners.append(InnerSplit(outer_fold=fold, data_manifest=manifest, entity_graph=graph, split=split, buffer_excluded_ids=()))
    # All covariates must satisfy the registered SCM numerical domain.
    columns = tuple(c.name for c in prepared.data.manifest.schema)
    xindex = columns.index("x")
    rows = tuple(tuple(.1 if j == xindex else value for j, value in enumerate(row)) for row in prepared.data.rows)
    data = replace(prepared.data, rows=rows)
    prepared = replace(prepared, data=data, inner=tuple(inners))
    target = prepared.outer.original_ids
    geo = {r.original_id: r for r in prepared.geography.rows}
    policy = dict(zip(prepared.policy_covariates.original_ids, zip(prepared.policy_covariates.geography_ids, prepared.policy_covariates.support_keys)))
    frame = CovariateFrame(original_ids=target, geography_ids=tuple(policy[i][0] for i in target),
        region_ids=tuple(geo[i].county for i in target), cluster_ids=tuple(policy[i][0] for i in target),
        coordinates=tuple((float(int(i[1:]) // 2), 0.) for i in target), columns=("x",),
        x=((.1,),) * len(target), support_keys=("s",) * len(target),
        weights=tuple(dict(zip(data.manifest.original_ids, data.column("w")))[i] for i in target),
        outcome_available=(True,) * len(target), biomarker_available=(True,) * len(target))
    geography = GeographyTable(rows=tuple(replace(r, latitude=float(int(r.original_id[1:]) // 2))
        if r.original_id in target else r for r in prepared.geography.rows),
        data_manifest_hash=prepared.data.manifest.content_hash, county_field="county", approval_reference="synthetic", mapping_review_id="synthetic")
    return replace(prepared, geography=geography), frame


def smoke_recipe(endpoint, frame):
    return {"endpoint_hash": endpoint.content_hash, "frame_hash": frame.content_hash,
        "nested_cv": {"ssl_epochs": 1, "frozen_epochs": 1, "synthetic": True},
        "inference": {"primary_bandwidth_km": 100, "county_locations": {"c": [0., 0.], "d": [5., 5.]}}}


@pytest.mark.parametrize("failure", [FloatingPointError, np.linalg.LinAlgError, ValueError])
def test_draw_executes_once_and_numerical_failure_is_retained(tmp_path, monkeypatch, failure):
    endpoint, frame = synthetic_endpoint(tmp_path)
    scenario = SCMConfig(name="null_effect", active_mechanisms=("null",), effect="null")
    draw = coverage.repetition_plan("failure-test", scenario.name, 1)[0]
    calls = []
    def fail(*args):
        calls.append(1)
        raise failure("injected estimator failure")
    monkeypatch.setattr(coverage, "estimate_repetition", fail)
    result = coverage.execute_draw(draw, frame, scenario, endpoint, smoke_recipe(endpoint, frame), tmp_path, float("inf"))
    assert calls == [1]
    assert result["status"] == "numerical_failure"
    assert "injected estimator failure" in result["reason"]
    assert result["draw"] == draw


def test_incomplete_fold_stops_repetition_without_estimating(tmp_path, monkeypatch):
    endpoint, frame = synthetic_endpoint(tmp_path)
    sample = coverage.generate_suite_a(frame, SCMConfig(name="null_effect", active_mechanisms=("null",), effect="null"), endpoint.policy)
    prepared = coverage.bind_observations(endpoint, sample.observations)
    calls = []
    class Incomplete:
        complete = False
    def fake(config, outer, seed, **kwargs):
        calls.append((config.fold, seed))
        return Incomplete()
    monkeypatch.setattr(coverage.nested_cv, "run_fold", fake)
    assert coverage.estimate_repetition(prepared, smoke_recipe(endpoint, frame), tmp_path, float("inf")) is None
    assert calls == [(0, 1103)]


def test_complete_controller_counts_fifteen_fits_as_one_repetition(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from oxyformer.contracts import OOFNuisances
    endpoint, frame = synthetic_endpoint(tmp_path)
    sample = coverage.generate_suite_a(frame, SCMConfig(name="null_effect", active_mechanisms=("null",), effect="null"), endpoint.policy)
    prepared = coverage.bind_observations(endpoint, sample.observations)
    calls = []
    def fake_fit(config, outer, seed, **kwargs):
        calls.append((config.fold, seed))
        held = tuple(i for i, f in zip(outer.original_ids, outer.fold_ids) if f == config.fold)
        return SimpleNamespace(complete=True, prediction_inputs=SimpleNamespace(original_ids=held), fold=config.fold, seed=seed)
    def fake_predict(artifact, view, policy):
        ids = view.original_ids
        weights = dict(zip(prepared.data.manifest.original_ids, prepared.data.column("w")))
        return OOFNuisances(spec=view.spec, original_ids=ids, fold_ids=(artifact.fold,) * len(ids),
            seed_ids=(artifact.seed,) * len(ids), mu_a=(50.,) * len(ids), mu_d=(50.,) * len(ids),
            r_a=(1.,) * len(ids), r_d=(1.,) * len(ids), origin_weights=tuple(weights[i] for i in ids),
            lineage=replace(prepared.data.manifest.lineage, unit_ids=ids, split_hash=prepared.outer.content_hash))
    monkeypatch.setattr(coverage.nested_cv, "run_fold", fake_fit)
    monkeypatch.setattr(coverage.nested_cv, "predict", fake_predict)
    result = coverage.estimate_repetition(prepared, smoke_recipe(endpoint, frame), tmp_path, coverage.time.monotonic() + 60)
    assert calls == [(fold, seed) for fold in range(5) for seed in coverage.nested_cv.SEEDS]
    assert set(result) == set(coverage.METHODS)
    assert all(type(value["value"]) is float and type(value["se"]) is float for value in result.values())
    assert result["one_step"]["value"] == 0
    # FP64 targeting centers and subtracts means around 50; one ULP there
    # bounds the observed two-femtounit cancellation residual on both builds.
    assert result["cv_tmle"]["value"] == pytest.approx(0., abs=np.spacing(50.))
    assert set(result["one_step"]["spatial_se"]) == {"50.0", "100.0", "200.0"}


def test_real_nested_fold_accepts_rebound_scm_endpoint(tmp_path):
    import torch
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        endpoint, frame = synthetic_endpoint(tmp_path)
        sample = coverage.generate_suite_a(frame, SCMConfig(name="null_effect", active_mechanisms=("null",), effect="null"), endpoint.policy)
        prepared = coverage.bind_observations(endpoint, sample.observations)
        recipe = smoke_recipe(endpoint, frame)
        config = prepared.configuration(0, tmp_path / "fit", **coverage.nested_cv._settings(recipe, prepared))
        result = coverage.nested_cv.run_fold(config, prepared.outer, 1103, geography=prepared.geography)
        assert result.complete
        view = coverage.nested_cv.subset(prepared.data.covariates(("x",)), result.prediction_inputs.original_ids)
        output = coverage.nested_cv.predict(result, view, prepared.policy)
        assert len(output.original_ids) == 12
        assert set(output.seed_ids) == {1103}
    finally:
        torch.set_num_threads(previous)


def test_stage_atomically_retains_each_declared_draw(tmp_path, monkeypatch):
    from test_campaign import request, STAMPS
    endpoint, frame = synthetic_endpoint(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "endpoint.json").write_text(endpoint.to_json())
    (inputs / "frame.json").write_text(frame.to_json())
    scenario = SCMConfig(name="null_effect", active_mechanisms=("null",), effect="null").to_dict()["payload"]
    draws = coverage.repetition_plan("smoke-stage", "null_effect", 2)
    task = {"id": "simulation-smoke", "stage": "simulation-smoke", "parameters": {
        "mode": "smoke", "recipe": smoke_recipe(endpoint, frame),
        "scenario": {"name": "null_effect", "active_mechanisms": ["null"], "effect": "null"}, "draws": draws,
        "wall_seconds": 60, "endpoint_input": {"dependency": "input", "path": "endpoint.json"},
        "frame_input": {"dependency": "input", "path": "frame.json"}}}
    req = request(tmp_path / "request", task, {"input": inputs})
    calls = []
    def estimate(*args):
        calls.append(1)
        if len(calls) == 1:
            raise FloatingPointError("injected draw failure")
        return {m: {"value": 0., "se": 1.} for m in coverage.METHODS}
    monkeypatch.setattr(coverage, "estimate_repetition", estimate)
    monkeypatch.setattr(campaign, "fingerprint", lambda: deepcopy(STAMPS))
    result = coverage.run_stage(req)
    assert result.status == "pass", result.message
    result.verify(req)
    assert len(calls) == 2
    root = Path(req.output_dir)
    summary = json.loads((root / "result.json").read_text())
    assert summary["scenario"] == scenario
    assert not summary["certifies_production_coverage"]
    assert not summary["summary"]["production_equivalent"]
    assert summary["summary"]["counts"]["numerical_failure"] == 1
    assert len(list(root.glob("repetitions/*/result.json"))) == 2
    assert len([r for r in result.artifacts if r.kind == "repetition"]) == 2
    assert coverage.run_stage(req).status == "blocked"
    assert len(calls) == 2  # An existing failed draw is never rerun until success.


@pytest.mark.parametrize("failure", [KeyError, TypeError, OSError])
def test_execution_faults_remain_incomplete_not_numerical(tmp_path, monkeypatch, failure):
    endpoint, frame = synthetic_endpoint(tmp_path)
    scenario = SCMConfig(name="null_effect", active_mechanisms=("null",), effect="null")
    draw = coverage.repetition_plan("execution-fault", scenario.name, 1)[0]
    def fail(*args):
        raise failure("injected execution fault")
    monkeypatch.setattr(coverage, "estimate_repetition", fail)
    result = coverage.execute_draw(draw, frame, scenario, endpoint, smoke_recipe(endpoint, frame), tmp_path, float("inf"))
    assert result["status"] == "incomplete"
    assert result["draw"] == draw
    assert "execution failure" in result["reason"]


def test_execution_runtime_error_cannot_be_tolerated_as_numerical_failure(tmp_path, monkeypatch):
    endpoint, frame = synthetic_endpoint(tmp_path)
    scenario = SCMConfig(name="null_effect", active_mechanisms=("null",), effect="null")
    draws, records = successful_records()
    def fail(*args):
        raise RuntimeError("worker process disconnected")
    monkeypatch.setattr(coverage, "estimate_repetition", fail)
    records[0] = coverage.execute_draw(draws[0], frame, scenario, endpoint, smoke_recipe(endpoint, frame), tmp_path, float("inf"))
    result = summarize(draws, records)
    assert not result["coverage_pass"]
    assert result["counts"]["incomplete"] == 1
    assert result["counts"]["numerical_failure"] == 0


def test_profile_measures_final_publication_and_verification(tmp_path, monkeypatch):
    from test_campaign import request, STAMPS
    endpoint, frame = synthetic_endpoint(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "endpoint.json").write_text(endpoint.to_json())
    (inputs / "frame.json").write_text(frame.to_json())
    recipe = smoke_recipe(endpoint, frame)
    recipe["nested_cv"] = {}
    task = {"id": "simulation-smoke", "stage": "simulation-smoke", "parameters": {
        "mode": "profile", "recipe": recipe,
        "scenario": {"name": "null_effect", "effect": "null", "active_mechanisms": ["null"]},
        "draws": coverage.repetition_plan("measured-publication", "null_effect", 2), "wall_seconds": 1000,
        "endpoint_input": {"dependency": "input", "path": "endpoint.json"},
        "frame_input": {"dependency": "input", "path": "frame.json"}}}
    req = request(tmp_path / "request", task, {"input": inputs})
    clock = [0.]
    monkeypatch.setattr(coverage.time, "monotonic", lambda: clock[0])
    deadlines = []
    original_verify = coverage.StageRequest.verify_inputs
    preflight_calls = []
    def delayed_preflight(self):
        if not preflight_calls:
            clock[0] += 5.
        preflight_calls.append(1)
        return original_verify(self)
    monkeypatch.setattr(coverage.StageRequest, "verify_inputs", delayed_preflight)
    def estimate(*args):
        deadlines.append(args[-1])
        clock[0] += 10.
        return {method: {"value": 0., "se": 1.} for method in coverage.METHODS}
    def stamp():
        clock[0] += 50.
        return deepcopy(STAMPS)
    original_publish = coverage.publish
    def delayed_publish(*args, **kwargs):
        clock[0] += 2.
        return original_publish(*args, **kwargs)
    monkeypatch.setattr(coverage, "estimate_repetition", estimate)
    monkeypatch.setattr(campaign, "fingerprint", stamp)
    monkeypatch.setattr(coverage, "publish", delayed_publish)
    result = coverage.run_stage(req)
    assert result.status == "pass", result.message
    result.verify(req)
    timing = json.loads((Path(req.output_dir) / "timing.json").read_text())
    assert deadlines == [1000., 1000.]  # Public-entry preflight consumes the same budget.
    assert timing["wall_seconds"] == 77.  # Includes preflight and normal leaf publication/verification.
    assert timing["complete_repetition_seconds"] == [10., 10.]
