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



def frame_subset(frame, ids):
    positions = [frame.original_ids.index(i) for i in ids]
    names = ("original_ids", "geography_ids", "region_ids", "cluster_ids", "coordinates",
             "x", "support_keys", "weights", "outcome_available", "biomarker_available")
    return replace(frame, **{name: tuple(getattr(frame, name)[j] for j in positions) for name in names})


def role_endpoint(tmp_path):
    endpoint, full = synthetic_endpoint(tmp_path)
    # Keep dependence pairs atomic; the last pair is inert, the first training-only.
    roles = ("training",) * 2 + ("target",) * (len(full.original_ids) - 4) + ("inert",) * 2
    by_id = dict(zip(full.original_ids, roles))
    def split_roles(split):
        pairs = [(i, f) for i, f in zip(split.original_ids, split.fold_ids) if by_id[i] == "target"]
        return replace(split, original_ids=tuple(i for i, _ in pairs), fold_ids=tuple(f for _, f in pairs),
            frame_ids=split.original_ids, frame_fold_ids=split.fold_ids,
            membership_roles=tuple(by_id[i] for i in split.original_ids))
    names = tuple(c.name for c in endpoint.data.manifest.schema)
    yi = names.index(endpoint.data.manifest.outcome_field)
    rows = tuple(tuple(None if j == yi else v for j, v in enumerate(row)) for row in endpoint.data.rows)
    endpoint = replace(endpoint, outer=split_roles(endpoint.outer), data=replace(endpoint.data, rows=rows),
        inner=tuple(replace(inner, split=split_roles(inner.split)) for inner in endpoint.inner))
    frame = frame_subset(full, tuple(i for i in full.original_ids if by_id[i] != "inert"))
    frame = replace(frame, weights=tuple(w if by_id[i] == "target" else 0.
                                         for i, w in zip(frame.original_ids, frame.weights)))
    return endpoint, frame


@pytest.mark.parametrize("role", ["target", "training"])
def test_scm_frame_checks_target_and_training_memberships_separately(tmp_path, role):
    endpoint, frame = role_endpoint(tmp_path)
    removed = next(i for i, r in zip(endpoint.outer.frame_original_ids, endpoint.outer.roles) if r == role)
    incomplete = frame_subset(frame, tuple(i for i in frame.original_ids if i != removed))
    sample = coverage.generate_suite_a(incomplete,
        SCMConfig(name="null_effect", active_mechanisms=("null",), effect="null"), endpoint.policy)
    with pytest.raises(ContractError, match="SCM frame " + role + " rows"):
        coverage.bind_observations(endpoint, sample.observations)


def test_scm_training_rows_get_y_and_zero_target_mass_while_inert_rows_get_no_y(tmp_path):
    endpoint, frame = role_endpoint(tmp_path)
    config = SCMConfig(name="nonlinear", active_mechanisms=("nonlinear",), effect="nonlinear")
    sample = coverage.generate_suite_a(frame, config, endpoint.policy)
    target_only = coverage.generate_suite_a(frame_subset(frame, endpoint.outer.original_ids), config, endpoint.policy)
    assert sample.observed_law_truth.value == target_only.observed_law_truth.value
    assert sample.structural_causal_truth.value == target_only.structural_causal_truth.value
    rebound = coverage.bind_observations(endpoint, sample.observations)
    y = dict(zip(rebound.data.manifest.original_ids, rebound.data.column(rebound.data.manifest.outcome_field)))
    for i, role in zip(endpoint.outer.frame_original_ids, endpoint.outer.roles):
        assert (y[i] is None) == (role == "inert")
    for fold in range(5):
        assert all(y[i] is not None for i in rebound.outer.training_ids(fold))
    # A training-only row cannot accidentally contribute mass to the estimand.
    wrong = replace(frame, weights=(1.,) + frame.weights[1:])
    with pytest.raises(ContractError, match="SCM weight drift"):
        coverage.bind_observations(endpoint, replace(sample.observations, frame=wrong))

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


@pytest.mark.parametrize("seed_count", [3, 1])
def test_complete_controller_counts_fifteen_fits_as_one_repetition(tmp_path, monkeypatch, seed_count):
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
    from oxyformer.training.fit import RecipeStudy
    result = coverage.estimate_repetition(prepared, smoke_recipe(endpoint, frame), tmp_path,
        coverage.time.monotonic() + 60, study=RecipeStudy(seed_count=seed_count))
    assert calls == [(fold, seed) for fold in range(5) for seed in coverage.nested_cv.SEEDS[:seed_count]]
    assert prepared.outer.seed_ids == coverage.nested_cv.SEEDS
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


@pytest.mark.parametrize("budget", [1000, 60])
def test_profile_measures_final_publication_and_verification(tmp_path, monkeypatch, budget):
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
        "draws": coverage.repetition_plan("measured-publication", "null_effect", 2), "wall_seconds": budget,
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
        if clock[0] + 10. > args[-1]:
            return None
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
    assert deadlines == [float("inf"), float("inf")]  # Planning estimates do not truncate profiles.
    assert timing["wall_seconds"] == 77.  # Includes preflight and normal leaf publication/verification.
    assert timing["complete_repetition_seconds"] == [10., 10.]
    assert timing["setup_seconds"] == 55.
    assert timing["publication_verification_seconds"] == 2.


@pytest.mark.parametrize("field,bad", [("value", float("nan")), ("se", float("inf")),
                                     ("spatial_se", float("nan")), ("se", -1.)])
def test_nonfinite_estimator_outputs_are_retained_as_numerical_failures(tmp_path, monkeypatch, field, bad):
    from test_campaign import request, STAMPS
    endpoint, frame = synthetic_endpoint(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "endpoint.json").write_text(endpoint.to_json())
    (inputs / "frame.json").write_text(frame.to_json())
    draws = coverage.repetition_plan("nonfinite-stage", "null_effect", 3)
    task = {"id": "simulation-smoke", "stage": "simulation-smoke", "parameters": {
        "mode": "smoke", "recipe": smoke_recipe(endpoint, frame),
        "scenario": {"name": "null_effect", "active_mechanisms": ["null"], "effect": "null"},
        "draws": draws, "wall_seconds": 60,
        "endpoint_input": {"dependency": "input", "path": "endpoint.json"},
        "frame_input": {"dependency": "input", "path": "frame.json"}}}
    req = request(tmp_path / "request", task, {"input": inputs})
    calls = []
    def estimate(*args):
        calls.append(1)
        result = {m: {"value": 0., "se": 1., "spatial_se": {"50": 1., "100": 1., "200": 1.}}
                  for m in coverage.METHODS}
        if len(calls) == 1:
            if field == "spatial_se":
                result["one_step"][field]["50"] = bad
            else:
                result["one_step"][field] = bad
        return result
    monkeypatch.setattr(coverage, "estimate_repetition", estimate)
    monkeypatch.setattr(campaign, "fingerprint", lambda: deepcopy(STAMPS))
    result = coverage.run_stage(req)
    assert result.status == "pass", result.message
    result.verify(req)
    assert len(calls) == 3
    root = Path(req.output_dir)
    aggregate = coverage.read_json(root / "result.json")
    summary = aggregate["summary"]
    assert summary["counts"]["numerical_failure"] == 1
    assert summary["counts"]["success"] == 2
    assert summary["declared_repetitions"] == 3
    assert summary["numerical_failure_upper"] == pytest.approx(coverage.binomial_bound(1, 3, side="upper"))
    failure = coverage.read_json(root / "repetitions" / draws[0]["repetition_id"] / "result.json")
    assert failure["status"] == "numerical_failure" and failure["estimates"] == {}
    assert "invalid" in failure["reason"] or "negative" in failure["reason"]
    assert (root / "runtime.json").is_file()
    assert coverage.run_stage(req).status == "blocked"
    assert len(calls) == 3  # Never redraw the failed sample to obtain a success.


@pytest.mark.parametrize("field", ["observed_absolute_difference", "quadrature_tail_absolute_bound",
                                   "truth_serialization_absolute_bound", "converged"])
def test_truth_error_components_must_be_available(tmp_path, monkeypatch, field):
    endpoint, frame = synthetic_endpoint(tmp_path)
    scenario = SCMConfig(name="null_effect", active_mechanisms=("null",), effect="null")
    sample = coverage.generate_suite_a(frame, scenario, endpoint.policy)
    broken = replace(sample, integration_uncertainty=replace(sample.integration_uncertainty,
                     **{field: False if field == "converged" else None}))
    monkeypatch.setattr(coverage, "generate_suite_a", lambda *args, **kwargs: broken)
    calls = []
    def estimate(*args):
        calls.append(1)
        return {m: {"value": 0., "se": 1.} for m in coverage.METHODS}
    monkeypatch.setattr(coverage, "estimate_repetition", estimate)
    draws, records = successful_records()
    records[0] = coverage.execute_draw(draws[0], frame, scenario, endpoint,
                                       smoke_recipe(endpoint, frame), tmp_path, float("inf"))
    summary = summarize(draws, records)
    assert not summary["coverage_pass"], "Unavailable error components cannot become zero-error coverage"
    assert records[0]["status"] == "incomplete"
    assert records[0]["estimates"] == {}
    assert summary["declared_repetitions"] == 1000
    assert calls == []


def test_profile_adapter_mismatch_is_incomplete(tmp_path, monkeypatch):
    from test_campaign import request, STAMPS
    endpoint, frame = synthetic_endpoint(tmp_path)
    frame = replace(frame, x=((.2,),) * len(frame.original_ids))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "endpoint.json").write_text(endpoint.to_json())
    (inputs / "frame.json").write_text(frame.to_json())
    recipe = smoke_recipe(endpoint, frame)
    recipe["nested_cv"] = {}
    task = {"id": "simulation-smoke", "stage": "simulation-smoke", "parameters": {
        "mode": "profile", "recipe": recipe,
        "scenario": {"name": "null_effect", "active_mechanisms": ["null"], "effect": "null"},
        "draws": coverage.repetition_plan("mismatched-adapter", "null_effect", 1), "wall_seconds": 60,
        "endpoint_input": {"dependency": "input", "path": "endpoint.json"},
        "frame_input": {"dependency": "input", "path": "frame.json"}}}
    req = request(tmp_path / "request", task, {"input": inputs})
    monkeypatch.setattr(campaign, "fingerprint", lambda: deepcopy(STAMPS))
    def no_fitting(*args):
        raise AssertionError("adapter mismatch must precede fitting")
    monkeypatch.setattr(coverage, "estimate_repetition", no_fitting)
    result = coverage.run_stage(req)
    assert result.status == "fail", result.message
    result.verify(req)
    aggregate = coverage.read_json(Path(req.output_dir) / "result.json")
    assert aggregate["summary"]["counts"]["incomplete"] == 1
    assert aggregate["summary"]["counts"]["numerical_failure"] == 0
    assert "fixed-X template drift" in aggregate["records"][0]["reason"]
    assert (Path(req.output_dir) / "_profiled/runtime.json").is_file()


@pytest.mark.parametrize("failure", [OSError, ValueError])
def test_failed_publication_does_not_expose_success_estimates(tmp_path, monkeypatch, failure):
    endpoint, frame = synthetic_endpoint(tmp_path)
    scenario = SCMConfig(name="null_effect", active_mechanisms=("null",), effect="null")
    draw = coverage.repetition_plan("failed-publication", scenario.name, 1)[0]
    monkeypatch.setattr(coverage, "estimate_repetition", lambda *args:
                        {m: {"value": 0., "se": 1.} for m in coverage.METHODS})
    def fail_write(*args):
        raise failure("injected publication fault")
    monkeypatch.setattr(coverage, "atomic_write", fail_write)
    record = coverage.execute_draw(draw, frame, scenario, endpoint, smoke_recipe(endpoint, frame),
                                   tmp_path, float("inf"))
    assert record["status"] == "incomplete"
    assert record["estimates"] == {}
    assert "truth" not in record
    assert "injected publication fault" in record["reason"]


def chain_request(tmp_path, task, dependencies):
    from test_campaign import request
    req = request(tmp_path, task, dependencies)
    paths = tuple(str(p) for root in dependencies.values() for p in sorted(root.iterdir()) if p.is_file())
    return replace(req, dependency_paths=paths, dependency_hashes=tuple(map(file_hash, paths)))


def chain_fixture(tmp_path):
    endpoint, frame = synthetic_endpoint(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "endpoint.json").write_text(endpoint.to_json())
    (inputs / "frame.json").write_text(frame.to_json())
    task = {"id": "chain-0", "stage": "simulation-smoke", "outputs": ["continuation.tar", "result.json", "timing.json"],
        "continuation": {"owner": "coverage-test", "step": 0, "predecessor": None},
        "parameters": {"mode": "smoke", "recipe": smoke_recipe(endpoint, frame),
            "study": {"seed_count": 1, "grid_points": 2},
            "scenario": {"name": "null_effect", "active_mechanisms": ["null"], "effect": "null"},
            "draws": coverage.repetition_plan("coverage-chain", "null_effect", 1), "wall_seconds": 14400,
            "endpoint_input": {"dependency": "inputs", "path": "endpoint.json"},
            "frame_input": {"dependency": "inputs", "path": "frame.json"}}}
    return task, inputs


def test_coverage_chain_bit_equivalence(tmp_path, monkeypatch):
    """Actual training; deadlines cut inside a fold and after the first outer fold."""
    import os
    import struct
    import time
    import torch
    from oxyformer.training import fit
    from oxyformer.validation.leakage import assert_fitted_invariant, model_hashes
    from test_campaign import STAMPS
    monkeypatch.setattr(campaign, "fingerprint", lambda: deepcopy(STAMPS))
    monkeypatch.setenv("OXYFORMER_DEVICE", os.environ.get("OXYFORMER_CHAIN_DEVICE", "cpu"))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    task, inputs = chain_fixture(tmp_path)
    fitted = {"reference": [], "chain": []}
    phase = ["reference"]
    original_fold = coverage.nested_cv.run_fold
    def capture(*args, **kwargs):
        artifact = original_fold(*args, **kwargs)
        if artifact.complete:
            fitted[phase[0]].append(artifact)
        return artifact
    monkeypatch.setattr(coverage.nested_cv, "run_fold", capture)
    try:
        reference = chain_request(tmp_path / "reference", task, {"inputs": inputs})
        result = coverage.run_stage(reference)
        assert result.status == "pass", result.message
        expected = coverage.read_json(Path(reference.output_dir) / "result.json")
        assert expected["terminal"]
        phase[0] = "chain"
        clock = [0.]
        monotonic = time.monotonic
        monkeypatch.setattr(coverage.time, "monotonic", lambda: monotonic() + clock[0])
        step = [0]
        original_reason = fit._Budget.reason
        cut = [False, False]
        def deadline(self):
            if step[0] == 0 and self.batches >= 3 and not cut[0]:
                clock[0] += 14400
                cut[0] = True
            return original_reason(self)
        monkeypatch.setattr(fit._Budget, "reason", deadline)
        original_predict = coverage.nested_cv.predict
        def predict(*args, **kwargs):
            value = original_predict(*args, **kwargs)
            if step[0] == 1 and not cut[1]:
                clock[0] += 14400
                cut[1] = True
            return value
        monkeypatch.setattr(coverage.nested_cv, "predict", predict)
        requests = []
        for index in range(3):
            step[0] = index
            current = deepcopy(task)
            current["id"] = f"chain-{index}"
            current["continuation"] = {"owner": "coverage-test", "step": index,
                "predecessor": None if index == 0 else f"chain-{index - 1}"}
            dependencies = {"inputs": inputs}
            if requests:
                dependencies[f"chain-{index - 1}"] = Path(requests[-1].output_dir)
                monkeypatch.setattr(coverage, "generate_suite_a", lambda *a, **k: pytest.fail("redrew observations"))
            req = chain_request(tmp_path / f"slice-{index}", current, dependencies)
            done = coverage.run_stage(req)
            assert done.status == "pass", done.message
            done.verify(req)
            actual = coverage.read_json(Path(req.output_dir) / "result.json")
            assert actual["terminal"] == (index == 2)
            if index != 2:
                assert not list(Path(req.output_dir).glob("repetitions/*/result.json"))
            requests.append(req)
        assert cut == [True, True]
        assert len(fitted["reference"]) == len(fitted["chain"]) == 5
        rid = task["parameters"]["draws"][0]["repetition_id"]
        expected_oof = coverage.read_json(Path(reference.output_dir) / "repetitions" / rid / "nuisances.json")
        actual_oof = coverage.read_json(Path(requests[-1].output_dir) / "repetitions" / rid / "nuisances.json")
        # Every payload field is enumerated, including discrete identity fields.
        excluded = ["nuisances.payload.lineage.parent_hashes", "record.wall_seconds"]
        fields = {}
        def compare(a, b, name):
            if name in excluded:
                fields[name] = {"result": "excluded", "reason": "provenance" if "lineage" in name else "timing"}
            elif isinstance(a, dict):
                assert set(a) == set(b), name
                for key in sorted(a):
                    compare(a[key], b[key], name + "." + key)
            elif isinstance(a, list):
                assert len(a) == len(b), name
                for index, (x, y) in enumerate(zip(a, b)):
                    compare(x, y, f"{name}[{index}]")
            else:
                identical = struct.pack("!d", a) == struct.pack("!d", b) if isinstance(a, float) else type(a) == type(b) and a == b
                fields[name] = {"result": "identical" if identical else "different", "reference": a, "chained": b}
        compare(expected_oof, actual_oof, "nuisances")
        compare(expected["records"][0], actual["records"][0], "record")
        for fold, (a, b) in enumerate(zip(fitted["reference"], fitted["chain"])):
            compare(model_hashes(a), model_hashes(b), f"models.fold{fold}")
            states = [coverage.nested_cv.load_checkpoint(x.checkpoint, x.checkpoint.identity)["controller"] for x in (a, b)]
            compare(states[0]["selection"], states[1]["selection"], f"selection.fold{fold}")
            compare(states[0]["counts"], states[1]["counts"], f"counts.fold{fold}")
            for name in ("calibration", "transfer_diagnostics"):
                x, y = [json.loads(state[name]) for state in states]
                excluded.append(f"{name}.fold{fold}.payload.lineage.parent_hashes")
                compare(x, y, f"{name}.fold{fold}")
            for kind in ("outcome", "origin"):
                compare(states[0]["final"][kind]["preprocessing"], states[1]["final"][kind]["preprocessing"],
                        f"preprocessing.fold{fold}.{kind}")
        report = {"device": str(fit.resolve_device()), "leaves": 3,
            "cuts": ["inside fold", "between folds"], "excluded_fields": excluded,
            "passed": all(v["result"] != "different" for v in fields.values()), "fields": fields}
        report_path = os.environ.get("OXYFORMER_EQUIVALENCE_REPORT")
        if report_path:
            Path(report_path).write_text(json.dumps(report, indent=2))
        assert report["passed"], [k for k, v in fields.items() if v["result"] == "different"]
        for a, b in zip(fitted["reference"], fitted["chain"]):
            assert_fitted_invariant(a, b)
    finally:
        torch.set_num_threads(old_threads)


@pytest.fixture
def saved_coverage_chain(tmp_path, monkeypatch):
    from test_campaign import STAMPS
    monkeypatch.setattr(campaign, "fingerprint", lambda: deepcopy(STAMPS))
    task, inputs = chain_fixture(tmp_path)
    monkeypatch.setattr(coverage, "estimate_repetition", lambda *a, **k: None)
    req = chain_request(tmp_path / "first", task, {"inputs": inputs})
    result = coverage.run_stage(req)
    assert result.status == "pass", result.message
    archive = Path(req.output_dir) / "continuation.tar"
    import tarfile
    with tarfile.open(archive) as tar:
        header = json.loads(tar.extractfile("coverage.json").read())
    return req, task, inputs, header


@pytest.mark.parametrize("field", ["draws", "scenario", "endpoint_hash", "frame_hash", "recipe_hash", "study",
                                    "lock_hash", "code_identity", "mode", "wall_seconds", "scientific_fingerprint", "environment_hash"])
def test_coverage_binding_refuses_each_field(saved_coverage_chain, field):
    req, task, _, header = saved_coverage_chain
    binding = deepcopy(header["binding"])
    binding[field] = "changed"
    with pytest.raises(ContractError, match="binding mismatch"):
        coverage._read_continuation(Path(req.output_dir) / "continuation.tar", binding,
            {"owner": "coverage-test", "step": 1, "predecessor": task["id"]})


@pytest.mark.parametrize("field,value", [("owner", "other"), ("step", 0), ("step", 2), ("predecessor", "other")])
def test_coverage_chain_refuses_gap_reordering_and_owner(saved_coverage_chain, field, value):
    req, task, _, header = saved_coverage_chain
    chain = {"owner": "coverage-test", "step": 1, "predecessor": task["id"], field: value}
    with pytest.raises(ContractError, match="nonconsecutive"):
        coverage._read_continuation(Path(req.output_dir) / "continuation.tar", header["binding"], chain)


@pytest.mark.parametrize("change", ["blob", "observation", "draw_seed", "draw_id"])
def test_coverage_rejects_tampered_state(saved_coverage_chain, tmp_path, change):
    import tarfile
    from io import BytesIO
    req, task, _, header = saved_coverage_chain
    source = Path(req.output_dir) / "continuation.tar"
    with tarfile.open(source) as tar:
        blobs = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    if change == "blob":
        name = next(k for k in blobs if k.startswith("blobs/"))
        blobs[name] += b"x"
    else:
        binding = header["states"][0]["binding"]
        if change == "observation":
            binding["observation_hash"] = "0" * 64
        else:
            binding["draw"]["seed" if change == "draw_seed" else "repetition_id"] = 0
        blobs["coverage.json"] = canonical_json(header).encode()
    target = tmp_path / "tampered.tar"
    with tarfile.open(target, "w") as tar:
        for name, value in blobs.items():
            member = tarfile.TarInfo(name)
            member.size = len(value)
            tar.addfile(member, BytesIO(value))
    with pytest.raises(ContractError, match="tampered|binding drift"):
        coverage._read_continuation(target, header["binding"],
            {"owner": "coverage-test", "step": 1, "predecessor": task["id"]})


def test_coverage_retry_uses_last_export_without_redraw(saved_coverage_chain, tmp_path, monkeypatch):
    req, task, inputs, _ = saved_coverage_chain
    task = deepcopy(task)
    task.update(id="chain-1", continuation={"owner": "coverage-test", "step": 1, "predecessor": "chain-0"})
    dependencies = {"inputs": inputs, "chain-0": Path(req.output_dir)}
    monkeypatch.setattr(coverage, "generate_suite_a", lambda *a, **k: pytest.fail("redrew predecessor"))
    def failure(*a, **k):
        raise OSError("injected leaf failure")
    monkeypatch.setattr(coverage, "estimate_repetition", failure)
    failed = chain_request(tmp_path / "failed", task, dependencies)
    result = coverage.run_stage(failed)
    assert result.status == "blocked" and "injected leaf failure" in result.message
    assert not (Path(failed.output_dir) / "continuation.tar").exists()
    monkeypatch.setattr(coverage, "estimate_repetition", lambda *a, **k:
        {m: {"value": 0., "se": 1.} for m in coverage.METHODS})
    retry = chain_request(tmp_path / "retry", task, dependencies)
    result = coverage.run_stage(retry)
    assert result.status == "pass", result.message
    actual = coverage.read_json(Path(retry.output_dir) / "result.json")
    assert actual["terminal"] and actual["records"][0]["draw"] == task["parameters"]["draws"][0]


def test_coverage_predecessor_must_be_declared(saved_coverage_chain, tmp_path):
    req, task, inputs, _ = saved_coverage_chain
    task = deepcopy(task)
    task.update(id="chain-1", continuation={"owner": "coverage-test", "step": 1, "predecessor": "chain-0"})
    next_req = chain_request(tmp_path / "next", task, {"inputs": inputs, "chain-0": Path(req.output_dir)})
    paths = tuple(p for p in next_req.dependency_paths if not p.endswith("continuation.tar"))
    next_req = replace(next_req, dependency_paths=paths, dependency_hashes=tuple(map(file_hash, paths)))
    result = coverage.run_stage(next_req)
    assert result.status == "blocked" and "undeclared dependency" in result.message


@pytest.mark.parametrize("folds", [(1,), (1, 0), (0, 0)])
def test_coverage_prediction_parts_must_be_consecutive(saved_coverage_chain, tmp_path, folds):
    from oxyformer.contracts import OOFNuisances
    req, task, inputs, header = saved_coverage_chain
    chain = {"owner": "coverage-test", "step": 1, "predecessor": task["id"]}
    states = coverage._read_continuation(Path(req.output_dir) / "continuation.tar", header["binding"], chain)
    endpoint = coverage.nested_cv.PreparedEndpoint.from_json((inputs / "endpoint.json").read_text())
    parts = []
    for fold in folds:
        ids = endpoint.outer.evaluation_ids(fold)
        parts.append(OOFNuisances(spec=endpoint.data.manifest.spec, original_ids=ids, fold_ids=(fold,) * len(ids),
            seed_ids=(1103,) * len(ids), mu_a=(0.,) * len(ids), mu_d=(0.,) * len(ids),
            r_a=(1.,) * len(ids), r_d=(1.,) * len(ids), origin_weights=(1.,) * len(ids),
            lineage=replace(endpoint.data.manifest.lineage, unit_ids=ids,
                            split_hash=endpoint.outer.content_hash)).to_json())
    states[0]["parts"] = parts
    path = tmp_path / "reordered.tar"
    coverage._write_continuation(path, states, binding=header["binding"], chain=task["continuation"], task_id=task["id"])
    with pytest.raises(ContractError, match="reordered prediction"):
        coverage._read_continuation(path, header["binding"], chain)


def test_predecessor_cannot_fall_back_to_legacy_redraw(saved_coverage_chain, tmp_path, monkeypatch):
    req, task, inputs, _ = saved_coverage_chain
    task = deepcopy(task)
    task["outputs"].remove("continuation.tar")
    task.update(id="chain-1", continuation={"owner": "coverage-test", "step": 1, "predecessor": "chain-0"})
    monkeypatch.setattr(coverage, "generate_suite_a", lambda *a, **k: pytest.fail("redrew predecessor"))
    next_req = chain_request(tmp_path / "omitted-output", task, {"inputs": inputs, "chain-0": Path(req.output_dir)})
    result = coverage.run_stage(next_req)
    assert result.status == "blocked" and "must be declared" in result.message


def test_nested_checkpoint_import_is_inside_import_timer(saved_coverage_chain, tmp_path, monkeypatch):
    req, task, inputs, header = saved_coverage_chain
    predecessor = Path(req.output_dir)
    chain = {"owner": "coverage-test", "step": 1, "predecessor": "chain-0"}
    states = coverage._read_continuation(predecessor / "continuation.tar", header["binding"], chain)
    states[0]["pending"] = b"synthetic-checkpoint-for-timed-import"
    replacement = tmp_path / "timed-import.tar"
    coverage._write_continuation(replacement, states, binding=header["binding"], chain=task["continuation"], task_id=task["id"])
    replacement.replace(predecessor / "continuation.tar")
    task = deepcopy(task)
    task.update(id="chain-1", continuation=chain)
    clock = [0.]
    monkeypatch.setattr(coverage.time, "monotonic", lambda: clock[0])
    def import_checkpoint(*a, **k):
        clock[0] += 90.
        return object()
    monkeypatch.setattr(coverage.nested_cv, "import_continuation", import_checkpoint)
    def finish(draw, *a, continuation, **k):
        continuation.update(predecessor=None, pending=None)
        return {"draw": draw, "status": "success", "truth": 0., "causal_truth": 0.,
                "truth_integration_error": 0., "estimates": {m: {"value": 0., "se": 1.} for m in coverage.METHODS}}
    monkeypatch.setattr(coverage, "execute_draw", finish)
    next_req = chain_request(tmp_path / "timed-import", task, {"inputs": inputs, "chain-0": predecessor})
    result = coverage.run_stage(next_req)
    assert result.status == "pass", result.message
    leaf = coverage.read_json(Path(next_req.output_dir) / "timing.json")["leaves"][-1]
    assert leaf["import_seconds"] == 90.
    assert leaf["publication_seconds"] == 0.
    assert leaf["wall_seconds"] == 90.


def test_profile_continuation_refuses_changed_wall_budget(tmp_path, monkeypatch):
    from test_campaign import STAMPS
    monkeypatch.setattr(campaign, "fingerprint", lambda: deepcopy(STAMPS))
    task, inputs = chain_fixture(tmp_path)
    task["parameters"].update(mode="profile", wall_seconds=600)
    task["parameters"]["recipe"]["nested_cv"] = {}
    task["parameters"].pop("study")
    monkeypatch.setattr(coverage, "estimate_repetition", lambda *a, **k: None)
    first = chain_request(tmp_path / "first", task, {"inputs": inputs})
    result = coverage.run_stage(first)
    assert result.status == "pass", result.message
    task = deepcopy(task)
    task.update(id="chain-1", continuation={"owner": "coverage-test", "step": 1, "predecessor": "chain-0"})
    task["parameters"]["wall_seconds"] = 14400
    next_req = chain_request(tmp_path / "larger-budget", task, {"inputs": inputs, "chain-0": Path(first.output_dir)})
    result = coverage.run_stage(next_req)
    assert result.status == "blocked", result.message
    assert "binding mismatch" in result.message
    assert not (Path(next_req.output_dir) / "continuation.tar").exists()
