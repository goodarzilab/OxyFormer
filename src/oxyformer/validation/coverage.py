"""Independent complete Suite A repetitions and conservative coverage accounting.

A batch is an execution container, never a statistical replication. Numerical
failures remain in all unconditional denominators. A completed leaf publishes
its records even when they fail scientific gates; only collection certifies a
whole, prospectively declared campaign. Smoke output never certifies coverage.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import beta, norm
import yaml

from oxyformer.contracts import OOFNuisances, StageRequest, StageResult, source_lineage_hash
from oxyformer.data.loaders import LoadedData
from oxyformer.execution.paths import atomic_json, atomic_write, output_path
from oxyformer.provenance import ArtifactLineage, ArtifactRecord, ContractError, canonical_json, file_hash, require
from oxyformer.training import nested_cv
from oxyformer.validation.generators import generate_suite_a
from oxyformer.validation.scm import CovariateFrame, SCMConfig

MIN_REPETITIONS = 1000
STATUSES = {"success", "numerical_failure", "eligibility_rejection", "incomplete"}
METHODS = ("one_step", "cv_tmle")
Z95 = float(norm.ppf(.975))
RETRY_RULES = {"numerical_failure": "retain_no_retry", "eligibility_rejection": "retain_no_retry",
               "incomplete": "retain_no_replacement", "max_attempts": 1}


def digest(value):
    return sha256(canonical_json(value).encode()).hexdigest()


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid {name}")
    return value


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), f"nonfinite or invalid {name}")
    return value


def binomial_bound(events, total, *, side):
    """Exact one-sided 95% Clopper-Pearson bound, including empty samples."""
    integer(total, "total")
    integer(events, "events")
    require(events <= total and side in ("lower", "upper"), "invalid binomial counts or side")
    if side == "lower":
        return 0. if events == 0 else float(beta.ppf(.05, events, total - events + 1))
    return 1. if events == total else float(beta.ppf(.95, events + 1, total - events))


def repetition_plan(namespace, scenario, count):
    integer(count, "repetition count", 1)
    require(isinstance(namespace, str) and bool(namespace), "seed namespace required")
    require(isinstance(scenario, str) and bool(scenario), "scenario required")
    rows = []
    for index in range(count):
        identity = digest([namespace, scenario, index])
        rows.append({"repetition_id": identity, "seed": int(identity[:16], 16),
                     "scenario": scenario, "index": index})
    validate_draws(rows)
    return rows


def validate_draws(rows):
    require(isinstance(rows, list) and bool(rows), "declared repetitions required")
    ids, seeds = [], []
    for row in rows:
        require(type(row) is dict and set(row) == {"repetition_id", "seed", "scenario", "index"},
                "invalid draw declaration")
        require(isinstance(row["repetition_id"], str) and len(row["repetition_id"]) == 64,
                "invalid repetition ID")
        integer(row["seed"], "simulation seed")
        require(row["seed"] < 2**64, "simulation seed exceeds generator interface")
        integer(row["index"], "repetition index")
        require(isinstance(row["scenario"], str) and bool(row["scenario"]), "scenario required")
        ids.append(row["repetition_id"])
        seeds.append(row["seed"])
    require(len(set(ids)) == len(ids), "duplicate repetition IDs")
    require(len(set(seeds)) == len(seeds), "duplicate simulation seeds")


def summarize(records, declared, *, production_equivalent, null_scenario, expected_rejection=False):
    """Never infer the denominator from the successful subset.

    Missing/incomplete draws forbid passage. Failed/rejected draws count as
    noncoverage and, conservatively, potential null rejections. Bias and SE
    summaries are explicitly conditional on an available estimate.
    """
    validate_draws(declared)
    expected = {r["repetition_id"]: r for r in declared}
    observed = {}
    for record in records:
        identity = record["draw"]["repetition_id"]
        require(identity not in observed, "duplicate repetition result")
        require(identity in expected and record["draw"] == expected[identity], "undeclared or changed repetition")
        require(record["status"] in STATUSES, "unknown repetition status")
        if record["status"] == "success":
            finite(record["truth"], "observed-law truth")
            finite(record["causal_truth"], "structural truth")
            error = finite(record["truth_integration_error"], "truth-integration error")
            require(error >= 0, "negative truth-integration error")
            require(set(record["estimates"]) == set(METHODS), "incomplete estimator results")
            for estimate in record["estimates"].values():
                finite(estimate["value"], "estimate")
                require(finite(estimate["se"], "standard error") >= 0, "negative standard error")
            if null_scenario:
                require(abs(record["truth"]) <= error, "null scenario has nonzero observed-law truth")
        observed[identity] = record
    n = len(declared)
    counts = {status: sum(r["status"] == status for r in records) for status in sorted(STATUSES)}
    missing = n - len(records)
    complete = missing == 0 and counts["incomplete"] == 0
    failures = counts["numerical_failure"]
    failure_upper = binomial_bound(failures, n, side="upper")
    successes = [r for r in records if r["status"] == "success"]
    summaries = {}
    for method in METHODS:
        values = np.asarray([r["estimates"][method]["value"] for r in successes], dtype=float)
        errors = np.asarray([r["estimates"][method]["value"] - r["truth"] for r in successes], dtype=float)
        ses = np.asarray([r["estimates"][method]["se"] for r in successes], dtype=float)
        # Whole truth-integration interval must be covered; ambiguous endpoints
        # count against coverage. Integration diagnostics are not rigorous CIs.
        covered = sum(abs(r["estimates"][method]["value"] - r["truth"]) + r["truth_integration_error"]
                      <= Z95 * r["estimates"][method]["se"] for r in successes)
        rejected = sum(abs(r["estimates"][method]["value"]) > Z95 * r["estimates"][method]["se"]
                       for r in successes)
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
        bias = float(np.mean(errors)) if len(errors) else None
        mean_se = float(np.mean(ses)) if len(ses) else None
        ratio = abs(bias) / sd if sd is not None and sd > 0 else None
        se_ratio = mean_se / sd if sd is not None and sd > 0 else None
        lower = binomial_bound(int(covered), n, side="lower")
        upper = binomial_bound(int(rejected) + n - len(successes), n, side="upper") if null_scenario else None
        summaries[method] = {"coverage_count": int(covered), "coverage_lower": lower,
            "null_rejection_upper": upper, "bias": bias, "empirical_sd": sd,
            "abs_bias_over_sd": ratio, "mean_se": mean_se, "mean_se_over_sd": se_ratio,
            "bias_target_met": ratio is not None and ratio <= .10,
            "bias_investigation_required": ratio is None or ratio > .20,
            "pass": lower >= .925 and (upper is None or upper <= .075)
                and ratio is not None and ratio <= .20 and se_ratio is not None and .90 <= se_ratio <= 1.10}
    passing = bool(production_equivalent and n >= MIN_REPETITIONS and complete and failure_upper <= .01
                   and not expected_rejection and all(v["pass"] for v in summaries.values()))
    return {"declared_repetitions": n, "received_repetitions": len(records), "missing_repetitions": missing,
        "counts": counts, "complete": complete, "production_equivalent": bool(production_equivalent),
        "numerical_failure_upper": failure_upper, "methods": summaries, "coverage_pass": passing,
        "expected_eligibility_rejection": bool(expected_rejection),
        "eligibility_test_pass": bool(expected_rejection and complete and counts["eligibility_rejection"] == n),
        "conditional_summary_count": len(successes),
        "truth_integration_error_max": max((r["truth_integration_error"] for r in successes), default=None),
        "summary_interpretation": "Bias/SD and SE/SD use successful estimates only; gate bounds retain all declared draws."}


def dependency(request, config, reference):
    return nested_cv._dependency(request, config, reference)


def read_json(path):
    return json.loads(Path(path).read_text())


def publish(request, values, *, status, message, extra_artifacts=()):
    root = Path(request.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    lineage = ArtifactLineage(source_hashes=(request.config_hash, request.task_hash),
        unit_ids=(request.content_hash,), parent_hashes=request.dependency_hashes,
        split_hash=None, config_hash=request.config_hash, model_hash=None,
        environment=(("code_identity", request.code_identity),), seed=None, parameter_count=None)
    records = []
    for name, value in values.items():
        path = atomic_json(root, name, value)
        records.append(ArtifactRecord(path=name, kind=name.removesuffix(".json"), sha256=file_hash(path), lineage=lineage))
    # Individually published repetition records are explicitly hash-bound too.
    for path in sorted((root / "repetitions").glob("*/result.json")):
        records.append(ArtifactRecord(path=str(path.relative_to(root)), kind="repetition",
                                     sha256=file_hash(path), lineage=lineage))
    records.extend(extra_artifacts)
    if "artifact_manifest.json" not in values:
        path = atomic_json(root, "artifact_manifest.json", {"request_hash": request.content_hash,
            "artifacts": [r.to_dict() for r in records]})
        records.append(ArtifactRecord(path="artifact_manifest.json", kind="artifact_manifest", sha256=file_hash(path), lineage=lineage))
    result = StageResult(request_hash=request.content_hash, status=status, artifacts=tuple(records), message=message)
    result.verify(request)
    return result


def bind_observations(template, observations):
    """Rebind a reviewed synthetic endpoint to one draw, preserving X/design.

    This adapter handles fixed-X, fixed-split identity endpoints. Selection or
    additional measured confounders require their own reviewed endpoint adapter;
    they cannot silently change the frozen population or predictor registry.
    """
    frame = observations.frame
    require(template.data.manifest.spec.endpoint.startswith("synthetic"), "synthetic endpoint template required")
    require(template.family == "identity", "SCM adapter requires identity likelihood")
    require(tuple(frame.original_ids) == template.outer.original_ids, "SCM frame must equal frozen inference target")
    require(not observations.measured_columns, "measured confounders require a reviewed endpoint adapter")
    require(all(y is not None for y in observations.y), "selection requires a reviewed endpoint adapter")
    data = template.data
    manifest = data.manifest
    ids = manifest.original_ids
    names = tuple(c.name for c in manifest.schema)
    values = {oid: dict(zip(names, row)) for oid, row in zip(ids, data.rows)}
    features = tuple(n for n, _ in template.feature_kinds)
    require(features == frame.columns, "SCM adjustment columns differ from production predictors")
    policy_rows = dict(zip(template.policy_covariates.original_ids,
        zip(template.policy_covariates.geography_ids, template.policy_covariates.support_keys)))
    geography = {r.original_id: r for r in template.geography.rows}
    for index, oid in enumerate(frame.original_ids):
        row = values[oid]
        require(tuple(row[n] for n in features) == frame.x[index], "fixed-X template drift")
        require(policy_rows[oid] == (frame.geography_ids[index], frame.support_keys[index]), "SCM policy routing drift")
        require((1. if manifest.weight_field is None else row[manifest.weight_field]) == frame.weights[index], "SCM weight drift")
        require(geography[oid].county == frame.region_ids[index], "SCM region/county mismatch")
        require((geography[oid].latitude, geography[oid].longitude) == frame.coordinates[index], "SCM geography drift")
        row[manifest.exposure_field], row[manifest.outcome_field] = observations.a[index], observations.y[index]
    source = replace(manifest.sources[0], payload_hash=observations.content_hash,
                     source_id="synthetic-repetition", uri="synthetic://independent-repetition")
    spec = replace(manifest.spec, source_lineage_hash=source_lineage_hash((source,)))
    lineage = replace(manifest.lineage, source_hashes=(observations.content_hash,), parent_hashes=(observations.content_hash,))
    manifest = replace(manifest, sources=(source,), spec=spec, lineage=lineage)
    data = LoadedData(manifest=manifest, rows=tuple(tuple(values[oid][n] for n in names) for oid in ids))
    outer = replace(template.outer, spec=spec, lineage=replace(lineage, parent_hashes=(manifest.content_hash,)))
    inners = []
    for inner in template.inner:
        local_lineage = replace(lineage, unit_ids=inner.data_manifest.original_ids, parent_hashes=(manifest.content_hash,))
        local = replace(inner.data_manifest, spec=spec, sources=(source,), lineage=local_lineage)
        split = replace(inner.split, spec=spec, lineage=replace(local_lineage, parent_hashes=(local.content_hash,)))
        inners.append(replace(inner, data_manifest=local, split=split))
    return replace(template, data=data, outer=outer, inner=tuple(inners),
                   geography=replace(template.geography, data_manifest_hash=manifest.content_hash))


def estimate_repetition(endpoint, recipe, root, deadline):
    """All five folds, three neural seeds, the merged tuning and calibration."""
    from oxyformer.estimation.mtp import one_step
    from oxyformer.estimation.targeting import cv_tmle
    from oxyformer.estimation.covariance import align_estimates, spatial_sensitivities
    from oxyformer.training.fit import subset

    options = nested_cv._settings(recipe, endpoint)
    parts = []
    for fold in range(5):
        for neural_seed in nested_cv.SEEDS:
            remaining = min(14400., deadline - time.monotonic())
            if remaining <= 0:
                return None
            config = endpoint.configuration(fold, root / f"fold-{fold}-seed-{neural_seed}",
                **options, slice_seconds=remaining, checkpoint_margin_seconds=min(1., remaining / 10))
            artifact = nested_cv.run_fold(config, endpoint.outer, neural_seed, geography=endpoint.geography)
            if not artifact.complete:
                return None
            view = subset(endpoint.data.covariates(tuple(n for n, _ in endpoint.feature_kinds)),
                          artifact.prediction_inputs.original_ids)
            parts.append(nested_cv.predict(artifact, view, endpoint.policy))
    names = ("original_ids", "fold_ids", "seed_ids", "mu_a", "mu_d", "r_a", "r_d", "origin_weights")
    values = {name: tuple(v for part in parts for v in getattr(part, name)) for name in names}
    lineage = replace(parts[0].lineage, unit_ids=endpoint.outer.original_ids,
        split_hash=endpoint.outer.content_hash, parent_hashes=(endpoint.data.manifest.content_hash,
        *(part.content_hash for part in parts)), model_hash=None, seed=None, parameter_count=None)
    nuisances = OOFNuisances(spec=endpoint.data.manifest.spec, lineage=lineage, **values)
    weights = dict(zip(endpoint.data.manifest.original_ids,
        (1.,) * len(endpoint.data.rows) if endpoint.data.manifest.weight_field is None
        else endpoint.data.column(endpoint.data.manifest.weight_field)))
    weights = {oid: weights[oid] for oid in endpoint.outer.original_ids}
    common = dict(split=endpoint.outer, policy=endpoint.policy)
    estimates = {"one_step": one_step(nuisances, endpoint.data, weights, nuisances.spec, **common),
                 "cv_tmle": cv_tmle(nuisances, endpoint.data, weights, "identity", nuisances.spec, **common).estimate}
    by_id = {r.original_id: r for r in endpoint.geography.rows}
    groups = {oid: by_id[oid].county for oid in endpoint.outer.original_ids}
    # Representatives are frozen in the recipe, never selected from outcomes.
    locations = {k: tuple(v) for k, v in recipe["inference"]["county_locations"].items()}
    covariances = spatial_sensitivities(align_estimates(estimates), groups, locations, interpretation="geographic_process")
    primary = recipe["inference"]["primary_bandwidth_km"]
    require(primary in covariances, "unregistered primary spatial bandwidth")
    result = {}
    for index, (name, estimate) in enumerate(estimates.items()):
        result[name] = {"value": estimate.value, "se": covariances[primary].standard_errors[index],
            "spatial_se": {str(b): cov.standard_errors[index] for b, cov in covariances.items()}}
    return result


def execute_draw(draw, frame, scenario, template, recipe, root, deadline):
    started = time.monotonic()
    record = {"draw": draw, "status": "incomplete", "reason": "budget exhausted", "estimates": {}}
    if started >= deadline:
        return {**record, "wall_seconds": 0.}
    try:
        sample = generate_suite_a(frame, scenario, template.policy, seed=draw["seed"])
        if sample.observed_law_truth.status in ("design_rejected", "empty_target"):
            record.update(status="eligibility_rejection", reason=sample.observed_law_truth.reason)
        else:
            endpoint = bind_observations(template, sample.observations)
            estimates = estimate_repetition(endpoint, recipe, root, deadline)
            if estimates is not None:
                uncertainty = sample.integration_uncertainty
                error = (uncertainty.observed_absolute_difference or 0.) + (uncertainty.quadrature_tail_absolute_bound or 0.) + (uncertainty.truth_serialization_absolute_bound or 0.)
                record.update(status="success", reason="complete production procedure", estimates=estimates,
                    truth=sample.observed_law_truth.value, causal_truth=sample.structural_causal_truth.value,
                    truth_integration_error=error, integration_uncertainty=uncertainty.to_dict(),
                    observation_hash=sample.observations.content_hash)
            atomic_write(root, "observations.json", sample.observations.to_json())
            atomic_write(root, "observed_law_truth.json", sample.observed_law_truth.to_json())
            atomic_write(root, "structural_causal_truth.json", sample.structural_causal_truth.to_json())
    except (ArithmeticError, np.linalg.LinAlgError, ValueError) as exc:
        # No draw is retried and no exception is silently relabeled as success.
        record.update(status="numerical_failure", reason=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        # Adapter, I/O and unexpected execution faults are not estimates of
        # numerical failure probability. Retain the draw as incomplete so it
        # cannot vanish or certify coverage. Process cancellation still escapes.
        record.update(status="incomplete", reason=f"execution failure: {type(exc).__name__}: {exc}")
    record["wall_seconds"] = time.monotonic() - started
    return record


def _run_batch(request: StageRequest, started: float) -> StageResult:
    from oxyformer.validation.campaign import load_lock, validate_recipe, validate_leaf_task, fingerprint
    try:
        request.verify_inputs()
        require(request.stage in ("coverage", "simulation-smoke"), "unsupported coverage stage")
        config, task = yaml.safe_load(Path(request.config_path).read_text()), read_json(request.task_path)
        require(task["stage"] == request.stage, "stage mismatch")
        parameters = task["parameters"]
        mode = parameters["mode"]
        require(mode in ("smoke", "profile", "screening", "final"), "unknown repetition mode")
        require((request.stage == "simulation-smoke") == (mode in ("smoke", "profile")), "stage/mode mismatch")
        if mode in ("screening", "final"):
            lock = load_lock(request, config, task)
            validate_leaf_task(request, config, task, lock)
            recipe = lock["recipe"]
            expected_task = next((t for t in lock["batches"] if t["batch_id"] == parameters["batch_id"]), None)
            require(expected_task is not None and expected_task["mode"] == mode, "undeclared batch")
            require(parameters["draws"] == expected_task["draws"], "batch draw drift")
            require(parameters["scenario"] == expected_task["scenario"], "batch scenario drift")
            require(parameters["wall_seconds"] == expected_task["wall_seconds"], "batch budget drift")
        else:
            recipe = parameters["recipe"]
            lock = None
        production = mode != "smoke"
        validate_recipe(recipe, production=production)
        template_path = dependency(request, config, parameters["endpoint_input"])
        frame_path = dependency(request, config, parameters["frame_input"])
        template = nested_cv.PreparedEndpoint.from_json(template_path.read_text())
        frame = CovariateFrame.from_json(frame_path.read_text())
        require(recipe["endpoint_hash"] == template.content_hash and recipe["frame_hash"] == frame.content_hash,
                "recipe input drift")
        scenario = SCMConfig(**parameters["scenario"])
        if lock:
            require(parameters["scenario"] in lock["scenario_family"], "scenario family drift")
        draws = parameters["draws"]
        validate_draws(draws)
        require(all(d["scenario"] == scenario.name for d in draws), "draw/scenario mismatch")
        seconds = finite(parameters["wall_seconds"], "batch wall seconds")
        require(seconds > 0, "positive batch budget required")
        root = Path(request.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        records, repetition_seconds = [], []
        for draw in draws:
            draw_started = time.monotonic()
            destination = output_path(root, "repetitions/" + draw["repetition_id"])
            require(not destination.exists(), "repetition was already attempted; use declared continuation, never redraw")
            destination.mkdir(parents=True)
            record = execute_draw(draw, frame, scenario, template, recipe, destination, started + seconds)
            atomic_json(destination, "result.json", record)
            repetition_seconds.append(time.monotonic() - draw_started)
            records.append(record)
        summary = summarize(records, draws, production_equivalent=production,
            null_scenario=scenario.effect == "null", expected_rejection=scenario.assignment == "atoms")
        # A leaf reports completed execution and its honest failures. Campaign
        # coverage is evaluated only after the full declared fan-in is present.
        result = {"schema_version": 1, "mode": mode, "recipe_hash": digest(recipe),
            "lock_hash": digest(lock) if lock else None, "batch_id": parameters.get("batch_id"),
            "scenario": scenario.to_dict()["payload"], "draws": draws, "records": records, "summary": summary,
            "certifies_production_coverage": False}
        # Locked execution already verified this fingerprint in load_lock.
        # Repeating the full checkout/environment scan would add an unprofiled
        # second scan to every locked leaf.
        stamps = ({k: lock[k] for k in ("scientific_fingerprint", "environment_hash")}
                  if lock else fingerprint())
        timing = {"wall_seconds": time.monotonic() - started, "gpu_seconds": 0., "device": "cpu",
            "measurement_scope": "before_final_publication",
            "production_equivalent": production, "recipe_hash": digest(recipe),
            "complete_repetition_seconds": [seconds for r, seconds in zip(records, repetition_seconds) if r["status"] == "success"],
            "complete": summary["complete"], "all_successful": summary["counts"]["success"] == len(draws), **stamps}
        return publish(request, {"result.json": result, "timing.json": timing},
            status="pass" if summary["complete"] else "fail",
            message="Declared repetitions accounted for; production coverage requires complete campaign collection"
                if summary["complete"] else "Incomplete batch; all declared draws retained")
    except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
        return StageResult(request_hash=request.content_hash, status="blocked", artifacts=(), message=str(exc) or type(exc).__name__)



def run_stage(request: StageRequest) -> StageResult:
    """Time profiling from outside the normal batch handler's complete return.

    A timing file cannot measure its own final publication. A profile therefore
    executes the ordinary batch handler in an owned child directory, waits for
    its complete artifact publication and verification, then publishes the
    measured receipt at the requested root. The profiler's administrative
    publication is not work performed by ordinary screening/final leaves.
    """
    started = time.monotonic()
    try:
        request.verify_inputs()
        task = read_json(request.task_path)
        if task.get("parameters", {}).get("mode") != "profile":
            return _run_batch(request, started)
        root = Path(request.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        destination = output_path(root, "_profiled")
        require(not destination.exists(), "profile already attempted; never rerun failed draws")
        destination.mkdir()
        profiled_request = replace(request, output_dir=str(destination))
        completed = _run_batch(profiled_request, started)
        if not completed.artifacts:
            return StageResult(request_hash=request.content_hash, status=completed.status,
                               artifacts=(), message=completed.message)
        completed.verify(profiled_request)
        elapsed = time.monotonic() - started
        result = read_json(destination / "result.json")
        timing = read_json(destination / "timing.json")
        timing.update(wall_seconds=elapsed, measurement_scope="complete_stage_return",
                      profiled_request_hash=profiled_request.content_hash)
        result["profiled_request_hash"] = profiled_request.content_hash
        extra = tuple(replace(a, path="_profiled/" + a.path) for a in completed.artifacts)
        return publish(request, {"result.json": result, "timing.json": timing,
            "profile_receipt.json": {"request": profiled_request.to_dict(), "result": completed.to_dict()}},
            status=completed.status, message="Production profile measured through complete stage return; " + completed.message,
            extra_artifacts=extra)
    except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
        return StageResult(request_hash=request.content_hash, status="blocked", artifacts=(),
                           message=str(exc) or type(exc).__name__)
