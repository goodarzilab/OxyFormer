"""Prospective campaign locks and complete fan-in; this module submits no jobs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time

import yaml
import torch

from oxyformer.training.fit import checkpoint_margin_seconds, fit_environment, resolve_device
from oxyformer.training.nested_cv import PreparedEndpoint

from oxyformer.contracts import StageRequest, StageResult
from oxyformer.execution import identity
from oxyformer.execution.campaign import expand_campaign, validate_plan, concrete_id
from oxyformer.provenance import ContractError, file_hash, require
from oxyformer.validation.coverage import (MIN_REPETITIONS, RETRY_RULES, dependency, digest,
    finite, integer, publish, read_json, repetition_plan, summarize, validate_draws, prepare_batch, atomic_json, fixed_exposures,
    leaf_budget, leaf_allocation, maximum_leaf_stage_seconds, remaining_compute, load_budget, BUDGET_RECORD, usable_leaf_compute, RESUME_OVERHEAD_FLOOR_SECONDS)
from oxyformer.validation.scm import SCMConfig, CovariateFrame, sampled_support_facts

REGISTRY = Path(__file__).parents[3] / "configs/validation/final_scenarios.yaml"
REPOSITORY = Path(__file__).parents[3]
LEAF_OUTPUTS = [BUDGET_RECORD, "result.json", "timing.json", "artifact_manifest.json", "runtime.json"]
COLLECT_OUTPUTS = ["result.json", "gate.json", "artifact_manifest.json"]


SCREENING_LABEL = "screening; final 1,000-repetition cells deferred (owner decision 2026-10-12)"
SCREENING_CELL = {"reference_cap": 32, "frozen_epochs": 30, "grid_points": 2, "seed_count": 1}
OWNER_APPROVALS = REPOSITORY / "configs/approvals.yaml"
_REPOSITORY_APPROVALS = object()


def screening_decision(approvals=_REPOSITORY_APPROVALS):
    """Read, never amend, the owner decision. Absence is not an opt-in."""
    if approvals is _REPOSITORY_APPROVALS:
        approvals = yaml.safe_load(OWNER_APPROVALS.read_text())
    require(type(approvals) is dict, "missing screening campaign approval")
    owner = approvals.get("owner_decisions")
    require(type(owner) is dict, "missing screening campaign approval")
    decision = owner.get("screening_campaign")
    expected = {"scenarios", "repetitions_per_scenario", "recipe", "budget_h100_hours",
                "final_cells", "gpu_concurrency_max", "gpu_concurrency_exempt_partitions", "month1_sizing"}
    require(type(decision) is dict and set(decision) == expected, "missing or malformed screening campaign approval")
    require(decision["scenarios"] == [s["name"] for s in _registry()["scenarios"]],
            "screening approval must name the four registered scenarios")
    for name, value in (("repetitions_per_scenario", 25), ("budget_h100_hours", 3600), ("gpu_concurrency_max", 4)):
        require(type(decision[name]) is int and decision[name] == value, "malformed screening approval: " + name)
    require(type(decision["recipe"]) is dict and decision["recipe"] == SCREENING_CELL
            and all(type(v) is int for v in decision["recipe"].values()), "malformed screening study cell")
    require(decision["gpu_concurrency_exempt_partitions"] == ["preemptible"], "malformed screening partition exemption")
    require(decision["final_cells"] == "the 1,000-repetition final cells (minimum_final_repetitions) are deferred to month 2 with a separate allocation; this screening campaign certifies no release criterion",
            "malformed screening final-cell deferral")
    sizing = decision["month1_sizing"]
    expected_sizing = {
        "leaf_bound_scope": "per_scenario", "leaves_per_scenario_max": 40,
        "gpu_hours_per_leaf_max": 4, "repetitions_per_scenario_max": 25,
        "sizing_rule": "repetitions per scenario = the largest count, at most 25, whose measured cost fits 40 leaves of at most 4 GPU-hours per scenario; the count is fixed from the measured screening-cell profiles and reported to the owner before campaign-lock admits any leaf",
        "continuation": "a repetition may span several leaves through a coverage-owned continuation; every leaf stays at or under 4 GPU-hours",
        "deferred": "more repetitions, any further change to the leaf bounds, and use of the preemptible partition are separate owner decisions taken with measured numbers",
    }
    require(type(sizing) is dict and sizing == expected_sizing
            and all(type(sizing[k]) is type(v) for k, v in expected_sizing.items()),
            "missing or malformed screening month1_sizing")
    return deepcopy(decision)


def screening_safety_factor(approvals=_REPOSITORY_APPROVALS):
    """Return a JSON-safe copy of the owner sizing decision; never amend it."""
    name = "owner_decisions.screening_month1_safety_factor"
    if approvals is _REPOSITORY_APPROVALS:
        try:
            approvals = yaml.safe_load(OWNER_APPROVALS.read_text())
        except (OSError, yaml.YAMLError, ValueError) as exc:
            raise ContractError("missing or malformed " + name) from exc
    require(type(approvals) is dict, "missing " + name)
    owner = approvals.get("owner_decisions")
    require(type(owner) is dict, "missing " + name)
    decision = owner.get("screening_month1_safety_factor")
    expected = {"profile_safety_factor", "decided_on", "applies_to", "basis", "accepted_risk", "deferred"}
    require(type(decision) is dict and set(decision) == expected, "missing or malformed " + name)
    factor = decision["profile_safety_factor"]
    require(type(factor) in (int, float) and 1. <= factor <= 2. and math.isfinite(factor),
            "invalid " + name + ".profile_safety_factor")
    require(decision["applies_to"] == ["campaign-estimate", "campaign-lock"],
            "invalid " + name + ".applies_to")
    for field in ("basis", "accepted_risk", "deferred"):
        require(type(decision[field]) is str and bool(decision[field].strip()), "invalid " + name + "." + field)
    decided_on = decision["decided_on"]
    if type(decided_on) is str:
        try:
            # fromisoformat handles calendar/week dates, but not ordinal dates.
            if re.fullmatch(r"[0-9]{4}-?[0-9]{3}", decided_on):
                year, day = int(decided_on[:4]), int(decided_on[-3:])
                decided_on = date.fromordinal(date(year, 1, 1).toordinal() + day - 1)
                require(decided_on.year == year, "invalid " + name + ".decided_on")
            else:
                decided_on = date.fromisoformat(decided_on)
        except ValueError as exc:
            raise ContractError("invalid " + name + ".decided_on") from exc
    require(type(decided_on) is date, "invalid " + name + ".decided_on")
    return {**deepcopy(decision), "profile_safety_factor": float(factor), "decided_on": decided_on.isoformat()}


def month2_decision(approvals=_REPOSITORY_APPROVALS):
    """Read the owner-originated month-2 decision, normalized for JSON."""
    name = "owner_decisions.month2_followon"
    if approvals is _REPOSITORY_APPROVALS:
        try:
            approvals = yaml.safe_load(OWNER_APPROVALS.read_text())
        except (OSError, yaml.YAMLError, ValueError) as exc:
            raise ContractError("missing or malformed " + name) from exc
    require(type(approvals) is dict, "missing " + name)
    owner = approvals.get("owner_decisions")
    require(type(owner) is dict, "missing " + name)
    d = owner.get("month2_followon")
    keys = {"decided_on", "partitions", "gpu_concurrency_max", "gpu_hours_ceiling",
            "diagnostic_campaign", "recipe_cost_study", "final_cells"}
    require(type(d) is dict, "missing " + name)
    require(set(d) == keys, name + " keys: " + str(sorted(set(d) ^ keys)))
    d = deepcopy(d)
    def check(ok, key):
        require(ok, "invalid " + name + "." + key)
    check(d["partitions"] == ["standard", "preemptible"], "partitions")
    concurrency = d["gpu_concurrency_max"]
    check(type(concurrency) is dict and set(concurrency) == set(d["partitions"]), "gpu_concurrency_max")
    for partition, value in concurrency.items():
        check(type(value) is int and value > 0, "gpu_concurrency_max." + partition)
    check(d["gpu_hours_ceiling"] is None or d["gpu_hours_ceiling"] == "none", "gpu_hours_ceiling")
    d["gpu_hours_ceiling"] = None
    c = d["diagnostic_campaign"]
    ckeys = {"scenarios", "repetitions", "recipe", "partition", "gpu_hours_per_leaf_max",
             "leaves_per_chain_max", "parallel_chains", "draws_continue_from", "certifies_release_criteria", "purpose"}
    check(type(c) is dict, "diagnostic_campaign")
    check(set(c) == ckeys, "diagnostic_campaign keys: " + str(sorted(set(c) ^ ckeys)))
    screening = screening_decision(approvals)
    names = c["scenarios"]
    check(type(names) is list and bool(names) and all(type(n) is str for n in names)
          and len(set(names)) == len(names) and set(names) <= set(screening["scenarios"]), "diagnostic_campaign.scenarios")
    check(type(c["repetitions"]) is int and c["repetitions"] > 0, "diagnostic_campaign.repetitions")
    check(type(c["recipe"]) is dict and c["recipe"] == SCREENING_CELL
          and all(type(v) is int for v in c["recipe"].values()), "diagnostic_campaign.recipe")
    check(c["partition"] in d["partitions"], "diagnostic_campaign.partition")
    check(concurrency[c["partition"]] <= 16, "gpu_concurrency_max." + c["partition"])
    for key, ceiling in (("gpu_hours_per_leaf_max", 4), ("leaves_per_chain_max", 40)):
        check(type(c[key]) in (int, float) and math.isfinite(c[key]) and 0 < c[key] <= ceiling,
              "diagnostic_campaign." + key)
    check(c["parallel_chains"] is True, "diagnostic_campaign.parallel_chains")
    check(c["certifies_release_criteria"] is False, "diagnostic_campaign.certifies_release_criteria")
    for block, key, prefix in ((c, "draws_continue_from", "diagnostic_campaign."),
                              (c, "purpose", "diagnostic_campaign."),
                              (d, "recipe_cost_study", ""), (d, "final_cells", "")):
        check(type(block[key]) is str and bool(block[key].strip()), prefix + key)
    value = d["decided_on"]
    if type(value) is str:
        try:
            value = date.fromisoformat(value)
        except ValueError as exc:
            raise ContractError("invalid " + name + ".decided_on") from exc
    check(type(value) is date, "decided_on")
    d["decided_on"] = value.isoformat()
    return d


def validate_diagnostic(recipe, approvals):
    decision = month2_decision(approvals)
    repository = month2_decision()
    for key in decision:
        require(decision[key] == repository[key], "month2_followon drift: " + key)
    require(recipe.get("study") == decision["diagnostic_campaign"]["recipe"], "diagnostic study drift")
    return decision


def diagnostic_sizing(profiles, decision, wall, factor=2.0):
    """Pure offline arithmetic; never grants admission or changes the count."""
    c = decision["diagnostic_campaign"]
    require(set(profiles) == set(c["scenarios"]), "diagnostic profiles drift")
    hours = leaf_allocation(wall)["allocation_seconds"] / 3600
    require(hours <= c["gpu_hours_per_leaf_max"],
            f"diagnostic leaf requires {hours} GPU-hours > {c['gpu_hours_per_leaf_max']}")
    rows = {}
    for name in c["scenarios"]:
        leaves = slice_count(profiles[name], 1, wall, factor, enforce_limit=False)
        require(leaves <= c["leaves_per_chain_max"],
                f"diagnostic {name} chain requires {leaves} leaves > {c['leaves_per_chain_max']}")
        n = c["repetitions"]
        rows[name] = {"repetitions": n, "chains": n, "leaves": n * leaves,
                      "gpu_hours": n * leaves * hours,
                      "per_chain": {"repetitions": 1, "leaves": leaves, "gpu_hours": leaves * hours}}
    return rows


def diagnostic_estimate(request, config, parameters):
    recipe = parameters["recipe"]
    validate_recipe(recipe)
    decision = validate_diagnostic(recipe, config.get("approvals"))
    c = decision["diagnostic_campaign"]
    for key, value in (("study", c["recipe"]), ("certifies_release_criteria", False),
                       ("partition", c["partition"]), ("gpus", 1)):
        require(parameters.get(key) == value and type(parameters.get(key)) is type(value),
                "diagnostic " + key + " drift")
    require(type(parameters.get("campaign_label")) is str and "diagnostic" in parameters["campaign_label"],
            "diagnostic campaign_label drift")
    require(set(parameters["profiles"]) == set(c["scenarios"]), "diagnostic profiles drift")
    for key in ("repetitions", "screening_repetitions", "final_repetitions", "repetitions_per_leaf"):
        require(key not in parameters, "diagnostic repetitions come from month2_followon: " + key)
    require(resolve_device().type == "cuda", "diagnostic measurement requires CUDA")
    require(integer(parameters["cpus_per_task"], "CPUs", 1) == 8, "diagnostic requires eight CPUs")
    wall = integer(parameters["wall_seconds"], "leaf wall seconds", 1)
    factor = finite(parameters.get("profile_safety_factor", 2.0), "profile safety factor")
    require(factor >= 1, "diagnostic profile_safety_factor")
    records = []
    for key, cls, expected in (("endpoint_input", PreparedEndpoint, recipe["endpoint_hash"]),
                              ("frame_input", CovariateFrame, recipe["frame_hash"])):
        record = cls.from_json(dependency(request, config, parameters[key]).read_text())
        require(record.content_hash == expected, "diagnostic " + key + " hash mismatch")
        records.append(record)
    registered = {s["name"]: s for s in registered_scenarios(*records)}
    scenarios = [registered[name] for name in c["scenarios"]]
    stamps = fingerprint()
    profiles = {s["name"]: _profile(request, config, parameters["profiles"][s["name"]], s, recipe, stamps)
                for s in scenarios}
    from oxyformer.validation.campaign_probe import calibrate, profile_setup_sources
    for scenario in scenarios:
        name = scenario["name"]
        refs = parameters.get("probe_profiles", {}).get(name)
        profiles[name]["setup_profile_sources"] = (profile_setup_sources(request, config, refs,
            scenario, profiles[name], stamps) if refs is not None else {})
    profiles, calibration = calibrate(request, config, parameters, profiles)
    rows = diagnostic_sizing(profiles, decision, wall, factor)
    return {"schema_version": 1, **screening_metadata(recipe), "estimate_only": True,
        "admitted": False, "submission_performed": False, "certifies_production_coverage": False,
        "recipe": deepcopy(recipe), "recipe_hash": digest(recipe), "owner_decision": "month2_followon",
        "owner_decision_hash": digest(decision), "scenario_family": scenarios, **stamps,
        "profiles": profiles, "calibration": calibration, "per_scenario": rows,
        "total_repetitions": sum(r["repetitions"] for r in rows.values()),
        "projected_leaf_count": sum(r["leaves"] for r in rows.values()),
        "gpu_hours": math.fsum(r["gpu_hours"] for r in rows.values()),
        "partition": c["partition"], "gpus": 1, "cpus_per_task": 8, "wall_seconds": wall,
        "owner_safety_factor_decision": None, "profile_safety_factor": factor,
        "gpu_concurrency_max": decision["gpu_concurrency_max"][c["partition"]], "budget_h100_hours": None}


def approved_safety_factor(parameters, approvals):
    """Refuse a stale manifest instead of silently resizing it."""
    decision = screening_safety_factor(approvals)
    factor = finite(parameters.get("profile_safety_factor"), "profile safety factor")
    require(factor == decision["profile_safety_factor"],
            "profile_safety_factor drift from owner_decisions.screening_month1_safety_factor")
    return decision


def screening_metadata(recipe):
    if recipe.get("campaign_mode") == "diagnostic":
        d = month2_decision()
        return {"campaign_mode": "diagnostic", "study": deepcopy(d["diagnostic_campaign"]["recipe"]),
                "campaign_label": "diagnostic; no release criterion certified",
                "certifies_release_criteria": False, "production_equivalent": False,
                "final_cells": d["final_cells"]}
    if recipe.get("campaign_mode") != "screening":
        return {}
    return {"campaign_mode": "screening", "study": deepcopy(recipe["study"]),
            "campaign_label": SCREENING_LABEL, "certifies_release_criteria": False,
            "final_cells": screening_decision()["final_cells"]}


def validate_screening(recipe, approvals):
    decision = screening_decision(approvals)
    require(decision == screening_decision(), "dispatched screening approval differs from repository")
    require(screening_safety_factor(approvals) == screening_safety_factor(),
            "dispatched screening_month1_safety_factor approval differs from repository")
    require(recipe.get("study") == decision["recipe"], "screening study differs from approval")
    return decision


def validate_recipe(recipe, *, production=True):
    screening = recipe.get("campaign_mode") in ("screening", "diagnostic")
    extra = {"campaign_mode", "study"} if screening else set()
    require(set(recipe) == {"endpoint_hash", "frame_hash", "nested_cv", "inference"} | extra, "unknown recipe fields")
    if screening:
        require(production, "screening study cannot be a smoke recipe")
        require(recipe["study"] == screening_decision()["recipe"], "screening study differs from approval")
        require(all(type(v) is int for v in recipe["study"].values()), "malformed screening study cell")
    from oxyformer.provenance import check_hash
    check_hash(recipe["endpoint_hash"])
    check_hash(recipe["frame_hash"])
    settings = recipe["nested_cv"]
    require(set(settings) <= {"ssl_epochs", "frozen_epochs", "batch_size", "stopping_ids", "precision", "synthetic"},
            "unknown nested recipe fields")
    integer(settings.get("batch_size", 256), "batch size", 1)
    require(settings.get("precision", "fp32") == "fp32", "unregistered precision")
    if production:
        require(settings.get("ssl_epochs", 30) == 30 and settings.get("frozen_epochs", 150) == 150,
                "final/profile/screening requires production tuning and stopping")
        require(not settings.get("synthetic", False), "smoke recipe cannot certify production")
    else:
        require(settings.get("synthetic") is True, "smoke shortening must be explicit")
    inference = recipe["inference"]
    require(set(inference) == {"primary_bandwidth_km", "county_locations"}, "unknown inference fields")
    require(inference["primary_bandwidth_km"] in (50, 100, 200), "unregistered spatial bandwidth")
    require(type(inference["county_locations"]) is dict and len(inference["county_locations"]) >= 2,
            "frozen county representatives required")
    for county, location in inference["county_locations"].items():
        require(isinstance(county, str) and bool(county) and len(location) == 2, "invalid county representative")
        lat, lon = (finite(v, "coordinate") for v in location)
        require(-90 <= lat <= 90 and -180 <= lon <= 180, "coordinate outside degree bounds")

def fingerprint():
    return {"scientific_fingerprint": identity.scientific_fingerprint(REPOSITORY),
            "environment_hash": digest([identity.environment_record(), fit_environment()])}


def load_lock(request, config, task):
    reference = task["recipe_lock"]
    require(set(reference) == {"dependency", "path", "sha256"}, "invalid recipe reference")
    path = dependency(request, config, {k: reference[k] for k in ("dependency", "path")})
    require(file_hash(path) == reference["sha256"], "recipe lock hash mismatch")
    lock = read_json(path)
    require(lock["schema_version"] == 1 and lock["retry_rules"] == RETRY_RULES, "lock schema or retry drift")
    for profile in lock["budget"]["profiles"].values():
        measured = load_budget(request, config, profile["source_unit"])
        require(measured["record_hash"] == profile["budget_record_hash"]
            and measured["complete"] and measured["all_successful"], "locked profile budget drift")
    current = fingerprint()
    # A screening collector only verifies receipts and aggregates terminal
    # records. It runs on CPU; fitting leaves still require the exact measured
    # accelerator environment. Profile receipt hashes remain checked above.
    screening_collector = request.stage == "campaign-collect" and bool(screening_metadata(lock["recipe"]))
    require(all(lock[k] == v for k, v in current.items()
                if not screening_collector or k == "scientific_fingerprint"), "scientific code/environment recipe drift")
    validate_recipe(lock["recipe"])
    if lock["recipe"].get("campaign_mode") == "diagnostic":
        decision = validate_diagnostic(lock["recipe"], config.get("approvals"))
        require(lock["owner_decision_hash"] == digest(decision), "month2_followon decision hash drift")
        require(all(b["mode"] == "diagnostic" for b in lock["batches"]), "diagnostic batch mode drift")
    elif screening_metadata(lock["recipe"]):
        decision = validate_screening(lock["recipe"], config.get("approvals"))
        require(set(lock["instances"]) == set(decision["scenarios"])
                and all(b["mode"] == "screening" for b in lock["batches"]),
                "screening study cannot authorize final cells")
    require(lock["nested_cv"] == lock["recipe"]["nested_cv"], "production nested recipe drift")
    draws = [draw for batch in lock["batches"] for draw in batch["draws"]]
    validate_draws(draws)
    require(len({b["batch_id"] for b in lock["batches"]}) == len(lock["batches"]), "duplicate locked batches")
    if screening_metadata(lock["recipe"]):
        from oxyformer.validation.campaign_probe import require_admission
        require_admission(request, config, task, lock)
    return lock


def _registry():
    value = yaml.safe_load(REGISTRY.read_text())
    require(value["minimum_final_repetitions"] == MIN_REPETITIONS, "registry replication drift")
    return value


def registered_scenarios(endpoint, frame):
    """Resolve only real-frame consistency declarations from dispatched support."""
    registry = _registry()
    derived = {}
    if endpoint.data.manifest.spec.endpoint == "synthetic-suite-a-real-frame":
        declarations = registry.get("real_frame_declarations")
        require(declarations == {"support_gaps": "from_frame", "heterogeneous_eligibility": "from_frame"},
                "real-frame support declarations must be derived from frame")
        facts = sampled_support_facts(endpoint.policy, frame,
                                      fixed_exposures=fixed_exposures(endpoint, frame))
        derived = {name: facts[name] for name in declarations}
    scenarios = []
    for row in registry["scenarios"]:
        values = dict(row)
        if derived:
            mechanisms = [name for name in values.get("active_mechanisms", ("linear",)) if name not in derived]
            mechanisms.extend(name for name, enabled in derived.items() if enabled)
            values.update(derived, active_mechanisms=tuple(mechanisms))
        scenarios.append(SCMConfig(**values).to_dict()["payload"])
    return scenarios


def _profile(request, config, reference, scenario, recipe, stamps):
    record = load_budget(request, config, reference["dependency"])
    screening = bool(screening_metadata(recipe))
    require(record["complete"] and record["all_successful"]
            and (screening or record["profile"]["production_equivalent"] is True),
            "complete production profiling required")
    if screening:
        producer = read_json(dependency(request, config,
            {"dependency": reference["dependency"], "path": "_execution/task.json"}))
        require(record["chained"] and producer["parameters"].get("study") == recipe["study"]
                and record["profile"]["production_equivalent"] is False,
                "screening requires measured study continuation profiles")
        slots = screening_decision()["month1_sizing"]["leaves_per_scenario_max"]
        require(producer.get("continuation", {}).get("step") == slots - 1
                and len(record["leaves"]) == slots, "screening requires the terminal bootstrap profile slot")
    result_path = dependency(request, config, reference)
    require(file_hash(result_path) == record["payload_hash"], "profile scientific payload hash mismatch")
    result = read_json(result_path)
    binding = record["binding"]
    require(binding["mode"] == "profile" and SCMConfig(**binding["scenario"]).to_dict()["payload"] == scenario, "profile scenario mismatch")
    require(binding["recipe_hash"] == digest(recipe), "profile recipe drift")
    chained = record["chained"]
    timing = {**record["profile"], "leaves": record["leaves"], "wall_seconds": record["wall_seconds"],
        "setup_seconds": math.fsum(leaf["setup_seconds"] for leaf in record["leaves"]),
        "admitted_budget_seconds": binding["wall_seconds"]}
    timing["gpu_seconds"] = timing["wall_seconds"] if timing["device"].startswith("cuda") else 0.
    selected = resolve_device()
    require(timing["device"] == str(selected), "profile device differs from selected runtime")
    gpu_seconds = finite(timing["gpu_seconds"], "profile GPU seconds")
    require((selected.type == "cpu" and gpu_seconds == 0) or
            (selected.type == "cuda" and gpu_seconds > 0), "profile GPU accounting mismatch")
    require(all(timing.get(k) == v for k, v in stamps.items()), "profile scientific/environment drift")
    declared, records = binding["draws"], result["records"]
    require(result["draws"] == declared, "profile scientific draw mismatch")
    summary = summarize(records, declared, production_equivalent=True, null_scenario=scenario["effect"] == "null")
    measured = timing["complete_repetition_seconds"]
    require(len(measured) == len(declared) and all(finite(t, "profile time") > 0 for t in measured), "invalid profile times")
    wall = finite(timing["wall_seconds"], "profile wall seconds")
    require(wall >= math.fsum(measured), "profile wall time does not include its repetitions")
    setup = finite(timing["setup_seconds"], "profile setup seconds")
    overhead = wall - math.fsum(measured)
    require(0 <= setup <= overhead, "invalid profile setup/finalization accounting")
    extra = {}
    rate = max(measured)
    if chained:
        leaves = timing["leaves"]
        require(bool(leaves) and [v["step"] for v in leaves] == list(range(len(leaves))),
                "partial profile chain")
        require(len({v["task_id"] for v in leaves}) == len(leaves), "repeated profile leaf")
        names = ("compute_seconds", "setup_seconds", "import_seconds", "export_seconds", "publication_seconds")
        for leaf in leaves:
            require(all(finite(leaf[k], "profile leaf " + k) >= 0 for k in names), "negative profile leaf time")
            require(math.isclose(math.fsum(leaf[k] for k in names), leaf["wall_seconds"], rel_tol=1e-6, abs_tol=.01),
                    "profile leaf accounting mismatch")
            require(leaf["wall_seconds"] <= timing["admitted_budget_seconds"] <= 14400,
                    "profile leaf exceeds slice budget")
        require(math.isclose(math.fsum(v["wall_seconds"] for v in leaves), wall, rel_tol=1e-12),
                "profile chain total mismatch")
        # Include synchronization and between-draw compute bookkeeping as
        # well as the maximum individually measured repetition.
        rate = max(rate, math.fsum(v["compute_seconds"] for v in leaves) / len(declared))
        extra = {"leaf_overheads": {k: max(v[k] for v in leaves) for k in names[1:]},
                 "leaf_measurements": leaves, "resume_overhead_seconds": RESUME_OVERHEAD_FLOOR_SECONDS,
                 "resume_source": {"rule": "uncalibrated_floor", "floor_seconds": RESUME_OVERHEAD_FLOOR_SECONDS}}
    return {**extra, "seconds_per_repetition": rate, "setup_seconds": setup,
            "publication_verification_seconds": overhead - setup,
            "profile_repetitions": len(declared), "result_hash": file_hash(result_path),
            "budget_record_hash": record["record_hash"], "source_unit": reference["dependency"], "draws": declared}


def slice_count(profile, repetitions, seconds, factor, *, enforce_limit=True):
    """Pay measured leaf costs and reserve one successor beyond the estimate."""
    require(finite(factor, "profile safety factor") >= 1, "invalid profile safety factor")
    integer(repetitions, "repetitions", 1)
    resume = profile.get("resume_overhead_seconds", RESUME_OVERHEAD_FLOOR_SECONDS)
    capacity = usable_leaf_compute(seconds, profile["leaf_overheads"],
                                   resume_overhead_seconds=resume) - resume
    demand = repetitions * profile["seconds_per_repetition"] + profile["publication_verification_seconds"]
    require(finite(demand, "chain demand") > 0, "positive chain demand required")
    count = math.ceil(factor * demand / capacity) + 1
    require(not enforce_limit or count <= 40, "infeasible batch budget: required spare slice exceeds forty leaves")
    return count


def instance_batches(lock, instance):
    if lock["recipe"].get("campaign_mode") == "diagnostic":
        return [b for b in lock["batches"] if b["instance"] == instance]
    if screening_metadata(lock["recipe"]):
        return [b for b in lock["batches"] if b["scenario"]["name"] == instance]
    return [b for b in lock["batches"] if b["mode"] == instance]


def screening_estimate(request, config, parameters):
    """Same measured capacity and whole-minute reservations used by the lock."""
    recipe = parameters["recipe"]
    if recipe.get("campaign_mode") == "diagnostic":
        return diagnostic_estimate(request, config, parameters)
    validate_recipe(recipe)
    decision = validate_screening(recipe, config.get("approvals"))
    require(parameters.get("partition") == "standard", "screening partition must be standard")
    require(type(parameters["gpus"]) is int and parameters["gpus"] == 1,
            "screening requires single-GPU leaves")
    require(resolve_device().type == "cuda", "screening measurement requires CUDA")
    require(integer(parameters["cpus_per_task"], "CPUs", 1) == 8, "screening requires eight CPUs")
    wall = integer(parameters["wall_seconds"], "leaf wall seconds", 1)
    owner_wall = maximum_leaf_stage_seconds()
    allocation = leaf_allocation(wall)["allocation_seconds"]
    factor = finite(parameters["profile_safety_factor"], "profile safety factor")
    safety_decision = approved_safety_factor(parameters, config.get("approvals"))
    require(not {"screening_repetitions", "final_repetitions", "repetitions_per_leaf"}.intersection(parameters),
            "screening counts come only from measured month1 sizing")
    records = []
    for key, cls, expected in (("endpoint_input", PreparedEndpoint, recipe["endpoint_hash"]),
                                ("frame_input", CovariateFrame, recipe["frame_hash"])):
        record = cls.from_json(dependency(request, config, parameters[key]).read_text())
        require(record.content_hash == expected, "screening endpoint/frame hash mismatch")
        records.append(record)
    scenarios = registered_scenarios(*records)
    require([s["name"] for s in scenarios] == decision["scenarios"], "screening scenario family drift")
    require(set(parameters["profiles"]) == set(decision["scenarios"]), "all registered screening profiles required")
    stamps = fingerprint()
    profiles = {s["name"]: _profile(request, config, parameters["profiles"][s["name"]], s, recipe, stamps)
                for s in scenarios}
    from oxyformer.validation.campaign_probe import calibrate, profile_setup_sources
    for scenario in scenarios:
        name = scenario['name']
        refs = parameters.get('probe_profiles', {}).get(name)
        profiles[name]['setup_profile_sources'] = (profile_setup_sources(request, config, refs,
            scenario, profiles[name], stamps) if refs is not None else {})
    profiles, calibration = calibrate(request, config, parameters, profiles)
    rows = {}
    requested_counts = {}
    for name, profile in profiles.items():
        require("leaf_overheads" in profile, "screening requires measured chained costs")
        # Search the bounded set in descending order. slice_count is also used
        # below to construct the lock: there is no second cost approximation.
        requested_counts[name] = slice_count(profile,
            decision["month1_sizing"]["repetitions_per_scenario_max"], owner_wall, factor, enforce_limit=False)
        count = 0
        for candidate in range(decision["month1_sizing"]["repetitions_per_scenario_max"], 0, -1):
            if slice_count(profile, candidate, owner_wall, factor, enforce_limit=False) <= 40:
                count = candidate
                break
        if count == 0 and calibration['source'] == 'pre_lock_probe':
            without = deepcopy(profile)
            without['leaf_overheads']['setup_seconds'] = calibration['setup_without_headroom']
            unpadded = slice_count(without, 1, owner_wall, factor, enforce_limit=False)
            require(False, f"one screening repetition cannot fit forty leaves with 1.5x calibration headroom; "
                f"count without headroom: {unpadded}; declare a calibration set from sealed campaign-shaped leaves; "
                f"completed probe draws: {len(calibration['draws'])}; "
                + ("run the second calibration probe before re-estimating" if len(calibration['draws']) == 1
                   else "two-draw recovery exhausted; further draws require owner decision"))
        require(count > 0, "one screening repetition with its required spare slice cannot fit forty leaves: " + name)
        # A shorter requested wall may save allocation, but cannot lower the
        # count fixed by the owner's full permitted per-scenario capacity.
        leaves = slice_count(profile, count, wall, factor, enforce_limit=False)
        require(leaves <= 40, "requested leaf wall cannot fit the owner-maximal repetition count with its required spare slice: " + name)
        rows[name] = {"repetitions": count, "leaves": leaves,
                      "gpu_hours": leaves * allocation / 3600}
    from oxyformer.validation.coverage import projected_setup_allowance, require_setup_fits
    longest = max(row['leaves'] for row in rows.values())
    projections = []
    for chain in calibration.get('setup_chains', []):
        if len(chain['resumed']) < 3:
            projections.append({**chain, 'status': 'unmeasured', 'reason': 'fewer than three resumed leaves'})
            continue
        profile = profiles[chain['scenario']]
        projected = projected_setup_allowance(chain['fresh'], chain['resumed'], longest,
            leaf_indices=chain['leaf_indices'])
        evidence = require_setup_fits(projected, window=wall,
            resume_reserve=profile['resume_overhead_seconds'],
            usable_compute=profile['assumed_usable_per_leaf'])
        projections.append({**chain, **evidence, 'status': 'measured'})
    calibration['setup_projection'] = {'longest_chain': longest,
        'status': 'measured' if any(p['status'] == 'measured' for p in projections) else 'unmeasured',
        'chains': projections, 'policy': 'nonnegative median of all pairwise slopes; max observed plus remaining horizon; +60 seconds'}
    calibration['requested_batch_counts'] = requested_counts
    calibration['second_probe_recommended'] = (calibration['source'] == 'pre_lock_probe'
        and len(calibration['draws']) == 1 and any(n > 40 for n in requested_counts.values()))
    total = math.fsum(row["gpu_hours"] for row in rows.values())
    require(total <= 640 and total <= decision["budget_h100_hours"], "screening exceeds owner GPU-hour ceiling")
    return {"schema_version": 1, **screening_metadata(recipe), "estimate_only": True,
        "admitted": False, "submission_performed": False, "certifies_production_coverage": False,
        "recipe": deepcopy(recipe), "recipe_hash": digest(recipe), "owner_decision": decision,
        "owner_decision_hash": digest(decision), "scenario_family": scenarios, **stamps,
        "profiles": profiles, "calibration": calibration, "per_scenario": rows, "total_repetitions": sum(r["repetitions"] for r in rows.values()),
        "projected_leaf_count": sum(r["leaves"] for r in rows.values()), "gpu_hours": total,
        "partition": "standard", "gpus": 1, "cpus_per_task": 8, "wall_seconds": wall,
        "owner_safety_factor_decision": safety_decision,
        "profile_safety_factor": factor, "gpu_concurrency_max": decision["gpu_concurrency_max"],
        "budget_h100_hours": decision["budget_h100_hours"],
        "limitations": ["Planning estimate only; report per-scenario counts to owner before separate campaign-lock.",
            "GPU hours reserve whole leaves; collectors use CPU; profiles and reruns are separate accounting.",
            "A separate post-lock probe and admission receipt are required before screening execution."]}


def build_screening_lock(request, config, task):
    p = task["parameters"]
    estimate = screening_estimate(request, config, p)
    diagnostic = p["recipe"].get("campaign_mode") == "diagnostic"
    d = month2_decision(config.get("approvals")) if diagnostic else None
    c = d["diagnostic_campaign"] if diagnostic else None
    partition = c["partition"] if diagnostic else "standard"
    concurrency = estimate["gpu_concurrency_max"]
    estimate_path = dependency(request, config, p["estimate_input"])
    require(read_json(estimate_path) == estimate, "screening estimate changed; report a new estimate before lock")
    campaign_id = p["campaign_id"]
    concrete_id(campaign_id)
    require(len(campaign_id) <= 23, "campaign ID must leave room for scenario suffix")
    namespace = p["evaluation_namespace"]
    require(isinstance(namespace, str) and bool(namespace.strip()), "independent evaluation namespace required")
    require(p["retry_rules"] == RETRY_RULES, "unregistered retry rules")
    prior_seeds, prior_namespaces = set(), set()
    prior_matches = 0
    for reference in p["prior_locks"]:
        previous = read_json(dependency(request, config, reference))
        if diagnostic and previous["campaign_id"] == c["draws_continue_from"]:
            prior_matches += 1
        prior_namespaces.add(previous["evaluation_namespace"])
        prior_seeds.update(d["seed"] for b in previous["batches"] for d in b["draws"])
    require(not diagnostic or prior_matches == 1, "diagnostic draws_continue_from requires exactly one matching prior lock")
    require(namespace not in prior_namespaces, "evaluation namespace already used")
    profiles = estimate["profiles"]
    for profile in profiles.values():
        prior_seeds.update(d["seed"] for d in profile["draws"])
    seed_namespace = digest([request.content_hash, namespace, estimate])
    batches = []
    metadata = screening_metadata(p["recipe"])
    if diagnostic:
        for i, scenario in enumerate(estimate["scenario_family"]):
            name = scenario["name"]
            draws = repetition_plan(seed_namespace + ":diagnostic", name, c["repetitions"])
            for j, draw in enumerate(draws):
                instance = f"{name}-c{j}"
                batches.append({"batch_id": digest([campaign_id, namespace, name, j])[:24],
                    "instance": instance, "mode": "diagnostic", "scenario": scenario, "draws": [draw],
                    "wall_seconds": p["wall_seconds"], "gpus": 1, "partition": partition,
                    "slices": estimate["per_scenario"][name]["per_chain"]["leaves"], **metadata,
                    "leaf_budget": leaf_budget(p["wall_seconds"], estimate["profile_safety_factor"], profiles[name]["leaf_overheads"])})
    else:
        for scenario in estimate["scenario_family"]:
            name = scenario["name"]
            count = estimate["per_scenario"][name]["repetitions"]
            profile = profiles[name]
            slices = slice_count(profile, count, p["wall_seconds"], p["profile_safety_factor"])
            draws = repetition_plan(seed_namespace + ":screening", name, count)
            batches.append({"batch_id": digest([campaign_id, namespace, name])[:24], "mode": "screening",
                "scenario": scenario, "draws": draws, "wall_seconds": p["wall_seconds"], "gpus": 1,
                "partition": "standard", "slices": slices, **metadata,
                "leaf_budget": leaf_budget(p["wall_seconds"], p["profile_safety_factor"], profile["leaf_overheads"])})
    draws = [d for b in batches for d in b["draws"]]
    validate_draws(draws)
    if diagnostic:
        require(len({d["seed"] for d in draws}) == len(draws), "duplicate campaign seed")
    require(not prior_seeds.intersection(d["seed"] for d in draws), "evaluation draws overlap prior lock or profiling")
    budget = {"gpu_hours": estimate["gpu_hours"], "gpu_hours_per_leaf_max": 4,
        "leaves_per_instance_max": 40, "per_scenario": estimate["per_scenario"],
        "owner_safety_factor_decision": estimate["owner_safety_factor_decision"],
        "profile_safety_factor": estimate["profile_safety_factor"], "profiles": profiles,
        "calibration": estimate["calibration"],
        "concurrency_limits": {"gpu": 4, "cpu": 6}, "run_gpu_hours_ceiling": 3600,
        "collector_gpu_hours": 0,
        "cpu_wall_hours_ceiling": (estimate["projected_leaf_count"] * leaf_allocation(p["wall_seconds"])["allocation_seconds"] / 60 + 40) / 60}
    from oxyformer.validation.campaign_probe import expected_predecessor_shape
    counts = {batch['scenario']['name']: len(batch['draws']) for batch in batches}
    budget['expected_predecessor_shapes'] = {name: expected_predecessor_shape(profile,
        LEAF_OUTPUTS + ['continuation.tar'], draw_count=counts[name])
        for name, profile in profiles.items()}
    budget['setup_safety_net'] = {'noise_fraction_of_fresh': .1, 'consecutive_overruns': 2,
        'total_overruns': 3, 'overrunning_leaf_gpu_hours_bound': 12,
        'policy': 'conservative projection, three-receipt admission, next-leaf refusal'}
    if diagnostic:
        budget["cpu_wall_hours_ceiling"] = (estimate["projected_leaf_count"] * leaf_allocation(p["wall_seconds"])["allocation_seconds"] + len(batches) * 600) / 3600
        budget.update(**metadata, gpu_hours_per_leaf_max=c["gpu_hours_per_leaf_max"],
            leaves_per_instance_max=c["leaves_per_chain_max"], concurrency_limits={"gpu": concurrency, "cpu": 6},
            gpu_concurrency_max=concurrency, run_gpu_hours_ceiling=None, owner_decision="month2_followon",
            owner_decision_hash=estimate["owner_decision_hash"])
    else:
        require(budget["gpu_hours"] + budget["collector_gpu_hours"] <= 3600, "screening allocation exceeded")
    require(finite(p["cpu_wall_hours_budget"], "CPU hours budget") >= budget["cpu_wall_hours_ceiling"],
            "infeasible CPU budget")
    lock = {"schema_version": 1, "campaign_id": campaign_id, **metadata,
        "recipe": deepcopy(p["recipe"]), "nested_cv": deepcopy(p["recipe"]["nested_cv"]),
        **{k: estimate[k] for k in ("scientific_fingerprint", "environment_hash", "scenario_family")},
        "launch_code_identity": request.code_identity, "estimate_hash": file_hash(estimate_path),
        "estimate_input": deepcopy(p["estimate_input"]), "admission_receipt": deepcopy(p["admission_receipt"]), "owner_decision_hash": estimate["owner_decision_hash"],
        "evaluation_namespace": namespace, "seed_namespace": seed_namespace, "batches": batches, "budget": budget,
        "retry_rules": deepcopy(RETRY_RULES), "prior_namespaces": sorted(prior_namespaces),
        "instances": {name: campaign_id + "-s" + str(i) for i, name in enumerate(estimate["per_scenario"])}}
    if diagnostic:
        lock.update(partition=partition, concurrency_limits={"gpu": concurrency, "cpu": 6},
            gpu_concurrency_max=concurrency, owner_decision="month2_followon",
            instances={f"{scenario['name']}-c{j}": f"{campaign_id}-s{i}-c{j}"
                for i, scenario in enumerate(estimate["scenario_family"]) for j in range(c["repetitions"])})
    lock_ref = {"dependency": task["id"], "path": "recipe_lock.json", "sha256": digest(lock)}
    from oxyformer.validation.campaign_probe import LOCK_OUTPUTS
    inputs = {task["id"]: list(LOCK_OUTPUTS)}
    for key in ("endpoint_input", "frame_input", "estimate_input"):
        ref = p[key]
        inputs.setdefault(ref["dependency"], []).append(ref["path"])
    for ref in p["profiles"].values():
        inputs[ref["dependency"]] = ["_execution/task.json", "_execution/request.json", "_execution/result.json",
                                     BUDGET_RECORD, "result.json"]
    plans = []
    for batch in batches:
        work = {"id": batch["batch_id"], "stage": "coverage",
            "parameters": {**batch, "endpoint_input": p["endpoint_input"], "frame_input": p["frame_input"]},
            "outputs": LEAF_OUTPUTS + ["continuation.tar"],
            "slices": [{"gpus": 1, "wall_seconds": leaf_allocation(p["wall_seconds"])["allocation_seconds"]}] * batch["slices"]}
        spec = {"schema_version": 1, "id": lock["instances"][batch["instance"] if diagnostic else batch["scenario"]["name"]],
            "kind": "diagnostic" if diagnostic else "screening", **metadata, "partition": partition, "gpu_concurrency_max": concurrency,
            "prerequisites": [], "inputs": inputs, "recipe_lock": lock_ref, "work": [work],
            "admission_receipt": deepcopy(p["admission_receipt"]),
            "collector": {"stage": "campaign-collect", "outputs": COLLECT_OUTPUTS, "wall_seconds": 600, "gpus": 0}}
        plans.append(expand_campaign(spec, config["approvals"]))
    if diagnostic:
        # The first chain's collector for each scenario receives every chain's
        # leaves. Other collectors retain their local completion receipts.
        for scenario in estimate["scenario_family"]:
            indices = [i for i, plan in enumerate(plans)
                if plan["spec"]["work"][0]["parameters"]["scenario"] == scenario]
            spec = deepcopy(plans[indices[0]]["spec"])
            spec["collector_inputs"] = {leaf["id"]: ["_execution/task.json", "_execution/request.json",
                "_execution/result.json", *leaf["outputs"]]
                for i in indices[1:] for leaf in plans[i]["tasks"][:-1]}
            plans[indices[0]] = expand_campaign(spec, config["approvals"])
    return lock, plans, {**budget, **metadata, "lock_hash": digest(lock), "estimate_hash": lock["estimate_hash"]}


def _admit_locked_leaves(request, config, lock, plans):
    """Measure setup on the actual expanded shapes before releasing the lock.

    No draws run here. Private candidate inputs exercise the same validation
    and parsing used by coverage.run_stage, including the complete lock and
    expansion. The temporary passing screening gate is solely a preflight cost
    fixture, is never emitted, and cannot authorize a scientific leaf.
    """
    Path(request.output_dir).mkdir(parents=True, exist_ok=True)
    measurements = []
    with TemporaryDirectory(prefix=".admission-", dir=request.output_dir) as temporary:
        root = Path(temporary)
        lock_root = root / "lock"
        lock_root.mkdir()
        atomic_json(lock_root, "recipe_lock.json", lock)
        atomic_json(lock_root, "expanded_units.json", {"lock_hash": digest(lock), "plans": plans})
        screening_root = root / "screening-cost-fixture"
        screening_root.mkdir()
        atomic_json(screening_root, "gate.json", {"pass": True, "mode": "screening", "lock_hash": digest(lock)})
        for plan in plans:
            measured_shapes = set()
            for leaf in plan["tasks"][:-1]:
                parameters = leaf["parameters"]
                size = len(parameters["draws"])
                name = parameters["scenario"]["name"]
                shape = (name, size)
                if shape in measured_shapes:
                    continue
                measured_shapes.add(shape)
                sample = root / leaf["id"]
                sample.mkdir()
                dependencies = dict(config["dependencies"])
                dependencies[leaf["recipe_lock"]["dependency"]] = str(lock_root)
                if "screening_gate" in parameters:
                    dependencies[parameters["screening_gate"]["dependency"]] = str(screening_root)
                sample_config = {**config, "dependencies": dependencies}
                config_path = atomic_json(sample, "config.json", sample_config)
                task_path = atomic_json(sample, "task.json", leaf)
                paths = tuple(dict.fromkeys(str(Path(dependencies[dep]) / relative)
                    for dep, relatives in plan["spec"]["inputs"].items() for relative in relatives))
                sample_request = replace(request, stage="coverage", config_path=str(config_path),
                    config_hash=file_hash(config_path), task_path=str(task_path), task_hash=file_hash(task_path),
                    dependency_paths=paths, dependency_hashes=tuple(map(file_hash, paths)),
                    output_dir=str(sample / "out"))
                started = time.monotonic()
                # Match the public entry plus complete batch preflight.
                sample_request.verify_inputs()
                read_json(sample_request.task_path)
                prepare_batch(sample_request)
                setup = time.monotonic() - started
                profile = lock["budget"]["profiles"][name]
                publication = profile["publication_verification_seconds"] * max(1., size / profile["profile_repetitions"])
                if "leaf_overheads" in profile:
                    contract = parameters["leaf_budget"]
                    overheads = contract["overheads"]
                    publication = overheads["publication_seconds"]
                    capacity = remaining_compute(contract, setup_seconds=setup)
                    require(size * profile["seconds_per_repetition"] / parameters["slices"] <= capacity,
                            "infeasible batch budget: locked setup reduces admitted compute")
                    estimated = contract["reserve_seconds"] + (max(setup, overheads["setup_seconds"]) + overheads["import_seconds"]
                        + overheads["export_seconds"] + publication
                        + size * profile["seconds_per_repetition"] / parameters["slices"]) * contract["safety_factor"]
                else:
                    estimated = (setup + size * profile["seconds_per_repetition"] + publication) * lock["budget"]["profile_safety_factor"]
                measurements.append({"mode": parameters["mode"], "scenario": name, "repetitions": size,
                    "locked_setup_seconds": setup, "publication_verification_seconds": publication,
                    "estimated_seconds": estimated, "admitted_budget_seconds": parameters["wall_seconds"]})
                require(estimated <= parameters["wall_seconds"],
                        "infeasible batch budget: measured locked setup, repetitions and scaled publication/verification do not fit")
    return measurements


def build_lock(request, config, task):
    """Expand screening and final instances under one immutable recipe.

    Each instance has its own <=40 leaves. Final leaves depend on the completed
    screening collector. New locks use disjoint namespaces and must name all
    prior locks as declared inputs; prior final and profile seeds are excluded.
    """
    parameters = task["parameters"]
    if screening_metadata(parameters["recipe"]):
        return build_screening_lock(request, config, task)
    campaign_id = parameters["campaign_id"]
    concrete_id(campaign_id)
    require(len(campaign_id) <= 25, "campaign ID must leave room for instance suffix")
    recipe = deepcopy(parameters["recipe"])
    validate_recipe(recipe)
    inputs = {}
    for name, expected_hash, record_type in (("endpoint_input", recipe["endpoint_hash"], PreparedEndpoint),
                                             ("frame_input", recipe["frame_hash"], CovariateFrame)):
        record = record_type.from_json(dependency(request, config, parameters[name]).read_text())
        require(record.content_hash == expected_hash, "locked endpoint/frame hash mismatch")
        inputs[name] = record
    family = parameters["scenario_family"]
    registered = {row["name"]: row for row in registered_scenarios(inputs["endpoint_input"], inputs["frame_input"])}
    require(type(family) is list and bool(family), "scenario family required")
    scenarios = [SCMConfig(**row).to_dict()["payload"] for row in family]
    require(len({s["name"] for s in scenarios}) == len(scenarios), "duplicate scenarios")
    require(all(s["name"] in registered and s == registered[s["name"]] for s in scenarios), "unregistered final scenario")
    count = integer(parameters["final_repetitions"], "final repetitions", MIN_REPETITIONS)
    screen_count = integer(parameters["screening_repetitions"], "screening repetitions", 1)
    batch_size = integer(parameters["repetitions_per_leaf"], "repetitions per leaf", 1)
    # Reject infeasible declarations using integer arithmetic before allocating
    # seed dictionaries or expanded units, even for an accidentally huge count.
    for repetitions in (screen_count, count):
        leaves = len(scenarios) * ((repetitions + batch_size - 1) // batch_size)
        require(leaves <= 40, "campaign instance exceeds forty leaves; profile a feasible batching plan")
    gpus = parameters["gpus"]
    require(type(gpus) is int and gpus in (0, 1), "nested fitting supports zero or one GPU")
    require(gpus == int(resolve_device().type == "cuda"),
            "requested GPU count has no GPU device interface in the selected runtime")
    wall_seconds = integer(parameters["wall_seconds"], "leaf wall seconds", 1)
    require(gpus * math.ceil(wall_seconds / 60) * 60 <= 14400, "leaf exceeds four GPU-hours")
    factor = finite(parameters["profile_safety_factor"], "profile safety factor")
    safety_decision = approved_safety_factor(parameters, config.get("approvals"))
    namespace = parameters["evaluation_namespace"]
    require(isinstance(namespace, str) and bool(namespace.strip()), "independent evaluation namespace required")
    require(parameters["retry_rules"] == RETRY_RULES, "unregistered retry rules")
    stamps = fingerprint()
    prior_seeds, prior_namespaces = set(), set()
    for reference in parameters["prior_locks"]:
        previous = read_json(dependency(request, config, reference))
        prior_namespaces.add(previous["evaluation_namespace"])
        prior_seeds.update(d["seed"] for b in previous["batches"] for d in b["draws"])
    seed_namespace = digest([request.content_hash, namespace, recipe, scenarios, stamps])
    profiles = {}
    for scenario in scenarios:
        profiles[scenario["name"]] = _profile(request, config, parameters["profiles"][scenario["name"]], scenario, recipe, stamps)
        prior_seeds.update(d["seed"] for d in profiles[scenario["name"]]["draws"])
    require(not any("leaf_overheads" in p for p in profiles.values()) or wall_seconds <= 14400,
            "coverage slice exceeds four hours")
    batches = []
    for mode, repetitions in (("screening", screen_count), ("final", count)):
        mode_leaves = 0
        for scenario in scenarios:
            profile = profiles[scenario["name"]]
            require("leaf_overheads" in profile or (profile["publication_verification_seconds"] * max(1., min(batch_size, repetitions) / profile["profile_repetitions"])
                     + profile["seconds_per_repetition"] * min(batch_size, repetitions)) * factor <= wall_seconds,
                    "infeasible batch budget: measured complete repetitions do not fit")
            draws = repetition_plan(seed_namespace + ":" + mode, scenario["name"], repetitions)
            for start in range(0, repetitions, batch_size):
                concrete = draws[start:start + batch_size]
                slices = (slice_count(profile, len(concrete), wall_seconds, factor)
                          if "leaf_overheads" in profile else 1)
                extra = {"slices": slices, "leaf_budget": leaf_budget(wall_seconds, factor, profile["leaf_overheads"])} if "leaf_overheads" in profile else {}
                batches.append({"batch_id": digest([campaign_id, namespace, mode, scenario["name"], start])[:24],
                    "mode": mode, "scenario": scenario, "draws": concrete, "wall_seconds": wall_seconds, "gpus": gpus, **extra})
                mode_leaves += slices
        require(mode_leaves <= 40, "campaign instance including required spare slices exceeds forty leaves; profile a feasible batching plan")
    draws = [d for batch in batches for d in batch["draws"]]
    validate_draws(draws)
    require(not prior_seeds.intersection(d["seed"] for d in draws), "evaluation draws overlap prior lock or profiling")
    total_leaves = sum(b.get("slices", 1) for b in batches)
    allocation_seconds = sum(b.get("slices", 1) * (leaf_allocation(wall_seconds)["allocation_seconds"]
        if "slices" in b else math.ceil(wall_seconds / 60) * 60) for b in batches)
    budget = {"gpu_hours": gpus * allocation_seconds / 3600, "gpu_hours_per_leaf_max": 4., "leaves_per_instance_max": 40,
              "cpu_wall_hours_ceiling": (allocation_seconds / 60 + 20) / 60,
              "owner_safety_factor_decision": safety_decision,
              "profile_safety_factor": factor, "profiles": profiles,
              "admission_model": "measure locked setup at each leaf size; scale complete profile finalization by repetition count",
              "budget_semantics": "planning estimate; report overruns and retain draws; scheduler enforces binding limits",
              "concurrency_limits": {"gpu": 8, "cpu": 6}, "run_gpu_hours_ceiling": 2500}
    require(finite(parameters["cpu_wall_hours_budget"], "CPU hours budget") >= budget["cpu_wall_hours_ceiling"],
            "infeasible CPU budget")
    lock = {"schema_version": 1, "campaign_id": campaign_id, "recipe": recipe, "nested_cv": deepcopy(recipe["nested_cv"]),
        "scientific_fingerprint": stamps["scientific_fingerprint"], "environment_hash": stamps["environment_hash"],
        "launch_code_identity": request.code_identity, "scenario_family": scenarios,
        "evaluation_namespace": namespace, "seed_namespace": seed_namespace, "batches": batches, "budget": budget,
        "retry_rules": deepcopy(RETRY_RULES), "prior_namespaces": sorted(prior_namespaces),
        "screening_rule": "all declared screening draws finish successfully; no numerical failures; recipe immutable",
        "instances": {"screening": campaign_id + "-screen", "final": campaign_id + "-final"}}
    lock_ref = {"dependency": task["id"], "path": "recipe_lock.json", "sha256": digest(lock)}
    inputs = {task["id"]: ["recipe_lock.json", "expanded_units.json"]}
    for key in ("endpoint_input", "frame_input"):
        reference = parameters[key]
        inputs.setdefault(reference["dependency"], [])
        if reference["path"] not in inputs[reference["dependency"]]:
            inputs[reference["dependency"]].append(reference["path"])
    for reference in parameters["profiles"].values():
        inputs[reference["dependency"]] = ["_execution/task.json", "_execution/request.json",
            "_execution/result.json", BUDGET_RECORD, "result.json"]
    plans = []
    for mode in ("screening", "final"):
        stage_inputs = deepcopy(inputs)
        screening_gate = None
        if mode == "final":
            screening_gate = {"dependency": plans[0]["tasks"][-1]["id"], "path": "gate.json"}
            stage_inputs[screening_gate["dependency"]] = ["gate.json"]
        work = []
        for batch in batches:
            if batch["mode"] != mode:
                continue
            parameters_for_leaf = {**batch, "endpoint_input": parameters["endpoint_input"],
                                   "frame_input": parameters["frame_input"]}
            if screening_gate:
                parameters_for_leaf["screening_gate"] = screening_gate
            work.append({"id": batch["batch_id"], "stage": "coverage", "parameters": parameters_for_leaf,
                "outputs": LEAF_OUTPUTS + (["continuation.tar"] if "slices" in batch else []),
                "slices": [{"gpus": gpus, "wall_seconds": (leaf_allocation(wall_seconds)["allocation_seconds"]
                    if "slices" in batch else wall_seconds)}] * batch.get("slices", 1)})
        spec = {"schema_version": 1, "id": lock["instances"][mode],
            "kind": "screening" if mode == "screening" else "final-coverage",
            "prerequisites": [], "inputs": stage_inputs, "recipe_lock": lock_ref, "work": work,
            "collector": {"stage": "campaign-collect", "outputs": COLLECT_OUTPUTS, "wall_seconds": 600}}
        plans.append(expand_campaign(spec, config["approvals"]))
    measurements = _admit_locked_leaves(request, config, lock, plans)
    return lock, plans, {**budget, "lock_hash": digest(lock), "admission_measurements": measurements}


def _plan_for_task(request, config, task, lock):
    reference = {"dependency": task["recipe_lock"]["dependency"], "path": "expanded_units.json"}
    expansion = read_json(dependency(request, config, reference))
    require(expansion["lock_hash"] == digest(lock), "expansion lock drift")
    plans = expansion["plans"]
    require({p["spec"]["id"] for p in plans} == set(lock["instances"].values()), "missing campaign instance")
    for plan in plans:
        validate_plan(plan, config["approvals"])
        mode = next(k for k, v in lock["instances"].items() if v == plan["spec"]["id"])
        declared = instance_batches(lock, mode)
        work = plan["spec"]["work"]
        require(len(work) == len(declared), "expansion omitted locked batch")
        by_batch = {w["parameters"]["batch_id"]: w for w in work}
        require(len(by_batch) == len(work), "duplicate batch in expansion")
        for batch in declared:
            require(batch["batch_id"] in by_batch, "locked batch missing")
            actual = by_batch[batch["batch_id"]]
            require(all(actual["parameters"].get(k) == v for k, v in batch.items()), "locked batch drift")
            require(actual["slices"] == [{"gpus": batch.get("gpus", 0), "wall_seconds": (leaf_allocation(batch["wall_seconds"])["allocation_seconds"]
                        if "slices" in batch else batch["wall_seconds"])}]
                    * batch.get("slices", 1), "locked resources drift")
        if plan["spec"]["id"] == task["campaign"]:
            require(task in plan["tasks"], "task is not the prospectively expanded task")
            selected = plan
    require("selected" in locals(), "unknown campaign instance")
    return selected


def validate_leaf_task(request, config, task, lock):
    plan = _plan_for_task(request, config, task, lock)
    require(task["id"] in plan["expected_leaves"], "collector cannot execute a repetition batch")
    parameters = task["parameters"]
    if "slices" in parameters:
        profile = lock["budget"]["profiles"][parameters["scenario"]["name"]]
        require(parameters["leaf_budget"] == leaf_budget(parameters["wall_seconds"],
                lock["budget"]["profile_safety_factor"], profile["leaf_overheads"]), "locked leaf budget drift")
    if task["parameters"]["mode"] == "final":
        gate = read_json(dependency(request, config, task["parameters"]["screening_gate"]))
        require(gate["lock_hash"] == digest(lock) and gate["mode"] == "screening" and gate["pass"] is True,
                "screening did not admit the locked recipe")


def diagnostic_summary(records, declared, *, null_scenario=False):
    """Expose the existing complete-denominator statistics without certification."""
    from oxyformer.validation.coverage import METHODS, Z95, binomial_bound
    summary = summarize(records, declared, production_equivalent=False, null_scenario=null_scenario)
    successes = [r for r in records if r["status"] == "success"]
    n = len(declared)
    methods = {}
    for method in METHODS:
        rejected = sum(abs(r["estimates"][method]["value"]) > Z95 * r["estimates"][method]["se"] for r in successes)
        methods[method] = {**summary["methods"][method], "n": n,
            "mean_estimate": math.fsum(r["estimates"][method]["value"] for r in successes) / len(successes) if successes else None,
            "truth": math.fsum(r["truth"] for r in successes) / len(successes) if successes else None,
            "rejections": rejected, "rejection_upper": binomial_bound(rejected + n - len(successes), n, side="upper"),
            "numerical_failures": summary["counts"]["numerical_failure"],
            "certifies_release_criteria": False, "statement": "No release criterion is certified."}
    return {**summary, "coverage_pass": False, "coverage_pass_reason": "diagnostic", "diagnostic_summary": methods}


def collect(request, config, task):
    lock = load_lock(request, config, task)
    plan = _plan_for_task(request, config, task, lock)
    require(task == plan["tasks"][-1], "collection requires the declared collector")
    collected_leaves = list(plan["tasks"][:-1])
    if lock.get("campaign_mode") == "diagnostic" and "collector_inputs" in plan["spec"]:
        expansion = read_json(dependency(request, config, {
            "dependency": task["recipe_lock"]["dependency"], "path": "expanded_units.json"}))
        scenario = plan["spec"]["work"][0]["parameters"]["scenario"]
        others = [p for p in expansion["plans"] if p["spec"]["id"] != task["campaign"]
                  and p["spec"]["work"][0]["parameters"]["scenario"] == scenario]
        collected_leaves.extend(leaf for p in others for leaf in p["tasks"][:-1])
    require(task["expected_leaves"] == [leaf["id"] for leaf in collected_leaves], "partial collector fan-in")
    instance = next(k for k, v in lock["instances"].items() if v == task["campaign"])
    declared_batches = instance_batches(lock, instance)
    if lock.get("campaign_mode") == "diagnostic" and "collector_inputs" in plan["spec"]:
        declared_batches = [b for b in lock["batches"] if b["scenario"] == declared_batches[0]["scenario"]]
    mode = declared_batches[0]["mode"]
    if mode == "final":
        references = [{"dependency": dep, "path": "gate.json"} for dep, paths in plan["spec"]["inputs"].items()
                      if "gate.json" in paths]
        require(len(references) == 1, "final collection requires screening gate")
        gate = read_json(dependency(request, config, references[0]))
        require(gate["pass"] is True and gate["mode"] == "screening" and gate["lock_hash"] == digest(lock),
                "screening did not admit final collection")
    results, runtimes = [], []
    for leaf in collected_leaves:
        budget = load_budget(request, config, leaf["id"], expected_task=leaf)
        path = dependency(request, config, {"dependency": leaf["id"], "path": "result.json"})
        require(file_hash(path) == budget["payload_hash"], "leaf scientific payload hash mismatch")
        result = read_json(path)
        parameters = leaf["parameters"]
        require(result["lock_hash"] == digest(lock) and budget["binding"]["recipe_hash"] == digest(lock["recipe"]), "leaf recipe drift")
        require(result["batch_id"] == parameters["batch_id"] and result["mode"] == mode
                and result["scenario"] == parameters["scenario"] and result["draws"] == parameters["draws"], "leaf identity drift")
        measured = budget["leaves"][-1]["wall_seconds"]
        budget_seconds = parameters["wall_seconds"]
        runtimes.append({"leaf_id": leaf["id"], "batch_id": parameters["batch_id"],
            "scenario": parameters["scenario"]["name"], "wall_seconds": measured,
            "admitted_budget_seconds": budget_seconds, "overrun_seconds": max(0., measured - budget_seconds)})
        if "slices" in parameters and leaf["continuation"]["step"] + 1 != parameters["slices"]:
            continue
        result = {**result, "terminal": budget["complete"]}
        results.append(result)
    records = [r for result in results for r in result["records"]]
    declarations = [d for b in declared_batches for d in b["draws"]]
    validate_draws(declarations)
    require(len({r["draw"]["repetition_id"] for r in records}) == len(records), "duplicate collected repetition")
    summaries = {}
    for scenario in [s for s in lock["scenario_family"] if any(b["scenario"] == s for b in declared_batches)]:
        name = scenario["name"]
        selected = [r for r in records if r["draw"]["scenario"] == name]
        declared = [d for b in declared_batches if b["scenario"]["name"] == name for d in b["draws"]]
        summaries[name] = summarize(selected, declared, production_equivalent=not bool(screening_metadata(lock["recipe"])), null_scenario=scenario["effect"] == "null")
        if mode == "diagnostic":
            summaries[name] = diagnostic_summary(selected, declared, null_scenario=scenario["effect"] == "null")
    if mode == "final":
        passing = all(s["coverage_pass"] for s in summaries.values())
    else:
        passing = all(s["complete"] and s["counts"]["success"] == s["declared_repetitions"] for s in summaries.values())
    gate = {"pass": passing, "mode": mode, "lock_hash": digest(lock),
            "certifies_production_coverage": bool(passing and mode == "final"),
            "expected_leaves": task["expected_leaves"], "received_leaves": [t["id"] for t in collected_leaves],
            "scenario_summaries": summaries, "retry_rules": lock["retry_rules"],
            "incomplete_chains": [r["batch_id"] for r in results if not r.get("terminal", True)],
            "leaf_runtimes": runtimes, "budget_overruns": [r for r in runtimes if r["overrun_seconds"] > 0]}
    metadata = screening_metadata(lock["recipe"])
    gate.update(metadata)
    if mode == "diagnostic":
        gate.update(coverage_pass=False, coverage_pass_reason="diagnostic",
            diagnostic_summary={name: value["diagnostic_summary"] for name, value in summaries.items()})
    return {"result.json": {"records": records, "gate": gate,
        **({"diagnostic_summary": gate["diagnostic_summary"]} if mode == "diagnostic" else {}), **metadata}, "gate.json": gate}, passing


def run_stage(request: StageRequest) -> StageResult:
    try:
        request.verify_inputs()
        config = yaml.safe_load(Path(request.config_path).read_text())
        task = read_json(request.task_path)
        require(task["stage"] == request.stage, "stage mismatch")
        if request.stage == "campaign-lock":
            lock, plans, budget = build_lock(request, config, task)
            admission = None
            if screening_metadata(lock['recipe']):
                from oxyformer.validation.campaign_probe import setup_admission_plan
                admission = setup_admission_plan(lock, plans, task['id'], config['approvals'])
            values = {"recipe_lock.json": lock, "task_manifest.json": {"lock_hash": digest(lock),
                "tasks": [t for plan in plans for t in plan["tasks"]],
                **({'admission_tasks': admission['tasks']} if admission else {})},
                "expanded_units.json": {"lock_hash": digest(lock), "plans": plans,
                    **({'admission': admission} if admission else {})},
                "budget.json": budget, "gate.json": {"pass": True, "admitted": not bool(screening_metadata(lock["recipe"])),
                    "lock_hash": digest(lock), "submission_performed": False, "screening_complete": False,
                    "certifies_production_coverage": False}}
            metadata = screening_metadata(lock["recipe"])
            values = {name: {**value, **metadata} for name, value in values.items()}
            return publish(request, values, status="pass", message="Recipe locked prospectively; no jobs submitted")
        require(request.stage == "campaign-collect", "unknown campaign stage")
        values, passing = collect(request, config, task)
        return publish(request, values, status="pass" if passing else "fail",
            message="All declared leaves collected; gate passed" if passing else "Campaign gate failed; stop")
    except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
        return StageResult(request_hash=request.content_hash, status="blocked", artifacts=(), message=str(exc) or type(exc).__name__)
