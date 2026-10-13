"""Prospective campaign locks and complete fan-in; this module submits no jobs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import yaml
import torch

from oxyformer.training.fit import fit_environment, resolve_device
from oxyformer.training.nested_cv import PreparedEndpoint

from oxyformer.contracts import StageRequest, StageResult
from oxyformer.execution import identity
from oxyformer.execution.campaign import expand_campaign, validate_plan, concrete_id
from oxyformer.provenance import ContractError, file_hash, require
from oxyformer.validation.coverage import (MIN_REPETITIONS, RETRY_RULES, dependency, digest,
    finite, integer, publish, read_json, repetition_plan, summarize, validate_draws, prepare_batch, atomic_json, fixed_exposures,
    leaf_budget, remaining_compute, load_budget, BUDGET_RECORD)
from oxyformer.validation.scm import SCMConfig, CovariateFrame, sampled_support_facts

REGISTRY = Path(__file__).parents[3] / "configs/validation/final_scenarios.yaml"
REPOSITORY = Path(__file__).parents[3]
LEAF_OUTPUTS = [BUDGET_RECORD, "result.json", "timing.json", "artifact_manifest.json", "runtime.json"]
COLLECT_OUTPUTS = ["result.json", "gate.json", "artifact_manifest.json"]


def validate_recipe(recipe, *, production=True):
    require(set(recipe) == {"endpoint_hash", "frame_hash", "nested_cv", "inference"}, "unknown recipe fields")
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
    require(all(lock[k] == v for k, v in current.items()), "scientific code/environment recipe drift")
    validate_recipe(lock["recipe"])
    require(lock["nested_cv"] == lock["recipe"]["nested_cv"], "production nested recipe drift")
    draws = [draw for batch in lock["batches"] for draw in batch["draws"]]
    validate_draws(draws)
    require(len({b["batch_id"] for b in lock["batches"]}) == len(lock["batches"]), "duplicate locked batches")
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
    require(record["complete"] and record["all_successful"] and record["profile"]["production_equivalent"] is True,
            "complete production profiling required")
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
                 "leaf_measurements": leaves}
    return {**extra, "seconds_per_repetition": rate, "setup_seconds": setup,
            "publication_verification_seconds": overhead - setup,
            "profile_repetitions": len(declared), "result_hash": file_hash(result_path),
            "budget_record_hash": record["record_hash"], "source_unit": reference["dependency"], "draws": declared}


def slice_count(profile, repetitions, seconds, factor):
    """Conservative compute capacity after paying every leaf's measured costs."""
    capacity = leaf_budget(seconds, factor, profile["leaf_overheads"])["compute_seconds"]
    count = max(1, math.ceil(profile["seconds_per_repetition"] * repetitions / capacity))
    require(count <= 40, "infeasible batch budget: campaign instance exceeds forty leaves")
    return count


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
    require(factor >= 1, "profiling safety factor must be at least one")
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
        require(mode_leaves <= 40, "campaign instance exceeds forty leaves; profile a feasible batching plan")
    draws = [d for batch in batches for d in batch["draws"]]
    validate_draws(draws)
    require(not prior_seeds.intersection(d["seed"] for d in draws), "evaluation draws overlap prior lock or profiling")
    total_leaves = sum(b.get("slices", 1) for b in batches)
    budget = {"gpu_hours": gpus * total_leaves * math.ceil(wall_seconds / 60) / 60, "gpu_hours_per_leaf_max": 4., "leaves_per_instance_max": 40,
              "cpu_wall_hours_ceiling": (total_leaves * math.ceil(wall_seconds / 60) + 20) / 60,
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
            "_execution/result.json", BUDGET_RECORD]
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
                "slices": [{"gpus": gpus, "wall_seconds": wall_seconds}] * batch.get("slices", 1)})
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
        declared = [b for b in lock["batches"] if b["mode"] == mode]
        work = plan["spec"]["work"]
        require(len(work) == len(declared), "expansion omitted locked batch")
        by_batch = {w["parameters"]["batch_id"]: w for w in work}
        require(len(by_batch) == len(work), "duplicate batch in expansion")
        for batch in declared:
            require(batch["batch_id"] in by_batch, "locked batch missing")
            actual = by_batch[batch["batch_id"]]
            require(all(actual["parameters"].get(k) == v for k, v in batch.items()), "locked batch drift")
            require(actual["slices"] == [{"gpus": batch.get("gpus", 0), "wall_seconds": batch["wall_seconds"]}]
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


def collect(request, config, task):
    lock = load_lock(request, config, task)
    plan = _plan_for_task(request, config, task, lock)
    require(task == plan["tasks"][-1], "collection requires the declared collector")
    require(task["expected_leaves"] == plan["expected_leaves"], "partial collector fan-in")
    mode = next(k for k, v in lock["instances"].items() if v == task["campaign"])
    if mode == "final":
        references = [{"dependency": dep, "path": "gate.json"} for dep, paths in plan["spec"]["inputs"].items()
                      if "gate.json" in paths]
        require(len(references) == 1, "final collection requires screening gate")
        gate = read_json(dependency(request, config, references[0]))
        require(gate["pass"] is True and gate["mode"] == "screening" and gate["lock_hash"] == digest(lock),
                "screening did not admit final collection")
    results, runtimes = [], []
    for leaf in plan["tasks"][:-1]:
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
    declarations = [d for b in lock["batches"] if b["mode"] == mode for d in b["draws"]]
    validate_draws(declarations)
    require(len({r["draw"]["repetition_id"] for r in records}) == len(records), "duplicate collected repetition")
    summaries = {}
    for scenario in lock["scenario_family"]:
        name = scenario["name"]
        selected = [r for r in records if r["draw"]["scenario"] == name]
        declared = [d for b in lock["batches"] if b["mode"] == mode and b["scenario"]["name"] == name for d in b["draws"]]
        summaries[name] = summarize(selected, declared, production_equivalent=True, null_scenario=scenario["effect"] == "null")
    if mode == "final":
        passing = all(s["coverage_pass"] for s in summaries.values())
    else:
        passing = all(s["complete"] and s["counts"]["success"] == s["declared_repetitions"] for s in summaries.values())
    gate = {"pass": passing, "mode": mode, "lock_hash": digest(lock),
            "certifies_production_coverage": bool(passing and mode == "final"),
            "expected_leaves": plan["expected_leaves"], "received_leaves": [t["id"] for t in plan["tasks"][:-1]],
            "scenario_summaries": summaries, "retry_rules": lock["retry_rules"],
            "incomplete_chains": [r["batch_id"] for r in results if not r.get("terminal", True)],
            "leaf_runtimes": runtimes, "budget_overruns": [r for r in runtimes if r["overrun_seconds"] > 0]}
    return {"result.json": {"records": records, "gate": gate}, "gate.json": gate}, passing


def run_stage(request: StageRequest) -> StageResult:
    try:
        request.verify_inputs()
        config = yaml.safe_load(Path(request.config_path).read_text())
        task = read_json(request.task_path)
        require(task["stage"] == request.stage, "stage mismatch")
        if request.stage == "campaign-lock":
            lock, plans, budget = build_lock(request, config, task)
            values = {"recipe_lock.json": lock, "task_manifest.json": {"lock_hash": digest(lock),
                "tasks": [t for plan in plans for t in plan["tasks"]]},
                "expanded_units.json": {"lock_hash": digest(lock), "plans": plans},
                "budget.json": budget, "gate.json": {"pass": True, "admitted": True,
                    "lock_hash": digest(lock), "submission_performed": False, "screening_complete": False,
                    "certifies_production_coverage": False}}
            return publish(request, values, status="pass", message="Recipe locked prospectively; no jobs submitted")
        require(request.stage == "campaign-collect", "unknown campaign stage")
        values, passing = collect(request, config, task)
        return publish(request, values, status="pass" if passing else "fail",
            message="All declared leaves collected; gate passed" if passing else "Campaign gate failed; stop")
    except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
        return StageResult(request_hash=request.content_hash, status="blocked", artifacts=(), message=str(exc) or type(exc).__name__)
