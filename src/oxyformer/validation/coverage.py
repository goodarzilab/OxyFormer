"""Independent complete Suite A repetitions and conservative coverage accounting.

A batch is an execution container, never a statistical replication. Numerical
failures remain in all unconditional denominators. A completed leaf publishes
its records even when they fail scientific gates; only collection certifies a
whole, prospectively declared campaign. Smoke output never certifies coverage.
"""
from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import tarfile
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


BUDGET_RECORD = "budget_record.json"
MEASUREMENTS = "coverage_measurements.json"


def projected_setup_allowance(fresh, resumed, longest_chain, *, leaf_indices=None, extrapolate=True):
    """Conservative setup policy; admission deliberately does not extrapolate.

    Explicit leaf indices preserve slopes for sparse calibration selections.
    This changes sizing/admission evidence, never the sealed compute formula.
    """
    from statistics import median
    fresh = finite(fresh, "fresh setup")
    values = [finite(value, "resumed setup") for value in resumed]
    integer(longest_chain, "longest chain", 1)
    require(fresh >= 0 and bool(values) and all(value >= 0 for value in values), "invalid setup measurements")
    indices = list(range(2, len(values) + 2)) if leaf_indices is None else list(leaf_indices)
    require(len(indices) == len(values) and all(type(index) is int and index >= 2 for index in indices)
        and indices == sorted(set(indices)), "invalid setup leaf indices")
    slopes = [(values[j] - values[i]) / (indices[j] - indices[i])
        for i in range(len(values)) for j in range(i + 1, len(values))]
    growth = max(0., median(slopes)) if slopes else 0.
    horizon = max(0, longest_chain - indices[-1]) if extrapolate else 0
    projected = max(values) + growth * horizon
    return {"fresh": fresh, "points": values, "leaf_indices": indices,
        "growth": growth if extrapolate else values[-1] - values[0],
        "projected": projected, "longest_chain": longest_chain, "horizon": horizon,
        "allowance": max(fresh, projected) + 60., "extrapolated": extrapolate}


def require_setup_fits(projection, *, window, resume_reserve, usable_compute):
    ceiling = finite(window, "setup window") - finite(resume_reserve, "resume reserve") - finite(usable_compute, "sized usable compute")
    evidence = {**projection, "ceiling": ceiling}
    require(projection["allowance"] <= ceiling,
        "setup allowance does not fit: " + canonical_json(evidence))
    return evidence


def setup_overruns(leaves, allowance):
    """Derive the next-leaf refusal from authenticated timings, never flags."""
    allowance = finite(allowance, "admitted setup allowance")
    require(allowance >= 0 and bool(leaves), "setup safety net requires fresh measurements")
    fresh = finite(leaves[0]["setup_seconds"], "chain fresh setup")
    require(fresh >= 0, "negative fresh setup")
    tolerance = .1 * fresh
    flags = [finite(leaf["setup_seconds"], "chain resumed setup") > allowance + tolerance for leaf in leaves[1:]]
    consecutive = 0
    for flag in reversed(flags):
        if not flag:
            break
        consecutive += 1
    return {"allowance": allowance, "fresh": fresh, "tolerance": tolerance,
        "overruns": sum(flags), "consecutive_overruns": consecutive,
        "overrun": bool(flags and flags[-1]),
        "stopped": consecutive >= 2 or sum(flags) >= 3}


def screening_setup_guard(request, config, task, lock, history):
    """Refuse before continuation import or compute; diagnostics never authorize a budget."""
    if lock is None or 'setup_safety_net' not in lock.get('budget', {}):
        return {}
    from oxyformer.validation.campaign_probe import (
        require_admission, declared_publication_shape, compare_publication_shape)
    admission = require_admission(request, config, task, lock)
    allowance = admission['admission_measurements']['allowance_seconds']
    result = {'allowance_seconds': allowance}

    def refuse(reason, evidence):
        atomic_json(request.output_dir, BUDGET_RECORD, {'schema_version': 1,
            'diagnostic_only': True, 'status': reason, 'task_id': task['id'],
            'chain': task['continuation'], 'setup_safety': evidence,
            'authorizes_compute_or_reuse': False})
        require(False, reason + ': chain stopped before compute; ' + canonical_json(evidence))

    if history:
        safety = setup_overruns(history, allowance)
        if safety['stopped']:
            refuse('allowance-exceeded', safety)
        result['history'] = safety
        if task['continuation']['step'] == 1:
            expected = lock['budget']['expected_predecessor_shapes'][task['parameters']['scenario']['name']]
            observed = declared_publication_shape(request, config, task['continuation']['predecessor'])
            comparison = compare_publication_shape(expected, observed)
            result['admission_shape'] = comparison
            if comparison['status'] == 'admission-shape-mismatch':
                refuse('admission-shape-mismatch', comparison)
    return result


def publication_reserve(seconds):
    """Post-check publication uses part of the existing finalization reserve."""
    return min(30., seconds / 20)


def reserved_wall_limit(seconds):
    """One pre-publication wall limit for worker, runner and budget consumers."""
    return seconds - publication_reserve(seconds)


def check_worker_wall(completed, started, task):
    if time.monotonic() - started > reserved_wall_limit(task["parameters"]["wall_seconds"]):
        return replace(completed, status="fail", message="Coverage publication exceeded wall budget")
    return completed


OVERHEAD_PHASES = ("setup_seconds", "import_seconds", "export_seconds", "publication_seconds")


# Slot-0 Slurm 89833--89836 (2026-10-14): clone/environment/runner
# preparation took 94 s. D3 Slurm 89923/89980 measured at most 4.099 s
# from in-fold checkpoint return through sealed terminal publication.
# Both margins retain a conservative 300 s floor; see docs/EXECUTION.md.
LEAF_SETUP_SECONDS = 300
LEAF_FINALIZATION_SECONDS = 300
MAX_LEAF_ALLOCATION_SECONDS = 14400
BUDGET_FORMULA = "arc-2612-window"
# The final null-effect leaf (2026-10-19) had 809.556315 s excess per leaf
# while still unsealed; round upward from the required 800 s minimum.
RESUME_OVERHEAD_FLOOR_SECONDS = 810.


def leaf_allocation(stage_seconds):
    """Whole-minute allocation; finalization is inside the sealed stage wall."""
    require(0 < finite(stage_seconds, "leaf stage seconds"), "positive stage wall required")
    allocated = math.ceil((stage_seconds + LEAF_SETUP_SECONDS) / 60) * 60
    require(allocated <= MAX_LEAF_ALLOCATION_SECONDS, "leaf exceeds four hours including setup")
    return {"setup_seconds": LEAF_SETUP_SECONDS, "stage_seconds": stage_seconds,
            "finalization_seconds": LEAF_FINALIZATION_SECONDS, "allocation_seconds": allocated}


def maximum_leaf_stage_seconds():
    return MAX_LEAF_ALLOCATION_SECONDS - LEAF_SETUP_SECONDS


def leaf_budget(seconds, factor, overheads):
    """The immutable budget used by sizing, admission and chain execution."""
    require(0 < finite(seconds, "leaf wall seconds") <= 14400, "coverage slice exceeds four hours")
    require(finite(factor, "profile safety factor") >= 1, "invalid profile safety factor")
    require(set(overheads) == set(OVERHEAD_PHASES), "incomplete leaf overheads")
    require(all(finite(overheads[k], k) >= 0 for k in OVERHEAD_PHASES), "negative leaf overhead")
    reserve = LEAF_FINALIZATION_SECONDS
    capacity = (seconds - reserve) - math.fsum(overheads.values())
    require(capacity > 0, "infeasible batch budget: leaf overhead consumes slice")
    return {"wall_seconds": seconds, "reserve_seconds": reserve, "safety_factor": factor,
            "overheads": dict(overheads), "compute_seconds": capacity, "formula": BUDGET_FORMULA}


def validate_leaf_budget(budget):
    """Accept the old generation only where both formulas are identical."""
    expected = leaf_budget(budget["wall_seconds"], budget["safety_factor"], budget["overheads"])
    if "formula" not in budget:
        require(budget["safety_factor"] == 1.0 and budget["compute_seconds"] == expected["compute_seconds"],
                "leaf budget sealed under the pre-ARC-2612 window formula")
        expected.pop("formula")
    require(budget == expected, "leaf budget drift")
    return budget


def usable_leaf_compute(seconds, overheads, margin=None, *, resume_overhead_seconds=RESUME_OVERHEAD_FLOOR_SECONDS):
    """Sizing capacity before resume; one shared guard for every sizing caller."""
    from oxyformer.training.fit import checkpoint_margin_seconds, PHASE_UNIT_SECONDS, CHECKPOINT_WRITE_SECONDS
    margin = checkpoint_margin_seconds() if margin is None else finite(margin, "checkpoint margin")
    require(margin >= checkpoint_margin_seconds(), "insufficient checkpoint margin")
    partial = max(PHASE_UNIT_SECONDS.values())
    resume = finite(resume_overhead_seconds, "resume overhead")
    require(resume >= RESUME_OVERHEAD_FLOOR_SECONDS, "insufficient resume overhead")
    usable = leaf_budget(seconds, 1., overheads)["compute_seconds"] - margin - partial
    require(usable - resume >= partial + CHECKPOINT_WRITE_SECONDS,
            "infeasible batch budget: usable capacity cannot fit a phase unit and checkpoint")
    return usable


def read_leaf_measurements(path):
    """Authenticated measurements only: never grants execution budget authority."""
    from oxyformer.execution.integrity import verify_publication
    path = Path(path).resolve(strict=True)
    require(path.name == BUDGET_RECORD, "calibration requires a sealed budget record")
    root = path.parent
    terminal = StageResult.from_json((root / "_execution/result.json").read_text())
    require(terminal.status == "pass", "measurement publication did not pass")
    verify_publication(root, terminal)
    records = [a for a in terminal.artifacts if a.path == BUDGET_RECORD]
    require(len(records) == 1 and file_hash(path) == records[0].sha256, "measurement record hash mismatch")
    value = read_json(path)
    require(value.get("schema_version") == 1 and bool(value.get("leaves")), "invalid measurement record")
    own = value["leaves"][-1]
    require(own["task_id"] == value["binding"]["task_id"], "measurement leaf identity mismatch")
    require(all(finite(own[k], k) >= 0 for k in (*OVERHEAD_PHASES, "compute_seconds")),
            "negative leaf measurement")
    generation = (own.get("leaf_budget") or {}).get("formula", "pre-ARC-2612")
    return {**value, "record_hash": records[0].sha256, "budget_generation": generation}


def remaining_compute(budget, **measured):
    """Charge positive phase overruns once; never credit an underrun."""
    return max(0., budget["compute_seconds"] - math.fsum(
        max(0., seconds - budget["overheads"][phase]) for phase, seconds in measured.items()))


def uses_continuation(task):
    """Match execution routing, including archive-free generic step zero."""
    return ("continuation.tar" in task.get("outputs", ()) or
            task.get("continuation", {}).get("predecessor") is not None)


def task_leaf_budget(task):
    """The producer task authorizes the contract; measurements cannot replace it."""
    parameters = task["parameters"]
    seconds = parameters["wall_seconds"]
    budget = parameters.get("leaf_budget")
    locked = bool(task.get("recipe_lock"))
    require(not locked or budget is not None, "locked chain requires leaf budget")
    if budget is not None:
        validate_leaf_budget(budget)
        require(budget["wall_seconds"] == seconds, "leaf budget wall drift")
    else:
        budget = leaf_budget(seconds, 1., {k: 0. for k in OVERHEAD_PHASES})
    ceiling = parameters.get("compute_ceiling_seconds", budget["compute_seconds"])
    require(0 < finite(ceiling, "compute ceiling") <= budget["compute_seconds"], "invalid compute ceiling")
    require(not locked or ceiling == budget["compute_seconds"], "locked compute ceiling drift")
    return {**budget, "compute_seconds": ceiling}


def budget_binding(request, task, recipe_hash):
    parameters = task["parameters"]
    return {"request_hash": request.content_hash, "wall_seconds": parameters["wall_seconds"],
        "task_id": task["id"], "chain": task.get("continuation"),
        "recipe_hash": recipe_hash, "lock_hash": task.get("recipe_lock", {}).get("sha256"),
        "draws": parameters["draws"], "scenario": parameters["scenario"],
        "mode": parameters["mode"], "code_identity": request.code_identity, **output_metadata(task)}


def record_measurements(request, completed, started, task, timing, summary):
    """Worker measurements only. They cannot authorize any downstream decision."""
    value = {**output_metadata(task), "binding": budget_binding(request, task, timing["recipe_hash"]),
        "started_monotonic": started, "timing": timing,
        "counts": summary["counts"], "received": summary["received_repetitions"]}
    path = atomic_json(request.output_dir, MEASUREMENTS, value)
    completed = replace(completed, artifacts=(*completed.artifacts,
        ArtifactRecord(path=MEASUREMENTS, kind="coverage_measurements", sha256=file_hash(path),
            lineage=completed.artifacts[0].lineage)))
    return check_worker_wall(completed, started, task)


def load_budget(request, config, unit, *, expected_task=None):
    """The only budget-evidence loader: declared request, sealed pass, bound hash.

    Diagnostic result/timing/runtime files have no budget authority. Scientific
    payloads may be read by callers only after this returns successfully.
    """
    from oxyformer.execution.integrity import verify_publication
    def declared(name):
        path = dependency(request, config, {"dependency": unit, "path": name})
        expected = dict(zip(request.dependency_paths, request.dependency_hashes))[str(path)]
        require(file_hash(path) == expected, "budget dependency hash mismatch: " + name)
        return path
    producer = StageRequest.from_json(declared("_execution/request.json").read_text())
    root = Path(config["dependencies"][unit]).resolve(strict=True)
    require(Path(producer.output_dir).resolve() == root, "budget producer owner mismatch")
    task_path = declared("_execution/task.json")
    require(file_hash(task_path) == producer.task_hash, "budget producer task mismatch")
    task = read_json(task_path)
    require(task["id"] == unit, "budget producer identity mismatch")
    if expected_task is not None:
        require(task == expected_task, "budget expected task mismatch")
    terminal = StageResult.from_json(declared("_execution/result.json").read_text())
    require(terminal.status == "pass", "chain stopped: allowance-exceeded" if 'allowance-exceeded' in terminal.message
        else "budget stage did not pass")
    require(terminal.request_hash == producer.content_hash, "budget request mismatch")
    verify_publication(root, terminal)
    records = [a for a in terminal.artifacts if a.path == BUDGET_RECORD]
    require(len(records) == 1, "runner-finalized budget record missing")
    path = declared(BUDGET_RECORD)
    require(file_hash(path) == records[0].sha256, "budget record hash mismatch")
    value = read_json(path)
    binding = value["binding"]
    require(binding == budget_binding(producer, task, binding["recipe_hash"]), "budget binding mismatch")
    if "recipe" in task["parameters"]:
        require(binding["recipe_hash"] == digest(task["parameters"]["recipe"]), "budget recipe mismatch")
    require(value["schema_version"] == 1, "unknown budget record schema")
    wall = finite(binding["wall_seconds"], "budget wall binding")
    require(wall > 0, "invalid budget wall binding")
    leaves = value["leaves"]
    require(bool(leaves), "missing budget measurements")
    chain = binding["chain"]
    chained = uses_continuation(task)
    require(value["chained"] is chained, "budget execution mode mismatch")
    contract = task_leaf_budget(task) if chained else None
    if chained:
        require(chain is not None and len(leaves) == chain["step"] + 1
            and [row["step"] for row in leaves] == list(range(len(leaves)))
            and leaves[-1]["task_id"] == task["id"], "partial budget chain")
    else:
        require(len(leaves) == 1 and leaves[0]["task_id"] == task["id"]
            and leaves[0]["step"] == 0, "invalid single-leaf budget")
    for leaf in leaves:
        names = ("compute_seconds", *OVERHEAD_PHASES)
        require(all(finite(leaf[k], k) >= 0 for k in names), "negative budget measurement")
        elapsed = finite(leaf["wall_seconds"], "measured budget wall")
        require(math.isclose(math.fsum(leaf[k] for k in names), elapsed, rel_tol=1e-6, abs_tol=.01),
            "budget accounting mismatch")
        require(leaf["post_check_reserve_seconds"] == publication_reserve(wall), "publication reserve drift")
        require(elapsed <= reserved_wall_limit(wall), "coverage leaf exceeded admitted budget")
        observed = leaf.get("leaf_budget")
        if observed is not None and "formula" not in observed:
            validate_leaf_budget(observed)
        comparable = dict(contract) if contract is not None else None
        if comparable is not None and observed is not None and "formula" not in observed:
            comparable.pop("formula", None)
        require(observed == comparable, "budget contract differs from producer task")
        if contract is not None:
            phases = {k: leaf[k] for k in OVERHEAD_PHASES} if task["parameters"].get("leaf_budget") is not None else {
                k: leaf[k] for k in ("setup_seconds", "import_seconds")}
            require(leaf["compute_seconds"] <= remaining_compute(contract, **phases),
                "coverage compute exceeded admitted budget")
    counts = value["counts"]
    require(set(counts) == STATUSES and all(type(v) is int and v >= 0 for v in counts.values()),
        "invalid budget outcome counts")
    require(sum(counts.values()) == value["received"] <= len(binding["draws"]), "budget outcome count mismatch")
    complete = value["received"] == len(binding["draws"]) and counts["incomplete"] == 0
    successful = complete and counts["success"] == len(binding["draws"])
    payloads = [a.sha256 for a in terminal.artifacts if a.path == "result.json"]
    require(len(payloads) == 1, "coverage scientific payload missing")
    payload_path = declared("result.json")
    require(file_hash(payload_path) == payloads[0], "coverage scientific payload hash mismatch")
    # Reconcile the finalized counters with the actual scientific outcomes,
    # never with diagnostic summary flags. This check belongs to the same
    # loader so admission, continuation and reuse have one consistent verdict.
    scientific = read_json(payload_path)
    require(scientific["scenario"] == SCMConfig(**binding["scenario"]).to_dict()["payload"],
        "budget scientific scenario mismatch")
    require(scientific["mode"] == binding["mode"], "budget scientific mode mismatch")
    require(scientific["recipe_hash"] == binding["recipe_hash"], "budget scientific recipe mismatch")
    scientific_records = scientific["records"]
    scientific_counts = {status: sum(row["status"] == status for row in scientific_records)
        for status in STATUSES}
    require(counts == scientific_counts and value["received"] == len(scientific_records),
        "budget counts disagree with scientific outcomes")
    # Stored complete/success/eligibility flags are never consulted.
    return {**value, "complete": complete, "all_successful": successful,
        "wall_seconds": math.fsum(row["wall_seconds"] for row in leaves),
        "record_hash": file_hash(path), "payload_hash": payloads[0]}


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


def validate_success(record, *, null_scenario):
    """Validate numerical output before it can become an atomic success record."""
    finite(record["truth"], "observed-law truth")
    finite(record["causal_truth"], "structural truth")
    error = finite(record["truth_integration_error"], "truth-integration error")
    require(error >= 0, "negative truth-integration error")
    require(set(record["estimates"]) == set(METHODS), "incomplete estimator results")
    for estimate in record["estimates"].values():
        finite(estimate["value"], "estimate")
        require(finite(estimate["se"], "standard error") >= 0, "negative standard error")
        for se in estimate.get("spatial_se", {}).values():
            require(finite(se, "spatial standard error") >= 0, "negative spatial standard error")
    if null_scenario:
        require(abs(record["truth"]) <= error, "null scenario has nonzero observed-law truth")


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
            validate_success(record, null_scenario=null_scenario)
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


def output_metadata(task):
    from oxyformer.validation.campaign import screening_metadata
    recipe = task.get("parameters", {}).get("recipe", task.get("parameters", {}))
    return screening_metadata(task if task.get("campaign_mode") == "screening" else recipe)


def publish(request, values, *, status, message, extra_artifacts=()):
    metadata = output_metadata(read_json(request.task_path))
    values = {name: {**value, **metadata} for name, value in values.items()}
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
            "artifacts": [r.to_dict() for r in records], **metadata})
        records.append(ArtifactRecord(path="artifact_manifest.json", kind="artifact_manifest", sha256=file_hash(path), lineage=lineage))
    result = StageResult(request_hash=request.content_hash, status=status, artifacts=tuple(records), message=message)
    result.verify(request)
    return result


def fixed_exposures(template, frame):
    """Atlas A for unsupported training-only rows; no serialized API change."""
    support = dict(template.policy.components_by_key)
    training = {i for i, role in zip(template.outer.frame_original_ids, template.outer.roles)
                if role == "training"}
    exposures = dict(zip(template.data.manifest.original_ids,
                         template.data.column(template.data.manifest.exposure_field)))
    fixed = {i: exposures[i] for i, key in zip(frame.original_ids, frame.support_keys)
             if key in support and not support[key]}
    require(set(fixed) <= training, "only labelled training-only rows may have fixed exposure")
    return fixed


def bind_observations(template, observations):
    """Rebind a reviewed synthetic endpoint to one draw, preserving X/design.

    This adapter handles fixed-X, fixed-split identity endpoints. Selection or
    additional measured confounders require their own reviewed endpoint adapter;
    they cannot silently change the frozen population or predictor registry.
    """
    frame = observations.frame
    require(template.data.manifest.spec.endpoint.startswith("synthetic"), "synthetic endpoint template required")
    require(template.family == "identity", "SCM adapter requires identity likelihood")
    targets = set(template.outer.original_ids)
    require(tuple(i for i in frame.original_ids if i in targets) == template.outer.original_ids,
            "SCM frame target rows must equal frozen inference target")
    training = tuple(i for i, role in zip(template.outer.frame_original_ids, template.outer.roles)
                     if role == "training")
    require(tuple(i for i in frame.original_ids if i not in targets) == training,
            "SCM frame training rows must equal labelled training-only rows")
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
    fixed = fixed_exposures(template, frame)
    for index, oid in enumerate(frame.original_ids):
        row = values[oid]
        require(tuple(row[n] for n in features) == frame.x[index], "fixed-X template drift")
        require(policy_rows[oid] == (frame.geography_ids[index], frame.support_keys[index]), "SCM policy routing drift")
        weight = (1. if manifest.weight_field is None else row[manifest.weight_field]) if oid in targets else 0.
        require(weight == frame.weights[index], "SCM weight drift")
        require(geography[oid].county == frame.region_ids[index], "SCM region/county mismatch")
        require((geography[oid].latitude, geography[oid].longitude) == frame.coordinates[index], "SCM geography drift")
        if oid in fixed:
            require(observations.a[index] == fixed[oid], "fixed exposure differs from atlas value")
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


def estimate_repetition(endpoint, recipe, root, deadline, *, study=None, continuation=None):
    """All five folds with nested tuning; study overrides require explicit use."""
    from oxyformer.estimation.mtp import one_step
    from oxyformer.estimation.targeting import cv_tmle
    from oxyformer.estimation.covariance import align_estimates, spatial_sensitivities
    from oxyformer.training.fit import DeadlineTiming, RecipeStudy, subset

    study = RecipeStudy() if study is None else study
    require(type(study) is RecipeStudy, "RecipeStudy required")
    # Only chained leaves can resume a deliberately early checkpoint. Keep
    # the existing admission margin for archive-free complete repetitions.
    timing = None if continuation is None else continuation.get("deadline_timing") or DeadlineTiming(deadline)
    seeds = study.seeds
    scoring_split = endpoint.outer if study.seed_count == 3 else replace(endpoint.outer, seed_ids=seeds)
    options = nested_cv._settings(recipe, endpoint)
    parts = ([] if continuation is None else
             [OOFNuisances.from_json(value) for value in continuation["parts"]])
    require(len(parts) <= 5 * len(seeds), "too many prediction parts")
    position = 0
    for fold in range(5):
        for neural_seed in seeds:
            position += 1
            if position <= len(parts):
                part = parts[position - 1]
                require(set(part.fold_ids) == {fold} and set(part.seed_ids) == {neural_seed},
                        "reordered prediction parts")
                continue
            # Include configuration/identity/predecessor preparation in the
            # interval up to the nested controller's first cooperative check.
            if timing is not None:
                timing.boundary("nuisance")
            remaining = min(14400., deadline - time.monotonic())
            if remaining <= (timing.margin_seconds if timing is not None else 0.):
                if timing is not None:
                    timing.finish_unit()
                return None
            margin = timing.margin_seconds if timing is not None else min(1., remaining / 10)
            predecessor = None if continuation is None else continuation.get("predecessor")
            config = endpoint.configuration(fold, root / f"fold-{fold}-seed-{neural_seed}",
                **options, study=study, predecessor=predecessor,
                slice_seconds=remaining, checkpoint_margin_seconds=margin, deadline_timing=timing)
            artifact = nested_cv.run_fold(config, endpoint.outer, neural_seed, geography=endpoint.geography)
            if not artifact.complete:
                if continuation is not None:
                    continuation["predecessor"] = artifact
                return None
            if continuation is not None:
                continuation["pending"] = None
                continuation["predecessor"] = None
            view = subset(endpoint.data.covariates(tuple(n for n, _ in endpoint.feature_kinds)),
                          artifact.prediction_inputs.original_ids)
            if timing is not None:
                timing.boundary("evaluation")
            parts.append(nested_cv.predict(artifact, view, endpoint.policy))
            if timing is not None:
                timing.finish_unit()
            if continuation is not None:
                continuation["parts"].append(parts[-1].to_json())
    names = ("original_ids", "fold_ids", "seed_ids", "mu_a", "mu_d", "r_a", "r_d", "origin_weights")
    values = {name: tuple(v for part in parts for v in getattr(part, name)) for name in names}
    lineage = replace(parts[0].lineage, unit_ids=endpoint.outer.original_ids,
        split_hash=scoring_split.content_hash, parent_hashes=(endpoint.data.manifest.content_hash,
        *(part.content_hash for part in parts)), model_hash=None, seed=None, parameter_count=None)
    nuisances = OOFNuisances(spec=endpoint.data.manifest.spec, lineage=lineage, **values)
    if continuation is not None:
        atomic_write(root, "nuisances.json", nuisances.to_json())
    weights = dict(zip(endpoint.data.manifest.original_ids,
        (1.,) * len(endpoint.data.rows) if endpoint.data.manifest.weight_field is None
        else endpoint.data.column(endpoint.data.manifest.weight_field)))
    weights = {oid: weights[oid] for oid in endpoint.outer.original_ids}
    common = dict(split=scoring_split, policy=endpoint.policy)
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


class NumericalFailure(Exception):
    """A numerical computation failed, distinct from adapter or I/O faults."""


def numerical_call(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except (ArithmeticError, np.linalg.LinAlgError, ValueError) as exc:
        raise NumericalFailure(f"{type(exc).__name__}: {exc}") from exc


def execute_draw(draw, frame, scenario, template, recipe, root, deadline, *, continuation=None, study=None):
    started = time.monotonic()
    record = {"draw": draw, "status": "incomplete", "reason": "budget exhausted", "estimates": {}}
    if started >= deadline:
        return {**record, "wall_seconds": 0.}
    try:
        fixed = fixed_exposures(template, frame)
        options = {"fixed_exposures": fixed} if fixed else {}
        if continuation is not None and continuation.get("sample"):
            from oxyformer.validation.generators import (GeneratedSample, ObservedRecords,
                TruthArtifact, IntegrationUncertainty)
            saved = continuation["sample"]
            sample = GeneratedSample(ObservedRecords.from_json(saved["observations"]),
                TruthArtifact.from_json(saved["observed_law_truth"]),
                TruthArtifact.from_json(saved["structural_causal_truth"]),
                IntegrationUncertainty.from_json(saved["integration_uncertainty"]))
        else:
            sample = numerical_call(generate_suite_a, frame, scenario, template.policy, seed=draw["seed"], **options)
            if continuation is not None:
                continuation["sample"] = {name: getattr(sample, name).to_json() for name in
                    ("observations", "observed_law_truth", "structural_causal_truth", "integration_uncertainty")}
        if continuation is not None:
            expected = {**continuation["base_binding"], "draw": draw,
                "observation_hash": sample.observations.content_hash}
            require(continuation.get("binding", expected) == expected, "draw/observation binding drift")
            continuation["binding"] = expected
        if sample.observed_law_truth.status in ("design_rejected", "empty_target"):
            record.update(status="eligibility_rejection", reason=sample.observed_law_truth.reason)
        else:
            # Adapter consistency and available truth are prerequisites, not
            # estimates of numerical failure probability. Keep their failures
            # outside the numerical-operation boundary.
            endpoint = bind_observations(template, sample.observations)
            uncertainty = sample.integration_uncertainty
            require(uncertainty.converged, "truth integration diagnostics are not converged")
            components = [finite(getattr(uncertainty, name), "truth error " + name) for name in
                ("observed_absolute_difference", "quadrature_tail_absolute_bound", "truth_serialization_absolute_bound")]
            require(all(value >= 0 for value in components), "negative truth-integration error component")
            error = numerical_call(math.fsum, components)
            extra = {} if continuation is None else {"continuation": continuation, "study": study}
            estimates = numerical_call(estimate_repetition, endpoint, recipe, root, deadline, **extra)
            successful = None
            if estimates is not None:
                require(set(estimates) == set(METHODS), "incomplete estimator results")
                successful = {**record, "status": "success", "reason": "complete production procedure",
                    "estimates": estimates, "truth": sample.observed_law_truth.value,
                    "causal_truth": sample.structural_causal_truth.value, "truth_integration_error": error,
                    "integration_uncertainty": uncertainty.to_dict(), "observation_hash": sample.observations.content_hash}
                numerical_call(validate_success, successful, null_scenario=scenario.effect == "null")
            atomic_write(root, "observations.json", sample.observations.to_json())
            atomic_write(root, "observed_law_truth.json", sample.observed_law_truth.to_json())
            atomic_write(root, "structural_causal_truth.json", sample.structural_causal_truth.to_json())
            # A failed publication never leaves estimates in a failure record.
            if successful is not None:
                record = successful
    except NumericalFailure as exc:
        record.update(status="numerical_failure", reason=str(exc))
    except Exception as exc:
        # No retry or substitution. Cancellation escapes; ordinary contract,
        # adapter and publication faults remain incomplete and forbid passage.
        record.update(status="incomplete", reason=f"execution failure: {type(exc).__name__}: {exc}")
    record["wall_seconds"] = time.monotonic() - started
    return record


def prepare_batch(request: StageRequest):
    """Validate the exact leaf inputs without generating outcomes or fitting.

    Campaign admission measures this same path using the fully expanded lock
    and the requested leaf size; unlocked profiles cannot supply its cost.
    """
    from oxyformer.validation.campaign import load_lock, validate_recipe, validate_leaf_task, fingerprint
    request.verify_inputs()
    require(request.stage in ("coverage", "simulation-smoke"), "unsupported coverage stage")
    config, task = yaml.safe_load(Path(request.config_path).read_text()), read_json(request.task_path)
    require(task["stage"] == request.stage, "stage mismatch")
    parameters = task["parameters"]
    mode = parameters["mode"]
    if "device" in parameters:
        from oxyformer.training.fit import resolve_device
        require(resolve_device() == resolve_device(parameters["device"]),
                "runtime differs from task required device")
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
    from oxyformer.validation.campaign import screening_metadata, validate_screening
    if screening_metadata(recipe):
        decision = validate_screening(recipe, config.get("approvals"))
        require(mode in ("profile", "screening"), "screening cell cannot certify final coverage")
        require(parameters.get("study") == decision["recipe"], "screening task study drift")
        require(parameters.get("partition") == "standard", "screening partition must be standard")
        require(uses_continuation(task), "screening requires coverage continuation")
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
    stamps = ({k: lock[k] for k in ("scientific_fingerprint", "environment_hash")}
              if lock else fingerprint())
    return task, recipe, lock, template, frame, scenario, seconds, stamps


def _run_batch(request: StageRequest, started: float) -> StageResult:
    try:
        task, recipe, lock, template, frame, scenario, seconds, stamps = prepare_batch(request)
        parameters = task["parameters"]
        mode, draws = parameters["mode"], parameters["draws"]
        production = mode != "smoke"
        from oxyformer.training.fit import resolve_device, fit_environment
        device = resolve_device()
        bounded_profile = mode == "profile" and device.type == "cuda"
        root = Path(request.output_dir)
        setup_seconds = time.monotonic() - started
        records, repetition_seconds = [], []
        for draw in draws:
            draw_started = time.monotonic()
            destination = output_path(root, "repetitions/" + draw["repetition_id"])
            require(not destination.exists(), "repetition was already attempted; use declared continuation, never redraw")
            destination.mkdir(parents=True)
            # CPU profiles and locked leaves retain planning-only budgets.
            # CUDA probes must finish within their bounded GPU allocation.
            deadline = started + reserved_wall_limit(seconds) if mode == "smoke" or bounded_profile else math.inf
            record = execute_draw(draw, frame, scenario, template, recipe, destination, deadline)
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
        if device.type == "cuda":
            import torch
            torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        timing = {"wall_seconds": elapsed, "gpu_seconds": elapsed if device.type == "cuda" else 0.,
            "device": str(device), "environment": dict(fit_environment(device)),
            "gpu_seconds_scope": "single_device_elapsed_not_kernel_utilization",
            "measurement_scope": "before_final_publication", "setup_seconds": setup_seconds,
            "admitted_budget_seconds": seconds,
            "production_equivalent": production, "recipe_hash": digest(recipe),
            "complete_repetition_seconds": [seconds for r, seconds in zip(records, repetition_seconds) if r["status"] == "success"],
            "complete": summary["complete"], "all_successful": summary["counts"]["success"] == len(draws), **stamps}
        completed = publish(request, {"result.json": result, "timing.json": timing},
            status="pass" if summary["complete"] else "fail",
            message="Declared repetitions accounted for; production coverage requires complete campaign collection"
                if summary["complete"] else "Incomplete batch; all declared draws retained")
        return record_measurements(request, completed, started, task, timing, summary)
    except (ContractError, KeyError, TypeError, ValueError, OSError, tarfile.TarError) as exc:
        return StageResult(request_hash=request.content_hash, status="blocked", artifacts=(), message=str(exc) or type(exc).__name__)



def record_runtime(request, completed, started, task, *, budget_exceeded=False):
    """Record the measured complete batch, including publication/verification.

    This small telemetry receipt is emitted after measurement; its own write
    and the outer executor are explicitly outside the measured interval.
    Profiles additionally time this receipt, so admission includes its cost.
    """
    if not completed.artifacts:
        return completed
    parameters = task["parameters"]
    budget = finite(parameters["wall_seconds"], "admitted budget")
    aggregate = read_json(Path(request.output_dir) / "result.json")
    elapsed = time.monotonic() - started
    runtime = {**output_metadata(task), "measurement_scope": "complete_batch_return", "wall_seconds": elapsed,
        "admitted_budget_seconds": budget, "budget_exceeded": budget_exceeded or elapsed > budget,
        "overrun_seconds": max(0., elapsed - budget), "batch_id": parameters.get("batch_id"),
        "lock_hash": aggregate["lock_hash"], "request_hash": completed.request_hash,
        "excluded": ["runtime receipt publication", "outer executor and scheduler"]}
    path = atomic_json(request.output_dir, "runtime.json", runtime)
    artifact = ArtifactRecord(path="runtime.json", sha256=file_hash(path),
                              lineage=completed.artifacts[0].lineage, kind="runtime")
    message = completed.message
    if runtime["budget_exceeded"]:
        message += f"; planning budget overrun {runtime['overrun_seconds']:.6f}s (draws retained)"
    return replace(completed, artifacts=(*completed.artifacts, artifact), message=message)



def _chain(task):
    chain = task["continuation"]
    require(type(chain) is dict and set(chain) == {"owner", "step", "predecessor"}
            and isinstance(chain["owner"], str) and bool(chain["owner"])
            and type(chain["step"]) is int and chain["step"] >= 0, "invalid coverage chain")
    require((chain["step"] == 0) == (chain["predecessor"] is None), "missing coverage predecessor")
    return chain


def _read_continuation(path, binding, chain):
    """Read only digest-named blobs; never follow paths from an earlier attempt."""
    with tarfile.open(path, "r:") as archive:
        members = archive.getmembers()
        require(len({m.name for m in members}) == len(members), "duplicate coverage member")
        require(sum(m.size for m in members) <= 10 * 1024**3, "coverage continuation exceeds byte limit")
        require(all(m.isfile() and (m.name == "coverage.json" or
                    (m.name.startswith("blobs/") and len(m.name) == 70 and
                     all(c in "0123456789abcdef" for c in m.name[6:]))) for m in members),
                "unexpected coverage member")
        blobs = {m.name: archive.extractfile(m).read() for m in members}
    header = json.loads(blobs.pop("coverage.json"))
    require(header["version"] == 1 and header["binding"] == binding, "coverage binding mismatch")
    previous = header["chain"]
    require(header["task_id"] == chain["predecessor"] and previous["owner"] == chain["owner"]
            and previous["step"] + 1 == chain["step"], "nonconsecutive coverage predecessor")
    require(all(sha256(value).hexdigest() == name[6:] for name, value in blobs.items()),
            "tampered coverage blob")
    used = set()
    def read(key):
        name = "blobs/" + key
        used.add(name)
        return blobs[name]
    states = header["states"]
    require(len(states) == len(binding["draws"]), "coverage draw count drift")
    from oxyformer.validation.generators import ObservedRecords
    from oxyformer.training.fit import RecipeStudy
    study = RecipeStudy.from_json(canonical_json(binding["study"]))
    order = [(fold, seed) for fold in range(5) for seed in study.seeds]
    for draw, state in zip(binding["draws"], states):
        state["parts"] = [read(key).decode() for key in state["parts"]]
        state["sample"] = {name: read(key).decode() for name, key in state["sample"].items()}
        state["pending"] = read(state["pending"]) if state["pending"] else None
        require(len(state["parts"]) <= len(order), "too many prediction parts")
        for value, (fold, seed) in zip(state["parts"], order):
            part = OOFNuisances.from_json(value)
            require(set(part.fold_ids) == {fold} and set(part.seed_ids) == {seed}, "reordered prediction parts")
        if state["sample"]:
            observations = ObservedRecords.from_json(state["sample"]["observations"])
            expected = {**binding, "draw": draw, "observation_hash": observations.content_hash}
            require(state["binding"] == expected, "draw/observation binding drift")
        else:
            require(not state["parts"] and not state["pending"], "missing saved observations")
    require(used == set(blobs), "unreferenced coverage blob")
    return states


def _write_continuation(path, states, *, binding, chain, task_id):
    blobs, saved = {}, []
    def add(value):
        value = value.encode() if isinstance(value, str) else value
        key = sha256(value).hexdigest()
        blobs["blobs/" + key] = value
        return key
    for state in states:
        saved.append({"parts": [add(value) for value in state["parts"]],
            "sample": {name: add(value) for name, value in state["sample"].items()},
            "pending": add(state["pending"]) if state["pending"] else None,
            "binding": state.get("binding"), "record": state.get("record"),
            "compute_seconds": state["compute_seconds"]})
    header = {"version": 1, "binding": binding, "task_id": task_id, "chain": chain, "states": saved}
    blobs["coverage.json"] = canonical_json(header).encode()
    with Path(path).open("xb") as stream, tarfile.open(fileobj=stream, mode="w") as archive:
        for name, value in sorted(blobs.items()):
            member = tarfile.TarInfo(name)
            member.size, member.mode, member.mtime = len(value), 0o400, 0
            archive.addfile(member, BytesIO(value))


def _run_chain(request, started):
    """One bounded leaf of a prospectively declared coverage work item."""
    from oxyformer.training.fit import DeadlineTiming, RecipeStudy, resolve_device, fit_environment
    task, recipe, lock, template, frame, scenario, seconds, stamps = prepare_batch(request)
    chain = _chain(task)
    parameters, root = task["parameters"], Path(request.output_dir)
    require("continuation.tar" in task["outputs"], "coverage continuation must be declared")
    require(seconds <= 14400, "coverage slice exceeds four hours")
    study = RecipeStudy(**parameters.get("study", {}))
    require(parameters["mode"] != "final" or study == RecipeStudy(), "final recipe cannot use study overrides")
    # Locked budgets were authenticated by prepare_batch against the lock and
    # expansion. Bootstrap profiles need no measurements: their checkpoint
    # reserve covers finalization, and an optional fixed ceiling can cut them
    # earlier. Neither mode may grow its compute allowance across successors.
    budget = task_leaf_budget(task)
    binding = {"draws": parameters["draws"], "scenario": scenario.to_dict()["payload"],
        "endpoint_hash": template.content_hash, "frame_hash": frame.content_hash,
        "recipe_hash": digest(recipe), "study": study.to_dict(), "lock_hash": digest(lock) if lock else None,
        "code_identity": request.code_identity, "mode": parameters["mode"], "wall_seconds": seconds,
        "leaf_budget": budget, **stamps}
    for name in ("continuation.tar", "repetitions", "result.json", "timing.json"):
        require(not output_path(root, name).exists(), "coverage attempt already exists: " + name)
    config = yaml.safe_load(Path(request.config_path).read_text())
    imported_at = time.monotonic()
    setup = imported_at - started
    history = []
    setup_safety = {}
    if chain["predecessor"] is None:
        setup_safety = screening_setup_guard(request, config, task, lock, history)
        states = [{"parts": [], "sample": {}, "pending": None, "record": None, "compute_seconds": 0.}
                  for _ in parameters["draws"]]
    else:
        previous = load_budget(request, config, chain["predecessor"])
        require(previous["binding"]["chain"]["owner"] == chain["owner"]
            and previous["binding"]["chain"]["step"] + 1 == chain["step"], "budget chain drift")
        require(previous["binding"]["wall_seconds"] == seconds, "budget wall binding drift")
        history = previous["leaves"]
        setup_safety = screening_setup_guard(request, config, task, lock, history)
        archive = dependency(request, config, {"dependency": chain["predecessor"], "path": "continuation.tar"})
        states = _read_continuation(archive, binding, chain)
    destinations = []
    for draw, state in zip(parameters["draws"], states):
        destination = output_path(root, "repetitions/" + draw["repetition_id"])
        destination.mkdir(parents=True)
        destinations.append(destination)
        state.update(base_binding=binding, chain=chain, task_id=task["id"], predecessor=None)
        if state.get("binding") is None:
            state.pop("binding", None)
        if state["pending"]:
            imported = destination / "imported-fold"
            imported.mkdir()
            (destination / "pending.tar").write_bytes(state["pending"])
            state["predecessor"] = nested_cv.import_continuation(destination / "pending.tar", imported,
                binding=state["binding"], chain=chain)
    compute_at = time.monotonic()
    import_seconds = compute_at - imported_at
    capacity = remaining_compute(budget, setup_seconds=setup, import_seconds=import_seconds)
    deadline = min(compute_at + capacity, started + reserved_wall_limit(seconds))
    deadline_timing = DeadlineTiming(deadline)
    for state in states:
        state["deadline_timing"] = deadline_timing
    for draw, state, destination in zip(parameters["draws"], states, destinations):
        if state["record"] is not None:
            require(state["record"]["draw"] == draw and state["record"]["status"] != "incomplete",
                    "terminal draw drift")
            continue
        if time.monotonic() >= deadline:
            break
        draw_at = time.monotonic()
        record = execute_draw(draw, frame, scenario, template, recipe, destination, deadline,
                              continuation=state, study=study)
        state["compute_seconds"] += time.monotonic() - draw_at
        if record["status"] == "incomplete":
            require(record["reason"] == "budget exhausted", record["reason"])
            break
        state["record"] = record
        atomic_json(destination, "result.json", record)
    device = resolve_device()
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)
    export_at = time.monotonic()
    compute_seconds = export_at - compute_at
    atomic_json(root, "deadline-timing.json", deadline_timing.report())
    require(compute_seconds <= capacity, "coverage compute exceeded admitted budget; no usable leaf")
    for state, destination in zip(states, destinations):
        if state["predecessor"] is not None:
            path = destination / "next-fold.tar"
            nested_cv.export_continuation(state["predecessor"], path, binding=state["binding"],
                task_id=task["id"], chain=chain, allowed_root=root)
            state["pending"] = path.read_bytes()
    _write_continuation(root / "continuation.tar", states, binding=binding, chain=chain, task_id=task["id"])
    export_seconds = time.monotonic() - export_at
    records = [s["record"] for s in states if s["record"] is not None]
    summary = summarize(records, parameters["draws"],
        production_equivalent=parameters["mode"] != "smoke" and study == RecipeStudy(),
        null_scenario=scenario.effect == "null", expected_rejection=scenario.assignment == "atoms")
    result = {"schema_version": 1, "mode": parameters["mode"], "recipe_hash": digest(recipe),
        "lock_hash": binding["lock_hash"], "batch_id": parameters.get("batch_id"),
        "scenario": binding["scenario"], "draws": parameters["draws"], "records": records,
        "summary": summary, "chain": chain, "terminal": summary["complete"],
        "certifies_production_coverage": False}
    lineage = ArtifactLineage(source_hashes=(request.config_hash, request.task_hash),
        unit_ids=(request.content_hash,), parent_hashes=request.dependency_hashes, split_hash=None,
        config_hash=request.config_hash, model_hash=None, environment=(("code_identity", request.code_identity),),
        seed=None, parameter_count=None)
    archive_record = ArtifactRecord(path="continuation.tar", kind="coverage_continuation",
        sha256=file_hash(root / "continuation.tar"), lineage=lineage)
    completed = publish(request, {"result.json": result}, status="pass",
        message="Coverage work complete" if summary["complete"] else "Coverage slice checkpointed; repetition incomplete",
        extra_artifacts=(archive_record,))
    environment = dict(fit_environment(device))
    elapsed = time.monotonic() - started
    # Attribute final aggregation, archive hashing and publication together;
    # every measured second belongs to exactly one phase.
    publication_seconds = elapsed - setup - import_seconds - compute_seconds - export_seconds
    leaf = {"task_id": task["id"], "step": chain["step"], "compute_seconds": compute_seconds,
        "setup_seconds": setup, "import_seconds": import_seconds, "export_seconds": export_seconds,
        "publication_seconds": publication_seconds, "wall_seconds": elapsed,
        "compute_capacity_seconds": capacity, "leaf_budget": budget,
        "budget_exceeded": elapsed > seconds or (parameters.get("leaf_budget") is not None and
            compute_seconds > remaining_compute(budget, setup_seconds=setup, import_seconds=import_seconds,
                export_seconds=export_seconds, publication_seconds=publication_seconds))}
    if setup_safety:
        leaf['setup_safety'] = setup_safety
    leaves = [*history, leaf]
    exceeded = leaf["budget_exceeded"]
    timing = {"chain": chain, "leaves": leaves, "wall_seconds": math.fsum(v["wall_seconds"] for v in leaves),
        "gpu_seconds": math.fsum(v["wall_seconds"] for v in leaves) if device.type == "cuda" else 0.,
        "device": str(device), "environment": environment,
        "measurement_scope": "complete_chain_return", "excluded": ["timing/runtime receipt publication", "outer executor and scheduler"],
        "setup_seconds": math.fsum(v["setup_seconds"] for v in leaves),
        "publication_verification_seconds": math.fsum(v["publication_seconds"] for v in leaves),
        "admitted_budget_seconds": seconds,
        "production_equivalent": parameters["mode"] != "smoke" and study == RecipeStudy(),
        "study": study.to_dict(), **output_metadata(task),
        "complete_repetition_seconds": [s["compute_seconds"] for s in states if s["record"] is not None
                                        and s["record"]["status"] == "success"],
        "complete": summary["complete"] and not exceeded, "budget_exceeded": exceeded,
        "all_successful": summary["counts"]["success"] == len(states),
        "recipe_hash": digest(recipe), **stamps}
    path = atomic_json(root, "timing.json", timing)
    completed = replace(completed, artifacts=(*completed.artifacts,
        ArtifactRecord(path="timing.json", kind="timing", sha256=file_hash(path), lineage=lineage)))
    if exceeded:
        completed = replace(completed, status="fail", message="Coverage budget exceeded; no usable profile or leaf")
    completed = record_runtime(request, completed, started, task, budget_exceeded=exceeded)
    return record_measurements(request, completed, started, task, timing, summary)


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
        started = read_json(request.config_path).get("coverage_started_monotonic", started)
        task = read_json(request.task_path)
        if uses_continuation(task):
            return _run_chain(request, started)
        if task.get("parameters", {}).get("mode") != "profile":
            completed = _run_batch(request, started)
            return check_worker_wall(record_runtime(request, completed, started, task), started, task)
        root = Path(request.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        destination = output_path(root, "_profiled")
        require(not destination.exists(), "profile already attempted; never rerun failed draws")
        destination.mkdir()
        profiled_request = replace(request, output_dir=str(destination))
        completed = _run_batch(profiled_request, started)
        completed = record_runtime(profiled_request, completed, started, task)
        if not completed.artifacts:
            return StageResult(request_hash=request.content_hash, status=completed.status,
                               artifacts=(), message=completed.message)
        completed.verify(profiled_request)
        elapsed = time.monotonic() - started
        result = read_json(destination / "result.json")
        measurement = read_json(destination / MEASUREMENTS)
        timing = dict(measurement["timing"])
        timing.update(wall_seconds=elapsed,
                      gpu_seconds=elapsed if timing["device"].startswith("cuda") else 0.,
                      publication_verification_seconds=elapsed - timing["setup_seconds"] - math.fsum(timing["complete_repetition_seconds"]),
                      measurement_scope="complete_stage_return",
                      budget_exceeded=elapsed > timing["admitted_budget_seconds"],
                      profiled_request_hash=profiled_request.content_hash)
        if timing["device"].startswith("cuda"):
            exceeded = elapsed > timing["admitted_budget_seconds"]
            timing.update(budget_exceeded=exceeded,
                          overrun_seconds=max(0., elapsed - timing["admitted_budget_seconds"]))
            timing["complete"] = (measurement["received"] == len(task["parameters"]["draws"])
                and measurement["counts"]["success"] == len(task["parameters"]["draws"]) and not exceeded)
        result["profiled_request_hash"] = profiled_request.content_hash
        extra = tuple(replace(a, path="_profiled/" + a.path) for a in completed.artifacts)
        published = publish(request, {"result.json": result, "timing.json": timing,
            "profile_receipt.json": {"request": profiled_request.to_dict(), "result": completed.to_dict()}},
            status=completed.status, message="Production profile measured through complete stage return; " + completed.message,
            extra_artifacts=extra)
        return record_measurements(request, published, started, task, timing,
            {"counts": measurement["counts"], "received_repetitions": measurement["received"]})
    except (ContractError, KeyError, TypeError, ValueError, OSError, tarfile.TarError) as exc:
        return StageResult(request_hash=request.content_hash, status="blocked", artifacts=(),
                           message=str(exc) or type(exc).__name__)
