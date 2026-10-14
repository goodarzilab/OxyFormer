"""Prospective locking and full fan-in with synthetic, offline receipts."""
from copy import deepcopy
from dataclasses import replace
import json
import shutil
from tempfile import mkdtemp
from pathlib import Path

import pytest

from oxyformer.contracts import StageRequest, StageResult
from oxyformer.provenance import ContractError, canonical_json, file_hash
from oxyformer.validation import campaign, coverage
from oxyformer.validation.scm import SCMConfig
from test_coverage import successful_records, synthetic_endpoint, run_coverage

STAMPS = {"scientific_fingerprint": {"algorithm": "tracked-science-v2", "sha256": "a" * 64},
          "environment_hash": "b" * 64}


@pytest.fixture(autouse=True)
def frozen_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setenv("OXYFORMER_PUBLICATION_STORE", str(tmp_path / ".publications"))
    # Scientific fingerprinting itself is tested by the execution unit; this
    # unit verifies binding/drift while its own checkout is being edited.
    monkeypatch.setattr(campaign, "fingerprint", lambda: deepcopy(STAMPS))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value))
    return path


def seal_budget_fixture(root, unit, task=None):
    """Publish synthetic measurements through the real immutable sealing path."""
    from oxyformer.execution.integrity import publish_result, RESULT, FINGERPRINT
    from oxyformer.provenance import ArtifactRecord, ArtifactLineage
    result = coverage.read_json(root / 'result.json')
    timing = coverage.read_json(root / 'timing.json') if (root / 'timing.json').exists() else {}
    runtime = coverage.read_json(root / 'runtime.json') if (root / 'runtime.json').exists() else {}
    if task is None:
        chain = timing.get('chain')
        if timing.get('measurement_scope') == 'complete_chain_return' and chain is None:
            chain = {'owner': unit, 'step': len(timing['leaves']) - 1,
                'predecessor': timing['leaves'][-2]['task_id'] if len(timing['leaves']) > 1 else None}
        task = {'id': unit, 'stage': 'simulation-smoke', 'parameters': {
            'mode': result['mode'], 'scenario': result['scenario'], 'draws': result['draws'],
            'wall_seconds': timing.get('admitted_budget_seconds', 14400)}}
        if chain is not None:
            task['parameters']['wall_seconds'] = max(840, task['parameters']['wall_seconds'])
            task['continuation'] = chain
            task['outputs'] = ['result.json', 'continuation.tar']
    config = write(root / '_execution/config.json', {})
    task_path = write(root / '_execution/task.json', task)
    producer = StageRequest(stage=task['stage'], config_path=str(config), config_hash=file_hash(config),
        task_path=str(task_path), task_hash=file_hash(task_path), dependency_paths=(), dependency_hashes=(),
        output_dir=str(root), code_identity='c' * 40)
    (root / '_execution/request.json').write_text(producer.to_json())
    wall = task['parameters']['wall_seconds']
    chained = coverage.uses_continuation(task)
    elapsed = timing.get('wall_seconds', runtime.get('wall_seconds', 100.))
    leaf = {'task_id': unit, 'step': 0, 'compute_seconds': sum(timing.get('complete_repetition_seconds', [elapsed])),
        'setup_seconds': timing.get('setup_seconds', 0.), 'import_seconds': 0., 'export_seconds': 0.}
    leaf['publication_seconds'] = elapsed - sum(leaf[k] for k in ('compute_seconds', 'setup_seconds'))
    leaf['wall_seconds'] = elapsed
    leaves = deepcopy(timing['leaves']) if chained else [leaf]
    leaves[-1]['task_id'] = unit
    for row in leaves:
        row['post_check_reserve_seconds'] = coverage.publication_reserve(wall)
        if chained:
            row.setdefault('leaf_budget', coverage.task_leaf_budget(task))
            row.setdefault('measured_overheads', 'leaf_budget' in task['parameters'])
    # Older regression fixtures explicitly model unsuccessful/partial profiling.
    # Their scientific outcomes must agree with their finalized counters.
    for flag, status in (('complete', 'incomplete'), ('all_successful', 'numerical_failure')):
        if timing.get(flag) is False:
            successful = next((r for r in result['records'] if r['status'] == 'success'), None)
            if successful is not None:
                successful.update(status=status, reason='synthetic unsuccessful profile', estimates={})
    write(root / 'result.json', result)
    counts = {status: sum(r['status'] == status for r in result['records']) for status in coverage.STATUSES}
    binding = coverage.budget_binding(producer, task, timing.get('recipe_hash', result['recipe_hash']))
    value = {'schema_version': 1, 'binding': binding, 'chained': chained, 'leaves': leaves,
        'counts': counts, 'received': sum(counts.values()),
        'profile': {k: timing.get(k) for k in ('device', 'environment', 'production_equivalent',
            'complete_repetition_seconds', 'scientific_fingerprint', 'environment_hash')}}
    # These mutations describe failed producers, not authoritative stored flags.
    failed = timing.get('budget_exceeded', False) or any(v.get('budget_exceeded', False) for v in leaves)
    write(root / coverage.BUDGET_RECORD, value)
    lineage = ArtifactLineage(source_hashes=(producer.config_hash,), unit_ids=(producer.content_hash,),
        parent_hashes=(), split_hash=None, config_hash=producer.config_hash, model_hash=None,
        environment=(("fixture", "synthetic"),), seed=None, parameter_count=None)
    artifacts = tuple(ArtifactRecord(path=p.name, sha256=file_hash(p), kind='fixture', lineage=lineage)
        for p in sorted(root.glob('*.json')))
    terminal = StageResult(request_hash=producer.content_hash, status='fail' if failed else 'pass',
        artifacts=artifacts, message='synthetic measured fixture')
    for name in (RESULT, FINGERPRINT):
        (root / name).unlink(missing_ok=True)
    sealed = publish_result(root, terminal)
    assert sealed.status == terminal.status, sealed.message
    return producer


def profile_directory(req, unit='profile'):
    return Path(coverage.read_json(req.config_path)['dependencies'][unit])


def fork_profile(req, unit='profile', *, measurements=True):
    """A new attempt for revised fixture measurements; never overwrite a seal."""
    config = coverage.read_json(req.config_path)
    source = Path(config['dependencies'][unit])
    root = Path(mkdtemp(prefix=unit + '-revision-', dir=source.parent))
    shutil.copytree(source, root, dirs_exist_ok=True)
    if measurements:
        seal_budget_fixture(root, unit)
    else:
        from oxyformer.execution.integrity import RESULT, FINGERPRINT
        producer = StageRequest.from_json((root / '_execution/request.json').read_text())
        producer = replace(producer, output_dir=str(root), config_path=str(root / '_execution/config.json'),
            task_path=str(root / '_execution/task.json'))
        (root / '_execution/request.json').write_text(producer.to_json())
        value = coverage.read_json(root / coverage.BUDGET_RECORD)
        value['binding']['request_hash'] = producer.content_hash
        write(root / coverage.BUDGET_RECORD, value)
        terminal = StageResult.from_json((root / RESULT).read_text())
        (root / RESULT).write_text(replace(terminal, request_hash=producer.content_hash).to_json())
        (root / FINGERPRINT).unlink(missing_ok=True)
    config['dependencies'][unit] = str(root)
    write(Path(req.config_path), config)
    paths = tuple(str(p) for name in config['dependencies'].values() for p in sorted([
        *Path(name).glob('*.json'), *(Path(name) / '_execution').glob('*.json')]))
    return replace(req, config_hash=file_hash(req.config_path), dependency_paths=paths,
        dependency_hashes=tuple(map(file_hash, paths)))


def refresh_profile(req):
    task = coverage.read_json(req.task_path)
    for reference in task['parameters']['profiles'].values():
        req = fork_profile(req, reference['dependency'])
    return req


def request(root, task, dependencies, approvals=None):
    root.mkdir(parents=True, exist_ok=True)
    dependencies = dict(dependencies)
    for source in list(dependencies.values()):
        config_path = source.parent / 'config.json'
        if config_path.exists():
            upstream = coverage.read_json(config_path).get('dependencies', {})
            for unit in task.get('needs', {}):
                if unit not in dependencies and unit in upstream:
                    dependencies[unit] = Path(upstream[unit])
    config = write(root / "config.json", {"dependencies": {k: str(v) for k, v in dependencies.items()},
        "approvals": approvals or {"owner_decisions": {"campaign_allocations": {
            "test-final": {"kind": "final-coverage", "gpu_hours": 0}}}}})
    task_path = write(root / "task.json", task)
    paths = tuple(str(p) for directory in dependencies.values()
        for p in sorted([*directory.glob("*.json"), *(directory / '_execution').glob('*.json')]))
    output = root / "out"
    output.mkdir()
    return StageRequest(stage=task["stage"], config_path=str(config), config_hash=file_hash(config),
        task_path=str(task_path), task_hash=file_hash(task_path), dependency_paths=paths,
        dependency_hashes=tuple(map(file_hash, paths)), output_dir=str(output), code_identity="c" * 40)


def setup_lock(tmp_path, **changes):
    inputs = tmp_path / "inputs"
    endpoint_record, frame_record = synthetic_endpoint(tmp_path)
    endpoint = write(inputs / "endpoint.json", endpoint_record.to_dict())
    frame = write(inputs / "frame.json", frame_record.to_dict())
    recipe = {"endpoint_hash": file_hash(endpoint), "frame_hash": file_hash(frame), "nested_cv": {},
              "inference": {"primary_bandwidth_km": 100, "county_locations": {"c": [0, 0], "d": [5, 5]}}}
    scenario = SCMConfig(name="null_effect", effect="null", active_mechanisms=("null",)).to_dict()["payload"]
    profile = tmp_path / "profile"
    draws, records = successful_records(2)
    write(profile / "result.json", {"mode": "profile", "scenario": scenario, "recipe_hash": coverage.digest(recipe),
        "draws": draws, "records": records})
    write(profile / "timing.json", {"production_equivalent": True, "complete": True, "all_successful": True,
        "recipe_hash": coverage.digest(recipe), "device": "cpu", "gpu_seconds": 0,
        "complete_repetition_seconds": [10., 11.], "wall_seconds": 25.,
        "measurement_scope": "complete_stage_return", "setup_seconds": 3., **STAMPS})
    parameters = {"campaign_id": "test", "recipe": recipe, "scenario_family": [scenario],
        "final_repetitions": 1000, "screening_repetitions": 5, "repetitions_per_leaf": 25,
        "gpus": 0, "wall_seconds": 600, "profile_safety_factor": 2., "evaluation_namespace": "prospective-one",
        "retry_rules": deepcopy(coverage.RETRY_RULES), "prior_locks": [],
        "profiles": {"null_effect": {"dependency": "profile", "path": "result.json"}},
        "endpoint_input": {"dependency": "inputs", "path": "endpoint.json"},
        "frame_input": {"dependency": "inputs", "path": "frame.json"}, "cpu_wall_hours_budget": 10.}
    parameters.update(changes)
    task = {"id": "campaign-lock", "stage": "campaign-lock", "parameters": parameters}
    seal_budget_fixture(profile, "profile")
    req = request(tmp_path / "lock-request", task, {"inputs": inputs, "profile": profile})
    return req, task


def locked(tmp_path):
    req, task = setup_lock(tmp_path)
    result = campaign.run_stage(req)
    assert result.status == "pass", result.message
    result.verify(req)
    root = Path(req.output_dir)
    return root, coverage.read_json(root / "recipe_lock.json"), coverage.read_json(root / "expanded_units.json")["plans"]


def test_lock_emits_reviewable_bounded_manifests_without_submission(tmp_path):
    root, lock, plans = locked(tmp_path)
    assert {"task_manifest.json", "expanded_units.json", "recipe_lock.json", "budget.json", "gate.json"} <= {p.name for p in root.iterdir()}
    assert len(plans[0]["expected_leaves"]) == 1
    assert len(plans[1]["expected_leaves"]) == 40
    assert all(u["gpu_hours"] <= 4 and u["max_attempts"] == 1 for p in plans for u in p["units"])
    assert not coverage.read_json(root / "gate.json")["submission_performed"]
    screening = plans[0]["tasks"][-1]["id"]
    assert all(screening in t["needs"] for t in plans[1]["tasks"])
    all_draws = [d for b in lock["batches"] for d in b["draws"]]
    assert len(all_draws) == len({d["seed"] for d in all_draws}) == 1005
    assert lock["recipe"]["nested_cv"] == {}


@pytest.mark.parametrize("changes,reason", [
    ({"final_repetitions": 999}, "final repetitions"),
    ({"repetitions_per_leaf": 24}, "forty leaves"),
    ({"wall_seconds": 100}, "infeasible batch budget"),
    ({"cpu_wall_hours_budget": 1}, "infeasible CPU budget"),
    ({"gpus": 1}, "no GPU device interface"),
    ({"retry_rules": {"max_attempts": 2}}, "retry rules"),
])
def test_infeasible_or_weakened_campaign_is_blocked(tmp_path, changes, reason):
    req, _ = setup_lock(tmp_path, **changes)
    result = campaign.run_stage(req)
    assert result.status == "blocked"
    assert reason in result.message
    assert not list(Path(req.output_dir).glob("*.json"))


def test_smoke_profile_cannot_justify_final_batching(tmp_path):
    req, _ = setup_lock(tmp_path)
    path = profile_directory(req) / "timing.json"
    data = coverage.read_json(path)
    data["production_equivalent"] = False
    write(path, data)
    req = refresh_profile(req)
    assert "production profiling" in campaign.run_stage(req).message


def test_recipe_change_invalidates_profile(tmp_path):
    req, task = setup_lock(tmp_path)
    task["parameters"]["recipe"]["nested_cv"]["batch_size"] = 64
    write(Path(req.task_path), task)
    req = replace(req, task_hash=file_hash(req.task_path))
    result = campaign.run_stage(req)
    assert result.status == "blocked" and "profile recipe drift" in result.message


def collection_request(tmp_path, *, missing_leaf=False, missing_repetition=False, failures=0, mode="final"):
    root, lock, plans = locked(tmp_path)
    plan = plans[1 if mode == "final" else 0]
    task = deepcopy(plan["tasks"][-1])
    dependencies = {"campaign-lock": root}
    if mode == "final":
        screen_id = plans[0]["tasks"][-1]["id"]
        screening = tmp_path / "screening"
        write(screening / "gate.json", {"pass": True, "mode": "screening", "lock_hash": coverage.digest(lock)})
        dependencies[screen_id] = screening
    draws = [d for b in lock["batches"] if b["mode"] == mode for d in b["draws"]]
    _, records = successful_records(len(draws))
    for draw, record in zip(draws, records):
        record["draw"] = draw
    for record in records[:failures]:
        record.update(status="numerical_failure", estimates={})
    by_id = {r["draw"]["repetition_id"]: r for r in records}
    for index, leaf in enumerate(plan["tasks"][:-1]):
        if missing_leaf and index == 0:
            continue
        params = leaf["parameters"]
        destination = tmp_path / "leaves" / leaf["id"]
        selected = [by_id[d["repetition_id"]] for d in params["draws"]]
        if missing_repetition and index == 0:
            selected.pop()
        write(destination / "result.json", {"lock_hash": coverage.digest(lock), "recipe_hash": coverage.digest(lock["recipe"]),
            "batch_id": params["batch_id"], "mode": mode, "scenario": params["scenario"], "draws": params["draws"], "records": selected})
        write(destination / "runtime.json", {"batch_id": params["batch_id"],
            "lock_hash": coverage.digest(lock), "admitted_budget_seconds": params["wall_seconds"],
            "wall_seconds": 100., "measurement_scope": "complete_batch_return"})
        seal_budget_fixture(destination, leaf["id"], leaf)
        dependencies[leaf["id"]] = destination
    return request(tmp_path / "collection-request", task, dependencies), lock, plans


def test_complete_final_collection_can_pass(tmp_path):
    req, _, _ = collection_request(tmp_path)
    result = campaign.run_stage(req)
    assert result.status == "pass", result.message
    result.verify(req)
    gate = coverage.read_json(Path(req.output_dir) / "gate.json")
    assert gate["certifies_production_coverage"]
    assert len(gate["expected_leaves"]) == len(gate["received_leaves"]) == 40


@pytest.mark.parametrize("missing", ["leaf", "repetition"])
def test_partial_fan_in_never_passes(tmp_path, missing):
    req, _, _ = collection_request(tmp_path, missing_leaf=missing == "leaf", missing_repetition=missing == "repetition")
    result = campaign.run_stage(req)
    assert result.status != "pass"
    gate = Path(req.output_dir) / "gate.json"
    if gate.exists():
        assert not coverage.read_json(gate)["pass"]


def test_collected_failures_remain_counted(tmp_path):
    req, _, _ = collection_request(tmp_path, failures=12)
    result = campaign.run_stage(req)
    assert result.status == "fail"
    summary = coverage.read_json(Path(req.output_dir) / "gate.json")["scenario_summaries"]["null_effect"]
    assert summary["counts"]["numerical_failure"] == 12
    assert summary["declared_repetitions"] == 1000
    assert summary["numerical_failure_upper"] > .01


def test_screening_never_certifies_production_coverage(tmp_path):
    req, _, _ = collection_request(tmp_path, mode="screening")
    result = campaign.run_stage(req)
    assert result.status == "pass", result.message
    gate = coverage.read_json(Path(req.output_dir) / "gate.json")
    assert gate["pass"] and not gate["certifies_production_coverage"]


def test_environment_drift_blocks_collection(tmp_path, monkeypatch):
    req, _, _ = collection_request(tmp_path)
    monkeypatch.setattr(campaign, "fingerprint", lambda: {**STAMPS, "environment_hash": "d" * 64})
    result = campaign.run_stage(req)
    assert result.status == "blocked" and "environment recipe drift" in result.message


def test_altered_collector_cannot_drop_a_declared_leaf(tmp_path):
    req, _, _ = collection_request(tmp_path)
    task = coverage.read_json(req.task_path)
    task["expected_leaves"].pop()
    write(Path(req.task_path), task)
    req = replace(req, task_hash=file_hash(req.task_path))
    result = campaign.run_stage(req)
    assert result.status == "blocked" and "prospectively expanded" in result.message


def test_new_lock_uses_new_draws_even_with_same_namespace(tmp_path):
    _, first, _ = locked(tmp_path / "first")
    _, second, _ = locked(tmp_path / "second")
    a = {d["seed"] for b in first["batches"] for d in b["draws"]}
    b = {d["seed"] for batch in second["batches"] for d in batch["draws"]}
    assert not a & b
    assert first["nested_cv"] == first["recipe"]["nested_cv"]


def test_final_execution_is_blocked_by_failed_screening(tmp_path):
    root, lock, plans = locked(tmp_path)
    leaf = plans[1]["tasks"][0]
    screen_id = plans[0]["tasks"][-1]["id"]
    screening = tmp_path / "screening"
    write(screening / "gate.json", {"pass": False, "mode": "screening", "lock_hash": coverage.digest(lock)})
    req = request(tmp_path / "leaf-request", leaf, {"campaign-lock": root, "inputs": tmp_path / "inputs", screen_id: screening})
    result = run_coverage(req)
    assert result.status == "blocked" and "screening did not admit" in result.message
    assert not list(Path(req.output_dir).glob("repetitions/*"))


def test_lock_accepts_authenticated_noncanonical_input_serialization(tmp_path):
    from test_coverage import synthetic_endpoint
    req, task = setup_lock(tmp_path)
    endpoint, frame = synthetic_endpoint(tmp_path)
    (tmp_path / "inputs/endpoint.json").write_text(endpoint.to_json() + "\n")
    (tmp_path / "inputs/frame.json").write_text(frame.to_json() + "\n")
    recipe = task["parameters"]["recipe"]
    recipe.update(endpoint_hash=endpoint.content_hash, frame_hash=frame.content_hash)
    for name in ("result.json", "timing.json"):
        path = profile_directory(req) / name
        value = coverage.read_json(path)
        value["recipe_hash"] = coverage.digest(recipe)
        write(path, value)
    write(Path(req.task_path), task)
    req = replace(req, task_hash=file_hash(req.task_path), dependency_hashes=tuple(map(file_hash, req.dependency_paths)))
    req = refresh_profile(req)
    result = campaign.run_stage(req)
    assert result.status == "pass", result.message
    result.verify(req)


@pytest.mark.parametrize("seconds,status", [(270, "blocked"), (294, "pass")])
def test_lock_scales_publication_cost_with_leaf_size(tmp_path, monkeypatch, seconds, status):
    req, _ = setup_lock(tmp_path, profile_safety_factor=1., wall_seconds=seconds)
    path = profile_directory(req) / "timing.json"
    timing = coverage.read_json(path)
    timing.update(wall_seconds=40., setup_seconds=18., complete_repetition_seconds=[10., 10.])
    write(path, timing)
    req = refresh_profile(req)
    clock = [0.]
    monkeypatch.setattr(coverage.time, "monotonic", lambda: clock[0])
    def fingerprint():
        clock[0] += 18.
        return deepcopy(STAMPS)
    monkeypatch.setattr(campaign, "fingerprint", fingerprint)
    result = campaign.run_stage(req)
    # Actual locked setup + 25 draws + publication at one second per draw.
    assert result.status == status, result.message
    if status == "blocked":
        assert "infeasible batch budget" in result.message


def test_locked_leaf_profiles_the_environment_only_once(tmp_path, monkeypatch):
    root, lock, plans = locked(tmp_path)
    leaf = plans[1]["tasks"][0]
    screen_id = plans[0]["tasks"][-1]["id"]
    screening = tmp_path / "screening"
    write(screening / "gate.json", {"pass": True, "mode": "screening", "lock_hash": coverage.digest(lock)})
    req = request(tmp_path / "leaf-request", leaf, {"campaign-lock": root, "inputs": tmp_path / "inputs", screen_id: screening})
    calls = []
    def stamp():
        calls.append(1)
        return deepcopy(STAMPS)
    monkeypatch.setattr(campaign, "fingerprint", stamp)
    monkeypatch.setattr(coverage, "estimate_repetition", lambda *args:
        {method: {"value": 0., "se": 1.} for method in coverage.METHODS})
    result = run_coverage(req)
    assert result.status == "pass", result.message
    assert calls == [1]  # Same single fingerprint cost as the production profile.


def test_oversized_campaign_is_refused_before_draw_allocation(tmp_path, monkeypatch):
    req, _ = setup_lock(tmp_path, final_repetitions=100_000_000)
    original = campaign.repetition_plan
    allocations = []
    def allocation_sentinel(namespace, scenario, count):
        allocations.append(count)
        if count > 1000:
            raise MemoryError("synthetic sentinel: oversized draw allocation reached")
        return original(namespace, scenario, count)
    monkeypatch.setattr(campaign, "repetition_plan", allocation_sentinel)
    result = campaign.run_stage(req)
    assert result.status == "blocked" and "forty leaves" in result.message
    assert allocations == []


@pytest.mark.parametrize("case", ["locked_preflight", "scaled_publication"])
def test_review_round_three_admission_matches_complete_locked_leaf(tmp_path, monkeypatch, case):
    """Plan for the actual leaf; report residual latency without dropping draws."""
    budget = 271 if case == "locked_preflight" else 293
    req, task = setup_lock(tmp_path, profile_safety_factor=1., wall_seconds=budget)
    profile_count = 25 if case == "locked_preflight" else 2
    profile_path = profile_directory(req) / "result.json"
    profile = coverage.read_json(profile_path)
    profile["draws"], profile["records"] = successful_records(profile_count)
    write(profile_path, profile)
    timing_path = profile_directory(req) / "timing.json"
    timing = coverage.read_json(timing_path)
    timing.update(wall_seconds=20. + profile_count * 10.,
                  setup_seconds=20. if case == "locked_preflight" else 18.,
                  complete_repetition_seconds=[10.] * profile_count)
    write(timing_path, timing)
    req = refresh_profile(req)
    clock = [0.]
    monkeypatch.setattr(coverage.time, "monotonic", lambda: clock[0])
    def fingerprint():
        clock[0] += 20. if case == "locked_preflight" else 18.
        return deepcopy(STAMPS)
    original_validate = campaign.validate_leaf_task
    def validate(*args):
        value = original_validate(*args)
        if case == "locked_preflight":
            clock[0] += 1.
        return value
    monkeypatch.setattr(campaign, "fingerprint", fingerprint)
    monkeypatch.setattr(campaign, "validate_leaf_task", validate)
    admitted = campaign.run_stage(req)
    assert admitted.status == "pass", admitted.message
    root = Path(req.output_dir)
    admission = coverage.read_json(root / "budget.json")["admission_measurements"]
    final_cost = next(row for row in admission if row["mode"] == "final")
    assert final_cost["estimated_seconds"] == budget
    assert final_cost["locked_setup_seconds"] == (21 if case == "locked_preflight" else 18)
    assert final_cost["publication_verification_seconds"] == (0 if case == "locked_preflight" else 25)
    lock = coverage.read_json(root / "recipe_lock.json")
    plans = coverage.read_json(root / "expanded_units.json")["plans"]
    leaf = plans[1]["tasks"][0]
    screen_id = plans[0]["tasks"][-1]["id"]
    screening = tmp_path / "screening"
    write(screening / "gate.json", {"pass": True, "mode": "screening", "lock_hash": coverage.digest(lock)})
    leaf_request = request(tmp_path / "leaf-request", leaf,
        {"campaign-lock": root, "inputs": tmp_path / "inputs", screen_id: screening})
    def draw_result(draw, frame, scenario, template, recipe, destination, deadline):
        complete = clock[0] + 10. <= deadline
        clock[0] += 10. if complete else max(0., deadline - clock[0])
        return {"draw": draw, "status": "success" if complete else "incomplete",
            "reason": "synthetic ten-second complete procedure", "wall_seconds": 10.,
            "truth": 0., "causal_truth": 0., "truth_integration_error": 0.,
            "estimates": {m: {"value": float(draw["index"] % 2), "se": 1.} for m in coverage.METHODS}}
    original_publish = coverage.publish
    def publish(request, values, **kwargs):
        if case == "scaled_publication":
            clock[0] += len(values["result.json"]["records"]) + 3.  # Residual observed latency.
        return original_publish(request, values, **kwargs)
    monkeypatch.setattr(coverage, "execute_draw", draw_result)
    monkeypatch.setattr(coverage, "publish", publish)
    clock[0] = 0.
    result = run_coverage(leaf_request)
    result.verify(leaf_request)
    data = coverage.read_json(Path(leaf_request.output_dir) / "result.json")
    runtime = coverage.read_json(Path(leaf_request.output_dir) / "runtime.json")
    assert result.status == "fail", result.message
    assert data["summary"]["counts"]["success"] == 25
    assert data["summary"]["counts"]["incomplete"] == 0
    assert runtime["wall_seconds"] == clock[0] == (271 if case == "locked_preflight" else 296)
    assert runtime["admitted_budget_seconds"] == budget
    assert runtime["overrun_seconds"] == (0 if case == "locked_preflight" else 3)
    assert runtime["budget_exceeded"] == (case == "scaled_publication")


def test_collection_ignores_diagnostic_overruns_without_discarding_draws(tmp_path):
    req, _, plans = collection_request(tmp_path)
    leaves = plans[1]["tasks"][:-1]
    for leaf, extra in zip(leaves[:2], [3., 7.]):
        path = tmp_path / "leaves" / leaf["id"] / "runtime.json"
        runtime = coverage.read_json(path)
        runtime["wall_seconds"] = runtime["admitted_budget_seconds"] + extra
        write(path, runtime)
    req = replace(req, dependency_hashes=tuple(map(file_hash, req.dependency_paths)))
    result = campaign.run_stage(req)
    assert result.status == "pass", result.message
    gate = coverage.read_json(Path(req.output_dir) / "gate.json")
    assert gate["budget_overruns"] == []
    assert len(gate["leaf_runtimes"]) == 40
    assert gate["scenario_summaries"]["null_effect"]["declared_repetitions"] == 1000
    assert len(coverage.read_json(Path(req.output_dir) / "result.json")["records"]) == 1000


@pytest.mark.parametrize("change", ["missing", "budget", "identity"])
def test_collection_requires_bound_final_budget_evidence(tmp_path, change):
    req, _, plans = collection_request(tmp_path)
    leaf = plans[1]["tasks"][0]
    path = tmp_path / "leaves" / leaf["id"] / coverage.BUDGET_RECORD
    if change == "missing":
        path.unlink()
        paths = tuple(p for p in req.dependency_paths if p != str(path))
        req = replace(req, dependency_paths=paths, dependency_hashes=tuple(map(file_hash, paths)))
    else:
        runtime = coverage.read_json(path)
        runtime["binding"]["wall_seconds" if change == "budget" else "task_id"] = 1
        write(path, runtime)
        req = replace(req, dependency_hashes=tuple(map(file_hash, req.dependency_paths)))
    assert campaign.run_stage(req).status == "blocked"


def test_admission_measures_setup_at_each_leaf_size_without_running_draws(tmp_path, monkeypatch):
    req, _ = setup_lock(tmp_path, profile_safety_factor=1.)
    clock = [0.]
    monkeypatch.setattr(coverage.time, "monotonic", lambda: clock[0])
    original = campaign.prepare_batch
    shapes = []
    def prepare(request):
        result = original(request)
        parameters = result[0]["parameters"]
        size = len(parameters["draws"])
        shapes.append((parameters["mode"], size))
        clock[0] += size / 5
        return result
    def no_draws(*args):
        raise AssertionError("admission must not inspect simulated outcomes")
    monkeypatch.setattr(campaign, "prepare_batch", prepare)
    monkeypatch.setattr(coverage, "execute_draw", no_draws)
    result = campaign.run_stage(req)
    assert result.status == "pass", result.message
    assert shapes == [("screening", 5), ("final", 25)]
    measurements = coverage.read_json(Path(req.output_dir) / "budget.json")["admission_measurements"]
    assert [m["locked_setup_seconds"] for m in measurements] == [1., 5.]
    assert not list(Path(req.output_dir).glob(".admission-*"))


def test_late_locked_draws_remain_complete_and_reported(tmp_path, monkeypatch):
    root, lock, plans = locked(tmp_path)
    leaf = plans[0]["tasks"][0]
    req = request(tmp_path / "late-leaf", leaf, {"campaign-lock": root, "inputs": tmp_path / "inputs"})
    clock = [0.]
    monkeypatch.setattr(coverage.time, "monotonic", lambda: clock[0])
    def draw_result(draw, frame, scenario, template, recipe, destination, deadline):
        assert deadline == float("inf")  # The scheduler owns the binding limit.
        clock[0] += 150.
        return {"draw": draw, "status": "success", "reason": "complete late draw",
            "truth": 0., "causal_truth": 0., "truth_integration_error": 0.,
            "estimates": {m: {"value": 0., "se": 1.} for m in coverage.METHODS}}
    monkeypatch.setattr(coverage, "execute_draw", draw_result)
    result = run_coverage(req)
    assert result.status == "fail", result.message
    runtime = coverage.read_json(Path(req.output_dir) / "runtime.json")
    assert runtime["wall_seconds"] == 750.
    assert runtime["admitted_budget_seconds"] == 600.
    assert runtime["overrun_seconds"] == 150.
    records = coverage.read_json(Path(req.output_dir) / "result.json")["records"]
    assert len(records) == 5 and all(r["status"] == "success" for r in records)


def setup_chain_lock(tmp_path, **changes):
    req, task = setup_lock(tmp_path, repetitions_per_leaf=50,
        wall_seconds=1500 + int(2 * campaign.checkpoint_margin_seconds()), cpu_wall_hours_budget=100., **changes)
    path = profile_directory(req) / "timing.json"
    timing = coverage.read_json(path)
    leaves = [{"task_id": f"profile-{i}", "step": i, "compute_seconds": seconds,
        "setup_seconds": 1., "import_seconds": 2., "export_seconds": 2., "publication_seconds": 1.,
        "wall_seconds": seconds + 6.} for i, seconds in enumerate((10., 11.))]
    timing.update(measurement_scope="complete_chain_return", leaves=leaves,
                  admitted_budget_seconds=60, wall_seconds=33., setup_seconds=2.)
    write(path, timing)
    return refresh_profile(req), task


def test_chained_profile_sizes_slices_without_changing_instances_or_limits(tmp_path):
    req, _ = setup_chain_lock(tmp_path)
    result = campaign.run_stage(req)
    assert result.status == "pass", result.message
    root = Path(req.output_dir)
    lock = coverage.read_json(root / "recipe_lock.json")
    plans = coverage.read_json(root / "expanded_units.json")["plans"]
    assert lock["instances"] == {"screening": "test-screen", "final": "test-final"}
    assert [len(p["expected_leaves"]) for p in plans] == [2, 40]
    assert all(len(w["slices"]) == 2 for w in plans[1]["spec"]["work"])
    assert all("continuation.tar" in t["outputs"] for p in plans for t in p["tasks"][:-1])
    assert all("--partition=standard" in u["sbatch"] and u["gpu_hours"] <= 4 for p in plans for u in p["units"])
    allocation = coverage.leaf_allocation(lock["batches"][0]["wall_seconds"])["allocation_seconds"]
    assert lock["budget"]["cpu_wall_hours_ceiling"] == pytest.approx(42 * allocation / 3600 + 20 / 60)


def test_chained_profile_still_refuses_forty_one_leaves(tmp_path):
    req, task = setup_chain_lock(tmp_path)
    task["parameters"]["wall_seconds"] = 600 + int(2 * campaign.checkpoint_margin_seconds())
    write(Path(req.task_path), task)
    result = campaign.run_stage(replace(req, task_hash=file_hash(req.task_path)))
    assert result.status == "blocked" and "forty leaves" in result.message


def test_slicing_pays_each_leaf_overhead_safety_factor_and_checkpoint_reserve():
    profile = {"seconds_per_repetition": 24000., "leaf_overheads": {
        "setup_seconds": 100., "import_seconds": 100., "export_seconds": 100., "publication_seconds": 100.}}
    assert campaign.slice_count(profile, 1, 14400, 1.2) == 4
    with pytest.raises(ContractError, match="overhead"):
        campaign.slice_count(profile, 1, 300, 1.)


@pytest.mark.parametrize("incomplete", [False, True])
def test_collection_counts_chained_terminal_records_once(tmp_path, incomplete):
    req, _ = setup_chain_lock(tmp_path)
    result = campaign.run_stage(req)
    assert result.status == "pass", result.message
    root = Path(req.output_dir)
    lock = coverage.read_json(root / "recipe_lock.json")
    plans = coverage.read_json(root / "expanded_units.json")["plans"]
    plan = plans[1]
    screen = tmp_path / "screen"
    write(screen / "gate.json", {"pass": True, "mode": "screening", "lock_hash": coverage.digest(lock)})
    dependencies = {"campaign-lock": root, plans[0]["tasks"][-1]["id"]: screen}
    _, records = successful_records(1000)
    draws = [d for b in lock["batches"] if b["mode"] == "final" for d in b["draws"]]
    by_id = {}
    for record, draw in zip(records, draws):
        record["draw"] = draw
        by_id[draw["repetition_id"]] = record
    for index, leaf in enumerate(plan["tasks"][:-1]):
        destination = tmp_path / leaf["id"]
        params = leaf["parameters"]
        selected = [by_id[d["repetition_id"]] for d in params["draws"]]
        terminal = not (incomplete and index < 2)
        if not terminal:
            selected = selected[:-1]
        write(destination / "result.json", {"lock_hash": coverage.digest(lock), "recipe_hash": coverage.digest(lock["recipe"]),
            "batch_id": params["batch_id"], "mode": "final", "scenario": params["scenario"],
            "draws": params["draws"], "records": selected, "terminal": terminal, "chain": leaf["continuation"]})
        write(destination / "runtime.json", {"batch_id": params["batch_id"], "lock_hash": coverage.digest(lock),
            "admitted_budget_seconds": params["wall_seconds"], "wall_seconds": 500., "measurement_scope": "complete_batch_return"})
        predecessor = leaf['continuation']['predecessor']
        history = coverage.read_json(dependencies[predecessor] / coverage.BUDGET_RECORD)['leaves'] if predecessor else []
        measured = {'task_id': leaf['id'], 'step': leaf['continuation']['step'], 'compute_seconds': 100.,
            'setup_seconds': 1., 'import_seconds': 2., 'export_seconds': 2., 'publication_seconds': 1.,
            'wall_seconds': 106.}
        write(destination / 'timing.json', {'measurement_scope': 'complete_chain_return',
            'leaves': [*history, measured]})
        seal_budget_fixture(destination, leaf["id"], leaf)
        dependencies[leaf["id"]] = destination
    collector = request(tmp_path / "collector", plan["tasks"][-1], dependencies)
    result = campaign.run_stage(collector)
    assert result.status == ("fail" if incomplete else "pass"), result.message
    value = coverage.read_json(Path(collector.output_dir) / "result.json")
    assert len(value["records"]) == (999 if incomplete else 1000)
    assert len(value["gate"]["incomplete_chains"]) == int(incomplete)
    assert value["gate"]["scenario_summaries"]["null_effect"]["missing_repetitions"] == int(incomplete)


def test_producer_budget_pays_in_fold_headroom(tmp_path, monkeypatch):
    """Real lock -> expansion -> execution -> collection; synthetic draw costs.

    A 1,000-draw batch costs 800 s. This preserves the frozen final minimum
    and instance structure while testing the committee's 800 s work item.
    """
    from test_coverage import chain_request
    margin = campaign.checkpoint_margin_seconds()
    wall = 840 + int(1.2 * margin)
    req, task = setup_lock(tmp_path, screening_repetitions=1000, repetitions_per_leaf=1000,
                           wall_seconds=wall, profile_safety_factor=1.2, cpu_wall_hours_budget=100.)
    profile_root = profile_directory(req)
    data = coverage.read_json(profile_root / 'result.json')
    data['draws'], data['records'] = successful_records(1000)
    write(profile_root / 'result.json', data)
    timing = coverage.read_json(profile_root / 'timing.json')
    timing.update(measurement_scope='complete_chain_return', admitted_budget_seconds=840,
        complete_repetition_seconds=[.8] * 1000, wall_seconds=1100., setup_seconds=0.,
        leaves=[{'task_id': f'profile-{i}', 'step': i, 'compute_seconds': c,
                 'setup_seconds': 0., 'import_seconds': 0., 'export_seconds': 100.,
                 'publication_seconds': 0., 'wall_seconds': c + 100.}
                for i, c in enumerate([350., 350., 100.])])
    write(profile_root / 'timing.json', timing)
    req = refresh_profile(req)
    clock = [0.]
    monkeypatch.setattr(coverage.time, 'monotonic', lambda: clock[0])
    result = campaign.run_stage(req)
    assert result.status == 'pass', result.message
    lock_root = Path(req.output_dir)
    lock = coverage.read_json(lock_root / 'recipe_lock.json')
    plans = coverage.read_json(lock_root / 'expanded_units.json')['plans']
    assert [len(p['expected_leaves']) for p in plans] == [4, 4]
    assert all(b['leaf_budget']['compute_seconds'] == 350. + margin for b in lock['batches'])
    assert all('continuation.tar' in leaf['outputs'] for p in plans for leaf in p['tasks'][:-1])
    dependencies = {'campaign-lock': lock_root, 'inputs': tmp_path / 'inputs'}
    original_export = coverage._write_continuation
    def export(*a, **kw):
        clock[0] += 100.
        return original_export(*a, **kw)
    monkeypatch.setattr(coverage, '_write_continuation', export)
    finished, resumed, deadlines = set(), [], []
    def execute(draw, frame, scenario, template, recipe, root, deadline, *, continuation, study):
        rid = draw['repetition_id']
        assert rid not in finished  # Completed draws are carried, never rerun.
        deadline -= margin
        deadlines.append(deadline)
        used = continuation['compute_seconds']
        if used:
            resumed.append(rid)
        remaining = .8 - used
        consumed = min(remaining, max(0., deadline - clock[0] - 1.))
        clock[0] += consumed
        if consumed < remaining:
            return {'draw': draw, 'status': 'incomplete', 'reason': 'budget exhausted'}
        finished.add(rid)
        return {'draw': draw, 'status': 'success', 'reason': 'complete', 'wall_seconds': consumed,
                'truth': 0., 'causal_truth': 0., 'truth_integration_error': 0.,
                'estimates': {m: {'value': 0., 'se': 1.} for m in coverage.METHODS}}
    monkeypatch.setattr(coverage, 'execute_draw', execute)
    for index, leaf in enumerate(plans[0]['tasks'][:-1]):
        clock[0] = 0.
        leaf_req = chain_request(tmp_path / f'leaf-{index}', leaf, dependencies)
        done = run_coverage(leaf_req)
        assert done.status == 'pass', done.message
        done.verify(leaf_req)
        directory = Path(leaf_req.output_dir)
        row = coverage.read_json(directory / 'timing.json')['leaves'][-1]
        assert row['compute_capacity_seconds'] == 350. + margin
        assert row['compute_seconds'] <= 350.
        assert row['export_seconds'] == 100.
        assert row['wall_seconds'] <= 450. < wall
        assert coverage.read_json(directory / 'result.json')['terminal'] == (index == 3)
        dependencies[leaf['id']] = directory
    assert set(deadlines) == {350.}
    assert len(finished) == 1000 and len(resumed) == 2
    collector = chain_request(tmp_path / 'collector', plans[0]['tasks'][-1], dependencies)
    done = campaign.run_stage(collector)
    assert done.status == 'pass', done.message
    result = coverage.read_json(Path(collector.output_dir) / 'result.json')
    assert len(result['records']) == 1000
    assert result['gate']['incomplete_chains'] == result['gate']['budget_overruns'] == []


@pytest.mark.parametrize('field', ['compute_seconds', 'wall_seconds', 'reserve_seconds', 'safety_factor', 'overheads'])
def test_locked_budget_refuses_executor_parameter_drift(tmp_path, field):
    from test_coverage import chain_request
    req, _ = setup_chain_lock(tmp_path)
    assert campaign.run_stage(req).status == 'pass'
    lock_root = Path(req.output_dir)
    plans = coverage.read_json(lock_root / 'expanded_units.json')['plans']
    leaf = deepcopy(plans[0]['tasks'][0])
    leaf['parameters']['leaf_budget'][field] = 999
    leaf_req = chain_request(tmp_path / 'changed', leaf, {'campaign-lock': lock_root, 'inputs': tmp_path / 'inputs'})
    result = run_coverage(leaf_req)
    assert result.status == 'blocked', result.message
    assert not result.artifacts
    assert not list(Path(leaf_req.output_dir).glob('repetitions/*'))


@pytest.mark.parametrize('case', ['partial', 'unsuccessful', 'known_overrun', 'earlier_leaf_overrun', 'legacy_overrun'])
def test_partial_unsuccessful_or_overrun_profile_cannot_lock(tmp_path, case):
    req, _ = setup_chain_lock(tmp_path)
    path = profile_directory(req) / 'timing.json'
    timing = coverage.read_json(path)
    if case == 'partial':
        timing['complete'] = False
    elif case == 'unsuccessful':
        timing['all_successful'] = False
    elif case == 'known_overrun':
        timing['budget_exceeded'] = True
    elif case == 'earlier_leaf_overrun':
        timing['leaves'][0]['budget_exceeded'] = True
    else:
        timing.update(measurement_scope='complete_stage_return', admitted_budget_seconds=30.)
    write(path, timing)
    req = refresh_profile(req)
    result = campaign.run_stage(req)
    assert result.status == 'blocked', result.message
    assert not result.artifacts


def reseal_budget_fixture(root, status='pass'):
    """Construct disagreements before sealing, as a faulty producer could."""
    from oxyformer.execution.integrity import publish_result, RESULT, FINGERPRINT
    terminal = StageResult.from_json((root / RESULT).read_text())
    terminal = replace(terminal, status=status,
        artifacts=tuple(replace(a, sha256=file_hash(root / a.path))
            for a in terminal.artifacts if a.path != FINGERPRINT))
    for name in (RESULT, FINGERPRINT):
        (root / name).unlink(missing_ok=True)
    return publish_result(root, terminal)


@pytest.mark.parametrize('timing_bad', [False, True])
@pytest.mark.parametrize('runtime_bad', [False, True])
@pytest.mark.parametrize('terminal_bad', [False, True])
@pytest.mark.parametrize('record_bad', [False, True])
def test_budget_divergence_matrix(tmp_path, timing_bad, runtime_bad, terminal_bad, record_bad):
    req, _ = setup_lock(tmp_path)
    req = fork_profile(req, measurements=False)
    root = profile_directory(req)
    # Diagnostic disagreements are present at seal time. Only the terminal
    # status and finalized measured record determine eligibility.
    timing = coverage.read_json(root / 'timing.json')
    timing.update(wall_seconds=601. if timing_bad else 599., complete=not timing_bad,
        all_successful=not timing_bad, budget_exceeded=timing_bad)
    write(root / 'timing.json', timing)
    write(root / 'runtime.json', {'wall_seconds': 601. if runtime_bad else 599.,
        'budget_exceeded': runtime_bad, 'overrun_seconds': 1. if runtime_bad else 0.})
    record = coverage.read_json(root / coverage.BUDGET_RECORD)
    # The fixture's producer declared 14,400 s. Bind both sides to 600 s.
    task_path = root / '_execution/task.json'
    task = coverage.read_json(task_path)
    task['parameters']['wall_seconds'] = 600
    write(task_path, task)
    producer = StageRequest.from_json((root / '_execution/request.json').read_text())
    producer = replace(producer, task_hash=file_hash(task_path))
    (root / '_execution/request.json').write_text(producer.to_json())
    record['binding'] = coverage.budget_binding(producer, task, record['binding']['recipe_hash'])
    row = record['leaves'][0]
    row.update(wall_seconds=601. if record_bad else 500., post_check_reserve_seconds=30.,
        publication_seconds=(601. if record_bad else 500.) - row['compute_seconds'] - row['setup_seconds'])
    record.update(complete=False, success=False, eligible=False)  # Not authoritative flags.
    write(root / coverage.BUDGET_RECORD, record)
    from oxyformer.execution.integrity import RESULT
    terminal = StageResult.from_json((root / RESULT).read_text())
    (root / RESULT).write_text(replace(terminal, request_hash=producer.content_hash).to_json())
    reseal_budget_fixture(root, 'fail' if terminal_bad else 'pass')
    req = replace(req, dependency_hashes=tuple(map(file_hash, req.dependency_paths)))
    config = coverage.read_json(req.config_path)
    if terminal_bad or record_bad:
        with pytest.raises(ContractError):
            coverage.load_budget(req, config, 'profile')
        refused = campaign.run_stage(req)
        assert refused.status != 'pass'
        assert not (Path(req.output_dir) / 'recipe_lock.json').exists()
    else:
        loaded = coverage.load_budget(req, config, 'profile')
        assert loaded['complete'] and loaded['all_successful']
        assert loaded['wall_seconds'] == 500.


@pytest.mark.parametrize('change', ['missing', 'truncated', 'hash', 'pre_seal', 'wall', 'request'])
def test_budget_refuses_unsealed_damaged_or_reused_record(tmp_path, change):
    from oxyformer.execution.integrity import publication_receipt
    req, _ = setup_lock(tmp_path)
    req = fork_profile(req, measurements=False)
    root = profile_directory(req)
    path = root / coverage.BUDGET_RECORD
    if change in ('missing', 'truncated', 'hash'):
        reseal_budget_fixture(root)
    if change == 'missing':
        path.unlink()
    elif change == 'truncated':
        path.write_text('{')
    elif change == 'pre_seal':
        publication_receipt(root).unlink(missing_ok=True)
    else:
        value = coverage.read_json(path)
        if change == 'hash':
            value['leaves'][0]['wall_seconds'] += 1.
        elif change == 'wall':
            value['binding']['wall_seconds'] = 600
        else:
            value['binding']['request_hash'] = 'f' * 64
        write(path, value)
        if change != 'hash':
            reseal_budget_fixture(root)
    paths = tuple(p for p in req.dependency_paths if Path(p).exists())
    req = replace(req, dependency_paths=paths, dependency_hashes=tuple(map(file_hash, paths)))
    result = campaign.run_stage(req)
    assert result.status != 'pass'
    assert not (Path(req.output_dir) / 'recipe_lock.json').exists()


def test_pre_record_read_admits_no_lock_or_slices(tmp_path):
    req, _ = setup_lock(tmp_path)
    req = fork_profile(req, measurements=False)
    root = profile_directory(req)
    (root / coverage.BUDGET_RECORD).unlink()
    from oxyformer.execution.integrity import RESULT
    (root / RESULT).unlink()
    result = campaign.run_stage(req)
    assert result.status != 'pass'
    assert not list(Path(req.output_dir).glob('*.json'))


def test_receipt_overrun_599_601_cannot_justify_lock(tmp_path, monkeypatch):
    from test_coverage import chain_request
    monkeypatch.setattr(campaign, 'fingerprint', lambda: deepcopy(STAMPS))
    _, lock_task = setup_lock(tmp_path, repetitions_per_leaf=1000,
        screening_repetitions=1000, wall_seconds=600, profile_safety_factor=1.2)
    parameters = lock_task['parameters']
    profile_task = {'id': 'profile', 'stage': 'simulation-smoke',
        'outputs': ['result.json', 'timing.json', 'runtime.json', 'artifact_manifest.json', 'continuation.tar'],
        'continuation': {'owner': 'profile', 'step': 0, 'predecessor': None},
        'parameters': {'mode': 'profile', 'recipe': parameters['recipe'],
            'scenario': parameters['scenario_family'][0],
            'draws': coverage.repetition_plan('receipt-overrun-reproduction', 'null_effect', 1000),
            'wall_seconds': 600, 'endpoint_input': parameters['endpoint_input'],
            'frame_input': parameters['frame_input']}}
    clock = [0.]
    monkeypatch.setattr(coverage.time, 'monotonic', lambda: clock[0])
    def execute(draw, frame, scenario, template, recipe, root, deadline, **kw):
        clock[0] += .25
        return {'draw': draw, 'status': 'success', 'reason': 'synthetic completed draw',
            'wall_seconds': .25, 'truth': 0., 'causal_truth': 0., 'truth_integration_error': 0.,
            'estimates': {m: {'value': 0., 'se': 1.} for m in coverage.METHODS}}
    monkeypatch.setattr(coverage, 'execute_draw', execute)
    original_export, original_publish, original_json = coverage._write_continuation, coverage.publish, coverage.atomic_json
    def export(*a, **kw):
        clock[0] += 340.
        return original_export(*a, **kw)
    def publish(*a, **kw):
        result = original_publish(*a, **kw)
        clock[0] += 9.
        return result
    def atomic_json(root, name, data):
        path = original_json(root, name, data)
        if name == 'timing.json':
            clock[0] += 2.
        return path
    monkeypatch.setattr(coverage, '_write_continuation', export)
    monkeypatch.setattr(coverage, 'publish', publish)
    monkeypatch.setattr(coverage, 'atomic_json', atomic_json)
    profile = chain_request(tmp_path / 'profile-producer', profile_task, {'inputs': tmp_path / 'inputs'})
    result = run_coverage(profile)
    result.verify(profile)
    profile_root = Path(profile.output_dir)
    timing = coverage.read_json(profile_root / 'timing.json')
    runtime = coverage.read_json(profile_root / 'runtime.json')
    assert result.status == 'fail'
    assert timing['complete'] and not timing['budget_exceeded']
    assert timing['wall_seconds'] == 599.
    assert runtime['wall_seconds'] == 601. and runtime['budget_exceeded']
    monkeypatch.setattr(coverage, 'publish', original_publish)
    monkeypatch.setattr(coverage, 'atomic_json', original_json)
    lock_request = request(tmp_path / 'admission', lock_task,
        {'inputs': tmp_path / 'inputs', 'profile': profile_root})
    admitted = campaign.run_stage(lock_request)
    assert admitted.status != 'pass', admitted.message
    assert 'budget stage did not pass' in admitted.message
    assert not (Path(lock_request.output_dir) / 'recipe_lock.json').exists()
    assert not (Path(lock_request.output_dir) / 'expanded_units.json').exists()


@pytest.mark.parametrize('change', ['wall', 'request'])
def test_genuine_budget_record_cannot_be_reused_for_another_request(tmp_path, change):
    from oxyformer.execution.integrity import RESULT
    req, _ = setup_lock(tmp_path)
    original = (profile_directory(req) / coverage.BUDGET_RECORD).read_bytes()
    req = fork_profile(req, measurements=False)
    root = profile_directory(req)
    producer = StageRequest.from_json((root / '_execution/request.json').read_text())
    task_path = root / '_execution/task.json'
    if change == 'wall':
        task = coverage.read_json(task_path)
        task['parameters']['wall_seconds'] = 600
        write(task_path, task)
        producer = replace(producer, task_hash=file_hash(task_path))
    else:
        config_path = root / '_execution/config.json'
        write(config_path, {'different_request': True})
        producer = replace(producer, config_hash=file_hash(config_path))
    (root / '_execution/request.json').write_text(producer.to_json())
    # Preserve the genuine finalized bytes and their original artifact hash.
    (root / coverage.BUDGET_RECORD).write_bytes(original)
    terminal = StageResult.from_json((root / RESULT).read_text())
    (root / RESULT).write_text(replace(terminal, request_hash=producer.content_hash).to_json())
    reseal_budget_fixture(root)
    assert (root / coverage.BUDGET_RECORD).read_bytes() == original
    req = replace(req, dependency_hashes=tuple(map(file_hash, req.dependency_paths)))
    result = campaign.run_stage(req)
    assert result.status == 'blocked' and 'budget binding mismatch' in result.message
    assert not (Path(req.output_dir) / 'recipe_lock.json').exists()


def test_profile_diagnostics_do_not_replace_worker_measurements(tmp_path, monkeypatch):
    from test_coverage import chain_request
    _, lock_task = setup_lock(tmp_path)
    params = lock_task['parameters']
    task = {'id': 'profile', 'stage': 'simulation-smoke', 'outputs': ['result.json', 'timing.json'],
        'parameters': {'mode': 'profile', 'wall_seconds': 600, 'recipe': params['recipe'],
            'scenario': params['scenario_family'][0],
            'draws': coverage.repetition_plan('diagnostic-authority', 'null_effect', 1),
            'endpoint_input': params['endpoint_input'], 'frame_input': params['frame_input']}}
    producer = chain_request(tmp_path / 'producer', task, {'inputs': tmp_path / 'inputs'})
    monkeypatch.setattr(coverage, 'estimate_repetition', lambda *a, **k:
        {m: {'value': 0., 'se': 1.} for m in coverage.METHODS})
    original_json = coverage.atomic_json
    def corrupt_diagnostic(root, name, value):
        if name == 'timing.json':
            value = {**value, 'wall_seconds': 1e9, 'setup_seconds': 1e9,
                'complete': False, 'all_successful': False, 'budget_exceeded': True}
        return original_json(root, name, value)
    monkeypatch.setattr(coverage, 'atomic_json', corrupt_diagnostic)
    result = run_coverage(producer)
    assert result.status == 'pass', result.message
    assert coverage.read_json(Path(producer.output_dir) / 'timing.json')['setup_seconds'] == 1e9
    consumer = request(tmp_path / 'consumer', {'id': 'reader', 'stage': 'campaign-lock'},
        {'profile': Path(producer.output_dir)})
    record = coverage.load_budget(consumer, coverage.read_json(consumer.config_path), 'profile')
    assert record['complete'] and record['all_successful']
    assert record['leaves'][0]['setup_seconds'] < 600


def test_final_budget_counts_must_match_scientific_records(tmp_path):
    req, _ = setup_lock(tmp_path)
    req = fork_profile(req, measurements=False)
    root = profile_directory(req)
    payload = coverage.read_json(root / 'result.json')
    payload['records'][0].update(status='numerical_failure', reason='synthetic producer inconsistency', estimates={})
    write(root / 'result.json', payload)
    # Preserve the genuine all-success budget counters; publish a fresh seal
    # over the inconsistent scientific payload and unchanged measurements.
    reseal_budget_fixture(root)
    req = replace(req, dependency_hashes=tuple(map(file_hash, req.dependency_paths)))
    admitted = campaign.run_stage(req)
    assert admitted.status != 'pass', admitted.message
    assert not (Path(req.output_dir) / 'recipe_lock.json').exists()
    assert not (Path(req.output_dir) / 'expanded_units.json').exists()


@pytest.mark.parametrize('wall', [14401, 20000])
def test_nonchain_gpu_lock_already_refuses_four_hour_overrun(tmp_path, monkeypatch, wall):
    from types import SimpleNamespace
    req, _ = setup_lock(tmp_path, gpus=1, wall_seconds=wall, cpu_wall_hours_budget=500.)
    # Exercise GPU declaration admission on a CPU allocation, without any fit.
    monkeypatch.setattr(campaign, 'resolve_device', lambda: SimpleNamespace(type='cuda'))
    result = campaign.run_stage(req)
    assert result.status == 'blocked'
    assert result.message == 'leaf exceeds four GPU-hours'
    assert not list(Path(req.output_dir).glob('*.json'))


def finalize_profile_fixture(req, task, payload, timing, elapsed, monkeypatch, *, status='pass'):
    """Exercise actual runner finalization and the independent publication seal."""
    import tarfile
    from oxyformer.execution import runner, integrity
    from oxyformer.provenance import ArtifactRecord
    root = profile_directory(req)
    task_path = root / '_execution/task.json'
    write(task_path, task)
    producer = StageRequest.from_json((root / '_execution/request.json').read_text())
    producer = replace(producer, task_hash=file_hash(task_path))
    (root / '_execution/request.json').write_text(producer.to_json())
    for name in ('result.json', 'timing.json', coverage.MEASUREMENTS, coverage.BUDGET_RECORD):
        (root / name).unlink(missing_ok=True)
    completed = coverage.publish(producer, {'result.json': payload, 'timing.json': timing},
        status='pass', message='synthetic request-bound producer')
    if 'continuation.tar' in task.get('outputs', []):
        with tarfile.open(root / 'continuation.tar', 'w'):
            pass
        completed = replace(completed, artifacts=(*completed.artifacts,
            ArtifactRecord(path='continuation.tar', sha256=file_hash(root / 'continuation.tar'),
                kind='fixture', lineage=completed.artifacts[0].lineage)))
    timing.setdefault('environment', {'fixture': 'synthetic'})
    summary = coverage.summarize(payload['records'], task['parameters']['draws'],
        production_equivalent=True, null_scenario=True)
    with monkeypatch.context() as clock:
        clock.setattr(runner.time, 'monotonic', lambda: elapsed)
        measured = coverage.record_measurements(producer, completed, 0., task, timing, summary)
        final = runner.finalize_coverage_budget(producer, measured, 0.)
    for name in (integrity.RESULT, integrity.FINGERPRINT):
        (root / name).unlink(missing_ok=True)
    sealed = integrity.publish_result(root, final)
    assert sealed.status == status, sealed.message
    return replace(req, dependency_hashes=tuple(map(file_hash, req.dependency_paths)))


def profile_leaf(unit, step, compute, overheads, budget):
    return {'task_id': unit, 'step': step, 'compute_seconds': compute, **overheads,
        'wall_seconds': compute + sum(overheads.values()), 'leaf_budget': budget,
        'measured_overheads': False, 'post_check_reserve_seconds': 30.}


def test_stale_scope_cannot_release_single_slice_lock(tmp_path, monkeypatch):
    req, consumer_task = setup_lock(tmp_path, wall_seconds=604, profile_safety_factor=1.2,
        cpu_wall_hours_budget=12.)
    req = fork_profile(req, measurements=False)
    root = profile_directory(req)
    payload = coverage.read_json(root / 'result.json')
    draws, records = successful_records(25)
    payload.update(draws=draws, records=records)
    task = coverage.read_json(root / '_execution/task.json')
    task['parameters'].update(wall_seconds=840, draws=draws, compute_ceiling_seconds=390.,
        recipe=consumer_task['parameters']['recipe'])
    task.update(outputs=['result.json', 'continuation.tar'],
        continuation={'owner': 'profile-chain', 'step': 1, 'predecessor': 'profile-previous'})
    zeros = {k: 0. for k in coverage.OVERHEAD_PHASES}
    budget = {**coverage.leaf_budget(840, 1., zeros), 'compute_seconds': 390.}
    timing = coverage.read_json(root / 'timing.json')
    timing.update(measurement_scope='complete_stage_return', admitted_budget_seconds=840,
        complete_repetition_seconds=[16.] * 25, setup_seconds=0., wall_seconds=500.,
        leaves=[profile_leaf('profile-previous', 0, 200., {**zeros, 'publication_seconds': 60.}, budget),
                profile_leaf('profile', 1, 200., {**zeros, 'publication_seconds': 40.}, budget)])
    req = finalize_profile_fixture(req, task, payload, timing, 500., monkeypatch)
    result = campaign.run_stage(req)
    assert result.status != 'pass', result.message
    assert not (Path(req.output_dir) / 'recipe_lock.json').exists()
    assert not (Path(req.output_dir) / 'expanded_units.json').exists()


def test_record_contract_must_match_authorized_task(tmp_path, monkeypatch):
    req, consumer_task = setup_lock(tmp_path, wall_seconds=14100, cpu_wall_hours_budget=200.)
    req = fork_profile(req, measurements=False)
    root = profile_directory(req)
    payload = coverage.read_json(root / 'result.json')
    task = coverage.read_json(root / '_execution/task.json')
    phases = dict(zip(coverage.OVERHEAD_PHASES, (20., 30., 40., 10.)))
    authorized = coverage.leaf_budget(840, 1.2, phases)
    substituted = coverage.leaf_budget(840, 1., {k: 0. for k in coverage.OVERHEAD_PHASES})
    assert authorized['compute_seconds'] == 350. and substituted['compute_seconds'] == 540.
    task['parameters'].update(wall_seconds=840, leaf_budget=authorized,
        recipe=consumer_task['parameters']['recipe'])
    task.update(outputs=['result.json', 'continuation.tar'],
        continuation={'owner': 'profile', 'step': 0, 'predecessor': None})
    timing = coverage.read_json(root / 'timing.json')
    timing.update(measurement_scope='complete_chain_return', admitted_budget_seconds=840,
        complete_repetition_seconds=[200., 200.], setup_seconds=20., wall_seconds=500.,
        leaves=[profile_leaf('profile', 0, 400., phases, substituted)])
    req = finalize_profile_fixture(req, task, payload, timing, 500., monkeypatch, status='fail')
    result = campaign.run_stage(req)
    assert result.status != 'pass', result.message
    assert not (Path(req.output_dir) / 'recipe_lock.json').exists()


def test_scientific_recipe_must_match_profile_recipe(tmp_path, monkeypatch):
    req, consumer_task = setup_lock(tmp_path)
    req = fork_profile(req, measurements=False)
    root = profile_directory(req)
    payload = coverage.read_json(root / 'result.json')
    task = coverage.read_json(root / '_execution/task.json')
    task['parameters']['recipe'] = consumer_task['parameters']['recipe']
    cheaper = deepcopy(task['parameters']['recipe'])
    cheaper['nested_cv'].update(ssl_epochs=1, frozen_epochs=1, synthetic=True)
    payload['recipe_hash'] = coverage.digest(cheaper)
    timing = coverage.read_json(root / 'timing.json')
    req = finalize_profile_fixture(req, task, payload, timing, 25., monkeypatch)
    result = campaign.run_stage(req)
    assert result.status != 'pass', result.message
    assert not (Path(req.output_dir) / 'recipe_lock.json').exists()


@pytest.mark.parametrize('stale', ['chained', 'compute_ceiling'])
def test_loader_reconciles_sealed_record_with_producer_task(tmp_path, stale):
    req, _ = setup_chain_lock(tmp_path)
    req = fork_profile(req, measurements=False)
    root = profile_directory(req)
    record = coverage.read_json(root / coverage.BUDGET_RECORD)
    if stale == 'chained':
        record['chained'] = False
    else:
        # Internally valid but never authorized by this producer task.
        for leaf in record['leaves']:
            leaf['leaf_budget']['compute_seconds'] -= 1.
    write(root / coverage.BUDGET_RECORD, record)
    assert reseal_budget_fixture(root).status == 'pass'
    req = replace(req, dependency_hashes=tuple(map(file_hash, req.dependency_paths)))
    with pytest.raises(ContractError, match='execution mode mismatch|differs from producer task'):
        coverage.load_budget(req, coverage.read_json(req.config_path), 'profile')


@pytest.mark.parametrize('stale', ['scenario', 'mode', 'recipe', None])
def test_profile_payload_identity_matches_producer_and_lock(tmp_path, monkeypatch, stale):
    req, consumer = setup_lock(tmp_path, wall_seconds=14400, cpu_wall_hours_budget=200.)
    req = fork_profile(req, measurements=False)
    root = profile_directory(req)
    task = coverage.read_json(root / '_execution/task.json')
    payload = coverage.read_json(root / 'result.json')
    timing = coverage.read_json(root / 'timing.json')
    task['parameters'].update(wall_seconds=600, recipe=consumer['parameters']['recipe'])
    if stale == 'scenario':
        scenario = SCMConfig(name='nonlinear', effect='nonlinear', active_mechanisms=('nonlinear',)).to_dict()['payload']
        consumer['parameters'].update(scenario_family=[scenario],
            profiles={'nonlinear': {'dependency': 'profile', 'path': 'result.json'}})
        write(Path(req.task_path), consumer)
        req = replace(req, task_hash=file_hash(req.task_path))
        task['parameters']['scenario'] = scenario
        draws = coverage.repetition_plan('payload-identity', 'nonlinear', 2)
        task['parameters']['draws'] = payload['draws'] = draws
        for record, draw in zip(payload['records'], draws):
            record['draw'] = draw
        assert payload['scenario']['name'] == 'null_effect'
    elif stale == 'mode':
        payload['mode'] = 'smoke'
    elif stale == 'recipe':
        cheaper = deepcopy(task['parameters']['recipe'])
        cheaper['nested_cv'].update(ssl_epochs=1, frozen_epochs=1, synthetic=True)
        payload['recipe_hash'] = coverage.digest(cheaper)
    timing.update(complete_repetition_seconds=[10., 10.], setup_seconds=20., wall_seconds=50.)
    req = finalize_profile_fixture(req, task, payload, timing, 50., monkeypatch)
    config = coverage.read_json(req.config_path)
    if stale is not None:
        with pytest.raises(ContractError, match='scientific .* mismatch'):
            coverage.load_budget(req, config, 'profile')
    else:
        assert coverage.load_budget(req, config, 'profile')['all_successful']
    result = campaign.run_stage(req)
    assert (result.status == 'pass') == (stale is None), result.message
    for name in ('recipe_lock.json', 'expanded_units.json'):
        assert (Path(req.output_dir) / name).exists() == (stale is None)


@pytest.mark.parametrize('explicit_null', [False, True])
@pytest.mark.parametrize('finalize', [False, True])
def test_null_leaf_budget_is_bootstrap_in_finalizer_and_loader(tmp_path, monkeypatch, explicit_null, finalize):
    req, consumer = setup_lock(tmp_path)
    req = fork_profile(req, measurements=False)
    root = profile_directory(req)
    task = coverage.read_json(root / '_execution/task.json')
    payload = coverage.read_json(root / 'result.json')
    timing = coverage.read_json(root / 'timing.json')
    task['parameters'].update(wall_seconds=840, recipe=consumer['parameters']['recipe'])
    if explicit_null:
        task['parameters']['leaf_budget'] = None
    task.update(outputs=['result.json', 'continuation.tar'],
        continuation={'owner': 'profile', 'step': 0, 'predecessor': None})
    budget = coverage.task_leaf_budget(task)
    assert coverage.remaining_compute(budget, setup_seconds=20., import_seconds=0.) == 520.
    phases = dict(zip(coverage.OVERHEAD_PHASES, (20., 0., 100., 10.)))
    timing.update(measurement_scope='complete_chain_return', admitted_budget_seconds=840,
        complete_repetition_seconds=[210., 210.], setup_seconds=20., wall_seconds=550.,
        leaves=[profile_leaf('profile', 0, 420., phases, budget)])
    assert 420 <= 520 and 550 + coverage.publication_reserve(600) == 580 <= 600
    if finalize:
        req = finalize_profile_fixture(req, task, payload, timing, 550., monkeypatch)
    else:
        write(root / 'timing.json', timing)
        seal_budget_fixture(root, 'profile', task)
        req = replace(req, dependency_hashes=tuple(map(file_hash, req.dependency_paths)))
    record = coverage.load_budget(req, coverage.read_json(req.config_path), 'profile')
    assert record['complete'] and record['all_successful']
    assert record['leaves'][0]['compute_seconds'] == 420.


def test_chained_allocation_includes_setup_compute_and_finalization():
    wall = coverage.maximum_leaf_stage_seconds()
    allocation = coverage.leaf_allocation(wall)
    budget = coverage.leaf_budget(wall, 1., dict.fromkeys(coverage.OVERHEAD_PHASES, 0.))
    assert allocation['setup_seconds'] >= 300
    assert budget['reserve_seconds'] >= 300
    assert allocation['allocation_seconds'] <= 14400
    assert wall < allocation['allocation_seconds']
    # An injected 94-second clone/setup, bounded compute and 240-second
    # finalization all fit, even when setup consumes its entire declared margin.
    for setup in (94., allocation['setup_seconds']):
        clock = setup + budget['compute_seconds'] + 240.
        assert clock < allocation['allocation_seconds']
    with pytest.raises(ContractError, match='including setup'):
        coverage.leaf_allocation(14400)


def test_slice_sizing_pays_in_fold_stopping_headroom():
    from oxyformer.training.fit import DeadlineTiming
    wall = coverage.maximum_leaf_stage_seconds()
    overheads = dict.fromkeys(coverage.OVERHEAD_PHASES, 0.)
    compute = coverage.leaf_budget(wall, 1., overheads)['compute_seconds']
    clock = [0.]
    timing = DeadlineTiming(compute, clock=lambda: clock[0])
    while not timing.boundary('nuisance'):
        clock[0] += 30.  # A bounded unit much longer than the former 1 s margin.
    assert clock[0] < compute
    profile = {'seconds_per_repetition': compute, 'leaf_overheads': overheads}
    # A repetition needing the entire compute ceiling cannot fit a leaf which
    # must durably stop earlier. Sizing must pay this headroom on every leaf.
    assert campaign.slice_count(profile, 1, wall, 1.) == 3


@pytest.mark.parametrize("estimated_slices", [1, 2, 39])
def test_exact_slice_sizing_reserves_one_successor(estimated_slices):
    # Stage 1,990 = 300 reserve + 690 stopping margin + 1,000 useful seconds.
    profile = {"seconds_per_repetition": 1000.,
               "leaf_overheads": dict.fromkeys(coverage.OVERHEAD_PHASES, 0.)}
    assert campaign.slice_count(profile, estimated_slices, 1990, 1.) == estimated_slices + 1


def test_slice_sizing_refuses_forty_one_with_required_spare():
    profile = {"seconds_per_repetition": 1000.,
               "leaf_overheads": dict.fromkeys(coverage.OVERHEAD_PHASES, 0.)}
    with pytest.raises(ContractError, match="required spare slice exceeds forty leaves"):
        campaign.slice_count(profile, 40, 1990, 1.)
    # The screening search can inspect an infeasible count, never drop its spare.
    assert campaign.slice_count(profile, 40, 1990, 1., enforce_limit=False) == 41


def test_campaign_refuses_aggregate_spares_over_forty(tmp_path):
    req, task = setup_chain_lock(tmp_path)
    # Twenty work items each need two estimated slices plus their own spare.
    task["parameters"]["wall_seconds"] = 900 + int(2 * campaign.checkpoint_margin_seconds())
    write(Path(req.task_path), task)
    result = campaign.run_stage(replace(req, task_hash=file_hash(req.task_path)))
    assert result.status == "blocked" and "required spare slices exceeds forty leaves" in result.message
    assert not result.artifacts and not list(Path(req.output_dir).iterdir())
