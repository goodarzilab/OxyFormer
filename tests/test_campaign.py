"""Prospective locking and full fan-in with synthetic, offline receipts."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from oxyformer.contracts import StageRequest
from oxyformer.provenance import ContractError, canonical_json, file_hash
from oxyformer.validation import campaign, coverage
from oxyformer.validation.scm import SCMConfig
from test_coverage import successful_records, synthetic_endpoint

STAMPS = {"scientific_fingerprint": {"algorithm": "tracked-science-v2", "sha256": "a" * 64},
          "environment_hash": "b" * 64}


@pytest.fixture(autouse=True)
def frozen_fingerprint(monkeypatch):
    # Scientific fingerprinting itself is tested by the execution unit; this
    # unit verifies binding/drift while its own checkout is being edited.
    monkeypatch.setattr(campaign, "fingerprint", lambda: deepcopy(STAMPS))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value))
    return path


def request(root, task, dependencies, approvals=None):
    root.mkdir(parents=True, exist_ok=True)
    config = write(root / "config.json", {"dependencies": {k: str(v) for k, v in dependencies.items()},
        "approvals": approvals or {"owner_decisions": {"campaign_allocations": {
            "test-final": {"kind": "final-coverage", "gpu_hours": 0}}}}})
    task_path = write(root / "task.json", task)
    paths = tuple(str(p) for directory in dependencies.values() for p in sorted(directory.glob("*.json")))
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
        "complete_repetition_seconds": [10., 11.], "wall_seconds": 25., **STAMPS})
    parameters = {"campaign_id": "test", "recipe": recipe, "scenario_family": [scenario],
        "final_repetitions": 1000, "screening_repetitions": 5, "repetitions_per_leaf": 25,
        "gpus": 0, "wall_seconds": 600, "profile_safety_factor": 2., "evaluation_namespace": "prospective-one",
        "retry_rules": deepcopy(coverage.RETRY_RULES), "prior_locks": [],
        "profiles": {"null_effect": {"dependency": "profile", "path": "result.json"}},
        "endpoint_input": {"dependency": "inputs", "path": "endpoint.json"},
        "frame_input": {"dependency": "inputs", "path": "frame.json"}, "cpu_wall_hours_budget": 10.}
    parameters.update(changes)
    task = {"id": "campaign-lock", "stage": "campaign-lock", "parameters": parameters}
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
    path = tmp_path / "profile/timing.json"
    data = coverage.read_json(path)
    data["production_equivalent"] = False
    write(path, data)
    req = replace(req, dependency_hashes=tuple(map(file_hash, req.dependency_paths)))
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
    result = coverage.run_stage(req)
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
        path = tmp_path / "profile" / name
        value = coverage.read_json(path)
        value["recipe_hash"] = coverage.digest(recipe)
        write(path, value)
    write(Path(req.task_path), task)
    req = replace(req, task_hash=file_hash(req.task_path), dependency_hashes=tuple(map(file_hash, req.dependency_paths)))
    result = campaign.run_stage(req)
    assert result.status == "pass", result.message
    result.verify(req)


@pytest.mark.parametrize("seconds,status", [(250, "blocked"), (270, "pass")])
def test_lock_accounts_for_measured_setup_overhead(tmp_path, seconds, status):
    req, _ = setup_lock(tmp_path, profile_safety_factor=1., wall_seconds=seconds)
    path = tmp_path / "profile/timing.json"
    timing = coverage.read_json(path)
    timing.update(wall_seconds=40., complete_repetition_seconds=[10., 10.])
    write(path, timing)
    req = replace(req, dependency_hashes=tuple(map(file_hash, req.dependency_paths)))
    result = campaign.run_stage(req)
    # 20 seconds of measured setup + 25 * 10-second repetitions need 270 s.
    assert result.status == status
    if status == "blocked":
        assert "infeasible batch budget" in result.message
