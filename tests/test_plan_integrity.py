"""Offline checks of admission declarations and real bounded expansions.

Synthetic budgets below exercise the planner, not runtime admission or owner
authorization. Production admission needs measured profiles and the owner file.
"""
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest
import yaml

from oxyformer.execution.campaign import expand_campaign, validate_plan
from oxyformer.execution.runner import dependency_variable, resolve_dependencies
from oxyformer.provenance import ContractError
from oxyformer.reporting.records import STAGE_GATES
from oxyformer.validation.coverage import repetition_plan, summarize

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs/execution/phase_b_templates.json"


def catalog():
    return json.loads(CATALOG.read_text())


def expansion():
    # A planner-only synthetic two-work, two-slice instance. No jobs submitted.
    spec = {
        "schema_version": 1, "id": "synthetic-chain", "kind": "primary",
        "prerequisites": ["nested-cv"],
        "inputs": {"campaign-lock": ["recipe_lock.json"], "tract-gate": ["endpoint.json"]},
        "recipe_lock": {"dependency": "campaign-lock", "path": "recipe_lock.json", "sha256": "1" * 64},
        "work": [{"id": f"fold-{fold}-seed-1103", "stage": "primary",
                  "parameters": {"outer_fold": fold, "seed": 1103},
                  "outputs": ["continuation.tar", "progress.json", "artifact_manifest.json"],
                  "slices": [{"gpus": 0, "wall_seconds": 120}, {"gpus": 0, "wall_seconds": 120}]}
                 for fold in (0, 1)],
        "collector": {"stage": "audit-collect", "wall_seconds": 120,
                      "outputs": ["report.json", "report.html", "estimators.svg"]},
    }
    return expand_campaign(spec, {})


def test_catalog_preserves_science_and_admission_boundaries():
    value = catalog()
    approvals = yaml.safe_load((ROOT / "configs/approvals.yaml").read_text())
    fixed = approvals["plan_fixed"]
    assert value["executable"] is False
    assert len(value["production"]["outer_folds"]) == fixed["outer_folds"] == 5
    assert len(value["production"]["inner_folds"]) == fixed["inner_folds"] == 3
    assert value["production"]["seeds"] == fixed["seeds"]
    training = yaml.safe_load((ROOT / "configs/training/nested.yaml").read_text())
    for name in ("learning_rates", "dropouts"):
        assert value["production"][name] == training[name]
    assert value["production"]["ssl_max_epochs"] == training["ssl_epochs"] == 30
    assert value["production"]["nuisance_max_epochs"] == training["max_epochs"] == 150
    assert value["production"]["minimum_final_repetitions_per_scenario"] == approvals["owner_decisions"]["release_gates"]["min_repetitions_per_scenario"] == 1000
    assert value["limits"] == dict(leaves_per_instance=40, gpu_hours_per_leaf=4,
                                  run_gpu_hours=2500, gpu_concurrency=8, cpu_concurrency=6)
    assert set(value["final_collector"]["required_gates"]) == set(STAGE_GATES["tract_release"])
    assert set(value["final_collector"]["dependency_roles"]) == {
        "primary-collector", "final-coverage-collector", "anchor-review", "audit-collect"}
    by_kind = {t["kind"]: t for t in value["templates"]}
    assert set(by_kind) == {"primary", "ablation", "screening", "final-coverage", "anchor", "refit-audit"}
    variants = yaml.safe_load((ROOT / "configs/models/ablations.yaml").read_text())["variants"]
    assert set(by_kind["ablation"]["variants"]) == set(variants) - {"A0", "A8", "B0", "D0"}
    assert set(by_kind["ablation"]["unit_weight_only"]) == {
        name for name, settings in variants.items() if settings.get("weights") == "unit_only"}
    assert by_kind["screening"]["after"] == "campaign-lock"
    assert by_kind["screening"]["recipe_mutable"] is False
    for kind in ("final-coverage", "anchor", "refit-audit"):
        assert by_kind[kind]["owner_allocation_required"] is True
    assert value["shared_root_promotion"] == {"planned": False, "authority_if_requested": "PI"}


def test_final_collector_dependencies():
    plan = expansion()
    validate_plan(plan, {})
    leaves = plan["expected_leaves"]
    assert len(leaves) == 4
    assert set(leaves) <= set(plan["units"][-1]["needs"])
    assert plan["tasks"][-1]["expected_leaves"] == leaves
    for first, second in zip(plan["tasks"][:4:2], plan["tasks"][1:4:2]):
        assert second["continuation"] == dict(owner=first["continuation"]["owner"], step=1, predecessor=first["id"])
        assert first["id"] in second["needs"]
        assert second["parameters"] == first["parameters"]
        assert "continuation.tar" in second["needs"][first["id"]]


def test_final_coverage_collector_dependencies(tmp_path, monkeypatch):
    # Existing synthetic profile receipts exercise the real lock builder. The
    # fixture's allocation and fingerprint are not production authorizations.
    from test_campaign import STAMPS, locked
    from oxyformer.validation import campaign
    monkeypatch.setattr(campaign, "fingerprint", lambda: deepcopy(STAMPS))
    _, lock, plans = locked(tmp_path)
    plan = plans[-1]
    approvals = {"owner_decisions": {"campaign_allocations": {
        "test-final": {"kind": "final-coverage", "gpu_hours": 0}}}}
    validate_plan(plan, approvals)
    screening = plans[0]["tasks"][-1]["id"]
    assert len(plan["expected_leaves"]) == 40
    assert set(plan["expected_leaves"]) <= set(plan["units"][-1]["needs"])
    assert all(screening in u["needs"] for u in plan["units"])
    draws = [d for b in lock["batches"] if b["mode"] == "final" for d in b["draws"]]
    assert len(draws) == len({d["repetition_id"] for d in draws}) == len({d["seed"] for d in draws}) == 1000
    assert all(t["recipe_lock"] == plan["tasks"][0]["recipe_lock"] for t in plan["tasks"])


def test_concrete_scopes_clone_commands_and_runtime():
    plan = expansion()
    for unit in plan["units"]:
        assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}", unit["id"])
        assert unit["write_scopes"] == [unit["id"] + "/**"]
        assert unit["runtime"] == "oxyformer-env" and unit["max_attempts"] == 1
        assert "--partition=standard" in unit["sbatch"] and "--account=root" in unit["sbatch"]
        assert "--time=00:02:00" in unit["sbatch"]
        assert unit["gpu_hours"] <= 4
        for path in unit["outputs"]:
            assert not Path(path).is_absolute() and ".." not in Path(path).parts
            assert path.split("/")[0] not in {"outputs", "report", "src"}
        command = unit["command"]
        clone = 'git clone --depth 1 --branch dev https://github.com/goodarzilab/OxyFormer.git "$SWARM_UNIT_DIR/src"'
        stamp = 'git -C "$SWARM_UNIT_DIR/src" rev-parse HEAD > "$SWARM_UNIT_DIR/code_commit.txt"'
        imports = 'export PYTHONPATH="$SWARM_UNIT_DIR/src/src"'
        assert command.index(clone) < command.index(stamp) < command.index(imports) < command.index(" -m oxyformer.cli")
        assert '--repo "$SWARM_UNIT_DIR/src" --out "$SWARM_UNIT_DIR" --deps-env' in command


@pytest.mark.parametrize("change", ["scope", "id", "runtime", "output", "clone", "collector", "chain"])
def test_expansion_drift_is_rejected(change):
    plan = expansion()
    if change == "scope":
        plan["units"][0]["write_scopes"] = ["**"]
    elif change == "id":
        plan["units"][0]["id"] = "unresolved-{fold}"
    elif change == "runtime":
        plan["units"][0]["runtime"] = "other-python"
    elif change == "output":
        plan["units"][0]["outputs"].append("../shared.json")
    elif change == "clone":
        plan["units"][0]["command"] = "python scripts/run_stage.py"
    elif change == "collector":
        plan["units"][-1]["needs"].remove(plan["expected_leaves"][0])
    else:
        plan["tasks"][1]["continuation"]["predecessor"] = None
    with pytest.raises(ContractError):
        validate_plan(plan, {})


@pytest.mark.parametrize("change", ["leaves", "gpu_hours", "allocation", "normalization"])
def test_limits_block_admission(change):
    spec = deepcopy(expansion()["spec"])
    if change == "leaves":
        spec["work"][0]["slices"] *= 20  # 42 leaves including work 2
    elif change == "gpu_hours":
        spec["work"][0]["slices"][0] = {"gpus": 2, "wall_seconds": 7260}
    elif change == "allocation":
        spec["kind"] = "final-coverage"
    else:
        spec["prerequisites"] += ["nested_cv"]
    with pytest.raises(ContractError):
        expand_campaign(spec, yaml.safe_load((ROOT / "configs/approvals.yaml").read_text()))


def test_only_normalized_dependencies_resolve(tmp_path):
    assert dependency_variable("atlas-mid-atlantic") == "SWARM_DEP_ATLAS_MID_ATLANTIC"
    assert resolve_dependencies(["atlas-mid-atlantic"], {"SWARM_DEP_ATLAS_MID_ATLANTIC": str(tmp_path)}) == {"atlas-mid-atlantic": tmp_path}
    with pytest.raises(ContractError, match="missing dependency variable"):
        resolve_dependencies(["atlas-mid-atlantic"], {"atlas-mid-atlantic": str(tmp_path)})


def test_early_registry_path_has_no_integration_barrier():
    stages = yaml.safe_load((ROOT / "configs/execution/stages.yaml").read_text())["stages"]
    rule = catalog()["early_path"]
    for name in rule["stages"]:
        assert not set(stages[name].get("needs", {})).intersection(rule["must_not_depend_on"])
    needs = stages["atlas-collect"]["needs"]
    assert set(needs) == {"atlas-inputs", "atlas-new-england", "atlas-mid-atlantic",
        "atlas-east-north", "atlas-west-north", "atlas-south-atlantic",
        "atlas-east-south", "atlas-west-south", "atlas-mountain", "atlas-pacific"}
    assert needs["atlas-inputs"] == ["atlas_shards.json", "raster_metadata.json"]
    # The coordinator's external DAG must also be checked at admission; this
    # repository registry is not evidence of that external state.


def test_missing_final_repetition_cannot_be_hidden_by_batching():
    draws = repetition_plan("synthetic-final", "null", 1000)
    batches = [draws[start:start + 25] for start in range(0, 1000, 25)]
    assert len(batches) == 40
    assert len({row["seed"] for batch in batches for row in batch}) == 1000
    result = summarize([], draws, production_equivalent=True, null_scenario=True)
    assert not result["complete"] and not result["coverage_pass"]
