"""Offline tests of the real nested controller and its stage boundary."""
from dataclasses import replace
from pathlib import Path
import json
import shutil
import tarfile

import pytest
import torch
import yaml

from test_fit import make_case
from oxyformer.contracts import StageRequest
from oxyformer.design.eligibility import GeographyRow, GeographyTable
from oxyformer.provenance import ContractError, canonical_json, file_hash
from oxyformer.training import nested_cv as nested
from oxyformer.training.checkpoint import CheckpointArtifact, load_checkpoint
from oxyformer.training.fit import subset
from oxyformer.validation.leakage import assert_fitted_invariant, assert_prediction_invariant


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def endpoint(root):
    _, split, manifest, config = make_case(root)
    rows = tuple(GeographyRow(original_id=i, tract_id=i, county="c", state="s", subblock=str(j // 2),
        assignment_geography=i, latitude=float(j), longitude=0., outcome_flag=1,
        label_available=i != "unlabeled-acs") for j, i in enumerate(manifest.original_ids))
    return nested.PreparedEndpoint(data=config.data, entity_graph=config.entity_graph,
        geography=GeographyTable(rows=rows, data_manifest_hash=manifest.content_hash, county_field="county",
            approval_reference="synthetic", mapping_review_id="synthetic"), outer=split, inner=(config.inner,),
        policy=config.policy, policy_covariates=config.policy_covariates,
        treatment_design=config.treatment_design, feature_kinds=config.feature_kinds,
        families=config.families, county_field=config.county_field,
        exposure_assignment_level=config.exposure_assignment_level)


def tiny_config(prepared, root, **kwargs):
    return prepared.configuration(0, root, ssl_epochs=1,
        settings=nested.NuisanceSettings(batch_size=256, frozen_epochs=1), **kwargs)


def fitted(root, prepared=None, **kwargs):
    prepared = prepared or endpoint(root)
    config = tiny_config(prepared, root, **kwargs)
    return prepared, nested.run_fold(config, prepared.outer, 1103)


def predictions(prepared, artifact):
    view = subset(prepared.data.covariates(("x",)), artifact.prediction_inputs.original_ids)
    return nested.predict(artifact, view, prepared.policy)


@pytest.fixture(scope="session")
def full(tmp_path_factory):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    result = fitted(tmp_path_factory.mktemp("nested-full") / "work")
    torch.set_num_threads(old)
    return result


def state(artifact):
    return load_checkpoint(artifact.checkpoint, artifact.checkpoint.identity)["controller"]


def test_registered_schedule(full):
    prepared, artifact = full
    assert artifact.complete
    assert state(artifact)["counts"]["ssl_fits"] == 4
    assert state(artifact)["counts"]["nuisance_fits"] == 26
    assert len(state(artifact)["results"]) == 24
    assert state(artifact)["compute"]["cpu_seconds"] > 0
    assert prepared.outer.seed_ids == nested.SEEDS
    config = yaml.safe_load(Path("configs/training/nested.yaml").read_text())
    assert config["outer_folds"] == 5 and config["inner_folds"] == 3
    assert len(nested.NuisanceSettings().grid) == 4
    assert config["seeds"] == list(nested.SEEDS)


@pytest.mark.parametrize("batches", [1, 3, 15, 30])
def test_portable_continuation_matches_uninterrupted(full, tmp_path, batches):
    prepared, expected = full
    config = tiny_config(prepared, tmp_path / "old" / "work", max_batches=batches)
    partial = nested.run_fold(config, prepared.outer, 1103)
    assert not partial.complete
    archive = tmp_path / "continuation.tar"
    chain = {"owner": "synthetic-owner", "step": 0, "predecessor": None}
    nested.export_continuation(partial, archive, binding={"recipe": "fixed"}, task_id="first", chain=chain,
                               allowed_root=tmp_path / "old")
    shutil.rmtree(tmp_path / "old")
    destination = tmp_path / "next"
    destination.mkdir()
    restored = nested.import_continuation(archive, destination, binding={"recipe": "fixed"},
        chain={"owner": "synthetic-owner", "step": 1, "predecessor": "first"})
    actual = nested.run_fold(tiny_config(prepared, destination / "work", predecessor=restored), prepared.outer, 1103)
    assert_fitted_invariant(expected, actual)
    assert_prediction_invariant(predictions(prepared, expected), predictions(prepared, actual))
    assert state(expected)["counts"] == state(actual)["counts"]


@pytest.mark.parametrize("variant", ["A1", "A2", "A3", "A4"])
def test_registered_transformer_variants(tmp_path, variant):
    prepared = endpoint(tmp_path)
    artifact = nested.run_fold(tiny_config(prepared, tmp_path / "work"), prepared.outer, 1103, variant=variant)
    assert artifact.complete
    assert state(artifact)["counts"]["ssl_fits"] == (0 if variant == "A1" else 4)
    assert state(artifact)["counts"]["nuisance_fits"] == 26
    predictions(prepared, artifact)
    model = nested._build(state(artifact)["final"]["outcome"])
    assert (model.county_context is None) == (variant == "A2")
    if variant == "A3":
        assert type(model).__name__ == "EarlyFusionOutcome"
    if variant == "A4":
        assert type(model).__name__ == "VaryingCoefficientOutcome"


def request(root, prepared, *, predecessor=None, stage="primary", variant="A0", max_batches=None):
    root.mkdir()
    inputs = root / "inputs"
    inputs.mkdir()
    (inputs / "endpoint.json").write_text(prepared.to_json())
    lock = {"nested_cv": {"ssl_epochs": 1, "frozen_epochs": 1, "batch_size": 256, "synthetic": True}}
    (inputs / "recipe.json").write_text(canonical_json(lock))
    dependencies = {"input": str(inputs)}
    files = [inputs / "endpoint.json", inputs / "recipe.json"]
    if predecessor:
        dependencies["first"] = str(predecessor)
        files.append(predecessor / "continuation.tar")
    task = {"id": "second" if predecessor else "first", "stage": stage,
        "parameters": {"endpoint_input": {"dependency": "input", "path": "endpoint.json"},
            "endpoint": prepared.data.manifest.spec.endpoint, "target_id": prepared.data.manifest.spec.target_id,
            "outer_fold": 0, "seed": 1103, "variant": variant},
        "recipe_lock": {"dependency": "input", "path": "recipe.json", "sha256": file_hash(inputs / "recipe.json")},
        "continuation": {"owner": "test", "step": 1 if predecessor else 0,
                         "predecessor": "first" if predecessor else None},
        "slice": {"max_batches": max_batches}}
    (root / "task.json").write_text(canonical_json(task))
    (root / "config.json").write_text(canonical_json({"dependencies": dependencies}))
    out = root / "out"
    out.mkdir()
    return StageRequest(stage=stage, config_path=str(root / "config.json"), config_hash=file_hash(root / "config.json"),
        task_path=str(root / "task.json"), task_hash=file_hash(root / "task.json"),
        dependency_paths=tuple(map(str, files)), dependency_hashes=tuple(map(file_hash, files)),
        output_dir=str(out), code_identity="8c5342e2f28fc744f5af24a3f74a67ba2d963d1f")


def test_stage_exports_and_declared_predecessor(full, tmp_path):
    prepared, expected = full
    first = request(tmp_path / "first", prepared, max_batches=1)
    result = nested.run_stage(first)
    assert result.status == "pass", result.message
    result.verify(first)
    out = Path(first.output_dir)
    assert {a.path for a in result.artifacts} == {"continuation.tar", "progress.json", "artifact_manifest.json"}
    assert json.loads((out / "progress.json").read_text())["state"] == "checkpointed"
    second = request(tmp_path / "second", prepared, predecessor=out)
    result = nested.run_stage(second)
    assert result.status == "pass", result.message
    result.verify(second)
    complete = Path(second.output_dir)
    assert json.loads((complete / "progress.json").read_text())["complete"]
    assert {"nuisances.parquet", "model_bundle.tar", "metrics.json"} <= {a.path for a in result.artifacts}
    import pandas as pd
    frame = pd.read_parquet(complete / "nuisances.parquet")
    assert not {"y", "outcome", "life_expectancy_years"} & set(frame.columns)
    assert tuple(frame.mu_a) == predictions(prepared, expected).mu_a


@pytest.mark.parametrize("field", ["code_identity", "target_id", "seed", "environment", "recipe"])
def test_stale_continuation_rejected(full, tmp_path, field):
    prepared, _ = full
    first = request(tmp_path / "first", prepared, max_batches=1)
    assert nested.run_stage(first).status == "pass"
    second = request(tmp_path / "second", prepared, predecessor=Path(first.output_dir))
    if field == "code_identity":
        second = replace(second, code_identity="1" * 40)
    elif field == "environment":
        # Direct portable API must reject a changed environment binding.
        with pytest.raises(ContractError, match="fingerprint"):
            nested.import_continuation(Path(first.output_dir) / "continuation.tar", Path(second.output_dir),
                binding={"environment": "different"}, chain={"owner": "test", "step": 1, "predecessor": "first"})
        return
    elif field == "recipe":
        path = Path(second.dependency_paths[1])
        value = json.loads(path.read_text())
        value["nested_cv"]["batch_size"] = 16
        path.write_text(canonical_json(value))
        task_path = Path(second.task_path)
        task = json.loads(task_path.read_text())
        task["recipe_lock"]["sha256"] = file_hash(path)
        task_path.write_text(canonical_json(task))
        second = replace(second, dependency_hashes=tuple(map(file_hash, second.dependency_paths)), task_hash=file_hash(task_path))
    else:
        task_path = Path(second.task_path)
        task = json.loads(task_path.read_text())
        task["parameters"][field] = "different" if field == "target_id" else 2207
        task_path.write_text(canonical_json(task))
        second = replace(second, task_hash=file_hash(task_path))
    result = nested.run_stage(second)
    assert result.status == "fail"
    assert not (Path(second.output_dir) / "nuisances.parquet").exists()


def test_undeclared_predecessor_is_not_read(full, tmp_path):
    prepared, _ = full
    first = request(tmp_path / "first", prepared, max_batches=1)
    assert nested.run_stage(first).status == "pass"
    second = request(tmp_path / "second", prepared, predecessor=Path(first.output_dir))
    second = replace(second, dependency_paths=second.dependency_paths[:-1], dependency_hashes=second.dependency_hashes[:-1])
    result = nested.run_stage(second)
    assert result.status == "fail" and "undeclared" in result.message


def test_unsupported_variant_blocks_before_training(full, tmp_path):
    prepared, _ = full
    req = request(tmp_path / "request", prepared, stage="ablation", variant="A5")
    result = nested.run_stage(req)
    assert result.status == "blocked"
    assert not list(Path(req.output_dir).iterdir())
