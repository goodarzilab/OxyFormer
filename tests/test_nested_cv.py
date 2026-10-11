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
        assignment_geography=i, latitude=float(j), longitude=0., outcome_flag=1 if j % 2 == 0 else 3,
        label_available=i != "unlabeled-acs") for j, i in enumerate(manifest.original_ids))
    return nested.PreparedEndpoint(data=config.data, entity_graph=config.entity_graph,
        geography=GeographyTable(rows=rows, data_manifest_hash=manifest.content_hash, county_field="county",
            approval_reference="synthetic", mapping_review_id="synthetic"), outer=split, inner=(config.inner,),
        policy=config.policy, policy_covariates=config.policy_covariates,
        treatment_design=config.treatment_design, feature_kinds=config.feature_kinds,
        families=config.families, county_field=config.county_field,
        exposure_assignment_level=config.exposure_assignment_level)


def tiny_config(prepared, root, *, batch_size=256, **kwargs):
    return prepared.configuration(0, root, ssl_epochs=1,
        settings=nested.NuisanceSettings(batch_size=batch_size, frozen_epochs=1), **kwargs)


def fitted(root, prepared=None, **kwargs):
    prepared = prepared or endpoint(root)
    config = tiny_config(prepared, root, **kwargs)
    return prepared, nested.run_fold(config, prepared.outer, 1103, geography=prepared.geography)


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


@pytest.mark.parametrize("batches,batch_size", [(1, 256), (3, 256), (15, 256), (30, 256), (5, 8)])
def test_portable_continuation_matches_uninterrupted(full, tmp_path, batches, batch_size):
    prepared, expected = full
    if batch_size != 256:
        expected = nested.run_fold(tiny_config(prepared, tmp_path / "uninterrupted", batch_size=batch_size),
            prepared.outer, 1103, geography=prepared.geography)
    config = tiny_config(prepared, tmp_path / "old" / "work", max_batches=batches, batch_size=batch_size)
    partial = nested.run_fold(config, prepared.outer, 1103, geography=prepared.geography)
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
    actual = nested.run_fold(tiny_config(prepared, destination / "work", predecessor=restored, batch_size=batch_size), prepared.outer, 1103, geography=prepared.geography)
    assert_fitted_invariant(expected, actual)
    assert_prediction_invariant(predictions(prepared, expected), predictions(prepared, actual))
    assert state(expected)["counts"] == state(actual)["counts"]


@pytest.mark.parametrize("variant", ["A1", "A2", "A3", "A4"])
def test_registered_transformer_variants(tmp_path, variant):
    prepared = endpoint(tmp_path)
    artifact = nested.run_fold(tiny_config(prepared, tmp_path / "work"), prepared.outer, 1103, geography=prepared.geography, variant=variant)
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
    record = next(a for a in result.artifacts if a.kind == "oof_nuisances")
    assert record.lineage.unit_ids == tuple(frame.original_ids)


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


def two_county_endpoint(root):
    prepared = endpoint(root)
    routes = {oid: f"c{j % 2}" for j, oid in enumerate(prepared.data.manifest.original_ids)}
    return replace(prepared,
        data=replace(prepared.data, rows=tuple(row[:4] + (routes[row[0]],) + row[5:] for row in prepared.data.rows)),
        geography=replace(prepared.geography, rows=tuple(replace(row, county=routes[row.original_id])
            for row in prepared.geography.rows)))


def test_stopping_reservation_rechecks_county_minima(tmp_path):
    prepared = two_county_endpoint(tmp_path)
    fitting = prepared.inner[0].split.training_ids(0)
    # Whole linked pairs remain together. Each county falls from eight fitting
    # tracts to three after five pairs are reserved for stopping.
    with pytest.raises(ContractError, match="four.*stopping"):
        tiny_config(prepared, tmp_path / "work", stopping_ids=((0, fitting[:10]),))


def test_work_symlink_refused_before_any_external_write(tmp_path):
    prepared = endpoint(tmp_path)
    req = request(tmp_path / "request", prepared, max_batches=1)
    external = tmp_path / "external"
    external.mkdir()
    (Path(req.output_dir) / "work").symlink_to(external, target_is_directory=True)
    result = nested.run_stage(req)
    assert result.status == "fail"
    assert list(external.iterdir()) == [], "stage wrote outside its output directory"


@pytest.mark.parametrize("stale", ["tar", "work"])
def test_invalid_attempt_returns_failed_stage_result(tmp_path, stale):
    prepared = endpoint(tmp_path)
    if stale == "tar":
        previous = tmp_path / "previous"
        previous.mkdir()
        (previous / "continuation.tar").write_bytes(b"truncated checkpoint")
        req = request(tmp_path / "request", prepared, predecessor=previous)
    else:
        req = request(tmp_path / "request", prepared, max_batches=1)
        (Path(req.output_dir) / "work" / "nuisance").mkdir(parents=True)
    result = nested.run_stage(req)
    assert result.status == "fail"
    assert not result.artifacts


def test_direct_controller_cannot_bypass_stopping_minima(tmp_path):
    prepared = two_county_endpoint(tmp_path)
    config = tiny_config(prepared, tmp_path / "work")
    fitting = prepared.inner[0].split.training_ids(0)
    invalid = replace(config, stopping_ids=((0, fitting[:10]),))
    with pytest.raises(ContractError, match="four.*stopping"):
        nested.run_fold(invalid, prepared.outer, 1103, geography=prepared.geography)
    assert not (tmp_path / "work").exists()
    valid = tiny_config(prepared, tmp_path / "valid", stopping_ids=((0, fitting[:6]),))
    assert valid.stopping_ids == ((0, fitting[:6]),)


def test_parquet_symlink_refused_before_fitting(tmp_path):
    prepared = endpoint(tmp_path)
    req = request(tmp_path / "request", prepared, max_batches=1)
    external = tmp_path / "sentinel"
    external.write_bytes(b"unchanged")
    (Path(req.output_dir) / "nuisances.parquet").symlink_to(external)
    result = nested.run_stage(req)
    assert result.status == "fail"
    assert external.read_bytes() == b"unchanged"
    assert not (Path(req.output_dir) / "work").exists()


def test_controller_preserves_frozen_split_identity(tmp_path):
    prepared = endpoint(tmp_path)
    outer = replace(prepared.outer, original_ids=prepared.outer.original_ids[::-1],
                    fold_ids=prepared.outer.fold_ids[::-1])
    artifact = nested.run_fold(tiny_config(prepared, tmp_path / "work", max_batches=1),
        outer, 1103, geography=prepared.geography)
    assert artifact.split.content_hash == outer.content_hash


@pytest.mark.parametrize("level", ["outer", "inner"])
def test_repeated_tract_cannot_cross_folds(tmp_path, level):
    prepared = endpoint(tmp_path)
    source_id, repeated_id = (("o00", "o06") if level == "outer" else ("o06", "o08"))
    original = next(r for r in prepared.geography.rows if r.original_id == source_id)
    geography = replace(prepared.geography, rows=tuple(
        replace(original, original_id=repeated_id) if row.original_id == repeated_id else row
        for row in prepared.geography.rows))
    changed = replace(prepared, geography=geography)
    with pytest.raises(ContractError, match="geographic component"):
        tiny_config(changed, tmp_path / "work")


def test_direct_controller_rejects_global_county_partition(tmp_path):
    prepared = endpoint(tmp_path)
    routes = {oid: "A" if j < 18 else "B" for j, oid in enumerate(prepared.data.manifest.original_ids)}
    data = replace(prepared.data, rows=tuple(row[:4] + (routes[row[0]],) + row[5:] for row in prepared.data.rows))
    geography = replace(prepared.geography, rows=tuple(replace(row, county=routes[row.original_id])
        for row in prepared.geography.rows))
    # Bypass PreparedEndpoint.configuration exactly as in the reported direct API case.
    config = replace(make_case(tmp_path / "work")[-1], data=data, max_batches=1)
    with pytest.raises(ContractError, match="within counties"):
        nested.run_fold(config, prepared.outer, 1103, geography=geography)
    assert not (tmp_path / "work").exists()


def geographic_stopping_endpoint(root):
    """A valid split whose geographic groups are larger than entity links."""
    prepared = endpoint(root)
    inner = prepared.inner[0]
    ids = inner.split.training_ids(0)
    groups = [g for g in prepared.entity_graph.components() if set(g) <= set(ids)]
    stop = min(groups, key=lambda group: nested.digest([1103, group, "ssl-stopping"]))
    outer_folds = dict(zip(prepared.outer.original_ids, prepared.outer.fold_ids))
    other = next(g for g in groups if g != stop and outer_folds[g[0]] == outer_folds[stop[0]])
    inner_folds = dict(zip(inner.split.original_ids, inner.split.fold_ids))
    split = replace(inner.split, fold_ids=tuple(inner_folds[stop[0]] if i in other else inner_folds[i]
                                               for i in inner.split.original_ids))
    row = next(r for r in prepared.geography.rows if r.original_id == stop[0])
    geography = replace(prepared.geography, rows=tuple(
        replace(row, original_id=other[0]) if r.original_id == other[0] else r
        for r in prepared.geography.rows))
    return replace(prepared, inner=(replace(inner, split=split),), geography=geography), stop


def test_ssl_stopping_preserves_geographic_components(tmp_path):
    from oxyformer.design.splits import dependence_groups
    prepared, _ = geographic_stopping_endpoint(tmp_path)
    artifact = nested.run_fold(tiny_config(prepared, tmp_path / "work", max_batches=1),
        prepared.outer, 1103, geography=prepared.geography)
    pending = CheckpointArtifact.from_json(state(artifact)["ssl_pending"])
    saved = load_checkpoint(pending, pending.identity)
    for group in dependence_groups(prepared.geography.rows, prepared.entity_graph):
        assert not (set(group).intersection(saved["fitting_ids"]) and
                    set(group).intersection(saved["stopping_ids"])), "SSL splits a geographic component"


def test_nuisance_stopping_preserves_geographic_components(tmp_path):
    prepared, stop = geographic_stopping_endpoint(tmp_path)
    with pytest.raises(ContractError, match="geographic component"):
        tiny_config(prepared, tmp_path / "work", stopping_ids=((0, stop),))


@pytest.mark.parametrize("partition", ["training", "evaluation"])
def test_nested_primary_flags_reject_predicted_labels(tmp_path, partition):
    prepared = endpoint(tmp_path)
    prepared.validate(0)  # mixed flag-1/3 frame is accepted throughout fitting
    ids = (prepared.outer.training_ids(0) if partition == "training" else
           tuple(i for i, fold in zip(prepared.outer.original_ids, prepared.outer.fold_ids) if fold == 0))
    bad = replace(prepared, geography=replace(prepared.geography, rows=tuple(
        replace(r, outcome_flag=2) if r.original_id == ids[0] else r for r in prepared.geography.rows)))
    with pytest.raises(ContractError, match="non-primary or unavailable"):
        bad.validate(0)


@pytest.mark.parametrize("decision", [None, {}, {"primary": [1]}])
def test_nested_refuses_missing_or_malformed_outcome_decision(tmp_path, monkeypatch, decision):
    from oxyformer.data.adapters import usaleep
    prepared = endpoint(tmp_path)
    approvals = yaml.safe_load(usaleep.OWNER_APPROVALS.read_text())
    if decision is None:
        del approvals["owner_decisions"]["outcome_flags"]
    else:
        approvals["owner_decisions"]["outcome_flags"] = decision
    owner_file = tmp_path / "owner.yaml"
    owner_file.write_text(yaml.safe_dump(approvals))
    monkeypatch.setattr(usaleep, "OWNER_APPROVALS", owner_file)
    with pytest.raises(ValueError, match="owner_decisions.outcome_flags"):
        prepared.validate(0)


def mixed_role_endpoint(root):
    prepared = endpoint(root)
    frame = prepared.outer.original_ids
    roles = {i: "inert" if i == "o29" else "training" if j % 4 == 0 else "target"
             for j, i in enumerate(frame)}
    def add_roles(split):
        targets = [(i, f) for i, f in zip(split.original_ids, split.fold_ids) if roles[i] == "target"]
        return replace(split, original_ids=tuple(i for i, _ in targets), fold_ids=tuple(f for _, f in targets),
                       frame_ids=split.original_ids, frame_fold_ids=split.fold_ids,
                       membership_roles=tuple(roles[i] for i in split.original_ids))
    return replace(prepared, outer=add_roles(prepared.outer),
                   inner=tuple(replace(inner, split=add_roles(inner.split)) for inner in prepared.inner),
                   geography=replace(prepared.geography, rows=tuple(
                       replace(r, outcome_flag=2) if r.original_id == "o29" else r
                       for r in prepared.geography.rows)))


def test_nested_consumes_roles_and_classifier_only_trains_on_target(tmp_path, monkeypatch):
    prepared = mixed_role_endpoint(tmp_path)
    seen = []
    original = nested._train_transaction
    def observe(config, bundle, encoder_state, view, stopping, *args, **kwargs):
        seen.append((bundle["kind"], set(view.original_ids), set(stopping)))
        return original(config, bundle, encoder_state, view, stopping, *args, **kwargs)
    monkeypatch.setattr(nested, "_train_transaction", observe)
    _, artifact = fitted(tmp_path / "fit", prepared)
    assert artifact.complete
    assert artifact.prediction_inputs.original_ids == prepared.outer.evaluation_ids(0)
    allowed = set(prepared.outer.training_ids(0))
    targets = set(prepared.outer.target_ids)
    assert all(ids <= allowed and "o29" not in ids for _, ids, _ in seen)
    assert any(kind == "outcome" and ids - targets for kind, ids, _ in seen)
    assert all(ids | stops <= targets for kind, ids, stops in seen if kind == "origin")
    calibration = nested.AffineCalibration.from_json(state(artifact)["calibration"])
    assert set(calibration.original_ids) <= targets & allowed
    prediction = predictions(prepared, artifact)
    assert set(prediction.original_ids) <= targets


def test_nested_refuses_held_out_target_in_training_partition(tmp_path):
    prepared = mixed_role_endpoint(tmp_path)
    held = prepared.outer.evaluation_ids(0)[0]
    inner = prepared.inner[0]
    # Substitute the held-out target for one permitted training row, preserving
    # internally consistent IDs, source provenance, graph and role fields.
    removed = inner.split.original_ids[0]
    def rename(ids):
        return tuple(held if i == removed else i for i in ids)
    graph = replace(inner.entity_graph, original_ids=rename(inner.entity_graph.original_ids),
                    links=tuple(replace(link, observation_id=held) if link.observation_id == removed else link
                                for link in inner.entity_graph.links))
    manifest = replace(inner.data_manifest, original_ids=rename(inner.data_manifest.original_ids),
                       entity_graph_hash=graph.content_hash,
                       lineage=replace(inner.data_manifest.lineage, unit_ids=rename(inner.data_manifest.original_ids)))
    split = replace(inner.split, original_ids=rename(inner.split.original_ids), frame_ids=rename(inner.split.frame_ids),
                    entity_graph_hash=graph.content_hash,
                    lineage=replace(inner.split.lineage, unit_ids=manifest.original_ids, parent_hashes=(manifest.content_hash,)))
    bad = replace(prepared, inner=(replace(inner, data_manifest=manifest, entity_graph=graph, split=split),))
    with pytest.raises(ContractError, match="inner escapes outer training"):
        bad.validate(0)


def test_nested_accepts_empty_target_fold_with_labelled_frame_presence(tmp_path, monkeypatch):
    prepared = mixed_role_endpoint(tmp_path)
    inner = prepared.inner[0]
    demoted = {i for i, fold in zip(inner.split.original_ids, inner.split.fold_ids) if fold == 2}
    def demote(split):
        targets = [(i, f) for i, f in zip(split.original_ids, split.fold_ids) if i not in demoted]
        return replace(split, original_ids=tuple(i for i, _ in targets), fold_ids=tuple(f for _, f in targets),
                       membership_roles=tuple('training' if i in demoted else role
                                              for i, role in zip(split.frame_ids, split.membership_roles)))
    prepared = replace(prepared, outer=demote(prepared.outer),
                       inner=(replace(inner, split=demote(inner.split)),))
    assert set(prepared.inner[0].split.fold_ids) == {0, 1}
    assert set(prepared.inner[0].split.geographic_fold_ids) == {0, 1, 2}
    ssl_ids = []
    original = nested.fitting_module_ssl
    def observe(config, split, fitting, *args, **kwargs):
        ssl_ids.append(set(fitting))
        return original(config, split, fitting, *args, **kwargs)
    monkeypatch.setattr(nested, 'fitting_module_ssl', observe)
    _, artifact = fitted(tmp_path / 'fit', prepared)
    assert ssl_ids and all(demoted <= ids for ids in ssl_ids)
    assert artifact.complete
    calibration = nested.AffineCalibration.from_json(state(artifact)['calibration'])
    assert {p.fold for p in calibration.partitions} == {0, 1}
    assert set(calibration.original_ids) == set(prepared.outer.training_ids(0)) & set(prepared.outer.target_ids)
    assert set(predictions(prepared, artifact).original_ids) == set(prepared.outer.evaluation_ids(0))


def test_origin_stopping_regime_uses_only_target_rows(tmp_path, monkeypatch):
    prepared = mixed_role_endpoint(tmp_path)
    allowed = set(prepared.inner[0].split.training_ids(0))
    targets = set(prepared.outer.target_ids)
    groups = nested.dependence_groups(prepared.geography.rows, prepared.entity_graph)
    stopping = next(tuple(i for i in g if i in allowed) for g in groups
                    if set(g) & allowed and set(g) & allowed <= allowed - targets)
    config = tiny_config(prepared, tmp_path / 'fit', stopping_ids=((0, stopping),))
    seen = []
    original = nested._train_transaction
    def observe(config, bundle, encoder, view, stops, epochs, *args, **kwargs):
        seen.append((bundle['kind'], tuple(stops), epochs))
        if bundle['kind'] == 'origin':
            assert not stops
            assert epochs == config.settings.frozen_epochs
        # Bound the outcome-stopping fixture; scheduling and metadata are the
        # subject here, while the actual training loop is exercised elsewhere.
        return original(config, bundle, encoder, view, stops, min(epochs, 1), *args, **kwargs)
    monkeypatch.setattr(nested, '_train_transaction', observe)
    artifact = nested.run_fold(config, prepared.outer, 1103, geography=prepared.geography)
    assert artifact.complete
    assert any(kind == 'outcome' and stops == stopping and epochs == config.settings.max_epochs
               for kind, stops, epochs in seen)
    calibration = nested.AffineCalibration.from_json(state(artifact)['calibration'])
    for audit in calibration.partitions:
        assert not audit.checkpoint_ids
        assert audit.frozen_epochs == config.settings.frozen_epochs
        assert set(audit.fitting_ids) <= targets
    outcome = next(r for r in state(artifact)['results'] if r['kind'] == 'outcome' and r['fold'] == 0)
    audit = nested.CalibrationPartition.from_json(outcome['ownership'])
    assert audit.checkpoint_ids == stopping and audit.frozen_epochs is None


def test_cost_study_two_grid_points_keep_nested_tuning_and_cap_reload(tmp_path, full):
    from oxyformer.training.fit import RecipeStudy
    prepared, registered = full
    config = tiny_config(prepared, tmp_path / "candidate", study=RecipeStudy(grid_points=2, reference_cap=2))
    artifact = nested.run_fold(config, prepared.outer, 1103, geography=prepared.geography)
    controller = state(artifact)
    assert controller["counts"]["ssl_fits"] == 4
    assert controller["counts"]["nuisance_fits"] == 14
    assert len(controller["results"]) == 12
    assert {(r["fold"], r["kind"], r["grid"]) for r in controller["results"]} == {
        (f, k, g) for f in range(3) for k in ("outcome", "origin") for g in range(2)}
    for kind, bundle in controller["final"].items():
        assert bundle["dropout"] == 0. and bundle["learning_rate"] in (3e-4, 1e-3)
        assert bundle["reference_cap"] == 2
        model = nested._build(bundle)
        assert model.county_context.reference_cap == 2
        assert model.county_context.references.original_ids == nested._build(state(registered)["final"][kind]).county_context.references.original_ids
        assert all(len(model.county_context._reference_ids(i, c)) == 2 for i, c in model.county_context._routes.items())
    actual = predictions(prepared, artifact)
    assert actual.original_ids == predictions(prepared, registered).original_ids


def test_cost_study_frozen_schedule_reaches_controller_without_patience(tmp_path, full, monkeypatch):
    from oxyformer.training.fit import RecipeStudy, _Budget
    from oxyformer.training.checkpoint import CheckpointRequest
    prepared, registered = full
    config = tiny_config(prepared, tmp_path / "candidate", device="cpu",
                         study=RecipeStudy(frozen_epochs=30, grid_points=2))
    original = nested._train_transaction
    requested = []
    def shortened(config, bundle, encoder, view, stopping, epochs, seed, budget, saved=None):
        requested.append((bool(stopping), epochs))
        return original(config, bundle, encoder, view, stopping, 1, seed, budget, saved)
    monkeypatch.setattr(nested, "_train_transaction", shortened)
    result = nested.run_fold(config, prepared.outer, 1103, geography=prepared.geography)
    assert requested[:12] == [(False, 30)] * 12
    assert requested[12:] == [(False, 1)] * 2  # median of this test's shortened inner fits
    assert all(json.loads(r["ownership"])["payload"]["frozen_epochs"] == 30 for r in state(result)["results"])
    # Exercise the actual transaction for longer than registered patience=15.
    bundle = state(registered)["final"]["outcome"]
    ids = prepared.outer.training_ids(0)
    view = subset(prepared.data.covariates(("x",)), ids)
    _, trained, complete, reason = original(config, bundle, None, view, (), 30, 1103,
                                           _Budget(config, CheckpointRequest()))
    assert complete and reason == "max_epochs"
    assert trained["progress"]["epoch"] == 30 and trained["progress"]["bad_epochs"] == 0
    assert trained["progress"]["history"] == [None] * 30
