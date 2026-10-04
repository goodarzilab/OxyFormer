"""Synthetic, CPU-only acceptance tests; no external payload or network access."""
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
from pathlib import Path

import pytest
import yaml

from oxyformer.contracts import (
    ColumnSpec, CovariateView, DataManifest, EstimandSpec, Estimate, OOFNuisances,
    SourceManifest, SplitManifest, StageRequest, StageResult, source_lineage_hash,
)
from oxyformer.data.entity_graph import EntityGraph, EntityLink
from oxyformer.data.feature_roles import FeatureRegistry, FeatureRule
from oxyformer.data.loaders import join_outcomes, load_records, validate_oof, validate_split
from oxyformer.provenance import (
    ArtifactLineage, ArtifactRecord, ContractError, file_hash, read_artifact, write_artifact,
)


def digest(text):
    return sha256(text.encode()).hexdigest()


@pytest.fixture
def fixture():
    # These names and permissions describe synthetic test data only.
    rules = tuple(FeatureRule(name=name, role=role, endpoints=("synthetic",), uses=uses,
                              approval_id="synthetic-fixture") for name, role, uses in (
        ("id", "identifier", ("linkage",)), ("y", "outcome", ("score",)),
        ("a", "exposure", ("score",)), ("x", "predictor", ("nuisance", "ssl", "context")),
        ("county", "county", ("county_routing",)),
        ("weight", "outcome_metadata", ("linkage",)),
        ("y_se", "outcome_metadata", ("diagnostic",)),
    ))
    registry = FeatureRegistry(registry_id="synthetic", rules=rules)
    ids = ("o1", "o2", "o3", "o4")
    graph = EntityGraph(original_ids=ids, links=())
    source = SourceManifest(source_id="synthetic", version="1", uri="synthetic://fixture",
                            payload_hash=digest("payload"), license_hash=digest("license"),
                            schema_hash=digest("source-schema"), field_mapping=(("raw-x", "x"),),
                            mapping_status="reviewed", mapping_review_id="synthetic-fixture")
    spec = EstimandSpec(endpoint="synthetic", target_id="fixture-target", outcome_scale="years",
                        policy_id="fixture-policy", weight_id="fixture-weights",
                        adjustment_schema_hash=registry.content_hash, inference_unit="county",
                        source_lineage_hash=source_lineage_hash((source,)))
    lineage = ArtifactLineage(source_hashes=(source.payload_hash,), unit_ids=ids, parent_hashes=(),
                              split_hash=None, config_hash=digest("config"), model_hash=None,
                              environment=(("python", "test"),), seed=None, parameter_count=None)
    schema = tuple(ColumnSpec(name=name, dtype=dtype) for name, dtype in (
        ("id", "string"), ("y", "number"), ("a", "number"), ("x", "number"),
        ("county", "string"), ("weight", "number"), ("y_se", "number")))
    manifest = DataManifest(spec=spec, sources=(source,), schema=schema, registry=registry,
                            original_ids=ids, id_field="id", outcome_field="y", exposure_field="a",
                            weight_field="weight", entity_graph_hash=graph.content_hash, lineage=lineage)
    records = [dict(id=oid, y=float(i + 60), a=float(i), x=float(i + 1), county="c1",
                    weight=float(i + 1), y_se=0.5) for i, oid in enumerate(ids)]
    data = load_records(records, manifest, spec, manifest.schema_hash)
    split = SplitManifest(spec=spec, level="outer", original_ids=ids, fold_ids=(0, 1, 0, 1),
                          design_ids=(), excluded_ids=(), seed_ids=(1103, 2207, 3301),
                          entity_graph_hash=graph.content_hash,
                          lineage=replace(lineage, parent_hashes=(manifest.content_hash,)))
    # Deliberately reverse record order and include all seeds.
    oid = tuple(x for _ in split.seed_ids for x in reversed(ids))
    nuisance = OOFNuisances(spec=spec, original_ids=oid, fold_ids=(1, 0, 1, 0) * 3,
                           seed_ids=tuple(s for s in split.seed_ids for _ in ids),
                           mu_a=(60.0,) * 12, mu_d=(61.0,) * 12, r_a=(1.0,) * 12,
                           r_d=(1.2,) * 12, origin_weights=(4.0, 3.0, 2.0, 1.0) * 3,
                           lineage=replace(lineage, parent_hashes=(manifest.content_hash,),
                                           split_hash=split.content_hash, model_hash=digest("model"),
                                           parameter_count=123))
    return data, records, split, nuisance, graph


def test_all_contracts_round_trip_and_stable_bytes(fixture, tmp_path):
    data, _, split, nuisance, graph = fixture
    spec, lineage = data.manifest.spec, data.manifest.lineage
    config, task = tmp_path / "config", tmp_path / "task"
    config.write_text("config")
    task.write_text("task")
    request = StageRequest(stage="synthetic", config_path=str(config), config_hash=file_hash(config),
                           task_path=str(task), task_hash=file_hash(task), dependency_paths=(),
                           dependency_hashes=(), output_dir=str(tmp_path), code_identity="a" * 40)
    artifact = ArtifactRecord(path="estimate.json", sha256=digest("bytes"), lineage=lineage, kind="estimate")
    result = StageResult(request_hash=request.content_hash, status="pass", artifacts=(artifact,), message="ok")
    estimate = Estimate(spec=spec, method="synthetic", value=1, standard_error=0.2,
                        original_ids=data.manifest.original_ids, scores=(1, 1, 1, 1),
                        influence=(0, 0, 0, 0), seed_ids=split.seed_ids, lineage=lineage)
    objects = (spec, data.manifest.sources[0], data.manifest, split, data.covariates(("x",)),
               nuisance, estimate, request, result, graph, data)
    for i, value in enumerate(objects):
        restored = type(value).from_json(value.to_json())
        assert restored == value
        assert restored.content_hash == value.content_hash
        assert restored.to_json() == value.to_json()
        path = tmp_path / str(i)
        assert write_artifact(path, value) == value.content_hash
        assert read_artifact(path, type(value), value.content_hash) == value
        with pytest.raises(FileExistsError):
            write_artifact(path, value)
    assert replace(lineage, environment=(("z", "2"), ("a", "1"))).content_hash == replace(
        lineage, environment=(("a", "1"), ("z", "2"))).content_hash


def test_deep_immutability(fixture):
    data = fixture[0]
    view = data.covariates(("x",))
    rows = [list(row) for row in view.values]
    clone = replace(view, values=rows)
    rows[0][0] = 999
    assert clone.values == view.values
    with pytest.raises(FrozenInstanceError):
        clone.columns = ("y",)
    with pytest.raises(TypeError):
        clone.values[0][0] = 999
    exported = view.to_dict()
    exported["payload"]["values"][0][0] = 999
    assert view.values[0][0] != 999


@pytest.mark.parametrize("name", ["y", "a", "y_se", "county", "weight", "unknown"])
def test_label_free_view_rejects_forbidden_access(fixture, name):
    data = fixture[0]
    view = data.covariates(("x",))
    with pytest.raises(ContractError):
        view[name]
    with pytest.raises(ContractError):
        data.covariates((name,))
    with pytest.raises(ContractError):
        replace(view, columns=(name,))
    assert not hasattr(view, "outcomes")
    assert not hasattr(view, "data")
    assert not hasattr(view, "manifest")


def test_changing_labels_does_not_change_view(fixture):
    data, records, *_ = fixture
    perturbed = [dict(r, y=-9999, y_se=999) for r in records]
    changed = load_records(perturbed, data.manifest, data.manifest.spec, data.manifest.schema_hash)
    assert data.covariates(("x",)).values == changed.covariates(("x",)).values
    assert data.county_routing("county") == ("c1",) * 4
    with pytest.raises(ContractError):
        data.county_routing("x")


@pytest.mark.parametrize("role", ["exposure_proxy", "precise_geography", "downstream_health",
                                  "outcome_metadata", "outcome", "exposure", "county", "identifier"])
@pytest.mark.parametrize("use", ["nuisance", "ssl", "context"])
def test_forbidden_role_cannot_be_approved_as_predictor(role, use):
    with pytest.raises(ContractError, match="forbidden permissions"):
        FeatureRule(name="synthetic", role=role, endpoints=("e",), uses=(use,), approval_id="approval")


def test_unknown_and_unreviewed_permissions_fail_closed():
    registry = FeatureRegistry(registry_id="pending", rules=())
    with pytest.raises(ContractError, match="unknown feature"):
        registry.require("x", "e", "nuisance")
    rule = FeatureRule(name="x", role="predictor", endpoints=("e",), uses=(), approval_id=None)
    registry = replace(registry, rules=(rule,))
    with pytest.raises(ContractError, match="unapproved"):
        registry.require("x", "e", "nuisance")
    with pytest.raises(ContractError, match="endpoint approval"):
        replace(rule, uses=("nuisance",))


@pytest.mark.parametrize("field,value", [
    ("endpoint", "other"), ("target_id", "other"), ("outcome_scale", "grams"),
    ("weight_id", "population"), ("adjustment_schema_hash", digest("other")),
    ("inference_unit", "psu"), ("source_lineage_hash", digest("other")),
])
def test_incompatible_estimands_cannot_join(fixture, field, value):
    data, records, split, nuisance, _ = fixture
    other = replace(data.manifest.spec, **{field: value})
    with pytest.raises(ContractError, match=f"{field} mismatch"):
        load_records(records, data.manifest, other, data.manifest.schema_hash)
    with pytest.raises(ContractError, match=f"{field} mismatch"):
        join_outcomes(replace(nuisance, spec=other), data, split, data.manifest.spec)


def test_policy_id_mismatch_rejected(fixture):
    data, records, split, nuisance, _ = fixture
    other = replace(data.manifest.spec, policy_id="different-policy")
    with pytest.raises(ContractError, match="policy_id mismatch"):
        load_records(records, data.manifest, other, data.manifest.schema_hash)
    with pytest.raises(ContractError, match="policy_id mismatch"):
        join_outcomes(replace(nuisance, spec=other), data, split, data.manifest.spec)


def test_schema_mismatches(fixture):
    data, records, *_ = fixture
    with pytest.raises(ContractError, match="data schema mismatch"):
        load_records(records, data.manifest, data.manifest.spec, digest("wrong"))
    for changed in ([dict(r, extra=0) for r in records], [dict(r, x="wrong") for r in records]):
        with pytest.raises(ContractError, match="schema mismatch|dtype mismatch"):
            load_records(changed, data.manifest, data.manifest.spec, data.manifest.schema_hash)
    with pytest.raises(ContractError, match="order mismatch"):
        load_records(records[::-1], data.manifest, data.manifest.spec, data.manifest.schema_hash)


def test_unreviewed_source_cannot_load(fixture):
    data, records, *_ = fixture
    source = replace(data.manifest.sources[0], mapping_status="unreviewed", field_mapping=(), mapping_review_id=None)
    spec = replace(data.manifest.spec, source_lineage_hash=source_lineage_hash((source,)))
    manifest = replace(data.manifest, sources=(source,), spec=spec)
    with pytest.raises(ContractError, match="unreviewed source mapping"):
        load_records(records, manifest, spec, manifest.schema_hash)


def test_artifact_hash_binds_source_and_parent_lineage(fixture, tmp_path):
    data = fixture[0]
    manifest = data.manifest
    changed = replace(manifest, lineage=replace(manifest.lineage, parent_hashes=(digest("new-parent"),)))
    assert changed.content_hash != manifest.content_hash
    source = replace(manifest.sources[0], payload_hash=digest("different-source"))
    assert source_lineage_hash((source,)) != manifest.spec.source_lineage_hash
    with pytest.raises(ContractError, match="source lineage mismatch"):
        replace(manifest, sources=(source,))
    path = tmp_path / "manifest"
    write_artifact(path, manifest)
    path.write_text(changed.to_json())
    with pytest.raises(ContractError, match="artifact hash mismatch"):
        read_artifact(path, DataManifest, manifest.content_hash)


@pytest.mark.parametrize("relation", ["household", "psu", "municipality", "repeated_geography", "outcome_lineage"])
def test_entity_graph_preserves_each_lineage(relation):
    links = tuple(EntityLink(observation_id=x, relation=relation, namespace="survey-year", entity_id="g")
                  for x in ("a", "b"))
    graph = EntityGraph(original_ids=("a", "b", "c"), links=links)
    assert graph.components() == (("a", "b"), ("c",))
    graph.assert_partition({"a": 0, "b": 0, "c": 1})
    with pytest.raises(ContractError, match="crosses partitions"):
        graph.assert_partition({"a": 0, "b": 1, "c": 1})


def test_entity_links_transitive_and_namespaced():
    links = tuple(EntityLink(observation_id=x, relation=kind, namespace=ns, entity_id=entity)
                  for x, kind, ns, entity in (
                      ("a", "household", "s", "1"), ("b", "household", "s", "1"),
                      ("b", "outcome_lineage", "s", "2"), ("c", "outcome_lineage", "s", "2"),
                      ("d", "household", "another-survey", "1")))
    graph = EntityGraph(original_ids=("a", "b", "c", "d"), links=links)
    assert graph.components() == (("a", "b", "c"), ("d",))


def test_oof_alignment_and_seed_identity(fixture):
    data, _, split, nuisance, graph = fixture
    validate_split(split, data.manifest, graph)
    validate_oof(nuisance, data, split)
    assert join_outcomes(nuisance, data, split, data.manifest.spec) == (63, 62, 61, 60) * 3
    with pytest.raises(ContractError, match="duplicate original ID/seed"):
        replace(nuisance, seed_ids=(1103,) * 12)
    with pytest.raises(ContractError, match="held-out fold mismatch"):
        validate_oof(replace(nuisance, fold_ids=(0, 1, 0, 1) * 3), data, split)
    with pytest.raises(ContractError, match="origin weight mismatch"):
        validate_oof(replace(nuisance, origin_weights=(1,) * 12), data, split)
    with pytest.raises(ContractError, match="OOF split mismatch"):
        validate_oof(replace(nuisance, lineage=replace(nuisance.lineage, split_hash=digest("other"))), data, split)
    with pytest.raises(ContractError, match="OOF alignment mismatch"):
        replace(nuisance, r_d=(1,))


def test_split_seals_design_and_group_lineage(fixture):
    data, _, split, _, graph = fixture
    links = tuple(EntityLink(observation_id=x, relation="outcome_lineage", namespace="s", entity_id="g")
                  for x in ("o1", "o2"))
    graph = replace(graph, links=links)
    manifest = replace(data.manifest, entity_graph_hash=graph.content_hash)
    split = replace(split, entity_graph_hash=graph.content_hash,
                    lineage=replace(split.lineage, parent_hashes=(manifest.content_hash,)))
    with pytest.raises(ContractError, match="crosses partitions"):
        validate_split(split, manifest, graph)
    split = replace(split, original_ids=("o2", "o3", "o4"), fold_ids=(0, 1, 0), design_ids=("o1",))
    assert "o1" not in split.training_ids(0)
    with pytest.raises(ContractError, match="crosses partitions"):
        validate_split(split, manifest, graph)


def test_reject_nonfinite_unknown_fields_and_versions(fixture):
    view = fixture[0].covariates(("x",))
    with pytest.raises(ContractError):
        replace(view, values=((float("nan"),),) * 4)
    for key, value in (("schema_version", 2), ("type", "SourceManifest")):
        payload = view.to_dict()
        payload[key] = value
        with pytest.raises(ContractError):
            CovariateView.from_json(json.dumps(payload))
    payload = view.to_dict()
    payload["payload"]["labels"] = [1, 2, 3, 4]
    with pytest.raises(ContractError):
        CovariateView.from_json(json.dumps(payload))
    with pytest.raises(ContractError):
        CovariateView.from_json('{"type":"a","type":"b"}')


def test_stage_result_hashes_paths_and_status(fixture, tmp_path):
    manifest = fixture[0].manifest
    config, task = tmp_path / "config", tmp_path / "task"
    config.write_text("config")
    task.write_text("task")
    request = StageRequest(stage="contracts", config_path=str(config), config_hash=file_hash(config),
                           task_path=str(task), task_hash=file_hash(task), dependency_paths=(),
                           dependency_hashes=(), output_dir=str(tmp_path), code_identity="a" * 40)
    artifact_path = tmp_path / "artifact.json"
    write_artifact(artifact_path, manifest)
    record = ArtifactRecord(path="artifact.json", sha256=file_hash(artifact_path),
                            lineage=manifest.lineage, kind="data_manifest")
    result = StageResult(request_hash=request.content_hash, status="pass", artifacts=(record,), message="ok")
    result.verify(request)
    for status in ("fail", "blocked"):
        assert replace(result, status=status, artifacts=()).status == status
    with pytest.raises(ContractError, match="declare artifacts"):
        replace(result, artifacts=())
    with pytest.raises(ContractError):
        replace(result, status="completed")
    for path in ("/absolute", "../escape", "a/../b", "./a", "a//b"):
        with pytest.raises(ContractError, match="relative path"):
            replace(record, path=path)
    task.write_text("changed")
    with pytest.raises(ContractError, match="input hash mismatch"):
        result.verify(request)
    task.write_text("task")
    artifact_path.write_text("changed")
    with pytest.raises(ContractError, match="artifact hash mismatch"):
        result.verify(request)


def test_configs_remain_explicitly_unusable():
    root = Path(__file__).resolve().parents[1]
    endpoints = yaml.safe_load((root / "configs/endpoints.yaml").read_text())
    roles = yaml.safe_load((root / "configs/feature_roles.yaml").read_text())
    approvals = yaml.safe_load((root / "configs/approvals.yaml").read_text())
    endpoint = endpoints["endpoints"]["usaleep_life_expectancy"]
    assert endpoint["usable"] is False and endpoint["source_mapping"] is None
    assert roles["rules"] == [] and roles["county_routing"]["usable"] is False
    assert endpoint["shift_mmhg"] == approvals["plan_fixed"]["tract_shift_mmhg"]
    assert endpoint["policy_form"] == approvals["plan_fixed"]["policy_form"]


def test_join_rejects_unbound_split(fixture):
    data, _, split, nuisance, _ = fixture
    split = replace(split, lineage=replace(split.lineage, parent_hashes=(digest("different-data"),)))
    nuisance = replace(nuisance, lineage=replace(nuisance.lineage, split_hash=split.content_hash))
    with pytest.raises(ContractError, match="missing data parent"):
        join_outcomes(nuisance, data, split, data.manifest.spec)


def test_oof_partial_seed_cannot_enter_estimation(fixture):
    data, _, split, nuisance, _ = fixture
    partial = replace(nuisance, **{name: getattr(nuisance, name)[:4] for name in (
        "original_ids", "fold_ids", "seed_ids", "mu_a", "mu_d", "r_a", "r_d", "origin_weights")})
    with pytest.raises(ContractError, match="coverage"):
        join_outcomes(partial, data, split, data.manifest.spec)


def test_stage_result_rejects_symlink_escape(fixture, tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("outside")
    output = tmp_path / "out"
    output.mkdir()
    (output / "link").symlink_to(outside)
    request = StageRequest(stage="synthetic", config_path=str(outside), config_hash=file_hash(outside),
                           task_path=str(outside), task_hash=file_hash(outside), dependency_paths=(),
                           dependency_hashes=(), output_dir=str(output), code_identity="a" * 40)
    record = ArtifactRecord(path="link", sha256=file_hash(outside),
                            lineage=fixture[0].manifest.lineage, kind="synthetic")
    result = StageResult(request_hash=request.content_hash, status="pass", artifacts=(record,), message="ok")
    with pytest.raises(ContractError, match="escapes output directory"):
        result.verify(request)


def test_namespace_subpackages_discovered():
    from setuptools import find_namespace_packages
    root = Path(__file__).resolve().parents[1]
    assert "oxyformer.data" in find_namespace_packages(where=str(root / "src"))
