"""Deterministic artificial inputs for execution smoke tests only.

This is not the fixed-real-covariate Suite A of plan section 6.2. It supplies
no production timing, tract eligibility evidence, or campaign approval. The
registered smoke task pins both content hashes and explicitly shortens epochs.
"""
from dataclasses import replace
import json
from pathlib import Path

from oxyformer.contracts import (ColumnSpec, DataManifest, EstimandSpec,
    SourceManifest, SplitManifest, StageRequest, StageResult, source_lineage_hash)
from oxyformer.data.entity_graph import EntityGraph, EntityLink
from oxyformer.data.feature_roles import FeatureRegistry, FeatureRule
from oxyformer.data.loaders import load_records
from oxyformer.design.eligibility import GeographyRow, GeographyTable
from oxyformer.design.policies import PolicyCovariates, ShiftOrStayPolicy
from oxyformer.design.splits import InnerSplit
from oxyformer.models.treatment_query import TreatmentDesign
from oxyformer.provenance import ArtifactLineage, ContractError, require
from oxyformer.training.nested_cv import PreparedEndpoint, SEEDS
from oxyformer.validation.coverage import digest, publish
from oxyformer.validation.scm import CovariateFrame

FIXTURE = "suite-a-smoke-v1"


def build_inputs():
    """Two artificial counties, 30 tracts each, with paired geographic groups.

    All five outer folds and three inner folds stay within each county. A pair
    never crosses a partition. X and geography are fixed before generating A/Y;
    placeholder A/Y are replaced by coverage.bind_observations for every draw.
    """
    ids = tuple(f"synthetic-{i:02d}" for i in range(60))
    counties = tuple("c" if i < 30 else "d" for i in range(60))
    groups = tuple(str(i // 2) for i in range(60))
    coordinates = tuple((float(i // 2) / 10, 0.) for i in range(60))
    records = [dict(id=oid, y=0., a=5., x=float((i // 2) % 5 - 2) / 2,
                    county=counties[i], w=1.) for i, oid in enumerate(ids)]
    schema = tuple(ColumnSpec(name=n, dtype=t) for n, t in (
        ("id", "string"), ("y", "number"), ("a", "number"), ("x", "number"),
        ("county", "string"), ("w", "number")))
    rules = tuple(FeatureRule(name=n, role=r, endpoints=("synthetic-smoke",), uses=u,
        approval_id=FIXTURE) for n, r, u in (
        ("id", "identifier", ("linkage",)), ("y", "outcome", ("score",)),
        ("a", "exposure", ("score",)), ("x", "predictor", ("nuisance", "ssl", "context")),
        ("county", "county", ("county_routing",)), ("w", "outcome_metadata", ("linkage",))))
    registry = FeatureRegistry(registry_id=FIXTURE, rules=rules)
    graph = EntityGraph(original_ids=ids, links=tuple(EntityLink(observation_id=oid,
        relation="repeated_geography", namespace=FIXTURE, entity_id=group)
        for oid, group in zip(ids, groups)))
    source = SourceManifest(source_id=FIXTURE, version="1", uri="synthetic://" + FIXTURE,
        payload_hash=digest(records), license_hash=digest("artificial-no-source-data"),
        schema_hash=digest([c.to_dict() for c in schema]),
        field_mapping=tuple((c.name, c.name) for c in schema),
        mapping_status="reviewed", mapping_review_id=FIXTURE)
    design_hash = digest([FIXTURE, "fixed-artificial-support", 0., 10.])
    policy = ShiftOrStayPolicy(support_design_hash=design_hash,
        components_by_key=(("s", ((0., 10.),)),), delta_mmhg=2.)
    spec = EstimandSpec(endpoint="synthetic-smoke", target_id=FIXTURE,
        outcome_scale="synthetic-units", policy_id=policy.policy_id, weight_id="equal-tract",
        adjustment_schema_hash=registry.content_hash, inference_unit="tract",
        source_lineage_hash=source_lineage_hash((source,)))
    lineage = ArtifactLineage(source_hashes=(source.payload_hash,), unit_ids=ids,
        parent_hashes=(), split_hash=None, config_hash=digest(FIXTURE), model_hash=None,
        environment=(("fixture", FIXTURE),), seed=None, parameter_count=None)
    manifest = DataManifest(spec=spec, sources=(source,), schema=schema, registry=registry,
        original_ids=ids, id_field="id", outcome_field="y", exposure_field="a", weight_field="w",
        entity_graph_hash=graph.content_hash, lineage=lineage)
    data = load_records(records, manifest, spec, manifest.schema_hash)
    outer = SplitManifest(spec=spec, level="outer", original_ids=ids,
        fold_ids=tuple((i % 30) // 6 for i in range(60)), design_ids=(), excluded_ids=(),
        seed_ids=SEEDS, entity_graph_hash=graph.content_hash,
        lineage=replace(lineage, parent_hashes=(manifest.content_hash,)))
    inner = []
    for fold in range(5):
        train = outer.training_ids(fold)
        local_graph = EntityGraph(original_ids=train,
            links=tuple(link for link in graph.links if link.observation_id in train))
        local_manifest = replace(manifest, original_ids=train, entity_graph_hash=local_graph.content_hash,
            lineage=replace(lineage, unit_ids=train, parent_hashes=(manifest.content_hash,)))
        split = SplitManifest(spec=spec, level="inner", original_ids=train,
            fold_ids=tuple((i // 2) % 3 for i in range(len(train))), design_ids=(), excluded_ids=(),
            seed_ids=SEEDS, entity_graph_hash=local_graph.content_hash,
            lineage=replace(local_manifest.lineage, parent_hashes=(local_manifest.content_hash,)))
        inner.append(InnerSplit(outer_fold=fold, data_manifest=local_manifest,
            entity_graph=local_graph, split=split, buffer_excluded_ids=()))
    geography = GeographyTable(rows=tuple(GeographyRow(original_id=oid, tract_id=oid,
        county=counties[i], state="synthetic-state", subblock=groups[i],
        assignment_geography=groups[i], latitude=coordinates[i][0], longitude=coordinates[i][1],
        outcome_flag=1, label_available=True) for i, oid in enumerate(ids)),
        data_manifest_hash=manifest.content_hash, county_field="county",
        approval_reference=FIXTURE, mapping_review_id=FIXTURE)
    endpoint = PreparedEndpoint(data=data, entity_graph=graph, geography=geography,
        outer=outer, inner=tuple(inner), policy=policy,
        policy_covariates=PolicyCovariates(original_ids=ids, geography_ids=groups, support_keys=("s",) * 60),
        treatment_design=TreatmentDesign(center=5., scale=5., knots=(0., 2., 4., 6., 8., 10.),
            design_hash=design_hash), feature_kinds=(("x", "numeric"),), families=(("x",),),
        county_field="county", exposure_assignment_level="tract")
    frame = CovariateFrame(original_ids=ids, geography_ids=groups, region_ids=counties,
        cluster_ids=groups, coordinates=coordinates, columns=("x",),
        x=tuple((row["x"],) for row in records), support_keys=("s",) * 60,
        weights=(1.,) * 60, outcome_available=(True,) * 60, biomarker_available=(True,) * 60)
    for fold in range(5):
        endpoint.validate(fold)
    return endpoint, frame


def run_stage(request: StageRequest) -> StageResult:
    try:
        request.verify_inputs()
        task = json.loads(Path(request.task_path).read_text())
        require(request.stage == task["stage"] == "simulation-inputs", "stage mismatch")
        require(task["parameters"] == {"fixture": FIXTURE}, "unregistered synthetic fixture")
        endpoint, frame = build_inputs()
        return publish(request, {"endpoint.json": endpoint.to_dict(), "frame.json": frame.to_dict()},
            status="pass", message="Artificial smoke inputs only; no production or coverage evidence")
    except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
        return StageResult(request_hash=request.content_hash, status="blocked", artifacts=(),
            message=str(exc) or type(exc).__name__)
