"""Synthetic CPU-only design tests. No production payloads or network calls."""
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
from pathlib import Path

import pytest
import yaml

from oxyformer.contracts import ColumnSpec, CovariateView, DataManifest, EstimandSpec, StageRequest, StageResult, SourceManifest, source_lineage_hash
from oxyformer.data.entity_graph import EntityGraph
from oxyformer.data.feature_roles import FeatureRegistry, FeatureRule
from oxyformer.design.eligibility import AtlasRow, CollectedAtlas, GeographyRow, GeographyTable, county_screen
from oxyformer.design.gate import FrozenDesign, OWNER_APPROVALS, run_stage, validate_approvals, MissingPrerequisite
from oxyformer.design.policies import PolicyCovariates
from oxyformer.design.splits import reserve_design
from oxyformer.design.support import SupportRecipe, connected_components, freeze_support, policy_for
from oxyformer.provenance import ArtifactLineage, ContractError, canonical_json, file_hash, write_artifact

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return sha256(str(value).encode()).hexdigest()


def make_inputs(*, label_offset=0):
    # Every block contains an exposure gradient. Tail evidence exists only in
    # the sealed blocks; tail observations do not set the evaluation target.
    names = [f"b{i:02}" for i in range(25)]
    sealed = set(sorted(names, key=lambda b: digest(canonical_json([1103, "c1", b])))[:5])
    rows, atlas_rows = [], []
    for block, name in enumerate(names):
        for dose in (range(12) if name in sealed else range(1, 11)):
            for rep in range(2):
                oid = f"{name}-{dose:02}-{rep}"
                rows.append(GeographyRow(original_id=oid, tract_id=oid, county="c1", state="s1",
                                         subblock=name, assignment_geography=oid,
                                         latitude=40, longitude=-100 + block * .003,
                                         outcome_flag=1, label_available=True))
                atlas_rows.append(AtlasRow(tract_id=oid, exposure_mmhg=dose + .25,
                                          inhabited_elevation_m=1500 - dose * 120,
                                          population=100, allocation_qualified=True))
    ids = tuple(r.original_id for r in rows)
    graph = EntityGraph(original_ids=ids, links=())
    endpoint = "usaleep_life_expectancy"
    rules = tuple(FeatureRule(name=name, role=role, endpoints=(endpoint,), uses=uses,
                              approval_id="synthetic-fixture-only") for name, role, uses in (
        ("id", "identifier", ("linkage",)), ("y", "outcome", ("score",)),
        ("a", "exposure", ("score",)), ("female_share", "predictor", ("nuisance", "ssl", "context")),
        ("county", "county", ("county_routing",)),
    ))
    registry = FeatureRegistry(registry_id="synthetic-design-fixture", rules=rules)
    # Simulate a change to the upstream Y payload and all its provenance. The
    # gate receives its manifest, never those outcome values.
    source = SourceManifest(source_id="synthetic", version="1", uri="synthetic://fixture",
                            payload_hash=digest(f"Y:{label_offset}"), license_hash=digest("license"),
                            schema_hash=digest("schema"), field_mapping=(("raw_x", "female_share"),),
                            mapping_status="reviewed", mapping_review_id="fixture-only")
    spec = EstimandSpec(endpoint=endpoint, target_id="unresolved-input-frame", outcome_scale="years",
                        policy_id="unresolved-design-policy", weight_id="equal-tract-fixture", inference_unit="county",
                        adjustment_schema_hash=registry.content_hash, source_lineage_hash=source_lineage_hash((source,)))
    lineage = ArtifactLineage(source_hashes=(source.payload_hash,), unit_ids=ids, parent_hashes=(),
                              split_hash=None, config_hash=digest("config"), model_hash=None,
                              environment=(("python", "synthetic"),), seed=None, parameter_count=None)
    schema = tuple(ColumnSpec(name=name, dtype=dtype) for name, dtype in (
        ("id", "string"), ("y", "number"), ("a", "number"), ("female_share", "number"), ("county", "string")))
    manifest = DataManifest(spec=spec, sources=(source,), schema=schema, registry=registry, original_ids=ids,
                            id_field="id", outcome_field="y", exposure_field="a", weight_field=None,
                            entity_graph_hash=graph.content_hash, lineage=lineage)
    covariates = CovariateView(spec=spec, registry=registry, original_ids=ids, columns=("female_share",),
                              values=((.5,),) * len(ids), use="nuisance",
                              lineage=replace(lineage, parent_hashes=(manifest.content_hash,)))
    geography = GeographyTable(rows=tuple(rows), data_manifest_hash=manifest.content_hash, county_field="county",
                                approval_reference="configs/approvals.yaml#plan_fixed.county_membership",
                                mapping_review_id="synthetic-reviewed-geography")
    atlas = CollectedAtlas(rows=tuple(atlas_rows), source_hashes=(digest("synthetic-atlas"),),
                           footprint="nine_census_divisions_contiguous_us_plus_dc", expected_tract_ids=ids,
                           missing_tract_ids=(), coverage_complete=True, mapping_review_id="synthetic-atlas-review")
    return dict(data_manifest=manifest, covariates=covariates, geography=geography, atlas=atlas, entity_graph=graph)


def make_request(tmp_path, inputs=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    inputs = make_inputs() if inputs is None else inputs
    paths = {"approvals": str(OWNER_APPROVALS)}
    for role, value in inputs.items():
        path = tmp_path / f"{role}.json"
        write_artifact(path, value)
        paths[role] = str(path)
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"dependencies": paths}))
    config = ROOT / "configs/design.yaml"
    return StageRequest(stage="tract_design", config_path=str(config), config_hash=file_hash(config),
                        task_path=str(task), task_hash=file_hash(task), dependency_paths=tuple(paths.values()),
                        dependency_hashes=tuple(file_hash(p) for p in paths.values()), output_dir=str(tmp_path / "out"),
                        code_identity="a" * 40)


def test_county_preliminary_screens_use_allocation_qualified_unique_tracts():
    values = make_inputs()
    rows = values["geography"].rows[:3]
    a = {r.tract_id: r for r in values["atlas"].rows}
    assert "fewer_than_four_observed_input_tracts" in county_screen(rows, a)["c1"]["reasons"]
    rows = values["geography"].rows
    assert not county_screen(rows, a)["c1"]["reasons"]
    bad = {key: replace(value, allocation_qualified=False) for key, value in a.items()}
    assert county_screen(rows, bad)["c1"]["tract_count"] == 0
    flat = {key: replace(value, inhabited_elevation_m=10) for key, value in a.items()}
    assert "inhabited_relief_below_300m" in county_screen(rows, flat)["c1"]["reasons"]


def test_disconnected_support_does_not_bridge_extrema_or_singletons():
    exposures = [x + .25 for x in (0, 1, 2, 3, 7, 8, 9, 10) for _ in range(2)]
    components = connected_components(exposures, [f"t{i}" for i in range(len(exposures))])
    assert components == ((1., 4.), (7., 10.))
    assert connected_components(exposures, ["same-tract"] * len(exposures)) == ()
    # Isolated single-tract bins never close a hole.
    assert connected_components(exposures + [4.2, 5.2, 6.2],
                                [f"t{i}" for i in range(len(exposures) + 3)]) == components


def test_reserve_twenty_percent_and_freeze_design_only():
    v = make_inputs()
    rows, graph = v["geography"].rows, v["entity_graph"]
    reservation = reserve_design(rows, graph)
    assert len(reservation.selected_subblocks) == len(reservation.sealed_subblocks) == 5
    assert reservation.total_subblocks == 25
    atlas = {r.tract_id: r for r in v["atlas"].rows}
    frozen = freeze_support(rows, reservation.design_ids, atlas, v["covariates"])
    sealed = set(reservation.design_ids)
    changed = {key: value if key in sealed else replace(value, exposure_mmhg=9999) for key, value in atlas.items()}
    again = freeze_support(rows, reservation.design_ids, changed, v["covariates"])
    assert again == frozen  # Evaluation A cannot fit the policy or spline knots.
    with pytest.raises(FrozenInstanceError):
        frozen.scales = (100,)
    assert FrozenDesign.from_json(FrozenDesign(reservation=reservation, support=frozen,
                                               approved_features=v["data_manifest"].registry).to_json()).support == frozen


def test_grouped_policy_intersects_personal_support_and_refuses_inconsistent_doses():
    v = make_inputs()
    rows = list(v["geography"].rows)
    reservation = reserve_design(rows, v["entity_graph"])
    target = next(r for r in rows if r.original_id not in reservation.design_ids)
    twin = replace(target, original_id="twin")
    rows.append(twin)
    cov = replace(v["covariates"], original_ids=v["covariates"].original_ids + ("twin",),
                  values=v["covariates"].values + ((100.,),),
                  lineage=replace(v["covariates"].lineage, unit_ids=v["covariates"].original_ids + ("twin",)))
    atlas = {r.tract_id: r for r in v["atlas"].rows}
    frozen = freeze_support(rows, reservation.design_ids, atlas, cov)
    assert dict(frozen.policy.components_by_key)[target.assignment_geography] == ()
    result = policy_for([target, twin], atlas, frozen)
    assert result.moved == (False, False)
    pc = PolicyCovariates(original_ids=("one", "two"), geography_ids=("g", "g"),
                          support_keys=(target.assignment_geography,) * 2)
    with pytest.raises(ContractError, match="inconsistent exposure"):
        frozen.policy.apply([1, 2], pc)


def test_stage_outputs_frozen_exclusions_and_buffer_target_changes(tmp_path):
    request = make_request(tmp_path)
    result = run_stage(request)
    assert result.status == "pass", result.message
    result.verify(request)
    output = Path(request.output_dir)
    assert {p.name for p in output.iterdir()} == {
        "design.json", "splits.json", "support_report.json", "gate.json", "artifact_manifest.json"}
    committed = StageResult.from_json((output / "artifact_manifest.json").read_text())
    committed.verify(request)
    design = FrozenDesign.from_json((output / "design.json").read_text())
    report = json.loads((output / "support_report.json").read_text())
    scenarios = json.loads((output / "splits.json").read_text())["scenarios"]
    assert report["reservation"]["sealed_fraction"] == .2
    assert scenarios[0]["payload"]["status"] == "pass"
    assert all(s["payload"]["status"] == "fail" for s in scenarios[1:])
    primary = scenarios[0]["payload"]["outer"]
    assert not set(primary["original_ids"]) & set(design.reservation.design_ids)
    for summary in report["scenarios"]:
        assert summary["evaluation_count"] + summary["sealed_design_count"] + summary["excluded_count"] == len(make_inputs()["geography"].rows)
    for audit in scenarios[0]["payload"]["count_audit"]:
        assert audit["outer_training_tracts"] >= 8
        assert min(audit["inner_fitting_tracts"]) >= 4
    assert json.loads((output / "gate.json").read_text())["effect_release_authorized"] is False
    before = (output / "design.json").read_bytes()
    with pytest.raises(ContractError, match="already exists"):
        run_stage(request)
    assert (output / "design.json").read_bytes() == before


def test_y_perturbation_leaves_design_and_split_decisions_identical(tmp_path):
    left, right = make_request(tmp_path / "a", make_inputs(label_offset=0)), make_request(tmp_path / "b", make_inputs(label_offset=9999))
    a, b = run_stage(left), run_stage(right)
    assert a.status == b.status == "pass"
    assert (Path(left.output_dir) / "design.json").read_bytes() == (Path(right.output_dir) / "design.json").read_bytes()
    l = json.loads((Path(left.output_dir) / "splits.json").read_text())
    r = json.loads((Path(right.output_dir) / "splits.json").read_text())
    for ls, rs in zip(l["scenarios"], r["scenarios"]):
        for key in ("target_id", "exclusions", "count_audit"):
            assert ls["payload"][key] == rs["payload"][key]
        if ls["payload"]["outer"]:
            assert ls["payload"]["outer"]["fold_ids"] == rs["payload"]["outer"]["fold_ids"]
    assert a.artifacts[0].lineage != b.artifacts[0].lineage  # honest source provenance remains different


def test_missing_dependencies_blocks_and_still_writes_all_reports(tmp_path):
    request = make_request(tmp_path)
    task = Path(request.task_path)
    task.write_text('{"dependencies": {}}')
    request = replace(request, task_hash=file_hash(task))
    result = run_stage(request)
    assert result.status == "blocked"
    assert len(result.artifacts) == 5
    result.verify(request)


def test_incomplete_atlas_is_explicitly_blocked(tmp_path):
    v = make_inputs()
    atlas = v["atlas"]
    # Extra footprint tract missing outside this candidate frame must still be
    # disclosed: a passing subset cannot certify national atlas completion.
    v["atlas"] = replace(atlas, expected_tract_ids=atlas.expected_tract_ids + ("missing",),
                         missing_tract_ids=("missing",), coverage_complete=False)
    request = make_request(tmp_path, v)
    result = run_stage(request)
    assert result.status == "blocked"
    coverage = json.loads((Path(request.output_dir) / "support_report.json").read_text())["coverage"]
    assert coverage["missing_tract_ids"] == ["missing"]
    assert coverage["complete"] is False


def test_missing_feature_and_geography_approval_are_not_assumed():
    v = make_inputs()
    config = yaml.safe_load((ROOT / "configs/design.yaml").read_text())
    approvals = yaml.safe_load(OWNER_APPROVALS.read_text())
    approvals["owner_decisions"]["endpoint_covariates"] = {}
    with pytest.raises(MissingPrerequisite, match="feature"):
        validate_approvals(config, approvals, v["data_manifest"], v["covariates"], v["geography"], v["atlas"])
    approvals = yaml.safe_load(OWNER_APPROVALS.read_text())
    with pytest.raises(MissingPrerequisite, match="geography"):
        validate_approvals(config, approvals, v["data_manifest"], v["covariates"],
                           replace(v["geography"], approval_reference="unapproved"), v["atlas"])


def test_outcomes_cannot_be_requested_as_design_covariates():
    v = make_inputs()
    with pytest.raises(ContractError, match="unapproved"):
        replace(v["covariates"], columns=("y",))


def test_fold_support_uses_raw_covariates_and_keeps_policy_fixed():
    from oxyformer.design.support import recheck_support
    v = make_inputs()
    rows = v["geography"].rows
    reservation = reserve_design(rows, v["entity_graph"])
    atlas = {r.tract_id: r for r in v["atlas"].rows}
    frozen = freeze_support(rows, reservation.design_ids, atlas, v["covariates"])
    target = next(r for r in rows if r.original_id not in reservation.design_ids)
    training = [r for r in rows if r.original_id != target.original_id and r.tract_id != target.tract_id]
    before = frozen.to_json()
    assert not recheck_support([target], training, atlas, v["covariates"], frozen)
    assert recheck_support([target], [], atlas, v["covariates"], frozen) == (target.original_id,)
    assert frozen.to_json() == before


def test_primary_flag_exclusions_are_explicit(tmp_path):
    v = make_inputs()
    reservation = reserve_design(v["geography"].rows, v["entity_graph"])
    block = next(r.subblock for r in v["geography"].rows if r.original_id not in reservation.design_ids)
    rows = tuple(replace(r, outcome_flag=2) if r.subblock == block else r for r in v["geography"].rows)
    v["geography"] = replace(v["geography"], rows=rows)
    request = make_request(tmp_path, v)
    result = run_stage(request)
    assert result.status == "pass", result.message
    report = json.loads((Path(request.output_dir) / "support_report.json").read_text())
    bad = {r.original_id for r in rows if r.outcome_flag == 2}
    primary = json.loads((Path(request.output_dir) / "splits.json").read_text())["scenarios"][0]["payload"]
    assert bad <= set(primary["outer"]["excluded_ids"])
    assert not bad.intersection(primary["outer"]["original_ids"])
    assert report["scenarios"][0]["exclusion_counts"]["flag_label_or_allocation_unusable"] == len(bad)


def test_tampered_dependency_fails_before_design(tmp_path):
    request = make_request(tmp_path)
    path = Path(request.dependency_paths[1])
    path.write_text(path.read_text() + " ")
    result = run_stage(request)
    assert result.status == "fail"
    assert "input hash mismatch" in result.message


def test_missing_covariates_are_conditioning_patterns_not_full_frame_imputation():
    v = make_inputs()
    rows = v["geography"].rows
    reservation = reserve_design(rows, v["entity_graph"])
    cov = replace(v["covariates"], values=((None,),) * len(rows))
    atlas = {r.tract_id: r for r in v["atlas"].rows}
    frozen = freeze_support(rows, reservation.design_ids, atlas, cov)
    assert frozen.scales == (0.0,)
    assert any(components for _, components in frozen.policy.components_by_key)
    # A newly observed query cannot borrow the all-missing conditional support.
    index = next(i for i, r in enumerate(rows) if r.original_id not in reservation.design_ids)
    values = list(cov.values)
    values[index] = (.5,)
    different = freeze_support(rows, reservation.design_ids, atlas, replace(cov, values=values))
    assert dict(different.policy.components_by_key)[rows[index].assignment_geography] == ()


def test_assignment_geography_can_span_distinct_tract_centroids():
    v = make_inputs()
    rows = list(v["geography"].rows)
    first = rows[0]
    atlas = {r.tract_id: r.exposure_mmhg for r in v["atlas"].rows}
    second = next(i for i, r in enumerate(rows) if r.subblock != first.subblock
                  and atlas[r.tract_id] == atlas[first.tract_id])
    rows[second] = replace(rows[second], assignment_geography=first.assignment_geography)
    reviewed = replace(v["geography"], rows=rows)
    assert reviewed.rows[0].longitude != reviewed.rows[second].longitude
    assert reviewed.rows[0].assignment_geography == reviewed.rows[second].assignment_geography


def test_zero_shift_is_disclosed_without_an_unapproved_failure_threshold(tmp_path):
    v = make_inputs()
    reservation = reserve_design(v["geography"].rows, v["entity_graph"])
    sealed = set(reservation.design_ids)
    v["atlas"] = replace(v["atlas"], rows=tuple(
        replace(r, exposure_mmhg=(5.25 + i % 3 if r.tract_id in sealed else 6.25 + .5 * (i % 2)))
        for i, r in enumerate(v["atlas"].rows)))
    request = make_request(tmp_path, v)
    result = run_stage(request)
    assert result.status == "pass", result.message
    report = json.loads((Path(request.output_dir) / "support_report.json").read_text())
    assert report["scenarios"][0]["shifted_fraction"] == 0
    assert report["scenarios"][0]["warning"] == "policy_moves_no_tracts"
    assert json.loads((Path(request.output_dir) / "gate.json").read_text())["effect_release_authorized"] is False


def test_duplicate_endpoint_tract_is_not_counted_as_two_equal_weight_tracts(tmp_path):
    # This stage is the single-release equal-tract USALEEP endpoint. A duplicate
    # row in that frame is not an individual-record endpoint with group weights.
    v = make_inputs()
    rows = list(v["geography"].rows)
    rows[1] = replace(rows[0], original_id=rows[1].original_id)
    v["geography"] = replace(v["geography"], rows=rows)
    request = make_request(tmp_path, v)
    result = run_stage(request)
    assert result.status == "fail"
    assert "duplicate tract observations" in result.message


def test_numpy_values_cannot_bypass_the_merged_covariate_contract():
    import numpy as np
    v = make_inputs()
    covariates = v["covariates"]
    # The authoritative Immutable/Cell contract rejects numpy scalar instances
    # during construction, before any design function can receive such a view.
    for scalar in (np.float64(.5), np.float32(.5), np.int64(1)):
        with pytest.raises(ContractError, match="value does not match"):
            replace(covariates, values=((scalar,),) * len(covariates.original_ids))
    # Honest numpy-backed adapters normalize at the existing contract boundary.
    normalized = replace(covariates, values=np.full((len(covariates.original_ids), 1), .5).tolist())
    restored = CovariateView.from_json(normalized.to_json())
    assert all(type(row[0]) is float for row in restored.values)
    reservation = reserve_design(v["geography"].rows, v["entity_graph"])
    support = freeze_support(v["geography"].rows, reservation.design_ids,
                              {r.tract_id: r for r in v["atlas"].rows}, restored)
    assert any(components for _, components in support.policy.components_by_key)


def test_missing_all_sealed_atlas_records_is_blocked_not_failed(tmp_path):
    v = make_inputs()
    reservation = reserve_design(v["geography"].rows, v["entity_graph"])
    missing = set(reservation.design_ids)
    v["atlas"] = replace(v["atlas"], rows=tuple(r for r in v["atlas"].rows if r.tract_id not in missing),
                         missing_tract_ids=tuple(sorted(missing)), coverage_complete=False)
    request = make_request(tmp_path, v)
    result = run_stage(request)
    assert result.status == "blocked", result.message
    coverage = json.loads((Path(request.output_dir) / "support_report.json").read_text())["coverage"]
    assert set(coverage["missing_tract_ids"]) == missing


@pytest.mark.parametrize('unaccounted', [False, True])
def test_accounted_dem_exclusions_are_reported_and_unaccounted_omission_blocks(tmp_path, monkeypatch, unaccounted):
    from oxyformer.design import gate
    from test_tract_tasks import collected_design_request, install_coverage_approval
    install_coverage_approval(tmp_path, monkeypatch)
    values = make_inputs()
    reservation = reserve_design(values['geography'].rows, values['entity_graph'])
    sealed = set(reservation.design_ids)
    missing = (reservation.design_ids[0], next(r.original_id for r in values['geography'].rows
                                               if r.original_id not in sealed))
    unknown = next(r.original_id for r in values['geography'].rows if r.original_id not in {*missing, *sealed})
    request = collected_design_request(tmp_path, values, missing=missing, accounted=(*missing, 'outside-endpoint'),
                                       absent=(unknown,) if unaccounted else ())
    result = gate.run_stage(request)
    assert result.status == ('blocked' if unaccounted else 'pass'), result.message
    result.verify(request)
    out = Path(request.output_dir)
    report = json.loads((out / 'support_report.json').read_text())
    coverage = report['coverage']
    assert coverage['complete'] is not unaccounted
    assert not coverage['physical_coverage_complete']
    assert coverage['accounted_missing_dem_tract_ids'] == sorted(missing)
    assert coverage['unaccounted_missing_tract_ids'] == ([unknown] if unaccounted else [])
    assert (coverage['covered_tracts'] + coverage['accounted_missing_dem_tracts']
            + len(coverage['unaccounted_missing_tract_ids'])) == coverage['expected_tracts']
    assert {r['original_id']: r['sealed_design'] for r in report['allocation_exclusions']} == {
        missing[0]: True, missing[1]: False}
    assert {r['reason'] for r in report['allocation_exclusions']} == {'atlas_missing_dem_coverage'}
    design = FrozenDesign.from_json((out / 'design.json').read_text())
    assert design.reservation == reservation
    scenarios = json.loads((out / 'splits.json').read_text())['scenarios']
    for scenario, counts in zip(scenarios, report['scenarios']):
        scenario = scenario['payload']
        assert dict(scenario['exclusions'])[missing[1]] == 'atlas_missing_dem_coverage'
        assert counts['evaluation_count'] + counts['sealed_design_count'] + counts['excluded_count'] == len(values['geography'].rows)
        if scenario['outer']:
            assert not set(missing).intersection(scenario['outer']['original_ids'])
    assert not json.loads((out / 'gate.json').read_text())['effect_release_authorized']


@pytest.mark.parametrize('approval', ['missing', 'wrong', 'legacy'])
def test_accounted_dem_exclusion_requires_owner_decision(approval):
    from oxyformer.design.gate import atlas_coverage
    values = make_inputs()
    atlas = values['atlas']
    missing = atlas.rows[0].tract_id
    values['atlas'] = replace(atlas, rows=atlas.rows[1:], missing_tract_ids=(missing,), coverage_complete=False)
    values['atlas_missing_dem_tract_ids'] = (missing,)
    values['approvals'] = yaml.safe_load(OWNER_APPROVALS.read_text())
    owner = values['approvals']['owner_decisions']
    owner.pop('atlas_coverage', None)
    if approval == 'wrong':
        owner['atlas_coverage'] = {'missing_dem_tracts': 'block'}
    elif approval == 'legacy':
        owner['tract_design']['atlas_missing_dem_tracts'] = 'exclude_as_not_allocation_qualified'
    with pytest.raises(MissingPrerequisite, match='owner approval'):
        atlas_coverage(values)


def test_accounted_dem_omissions_of_all_sealed_records_remain_blocked(tmp_path, monkeypatch):
    from test_tract_tasks import collected_design_request, install_coverage_approval
    install_coverage_approval(tmp_path, monkeypatch)
    values = make_inputs()
    reservation = reserve_design(values['geography'].rows, values['entity_graph'])
    missing = reservation.design_ids
    request = collected_design_request(tmp_path, values, missing=missing, accounted=missing)
    result = run_stage(request)
    assert result.status == 'blocked', result.message
    result.verify(request)
    out = Path(request.output_dir)
    report = json.loads((out / 'support_report.json').read_text())
    assert report['coverage']['complete'] and not report['coverage']['physical_coverage_complete']
    assert {r['original_id'] for r in report['allocation_exclusions']} == set(missing)
    assert all(r['sealed_design'] for r in report['allocation_exclusions'])
    assert not json.loads((out / 'design.json').read_text())['available']
    assert json.loads((out / 'splits.json').read_text())['scenarios'] == []
    assert not json.loads((out / 'gate.json').read_text())['effect_release_authorized']
