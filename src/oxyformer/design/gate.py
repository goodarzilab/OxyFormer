"""Outcome-blind tract feasibility stage; a pass never authorizes effect release.

The task JSON maps dependency roles to absolute request.dependency_paths.
Scientific inputs are canonical existing DataManifest/CovariateView/EntityGraph
records plus the explicit outcome-free GeographyTable and CollectedAtlas in
eligibility.py. Producers remain responsible for reviewed raw field mappings.
No raw table, LoadedData, label column or outcome file is accepted by this stage.
Consumers must check StageResult.status before decoding or consuming scientific
artifacts. A nonpassing stage may publish an unavailable-design marker instead
of a FrozenDesign; no fitted design is fabricated for missing prerequisites.
"""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import yaml

from oxyformer.contracts import CovariateView, DataManifest, StageRequest, StageResult
from oxyformer.data.adapters.usaleep import primary_outcome_flags
from oxyformer.data.entity_graph import EntityGraph
from oxyformer.data.feature_roles import FeatureRegistry
from oxyformer.design.eligibility import CollectedAtlas, GeographyTable, county_screen, usable, _usable
from oxyformer.design.splits import Reservation, SplitScenarioDesign, build_scenario, reserve_design
from oxyformer.design.support import FrozenSupport, SupportRecipe, freeze_support, policy_for, supported_ids
from oxyformer.provenance import (
    ArtifactLineage, ArtifactRecord, ContractError, Immutable, canonical_json, file_hash,
    read_artifact, require,
)

ROOT = Path(__file__).resolve().parents[3]
OWNER_APPROVALS = ROOT / "configs" / "approvals.yaml"
DESIGN_CONFIG = ROOT / "configs" / "design.yaml"
INFERENCE_FRAME = (
    "A2 equal-tract inference is conditional on the realised allocation-qualified frame, "
    "approved raw-X registry, frozen support policy, atlas coverage and buffer-specific "
    "eligible primary (flag-1/3) tract frame. Sealed design outcomes enter neither nuisance fitting "
    "nor effect evaluation. Uncertainty concerns the declared geographic stochastic "
    "process, not a literal census sampling error. Every changed target has a new ID. "
    "Whole-county/state deletion requires full refitting; no unseen-county offsets."
)


class MissingPrerequisite(ContractError):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenDesign(Immutable):
    reservation: Reservation
    support: FrozenSupport
    approved_features: FeatureRegistry
    inference_frame: str = INFERENCE_FRAME
    effect_release_authorized: bool = False
    estimand_id: str = "usaleep-a2-2026-10-10"
    locked_analysis: str = "owner_decisions.tract_design_amendment/2026-10-10"


A2_ESTIMAND_ID = "usaleep-a2-2026-10-10"
# Published tract-gate-r2 primary target, retained even though infeasible.
FROZEN_TARGET_ID = "e2999854bbeb2d232ca59b46e058b55e69b9ca6c2ddb0896d805f018350181d5"


def amendment(approvals):
    owner = approvals.get("owner_decisions") if isinstance(approvals, dict) else None
    value = owner.get("tract_design_amendment") if isinstance(owner, dict) else None
    required = {"new_estimand", "supersedes", "eligibility_closure", "non_target_frame_rows",
                "county_minima_count", "support_pool", "support_recipe", "relief_screen",
                "declared_fallback", "census", "disclosures", "process_change"}
    valid = isinstance(value, dict) and set(value) == required
    if valid:
        expected = {"eligibility_closure": "tract", "county_minima_count": "labelled_training_tracts",
                    "support_pool": "all_allocation_qualified_frame_tracts",
                    "relief_screen": "reported_diagnostic",
                    "support_recipe": {"bin_width_mmhg": 1.0, "min_bin_tracts": 2, "raw_x_radius_iqr": 2.0},
                    "declared_fallback": {"raw_x_radius_iqr": 3.0}}
        valid = all(value[k] == v for k, v in expected.items())
        if not valid:
            raise MissingPrerequisite("missing or malformed owner_decisions.tract_design_amendment")
        valid &= all(isinstance(value[k], str) and bool(value[k].strip()) for k in
                     ("new_estimand", "supersedes", "non_target_frame_rows", "disclosures", "process_change"))
        valid &= value["new_estimand"] == ("the first application gets a new estimand ID and a locked analysis; "
                    "the current estimand ID is recorded as infeasible as specified and kept in the registry")
        valid &= value["non_target_frame_rows"] == ("primary-eligible labelled rows are training rows for the outcome "
                    "regression and SSL; all others are inert; the density-ratio classifier fits on "
                    "target-distribution rows only (section 4.2)")
        census = value["census"]
        valid &= isinstance(census, dict) and census == {
            "evaluable_tracts": 1752, "counties": 158, "states": 33, "max_county_share": .054,
            "effective_counties": 59.2, "target_hash": "c8ecb139b57de78d", "flag1_tracts": 79}
        recipe = value["support_recipe"]
        valid &= (type(recipe["min_bin_tracts"]) is int and
                  all(type(recipe[k]) in (int, float) for k in ("bin_width_mmhg", "raw_x_radius_iqr")))
    if not valid:
        raise MissingPrerequisite("missing or malformed owner_decisions.tract_design_amendment")
    return value


def census(rows, ids, screens):
    ids = set(ids)
    selected = [r for r in rows if r.original_id in ids]
    counties = Counter(r.county for r in selected)
    n = len(selected)
    relief = [screens[c]["p90_p10_m"] for c in counties]
    return {"tracts": n, "counties": len(counties), "states": len({r.state for r in selected}),
            "per_county": dict(sorted(counties.items())), "flag_counts": flag_counts(selected),
            "max_county_share": max(counties.values()) / n if n else None,
            "effective_counties": n*n / sum(v*v for v in counties.values()) if n else None,
            "target_hash": sha256(json.dumps(sorted(ids)).encode()).hexdigest()[:16],
            "county_relief_quantiles_p10_p50_p90_m": np.quantile(relief, [.1, .5, .9]).tolist() if relief else [],
            "tracts_in_counties_relief_below_m": {str(t): sum(v for c, v in counties.items()
                if screens[c]["p90_p10_m"] < t) for t in (50, 100, 300)}}


def validate_approvals(config, approvals, manifest, covariates, geography, atlas):
    try:
        primary_outcome_flags(approvals)
    except ValueError as exc:
        raise MissingPrerequisite(str(exc)) from exc
    a2 = amendment(approvals)
    fixed = approvals.get("plan_fixed", {})
    owner = approvals.get("owner_decisions", {})
    pairs = {
        "support_design_fraction": "support_design_fraction",
        "min_observed_tracts": "min_flag1_tracts_per_county",
        "min_relief_m": "min_inhabited_relief_m", "max_comparison_km": "max_local_comparison_km",
        "min_outer_training_tracts": "min_outer_training_tracts_per_county",
        "min_inner_fitting_tracts": "min_inner_partition_tracts_per_county",
        "outer_folds": "outer_folds", "inner_folds": "inner_folds", "seeds": "seeds",
        "shift_mmhg": "tract_shift_mmhg",
    }
    expected = {"support_design_fraction": .2, "min_observed_tracts": 4, "min_relief_m": 300,
                "max_comparison_km": 25, "min_outer_training_tracts": 8, "min_inner_fitting_tracts": 4,
                "outer_folds": 5, "inner_folds": 3, "seeds": [1103, 2207, 3301], "shift_mmhg": 2}
    for key, approval_key in pairs.items():
        if key not in config or fixed.get(approval_key) != expected[key] or config[key] != expected[key]:
            raise MissingPrerequisite(f"missing or contradictory fixed approval: {approval_key}")
    if (config.get("buffers_km") != [0, 10, 25]
            or fixed.get("policy_form") != "shift_or_stay"
            or fixed.get("county_membership") != "coarse_comparison_stratum"):
        raise MissingPrerequisite("missing or contradictory policy/geography approval")
    for key in ("eligibility_closure", "county_minima_count", "support_pool", "relief_screen"):
        if config.get(key) != a2[key]:
            raise MissingPrerequisite(f"configuration contradicts tract_design_amendment: {key}")
    historical = owner.get("design_support_recipe", {})
    if (any(historical.get(k) != v for k, v in
            {"bin_width_mmhg": 1.0, "min_bin_tracts": 2, "raw_x_radius_iqr": 1.0}.items())
            or fixed.get("primary_outcome_flags") != [1]):
        raise MissingPrerequisite("missing or contradictory historical support/flag approval")
    recipe = dict(a2["support_recipe"])
    recipe["raw_x_radius"] = recipe.pop("raw_x_radius_iqr")
    if config.get("support") != recipe:
        raise MissingPrerequisite("support recipe contradicts tract_design_amendment")
    endpoint = manifest.spec.endpoint
    concepts = owner.get("endpoint_covariates", {}).get(endpoint, {}).get("concepts", {})
    if not concepts or not covariates.columns or not set(covariates.columns) <= set(concepts):
        raise MissingPrerequisite("explicit endpoint-approved feature concepts required")
    approved = {r.name for r in manifest.registry.rules if r.role == "predictor" and endpoint in r.endpoints}
    if set(covariates.columns) != approved:
        raise MissingPrerequisite("every approved covariate must remain in conditional support")
    try:
        for name in covariates.columns:
            rule = manifest.registry.require(name, endpoint, "nuisance")
            if rule.role != "predictor":
                raise MissingPrerequisite(f"non-predictor feature: {name}")
        manifest.registry.require(geography.county_field, endpoint, "county_routing")
    except ContractError as exc:
        raise MissingPrerequisite(f"missing feature/geography permission: {exc}") from exc
    if geography.approval_reference != "configs/approvals.yaml#plan_fixed.county_membership":
        raise MissingPrerequisite("explicit county geography approval required")
    if atlas.footprint != owner.get("exposure_atlas_footprint"):
        raise MissingPrerequisite("atlas footprint is not owner approved")
    require(manifest.spec.outcome_scale == "years" and manifest.spec.inference_unit == "county",
            "tract gate requires years and county inference")
    require(manifest.weight_field is None, "primary tract gate requires equal-tract origin weights")


def _inputs(request, task):
    if request.stage == "tract-support-gate":
        return dispatched_values(request)
    roles = task.get("dependencies", {})
    types = {"data_manifest": DataManifest, "covariates": CovariateView,
             "geography": GeographyTable, "atlas": CollectedAtlas, "entity_graph": EntityGraph}
    needed = set(types) | {"approvals"}
    if set(roles) != needed:
        raise MissingPrerequisite("task must bind approvals, data_manifest, covariates, geography, atlas, entity_graph")
    dependencies = dict(zip(request.dependency_paths, request.dependency_hashes))
    if any(path not in dependencies for path in roles.values()):
        raise MissingPrerequisite("task dependency is not hash-bound by StageRequest")
    if Path(roles["approvals"]).resolve() != OWNER_APPROVALS:
        raise MissingPrerequisite("approval dependency must be this repository's read-only owner file")
    values = {role: read_artifact(roles[role], cls, dependencies[roles[role]]) for role, cls in types.items()}
    values["approvals"] = yaml.safe_load(Path(roles["approvals"]).read_text())
    return values


def collected_atlas(paths, tract_ids=None):
    """Return physical atlas coverage; gating also uses its accounted omissions."""
    return collected_atlas_inputs(paths, tract_ids)['atlas']


def collected_atlas_inputs(paths, tract_ids=None):
    """Map the collected product using the pinned owner decision, without imputation.

    Zero-population/incomplete rows are absent from the usable atlas and remain
    explicit in its missing-tract accounting. The original file hashes remain
    parents of the request and the conversion is entirely outcome blind.
    """
    import pandas as pd
    from oxyformer.data.tract_inputs import tract_decisions
    from oxyformer.design.eligibility import AtlasRow
    from oxyformer.execution.runner import read_mapping
    decision = tract_decisions()
    parquet, quality_path, manifest_path = (paths['atlas-collect', n] for n in
                                           ('atlas.parquet', 'quality.json', 'artifact_manifest.json'))
    publication = json.loads(manifest_path.read_text())
    require(publication['kind'] == 'atlas-collect' and publication['status'] == 'pass',
            'collected atlas publication did not pass')
    require(publication['files'] == {'atlas.parquet': file_hash(parquet), 'quality.json': file_hash(quality_path)},
            'collected atlas publication binding mismatch')
    # The passing, runner-sealed collector already reconciled every block in
    # quality.json. Recheck its file binding above, without materializing the
    # nationwide block ledger a second time in the tract design process.
    frame = pd.read_parquet(parquet)
    require(not frame.duplicated(['tract_id', 'scenario']).any(), 'duplicate atlas tract/scenario')
    require((frame.population >= 0).all() and (frame.missing_population >= 0).all(),
            'invalid atlas population accounting')
    selected = frame[frame.scenario == decision['placement_scenario']]
    require(len(selected) > 0 and not selected.tract_id.duplicated().any(), 'missing or duplicate design atlas rows')
    # Gate coverage is checked on the endpoint frame. Uninhabited Census
    # tracts outside that frame do not become missing endpoint observations.
    expected = tuple(sorted(selected.tract_id if tract_ids is None else tract_ids))
    selected = selected[selected.tract_id.isin(expected)]
    rows, missing_dem = [], set()
    missing = sorted(set(expected) - set(selected.tract_id))
    for record in selected.sort_values('tract_id').to_dict('records'):
        complete = record['population'] > 0 and record['missing_population'] == 0
        require(record['status'] == ('complete' if complete else
                'zero_population' if record['population'] == 0 else 'missing_dem'), 'inconsistent atlas status')
        if not complete:
            missing.append(record['tract_id'])
            if record['status'] == 'missing_dem':
                missing_dem.add(record['tract_id'])
            continue
        rows.append(AtlasRow(tract_id=record['tract_id'], exposure_mmhg=float(record[decision['exposure_field']]),
            inhabited_elevation_m=float(record['elevation_' + decision['inhabited_elevation'] + '_m']),
            population=float(record['population']), allocation_qualified=True))
    owner = read_mapping(OWNER_APPROVALS)['owner_decisions']
    atlas = CollectedAtlas(rows=tuple(rows), source_hashes=tuple(sorted(set(publication['source_identities'].values()))),
        footprint=owner['exposure_atlas_footprint'], expected_tract_ids=expected, missing_tract_ids=tuple(sorted(missing)),
        coverage_complete=not missing, mapping_review_id='configs/approvals.yaml#owner_decisions.tract_design')
    # The passing collector copies this validated quality.json summary verbatim
    # into its manifest (exposure.build.run_stage). Its file binding was checked
    # above; use the compact copy without loading the nationwide block ledger.
    accounted = publication.get('missing_dem_tracts', [])
    require(isinstance(accounted, list) and all(isinstance(t, str) for t in accounted),
            'invalid collected missing-DEM summary')
    # The quality summary spans placement scenarios. Only a selected-scenario
    # missing_dem row explains that scenario's omission; an absent or empty row
    # cannot borrow an explanation from another scenario.
    return {'atlas': atlas, 'atlas_missing_dem_tract_ids': tuple(sorted(missing_dem & set(accounted)))}


def atlas_coverage(values):
    """Keep physical coverage explicit while accounting for approved exclusions."""
    atlas = values['atlas']
    accounted = set(values.get('atlas_missing_dem_tract_ids', ()))
    require(accounted <= set(atlas.missing_tract_ids), 'accounted DEM omissions must be missing atlas tracts')
    if accounted:
        decision = values['approvals'].get('owner_decisions', {}).get('atlas_coverage', {})
        if decision.get('missing_dem_tracts') != 'exclude_as_not_allocation_qualified':
            raise MissingPrerequisite('missing owner approval for accounted DEM exclusions')
    unaccounted = set(atlas.missing_tract_ids) - accounted
    return {'complete': not unaccounted, 'physical_coverage_complete': atlas.coverage_complete,
            'footprint': atlas.footprint, 'expected_tracts': len(atlas.expected_tract_ids),
            'covered_tracts': len(atlas.rows), 'missing_tract_ids': atlas.missing_tract_ids,
            'accounted_missing_dem_tracts': len(accounted),
            'accounted_missing_dem_tract_ids': tuple(sorted(accounted)),
            'unaccounted_missing_tract_ids': tuple(sorted(unaccounted))}


def flag_counts(rows):
    counts = Counter(str(row.outcome_flag) for row in rows)
    return {str(flag): counts[str(flag)] for flag in (1, 2, 3)}


def allocation_exclusions(rows, coverage, design_ids):
    """Disclose DEM exclusions even when their labels are in the sealed reserve."""
    missing = set(coverage['accounted_missing_dem_tract_ids'])
    return [{'original_id': r.original_id, 'tract_id': r.tract_id,
             'reason': 'atlas_missing_dem_coverage', 'sealed_design': r.original_id in design_ids}
            for r in rows if r.tract_id in missing]


def dispatched_values(request):
    from oxyformer.data.tract_inputs import dispatch_inputs
    from oxyformer.execution.runner import read_mapping
    paths = dispatch_inputs(request)
    hashes = dict(zip(request.dependency_paths, request.dependency_hashes))
    values = {}
    for role, cls in [('data_manifest', DataManifest), ('covariates', CovariateView),
                      ('geography', GeographyTable), ('entity_graph', EntityGraph)]:
        path = paths['tract-inputs', role + '.json']
        values[role] = read_artifact(path, cls, hashes[str(path)])
    values.update(collected_atlas_inputs(paths, values['data_manifest'].original_ids))
    values['approvals'] = read_mapping(OWNER_APPROVALS)
    return values


def _validate_data(manifest, covariates, geography, atlas, graph):
    for source in manifest.sources:
        if source.mapping_status != "reviewed":
            raise MissingPrerequisite(f"unreviewed source mapping: {source.source_id}")
        source.assert_usable()
    manifest.spec.assert_compatible(covariates.spec)
    require(covariates.registry == manifest.registry and covariates.use == "nuisance",
            "approved nuisance covariate view required")
    ids = manifest.original_ids
    require(ids == covariates.original_ids == tuple(r.original_id for r in geography.rows),
            "validated input order mismatch")
    require(set(ids) == set(graph.original_ids) and graph.content_hash == manifest.entity_graph_hash,
            "entity graph mismatch")
    require(geography.data_manifest_hash == manifest.content_hash,
            "geography data binding mismatch")
    require(manifest.content_hash in covariates.lineage.parent_hashes,
            "covariate data parent required")
    require(set(covariates.lineage.source_hashes) == set(manifest.lineage.source_hashes),
            "covariate source mismatch")
    require(geography.county_field in {c.name for c in manifest.schema}, "unbound county field")
    require({r.tract_id for r in geography.rows} <= set(atlas.expected_tract_ids),
            "data geography outside declared atlas frame")
    # This is the tract endpoint: repeated tract rows cannot be counted as new
    # observations or receive repeated equal-tract mass. Grouped policy helpers
    # separately support individual records and enforce common actions.
    require(len({r.tract_id for r in geography.rows}) == len(ids), "duplicate tract observations")


def _publish(request, payloads, lineage, status, message):
    root = Path(request.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    names = ("design.json", "splits.json", "support_report.json", "gate.json", "artifact_manifest.json")
    require(not any((root / name).exists() for name in names), "immutable design output already exists")
    records = []
    # Publish complete bytes atomically per file, never overwrite. The manifest
    # is published last and commits the other four files; it cannot hash itself.
    with tempfile.TemporaryDirectory(prefix=".design-", dir=root) as staging:
        for name in names[:4]:
            data = payloads[name]
            text = data.to_json() if isinstance(data, Immutable) else canonical_json(data)
            path = Path(staging) / name
            path.write_text(text)
            records.append(ArtifactRecord(path=name, sha256=file_hash(path), lineage=lineage,
                                          kind=name.removesuffix(".json")))
        commit = StageResult(request_hash=request.content_hash, status=status,
                             artifacts=tuple(records), message=message)
        path = Path(staging) / names[-1]
        path.write_text(commit.to_json())
        records.append(ArtifactRecord(path=names[-1], sha256=file_hash(path), lineage=lineage,
                                      kind="artifact_manifest"))
        for name in names:
            os.link(Path(staging) / name, root / name)
    return StageResult(request_hash=request.content_hash, status=status, artifacts=tuple(records), message=message)


def design_configuration(request):
    """Read scientific settings only from this checkout's design configuration.

    The dispatcher envelope is transport metadata. It cannot supply a support
    recipe, threshold or approval override. Keep the direct tract_design API
    available for existing callers and the synthetic design tests.
    """
    from oxyformer.execution.runner import read_mapping
    from oxyformer.data.tract_inputs import tract_decisions, validate_dispatch_approvals
    tract_decisions()
    canonical = read_mapping(DESIGN_CONFIG)
    require(canonical.get("schema_version") == 1 and canonical.get("stage") == "tract_design",
            "invalid repository design configuration")
    submitted = read_mapping(request.config_path)
    if request.stage == "tract_design":
        require(submitted == canonical, "design settings differ from repository configuration")
        return canonical
    require(request.stage == submitted.get("stage") == "tract-support-gate",
            "invalid design stage configuration")
    require(submitted.get("settings", {}).get("module") == "oxyformer.design.gate",
            "invalid registered design module")
    validate_dispatch_approvals(submitted)
    return canonical


def run_stage(request: StageRequest) -> StageResult:
    status, message = "blocked", "missing design prerequisites"
    design = {"status": "blocked", "effect_release_authorized": False}
    split_payload = {"scenarios": []}
    report = {"coverage": "unknown", "inference_frame": INFERENCE_FRAME}
    lineage = ArtifactLineage(source_hashes=(request.config_hash,), unit_ids=(request.stage,),
                              parent_hashes=request.dependency_hashes, split_hash=None,
                              config_hash=request.config_hash, model_hash=None,
                              environment=(("python", platform.python_version()), ("code", request.code_identity)),
                              seed=1103, parameter_count=None)
    try:
        request.verify_inputs()
        config = design_configuration(request)
        values = _inputs(request, json.loads(Path(request.task_path).read_text()))
        manifest, covariates = values["data_manifest"], values["covariates"]
        geography, atlas, graph = values["geography"], values["atlas"], values["entity_graph"]
        _validate_data(manifest, covariates, geography, atlas, graph)
        validate_approvals(config, values["approvals"], manifest, covariates, geography, atlas)
        lineage = ArtifactLineage(source_hashes=tuple(sorted(set(manifest.lineage.source_hashes + atlas.source_hashes))),
                                  unit_ids=manifest.original_ids, parent_hashes=request.dependency_hashes,
                                  split_hash=None, config_hash=request.config_hash, model_hash=None,
                                  environment=lineage.environment, seed=1103, parameter_count=None)
        rows = geography.rows
        flags = primary_outcome_flags(values["approvals"])
        a2 = amendment(values["approvals"])
        a = {r.tract_id: r for r in atlas.rows}
        reservation = reserve_design(rows, graph)
        design_ids = set(reservation.design_ids)
        screens = county_screen(rows, a)
        coverage = atlas_coverage(values)
        accounted_missing = set(coverage["accounted_missing_dem_tract_ids"])
        report = {"coverage": coverage,
                  "primary_outcome_flags": list(primary_outcome_flags(values["approvals"])),
                  "registered_sensitivity": "flag1_only",
                  "mortality_input_flag": {r.original_id: r.outcome_flag for r in rows},
                  "flag_counts": flag_counts(rows),
                  "primary_eligible_flag_counts": flag_counts(r for r in rows if _usable(r, a, flags)),
                  "sealed_design_flag_counts": flag_counts(
                      r for r in rows if r.original_id in design_ids),
                  "allocation_exclusions": allocation_exclusions(rows, coverage, set(reservation.design_ids)),
                  "initial_county_screens": screens, "inference_frame": INFERENCE_FRAME,
                  "reservation": {"requested_fraction": .2, "rounding": "ceil per county then lineage closure",
                                  "total_subblocks": reservation.total_subblocks,
                                  "selected_subblocks": len(reservation.selected_subblocks),
                                  "sealed_subblocks": len(reservation.sealed_subblocks),
                                  "sealed_fraction": len(reservation.sealed_subblocks) / reservation.total_subblocks,
                                  "excluded_design_labels": len(reservation.design_ids)}}
        design_ids = set(reservation.design_ids)
        frozen = freeze_support(rows, reservation.design_ids, a, covariates,
                                recipe=SupportRecipe(**config["support"]))
        design = FrozenDesign(reservation=reservation, support=frozen, approved_features=manifest.registry)
        supported = supported_ids(rows, a, frozen)
        design_ids = set(reservation.design_ids)
        reasons = {}
        for row in rows:
            if row.original_id in design_ids:
                continue
            if row.tract_id in accounted_missing:
                reasons[row.original_id] = "atlas_missing_dem_coverage"
            elif not _usable(row, a, flags):
                reasons[row.original_id] = "flag_label_or_allocation_unusable"
            elif screens[row.county]["reasons"]:
                reasons[row.original_id] = "initial_county_screen"
            elif row.original_id not in supported:
                reasons[row.original_id] = "outside_frozen_conditional_support"
        excluded = set(reasons) - design_ids
        candidates = set(manifest.original_ids) - design_ids - excluded
        scenarios = tuple(build_scenario(rows, graph, manifest, reservation, candidates, reasons,
                                         radius_km=radius, policy_id=frozen.policy.policy_id,
                                         atlas=a, covariates=covariates, support=frozen)
                          for radius in config["buffers_km"])
        registry = [{"estimand_id": FROZEN_TARGET_ID, "status": "infeasible as specified",
                     "reference": "frozen section 4.1; tract-gate-r2/2026-10-10"},
                    {"estimand_id": A2_ESTIMAND_ID, "status": "locked analysis",
                     "reference": "configs/approvals.yaml#owner_decisions.tract_design_amendment"}]
        eligible = {r.original_id for r in rows if _usable(r, a, flags)}
        unsealed = eligible - design_ids
        screened = {r.original_id for r in rows if r.original_id in unsealed and not screens[r.county]["reasons"]}
        layers = [("eligibility", eligible), ("reservation", unsealed), ("screens", screened),
                  ("support", candidates)]
        report.update(estimand_registry=registry, estimand_id=A2_ESTIMAND_ID,
                      declared_fallback={**a2["declared_fallback"], "run": False},
                      support_pool=frozen.support_pool, raw_x_radius_iqr=frozen.recipe.raw_x_radius,
                      layers={key: census(rows, ids, screens) for key, ids in layers})
        split_payload = {"estimand_id": A2_ESTIMAND_ID, "estimand_registry": registry, "design_hash": design.content_hash, "scenarios": [SplitScenarioDesign.from_scenario(s).to_dict() for s in scenarios],
                         "buffer_rule": "common target excludes both sides of cross-fold boundaries, whole lineages",
                         "deletion_rule": "full analysis refit after deleting county/state"}
        diagnostics = []
        primary_ids = set(scenarios[0].outer.target_ids) if scenarios[0].outer else set()
        for scenario in scenarios:
            ids = set(scenario.outer.target_ids) if scenario.outer else set()
            chosen = [r for r in rows if r.original_id in ids]
            action = policy_for(chosen, a, frozen) if chosen else None
            counts = {}
            for _, reason in scenario.exclusions:
                counts[reason] = counts.get(reason, 0) + 1
            diagnostics.append({"buffer_km": scenario.buffer_km, "status": scenario.status,
                                "target_id": scenario.target_id, "evaluation_count": len(ids),
                                "target_hash": scenario.target_hash, "census": census(rows, ids, screens),
                                "layers": {k: census(rows, v, screens) for k, v in scenario.layer_ids},
                                "uninformative": not bool(ids),
                                "outer_buffer_excluded_count": len(scenario.outer_buffer_excluded_ids),
                                "outer_buffer_frame_census": census(rows, scenario.outer_buffer_excluded_ids, screens),
                                "evaluation_flag_counts": flag_counts(chosen),
                                "sealed_design_count": len(design_ids), "excluded_count": len(scenario.exclusions),
                                "exclusion_counts": counts, "removed_from_primary": sorted(primary_ids - ids),
                                "added_to_primary": sorted(ids - primary_ids),
                                "shifted_fraction": sum(action.moved) / len(ids) if action else None,
                                "achieved_average_shift_mmhg": 2 * sum(action.moved) / len(ids) if action else None,
                                "warning": "policy_moves_no_tracts" if action and not any(action.moved) else None})
        report["scenarios"] = diagnostics
        report["support_method"] = "replicated interior bins in raw-X neighborhoods; empirical screen, not a positivity guarantee"
        report["frozen_policy_id"] = frozen.policy.policy_id
        if not coverage["complete"]:
            status, message = "blocked", "incomplete atlas coverage; target accounting recorded"
        elif scenarios[0].status != "pass":
            status, message = "fail", "no tract target passes primary geographic, support and split minima"
        else:
            status, message = "pass", "primary tract feasibility passed; buffer-specific target changes are disclosed"
    except MissingPrerequisite as exc:
        status, message = "blocked", str(exc)
    except FileNotFoundError as exc:
        status, message = "blocked", f"missing prerequisite file: {exc.filename}"
    except (ContractError, KeyError, TypeError, ValueError) as exc:
        status, message = "fail", f"invalid design input or failed design: {exc}"
    if not isinstance(design, FrozenDesign):
        design = {"available": False, "status": status, "message": message, "effect_release_authorized": False}
    gate = {"status": status, "message": message, "effect_release_authorized": False,
            "request_hash": request.content_hash, "inference_frame": INFERENCE_FRAME}
    return _publish(request, {"design.json": design, "splits.json": split_payload,
                              "support_report.json": report, "gate.json": gate}, lineage, status, message)
