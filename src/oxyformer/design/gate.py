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
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import tempfile

import yaml

from oxyformer.contracts import CovariateView, DataManifest, StageRequest, StageResult
from oxyformer.data.entity_graph import EntityGraph
from oxyformer.data.feature_roles import FeatureRegistry
from oxyformer.design.eligibility import CollectedAtlas, GeographyTable, county_screen, usable
from oxyformer.design.splits import Reservation, build_scenario, close_groups, reserve_design
from oxyformer.design.support import FrozenSupport, SupportRecipe, freeze_support, policy_for, supported_ids
from oxyformer.provenance import (
    ArtifactLineage, ArtifactRecord, ContractError, Immutable, canonical_json, file_hash,
    read_artifact, require,
)

ROOT = Path(__file__).resolve().parents[3]
OWNER_APPROVALS = ROOT / "configs" / "approvals.yaml"
DESIGN_CONFIG = ROOT / "configs" / "design.yaml"
INFERENCE_FRAME = (
    "Equal-tract inference is conditional on the sealed geographic support design, "
    "approved raw-X registry, frozen support policy, atlas coverage and buffer-specific "
    "eligible flag-1 tract frame. Sealed design outcomes enter neither nuisance fitting "
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


def validate_approvals(config, approvals, manifest, covariates, geography, atlas):
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
    if (config.get("buffers_km") != [0, 10, 25] or fixed.get("primary_outcome_flags") != [1]
            or fixed.get("policy_form") != "shift_or_stay"
            or fixed.get("county_membership") != "coarse_comparison_stratum"):
        raise MissingPrerequisite("missing or contradictory policy/geography approval")
    endpoint = manifest.spec.endpoint
    concepts = owner.get("endpoint_covariates", {}).get(endpoint, {}).get("concepts", {})
    if not concepts or not covariates.columns or not set(covariates.columns) <= set(concepts):
        raise MissingPrerequisite("explicit endpoint-approved feature concepts required")
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
    rows, missing = [], sorted(set(expected) - set(selected.tract_id))
    for record in selected.sort_values('tract_id').to_dict('records'):
        complete = record['population'] > 0 and record['missing_population'] == 0
        require(record['status'] == ('complete' if complete else
                'zero_population' if record['population'] == 0 else 'missing_dem'), 'inconsistent atlas status')
        if not complete:
            missing.append(record['tract_id'])
            continue
        rows.append(AtlasRow(tract_id=record['tract_id'], exposure_mmhg=float(record[decision['exposure_field']]),
            inhabited_elevation_m=float(record['elevation_' + decision['inhabited_elevation'] + '_m']),
            population=float(record['population']), allocation_qualified=True))
    owner = read_mapping(OWNER_APPROVALS)['owner_decisions']
    return CollectedAtlas(rows=tuple(rows), source_hashes=tuple(sorted(set(publication['source_identities'].values()))),
        footprint=owner['exposure_atlas_footprint'], expected_tract_ids=expected, missing_tract_ids=tuple(sorted(missing)),
        coverage_complete=not missing, mapping_review_id='configs/approvals.yaml#owner_decisions.tract_design')


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
    values['atlas'] = collected_atlas(paths, values['data_manifest'].original_ids)
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
    from oxyformer.data.tract_inputs import tract_decisions
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
    require(submitted.get("approvals") == read_mapping(OWNER_APPROVALS),
            "dispatcher approvals differ from repository owner file")
    require(submitted.get("input_sources", {}).get(str(OWNER_APPROVALS)) == file_hash(OWNER_APPROVALS),
            "dispatcher owner file is not hash-bound")
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
        a = {r.tract_id: r for r in atlas.rows}
        reservation = reserve_design(rows, graph)
        screens = county_screen(rows, a)
        report = {"coverage": {"complete": atlas.coverage_complete, "footprint": atlas.footprint,
                               "expected_tracts": len(atlas.expected_tract_ids), "covered_tracts": len(atlas.rows),
                               "missing_tract_ids": atlas.missing_tract_ids},
                  "initial_county_screens": screens, "inference_frame": INFERENCE_FRAME,
                  "reservation": {"requested_fraction": .2, "rounding": "ceil per county then lineage closure",
                                  "total_subblocks": reservation.total_subblocks,
                                  "selected_subblocks": len(reservation.selected_subblocks),
                                  "sealed_subblocks": len(reservation.sealed_subblocks),
                                  "sealed_fraction": len(reservation.sealed_subblocks) / reservation.total_subblocks,
                                  "excluded_design_labels": len(reservation.design_ids)}}
        design_ids = set(reservation.design_ids)
        if not atlas.coverage_complete and not any(
                r.original_id in design_ids and r.tract_id in a
                and a[r.tract_id].allocation_qualified and a[r.tract_id].population > 0 for r in rows):
            raise MissingPrerequisite("incomplete atlas coverage leaves no qualified sealed design records")
        frozen = freeze_support(rows, reservation.design_ids, a, covariates,
                                recipe=SupportRecipe(**config["support"]))
        design = FrozenDesign(reservation=reservation, support=frozen, approved_features=manifest.registry)
        supported = supported_ids(rows, a, frozen)
        design_ids = set(reservation.design_ids)
        reasons = {}
        for row in rows:
            if row.original_id in design_ids:
                continue
            if not usable(row, a):
                reasons[row.original_id] = "flag_label_or_allocation_unusable"
            elif screens[row.county]["reasons"]:
                reasons[row.original_id] = "initial_county_screen"
            elif row.original_id not in supported:
                reasons[row.original_id] = "outside_frozen_conditional_support"
        excluded = close_groups(reasons, reservation.groups) - design_ids
        candidates = set(manifest.original_ids) - design_ids - excluded
        scenarios = tuple(build_scenario(rows, graph, manifest, reservation, candidates, reasons,
                                         radius_km=radius, policy_id=frozen.policy.policy_id,
                                         atlas=a, covariates=covariates, support=frozen)
                          for radius in config["buffers_km"])
        split_payload = {"design_hash": design.content_hash, "scenarios": [s.to_dict() for s in scenarios],
                         "buffer_rule": "common target excludes both sides of cross-fold boundaries, whole lineages",
                         "deletion_rule": "full analysis refit after deleting county/state"}
        diagnostics = []
        primary_ids = set(scenarios[0].outer.original_ids) if scenarios[0].outer else set()
        for scenario in scenarios:
            ids = set(scenario.outer.original_ids) if scenario.outer else set()
            chosen = [r for r in rows if r.original_id in ids]
            action = policy_for(chosen, a, frozen) if chosen else None
            counts = {}
            for _, reason in scenario.exclusions:
                counts[reason] = counts.get(reason, 0) + 1
            diagnostics.append({"buffer_km": scenario.buffer_km, "status": scenario.status,
                                "target_id": scenario.target_id, "evaluation_count": len(ids),
                                "sealed_design_count": len(design_ids), "excluded_count": len(scenario.exclusions),
                                "exclusion_counts": counts, "removed_from_primary": sorted(primary_ids - ids),
                                "added_to_primary": sorted(ids - primary_ids),
                                "shifted_fraction": sum(action.moved) / len(ids) if action else None,
                                "achieved_average_shift_mmhg": 2 * sum(action.moved) / len(ids) if action else None,
                                "warning": "policy_moves_no_tracts" if action and not any(action.moved) else None})
        report["scenarios"] = diagnostics
        report["support_method"] = "replicated interior bins in raw-X neighborhoods; empirical screen, not a positivity guarantee"
        report["frozen_policy_id"] = frozen.policy.policy_id
        if not atlas.coverage_complete:
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
