"""Outcome-free real-frame adapter for the registered Suite A profiles.

The target and policy use the merged tract design helpers. This is a simulation
endpoint: placeholder labels are never fitted, and only the SCM supplies A/Y.
The independent tract gate remains a prerequisite of any later campaign lock.
"""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from oxyformer.contracts import CovariateView, DataManifest, StageRequest, StageResult
from oxyformer.data.entity_graph import EntityGraph
from oxyformer.data.loaders import load_records
from oxyformer.data.tract_inputs import tract_decisions, validate_dispatch_approvals
from oxyformer.design import gate
from oxyformer.design.eligibility import GeographyTable, county_screen, usable
from oxyformer.design.policies import PolicyCovariates
from oxyformer.design.splits import build_scenario, close_groups, reserve_design
from oxyformer.design.support import SupportRecipe, freeze_support, supported_ids
from oxyformer.execution.runner import read_mapping
from oxyformer.models.treatment_query import TreatmentDesign
from oxyformer.provenance import ContractError, read_artifact, require
from oxyformer.training.nested_cv import PreparedEndpoint
from oxyformer.validation import campaign
from oxyformer.validation.coverage import dependency, digest, publish, repetition_plan
from oxyformer.validation.scm import CovariateFrame, SCMConfig

ROOT = Path(__file__).parents[3]
TASK_FILE = ROOT / 'configs/execution/tasks/campaign_profiles.yaml'
ENDPOINT = 'synthetic-suite-a-real-frame'


def prepare_inputs(request):
    """Read and validate transport/typed inputs; do not design, simulate or fit."""
    request.verify_inputs()
    config = read_mapping(request.config_path)
    task = json.loads(Path(request.task_path).read_text())
    registered = read_mapping(TASK_FILE)['tasks'][0]
    require(request.stage == task['stage'] == config['stage'] == 'real-frame-inputs', 'stage mismatch')
    require(task == registered, 'real-frame task differs from repository')
    validate_dispatch_approvals(config)
    tract_decisions()
    paths = {(unit, name): dependency(request, config, {'dependency': unit, 'path': name})
             for unit, names in task['needs'].items() for name in names}
    hashes = dict(zip(request.dependency_paths, request.dependency_hashes))
    values = {}
    for role, cls in [('data_manifest', DataManifest), ('covariates', CovariateView),
                      ('geography', GeographyTable), ('entity_graph', EntityGraph)]:
        path = paths['tract-inputs', role + '.json']
        values[role] = read_artifact(path, cls, hashes[str(path)])
    values['atlas'] = gate.collected_atlas(paths, values['data_manifest'].original_ids)
    values['approvals'] = read_mapping(gate.OWNER_APPROVALS)
    design = read_mapping(gate.DESIGN_CONFIG)
    gate._validate_data(values['data_manifest'], values['covariates'], values['geography'],
                        values['atlas'], values['entity_graph'])
    gate.validate_approvals(design, values['approvals'], values['data_manifest'],
                           values['covariates'], values['geography'], values['atlas'])
    return values, design


def build_inputs(values, design):
    """Freeze the primary target with unchanged support and geographic gates."""
    manifest, covariates = values['data_manifest'], values['covariates']
    geography, atlas, graph = values['geography'], values['atlas'], values['entity_graph']
    gate._validate_data(manifest, covariates, geography, atlas, graph)
    gate.validate_approvals(design, values['approvals'], manifest, covariates, geography, atlas)
    require(atlas.coverage_complete, 'incomplete atlas coverage')
    rows, a = geography.rows, {r.tract_id: r for r in atlas.rows}
    reservation = reserve_design(rows, graph)
    screens = county_screen(rows, a)
    frozen = freeze_support(rows, reservation.design_ids, a, covariates,
                            recipe=SupportRecipe(**design['support']))
    supported = supported_ids(rows, a, frozen)
    design_ids, reasons = set(reservation.design_ids), {}
    for row in rows:
        if row.original_id in design_ids:
            continue
        if not usable(row, a):
            reasons[row.original_id] = 'flag_label_or_allocation_unusable'
        elif screens[row.county]['reasons']:
            reasons[row.original_id] = 'initial_county_screen'
        elif row.original_id not in supported:
            reasons[row.original_id] = 'outside_frozen_conditional_support'
    excluded = close_groups(reasons, reservation.groups) - design_ids
    scenario = build_scenario(rows, graph, manifest, reservation,
        set(manifest.original_ids) - design_ids - excluded, reasons, radius_km=0,
        policy_id=frozen.policy.policy_id, atlas=a, covariates=covariates, support=frozen)
    require(scenario.status == 'pass', 'no tract target passes primary geographic, support and split minima')
    # Preserve source permissions; the synthetic name distinguishes generated
    # outcomes from the clinical endpoint. Add no predictor or geographic input.
    registry = replace(manifest.registry, rules=tuple(replace(rule,
        endpoints=tuple(dict.fromkeys((*rule.endpoints, ENDPOINT)))) for rule in manifest.registry.rules))
    spec = replace(scenario.data_manifest.spec, endpoint=ENDPOINT,
                   adjustment_schema_hash=registry.content_hash)
    template_manifest = replace(scenario.data_manifest, spec=spec, registry=registry)
    outer = replace(scenario.outer, spec=spec, lineage=replace(scenario.outer.lineage,
                    parent_hashes=(template_manifest.content_hash,)))
    inners = []
    for inner in scenario.inner:
        local = replace(inner.data_manifest, spec=spec, registry=registry,
            lineage=replace(inner.data_manifest.lineage, parent_hashes=(template_manifest.content_hash,)))
        split = replace(inner.split, spec=spec,
            lineage=replace(inner.split.lineage, parent_hashes=(local.content_hash,)))
        inners.append(replace(inner, data_manifest=local, split=split))
    x = dict(zip(covariates.original_ids, covariates.values))
    by_id = {r.original_id: r for r in rows}
    records = []
    for oid in template_manifest.original_ids:
        record = dict(zip(covariates.columns, x[oid]))
        record.update({manifest.id_field: oid, manifest.outcome_field: None,
            manifest.exposure_field: a[by_id[oid].tract_id].exposure_mmhg,
            geography.county_field: by_id[oid].county})
        records.append(record)
    data = load_records(records, template_manifest, spec, template_manifest.schema_hash)
    selected = tuple(by_id[oid] for oid in outer.original_ids)
    # Whole dependence components survive the frozen eligibility and partitions.
    clusters = {oid: digest(group) for group in reservation.groups for oid in group}
    frame = CovariateFrame(original_ids=outer.original_ids,
        geography_ids=tuple(r.assignment_geography for r in selected),
        region_ids=tuple(r.county for r in selected),
        cluster_ids=tuple(clusters[r.original_id] for r in selected),
        coordinates=tuple((r.latitude, r.longitude) for r in selected),
        columns=covariates.columns, x=tuple(x[r.original_id] for r in selected),
        support_keys=tuple(r.assignment_geography for r in selected),
        weights=(1.,) * len(selected), outcome_available=tuple(r.label_available for r in selected),
        biomarker_available=(True,) * len(selected))
    for row in campaign._registry()['scenarios']:
        SCMConfig(**row).validate_policy(frozen.policy, frame)
    knots = frozen.spline_knots
    sealed_a = [a[r.tract_id].exposure_mmhg for r in rows if r.original_id in design_ids]
    center = sum(sealed_a) / len(sealed_a)
    scale = (sum((v - center) ** 2 for v in sealed_a) / len(sealed_a)) ** .5
    families = values['approvals']['owner_decisions']['endpoint_covariates'][manifest.spec.endpoint]['masking_families']
    endpoint = PreparedEndpoint(data=data, entity_graph=graph,
        geography=replace(geography, data_manifest_hash=template_manifest.content_hash),
        outer=outer, inner=tuple(inners), policy=frozen.policy,
        policy_covariates=PolicyCovariates(original_ids=manifest.original_ids,
            geography_ids=tuple(r.assignment_geography for r in rows),
            support_keys=tuple(r.assignment_geography for r in rows)),
        treatment_design=TreatmentDesign(center=center, scale=scale, knots=knots, design_hash=frozen.content_hash),
        feature_kinds=tuple((name, 'numeric') for name in covariates.columns),
        families=tuple(tuple(n for n in names if n in covariates.columns)
                       for names in families.values() if set(names).intersection(covariates.columns)),
        county_field=geography.county_field, exposure_assignment_level='tract')
    for fold in range(5):
        endpoint.validate(fold)
    audit = {'input_rows': len(rows), 'evaluation_rows': len(selected),
        'design_ids': list(reservation.design_ids), 'exclusions': list(scenario.exclusions),
        'cluster_sizes': dict(Counter(frame.cluster_ids)),
        'missing_covariates': sum(v is None for row in frame.x for v in row),
        'target_id': spec.target_id, 'source_manifest_hash': manifest.content_hash,
        'source_covariates_hash': covariates.content_hash, 'source_geography_hash': geography.content_hash,
        'source_entity_graph_hash': graph.content_hash, 'atlas_hash': atlas.content_hash,
        'labels': 'No real outcomes read; generated A/Y replace placeholders for every draw',
        'certifies_production_coverage': False, 'effect_release_authorized': False}
    return endpoint, frame, frozen, audit


def profile_tasks(endpoint, frame):
    """Concrete hash-bound tasks, generated once after real inputs exist.

    ``resources`` is the scheduler request consumed by coordinator plan wiring;
    the merged stage runner executes inside that allocation and submits no job.
    Profile and projection resources share one declaration.
    """
    locations = {}
    for county in sorted(set(frame.region_ids)):
        points = [p for c, p in zip(frame.region_ids, frame.coordinates) if c == county]
        locations[county] = [sum(p[j] for p in points) / len(points) for j in range(2)]
    recipe = {'endpoint_hash': endpoint.content_hash, 'frame_hash': frame.content_hash,
        'nested_cv': {'ssl_epochs': 30, 'frozen_epochs': 150, 'batch_size': 256, 'precision': 'fp32'},
        'inference': {'primary_bandwidth_km': 100, 'county_locations': locations}}
    campaign.validate_recipe(recipe)
    common = {'endpoint_input': {'dependency': 'real-frame-inputs', 'path': 'endpoint.json'},
              'frame_input': {'dependency': 'real-frame-inputs', 'path': 'frame.json'}, 'recipe': recipe}
    inputs = {'real-frame-inputs': ['endpoint.json', 'frame.json', 'artifact_manifest.json']}
    # CPU leaves: 11.5 h of work within the 12 h scheduler limit. The four
    # GPU-hour mandate applies only to leaves that request GPUs.
    resources = {'cpus_per_task': 8, 'gpus': 0, 'wall_seconds': 41400}
    tasks, profiles = [], {}
    for row in campaign._registry()['scenarios']:
        scenario = SCMConfig(**row).to_dict()['payload']
        task_id = 'profile-' + scenario['name'].replace('_', '-')
        tasks.append({'id': task_id, 'stage': 'simulation-smoke', 'needs': inputs,
            'resources': dict(resources),
            'outputs': ['result.json', 'timing.json', 'profile_receipt.json', 'artifact_manifest.json'],
            'parameters': {**common, 'mode': 'profile', 'scenario': scenario,
                'draws': repetition_plan(digest(['real-frame-profile-v1', recipe]), scenario['name'], 1),
                'wall_seconds': resources['wall_seconds']}})
        profiles[scenario['name']] = {'dependency': task_id, 'path': 'result.json'}
    tasks.append({'id': 'campaign-estimate', 'stage': 'campaign-estimate',
        'resources': dict(resources),
        'needs': {**inputs, **{ref['dependency']: ['result.json', 'timing.json', 'artifact_manifest.json']
                             for ref in profiles.values()}},
        'outputs': ['budget_estimate.json', 'artifact_manifest.json'],
        'parameters': {**common, 'profiles': profiles, 'final_repetitions': 1000,
            **resources, 'profile_safety_factor': 2.}})
    return {'schema_version': 1, 'tasks': tasks}


def run_stage(request: StageRequest) -> StageResult:
    try:
        values, design = prepare_inputs(request)
        endpoint, frame, support, audit = build_inputs(values, design)
        return publish(request, {'endpoint.json': endpoint.to_dict(), 'frame.json': frame.to_dict(),
            'support.json': support.to_dict(), 'input_audit.json': audit,
            'profile_tasks.json': profile_tasks(endpoint, frame)}, status='pass',
            message='Real covariates and frozen primary design prepared for simulated outcomes; no campaign admitted')
    except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
        return StageResult(request_hash=request.content_hash, status='blocked', artifacts=(),
                           message=str(exc) or type(exc).__name__)
