"""Outcome-free real-frame adapter for the registered Suite A profiles.

The target and policy use the merged tract design helpers. This is a simulation
endpoint: placeholder labels are never fitted, and only the SCM supplies A/Y.
The independent tract gate remains a prerequisite of any later campaign lock.
"""
from collections import Counter
from copy import deepcopy
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
from oxyformer.design.splits import build_scenario, reserve_design
from oxyformer.design.support import SupportRecipe, freeze_support, supported_ids
from oxyformer.execution.runner import read_mapping
from oxyformer.models.treatment_query import TreatmentDesign
from oxyformer.provenance import ContractError, read_artifact, require
from oxyformer.training.nested_cv import PreparedEndpoint
from oxyformer.validation import campaign
from oxyformer.validation.coverage import dependency, digest, publish, repetition_plan
from oxyformer.validation.scm import CovariateFrame, SCMConfig, sampled_support_facts

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
    values.update(gate.collected_atlas_inputs(paths, values['data_manifest'].original_ids))
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
    atlas_coverage = gate.atlas_coverage(values)
    require(atlas_coverage['complete'], 'incomplete atlas coverage')
    accounted_missing = set(atlas_coverage['accounted_missing_dem_tract_ids'])
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
        if row.tract_id in accounted_missing:
            reasons[row.original_id] = 'atlas_missing_dem_coverage'
        elif not usable(row, a):
            reasons[row.original_id] = 'flag_label_or_allocation_unusable'
        elif screens[row.county]['reasons']:
            reasons[row.original_id] = 'initial_county_screen'
        elif row.original_id not in supported:
            reasons[row.original_id] = 'outside_frozen_conditional_support'
    excluded = set(reasons) - design_ids
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
    # Excluded/reserved rows remain in the lineage frame, but have no invented A.
    # The SCM supplies labels for target and labelled training rows only.
    schema = tuple(replace(col, nullable=True)
                   if accounted_missing and col.name == manifest.exposure_field else col
                   for col in scenario.data_manifest.schema)
    template_manifest = replace(scenario.data_manifest, spec=spec, registry=registry, schema=schema)
    outer = replace(scenario.outer, spec=spec, lineage=replace(scenario.outer.lineage,
                    parent_hashes=(template_manifest.content_hash,)))
    inners = []
    for inner in scenario.inner:
        local = replace(inner.data_manifest, spec=spec, registry=registry, schema=schema,
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
            manifest.exposure_field: (None if by_id[oid].tract_id in accounted_missing
                                      else a[by_id[oid].tract_id].exposure_mmhg),
            geography.county_field: by_id[oid].county})
        records.append(record)
    data = load_records(records, template_manifest, spec, template_manifest.schema_hash)
    labelled_ids = tuple(i for i, role in zip(outer.frame_original_ids, outer.roles) if role != 'inert')
    targets = set(outer.original_ids)
    selected = tuple(by_id[oid] for oid in labelled_ids)
    # Dependence components retain their identities across membership roles.
    clusters = {oid: digest(group) for group in reservation.groups for oid in group}
    frame = CovariateFrame(original_ids=labelled_ids,
        geography_ids=tuple(r.assignment_geography for r in selected),
        region_ids=tuple(r.county for r in selected),
        cluster_ids=tuple(clusters[r.original_id] for r in selected),
        coordinates=tuple((r.latitude, r.longitude) for r in selected),
        columns=covariates.columns, x=tuple(x[r.original_id] for r in selected),
        support_keys=tuple(r.assignment_geography for r in selected),
        weights=tuple(1. if r.original_id in targets else 0. for r in selected),
        outcome_available=tuple(r.label_available for r in selected),
        biomarker_available=(True,) * len(selected))
    components = dict(frozen.policy.components_by_key)
    fixed_exposures = {r.original_id: a[r.tract_id].exposure_mmhg for r in selected
                       if not components[r.assignment_geography]}
    knots = frozen.spline_knots
    sealed_a = [a[r.tract_id].exposure_mmhg for r in rows
                if r.original_id in design_ids and r.tract_id not in accounted_missing]
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
    for scenario_config in campaign.registered_scenarios(endpoint, frame):
        SCMConfig(**scenario_config).validate_policy(frozen.policy, frame, fixed_exposures=fixed_exposures)
    facts = sampled_support_facts(frozen.policy, frame, fixed_exposures=fixed_exposures)
    def membership_counts(ids):
        return {'total': len(ids), 'target': len(targets.intersection(ids)),
                'training_only': len(set(ids) - targets)}
    audit = {'input_rows': len(rows), 'evaluation_rows': len(targets),
        'simulation_rows': len(selected), 'training_only_rows': len(selected) - len(targets),
        'sampled_exposure_rows': len(selected) - len(fixed_exposures),
        'fixed_exposure_rows': len(fixed_exposures),
        'support_declarations': {name: facts[name] for name in ('support_gaps', 'heterogeneous_eligibility')},
        'support_declarations_source': 'from_frame',
        'sampled_rows_with_gaps': membership_counts(facts['gap_ids']),
        'sampled_rows_without_positive_width_shift': membership_counts(facts['no_positive_width_shift_ids']),
        'coverage': atlas_coverage,
        'allocation_exclusions': gate.allocation_exclusions(rows, atlas_coverage, design_ids),
        'design_ids': list(reservation.design_ids), 'exclusions': list(scenario.exclusions),
        'cluster_sizes': dict(Counter(frame.cluster_ids)),
        'missing_covariates': sum(v is None for row in frame.x for v in row),
        'target_id': spec.target_id, 'source_manifest_hash': manifest.content_hash,
        'source_covariates_hash': covariates.content_hash, 'source_geography_hash': geography.content_hash,
        'source_entity_graph_hash': graph.content_hash, 'atlas_hash': atlas.content_hash,
        'labels': 'No real outcomes read; Y generated for target/training rows; empty-support training A stays at atlas value',
        'certifies_production_coverage': False, 'effect_release_authorized': False}
    return endpoint, frame, frozen, audit


def profile_tasks(endpoint, frame):
    """Concrete hash-bound tasks, generated once after real inputs exist.

    ``resources`` is the scheduler request consumed by coordinator plan wiring;
    the merged stage runner executes inside that allocation and submits no job.
    Profile and projection resources share one declaration.
    """
    locations = {}
    targets = set(endpoint.outer.original_ids)
    target_locations = [(c, p) for i, c, p in zip(frame.original_ids, frame.region_ids, frame.coordinates)
                        if i in targets]
    for county in sorted({c for c, _ in target_locations}):
        points = [p for c, p in target_locations if c == county]
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
    for scenario in campaign.registered_scenarios(endpoint, frame):
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
            **resources, 'profile_safety_factor': campaign.screening_safety_factor()['profile_safety_factor']}})
    # A paired hardware probe, deliberately outside the CPU estimate fan-in.
    # Copy the already-declared draw, rather than assigning a new seed namespace.
    probe = deepcopy(next(task for task in tasks if task['id'] == 'profile-null-effect'))
    probe['id'] = 'profile-null-effect-gpu'
    probe['resources'] = {'cpus_per_task': 8, 'gpus': 1, 'wall_seconds': 13500}
    probe['parameters'].update(device='cuda', wall_seconds=13500)
    tasks.append(probe)
    return {'schema_version': 1, 'tasks': tasks}


def sizing_task_inputs(task, *, setup_calibration=None, calibration_probes=None, admission_receipt=None):
    """Bind sizing-only inputs without changing any bootstrap profile task."""
    task = deepcopy(task)
    require(task['stage'] in ('campaign-estimate', 'campaign-lock'), 'sizing inputs require estimate or lock')
    require(setup_calibration is None or calibration_probes is None, 'choose calibration records or probes')
    p = task['parameters']
    if setup_calibration is not None:
        p['setup_calibration'] = deepcopy(setup_calibration)
        p.pop('calibration_probes', None)
        refs = setup_calibration
    else:
        refs = calibration_probes or [{'dependency': 'campaign-calibration', 'path': 'setup_probe.json'}]
        p['calibration_probes'] = deepcopy(refs)
    for ref in refs:
        names = task.setdefault('needs', {}).setdefault(ref['dependency'], [])
        if ref['path'] not in names:
            names.append(ref['path'])
    if task['stage'] == 'campaign-lock':
        p['admission_receipt'] = deepcopy(admission_receipt or
            {'dependency': 'campaign-admit', 'path': 'admission.json'})
    else:
        require(admission_receipt is None, 'estimate cannot depend on post-lock admission')
    return task


def screening_profile_tasks(endpoint, frame, approvals, *, setup_calibration=None, calibration_probes=None):
    """Bounded bootstrap chains; the final receipts must contain complete draws.

    Each scenario has 40 prospectively declared slots. A completed repetition
    is carried unchanged through remaining slots, never fitted again. This
    supplies a concrete terminal dependency without predicting its cost.
    """
    from oxyformer.validation.coverage import BUDGET_RECORD, leaf_allocation, maximum_leaf_stage_seconds
    decision = campaign.screening_decision(approvals)
    require(decision == campaign.screening_decision(), 'screening approval drift')
    legacy = profile_tasks(endpoint, frame)['tasks']
    recipe = deepcopy(legacy[0]['parameters']['recipe'])
    recipe.update(campaign_mode='screening', study=deepcopy(decision['recipe']))
    campaign.validate_recipe(recipe)
    metadata = campaign.screening_metadata(recipe)
    stage_seconds = maximum_leaf_stage_seconds()
    allocation = leaf_allocation(stage_seconds)
    resources = {'cpus_per_task': 8, 'gpus': 1, 'wall_seconds': allocation['allocation_seconds'], 'partition': 'standard'}
    outputs = ['result.json', 'timing.json', 'runtime.json', 'artifact_manifest.json', BUDGET_RECORD, 'continuation.tar']
    receipt_inputs = ['_execution/task.json', '_execution/request.json', '_execution/result.json', *outputs]
    tasks, profiles, probe_profiles = [], {}, {}
    for original in legacy[:4]:
        parameters = {**deepcopy(original['parameters']), 'recipe': recipe, **metadata,
                      'study': decision['recipe'], 'wall_seconds': stage_seconds,
                      'partition': 'standard', 'device': 'cuda'}
        name = parameters['scenario']['name']
        parameters['draws'] = repetition_plan(digest(['month1-screening-profile', recipe]), name, 1)
        owner = digest(['month1-profile-chain', parameters])
        predecessor = None
        probe_profiles[name] = {}
        for step in range(decision['month1_sizing']['leaves_per_scenario_max']):
            task_id = original['id'] + '-' + str(step)
            needs = deepcopy(original['needs'])
            if predecessor:
                needs[predecessor] = list(receipt_inputs)
            tasks.append({'id': task_id, 'stage': 'simulation-smoke', **metadata,
                'resources': dict(resources), 'needs': needs, 'outputs': list(outputs),
                'parameters': deepcopy(parameters),
                'continuation': {'owner': owner, 'step': step, 'predecessor': predecessor}})
            predecessor = task_id
            if step < 2:
                probe_profiles[name][str(step + 1)] = {'dependency': task_id, 'path': BUDGET_RECORD}
        profiles[name] = {'dependency': predecessor, 'path': 'result.json'}
    tasks.append({'id': 'campaign-estimate', 'stage': 'campaign-estimate', **metadata,
        'resources': dict(resources),
        'needs': {**deepcopy(legacy[0]['needs']),
                  **{ref['dependency']: receipt_inputs for ref in profiles.values()}},
        'outputs': ['budget_estimate.json', 'artifact_manifest.json'],
        'parameters': {**{k: deepcopy(legacy[4]['parameters'][k]) for k in ('endpoint_input', 'frame_input')},
            'recipe': recipe, 'profiles': profiles, 'probe_profiles': probe_profiles, **resources, 'wall_seconds': stage_seconds,
            'profile_safety_factor': campaign.screening_safety_factor(approvals)['profile_safety_factor']}})
    from oxyformer.validation.campaign_probe import PROFILE_SETUP_INPUTS
    for refs in probe_profiles.values():
        for ref in refs.values():
            tasks[-1]['needs'][ref['dependency']] = list(PROFILE_SETUP_INPUTS)
    tasks[-1] = sizing_task_inputs(tasks[-1], setup_calibration=setup_calibration, calibration_probes=calibration_probes)
    return {'schema_version': 1, **metadata, 'gpu_concurrency_max': 4, 'tasks': tasks}


def diagnostic_profile_tasks(endpoint, frame, approvals, *, campaign_id, evaluation_namespace,
                             prior_locks, cpu_wall_hours_budget, setup_calibration=None,
                             calibration_probes=None):
    """Emit the approved profile subset and sizing/lock tasks, without submission."""
    d = campaign.month2_decision(approvals)
    require(d == campaign.month2_decision(), 'month2_followon drift')
    c = d['diagnostic_campaign']
    campaign.concrete_id(campaign_id)
    require(len(campaign_id) <= 23, 'campaign ID must leave room for chain suffix')
    original = screening_profile_tasks(endpoint, frame, approvals)
    tasks = []
    for source in original['tasks']:
        if source['stage'] != 'simulation-smoke' or source['parameters']['scenario']['name'] not in c['scenarios']:
            continue
        task = deepcopy(source)
        recipe = task['parameters']['recipe']
        recipe['campaign_mode'] = 'diagnostic'
        metadata = campaign.screening_metadata(recipe)
        task.update(metadata)
        task['parameters'].update(metadata, partition=c['partition'])
        task['resources']['partition'] = c['partition']
        task['parameters']['draws'] = repetition_plan(digest(['diagnostic-profile', recipe, campaign_id]),
            task['parameters']['scenario']['name'], 1)
        task['continuation']['owner'] = digest(['diagnostic-profile-chain', task['parameters']])
        tasks.append(task)
    recipe = deepcopy(tasks[0]['parameters']['recipe'])
    metadata = campaign.screening_metadata(recipe)
    estimate = deepcopy(original['tasks'][-1])
    estimate['id'] = campaign_id + '-estimate'
    campaign.concrete_id(estimate['id'])
    estimate.update(metadata)
    p = estimate['parameters']
    p.update(metadata, recipe=recipe, partition=c['partition'], profile_safety_factor=2.0)
    p['profiles'] = {n: p['profiles'][n] for n in c['scenarios']}
    p['probe_profiles'] = {n: p['probe_profiles'][n] for n in c['scenarios']}
    estimate['resources']['partition'] = c['partition']
    # Rebuild dependency declarations from the artifacts each consumer reads.
    estimate['needs'] = deepcopy(tasks[0]['needs'])
    from oxyformer.validation.campaign_probe import PROFILE_SETUP_INPUTS
    for ref in p['profiles'].values():
        estimate['needs'][ref['dependency']] = list(PROFILE_SETUP_INPUTS)
    for refs in p['probe_profiles'].values():
        for ref in refs.values():
            estimate['needs'][ref['dependency']] = list(PROFILE_SETUP_INPUTS)
    estimate = sizing_task_inputs(estimate, setup_calibration=setup_calibration, calibration_probes=calibration_probes)
    lock = deepcopy(estimate)
    lock.update(id=campaign_id + '-lock', stage='campaign-lock')
    campaign.concrete_id(lock['id'])
    from oxyformer.validation.campaign_probe import LOCK_OUTPUTS
    lock['outputs'] = list(LOCK_OUTPUTS)
    lock['parameters'].update(campaign_id=campaign_id, evaluation_namespace=evaluation_namespace,
        prior_locks=deepcopy(prior_locks), cpu_wall_hours_budget=cpu_wall_hours_budget,
        retry_rules=deepcopy(campaign.RETRY_RULES),
        estimate_input={'dependency': estimate['id'], 'path': 'budget_estimate.json'},
        admission_receipt={'dependency': campaign_id + '-admit', 'path': 'admission.json'})
    for unit, names in read_mapping(ROOT / 'configs/execution/stages.yaml')['stages']['campaign-lock'].get('needs', {}).items():
        lock['needs'][unit] = list(dict.fromkeys([*lock['needs'].get(unit, []), *names]))
    lock['needs'][estimate['id']] = ['budget_estimate.json']
    for ref in prior_locks:
        lock['needs'].setdefault(ref['dependency'], []).append(ref['path'])
    tasks.extend([estimate, lock])
    return {'schema_version': 1, **metadata,
            'gpu_concurrency_max': d['gpu_concurrency_max'][c['partition']], 'tasks': tasks}


def run_stage(request: StageRequest) -> StageResult:
    try:
        values, design = prepare_inputs(request)
        endpoint, frame, support, audit = build_inputs(values, design)
        return publish(request, {'endpoint.json': endpoint.to_dict(), 'frame.json': frame.to_dict(),
            'support.json': support.to_dict(), 'input_audit.json': audit,
            'profile_tasks.json': screening_profile_tasks(endpoint, frame, values['approvals'])}, status='pass',
            message='Real covariates and frozen primary design prepared for simulated outcomes; no campaign admitted')
    except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
        return StageResult(request_hash=request.content_hash, status='blocked', artifacts=(),
                           message=str(exc) or type(exc).__name__)
