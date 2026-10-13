"""Measured planning estimates only: no allocation, expansion, lock or admission."""
import json
import math
from pathlib import Path

from oxyformer.contracts import StageRequest, StageResult
from oxyformer.execution.runner import read_mapping
from oxyformer.provenance import ContractError, require
from oxyformer.training.nested_cv import PreparedEndpoint
from oxyformer.validation import campaign
from oxyformer.validation.coverage import dependency, digest, finite, integer, publish
from oxyformer.validation.scm import CovariateFrame


def screening_plan(profiles, decision, wall, factor):
    """Uniform breadth reduction with separate leaf feasibility reporting.

    Round projected leaves up to scheduler minutes. An overlong repetition
    still gets an honest cost estimate; it never becomes an admitted leaf.
    """
    require(set(profiles) == set(decision["scenarios"]), "all four screening profiles required")
    count = decision["repetitions_per_scenario"]
    integer(wall, "screening leaf wall", 1)
    require(math.ceil(wall / 60) * 60 <= 14400, "leaf exceeds four GPU-hours")
    require(finite(factor, "profile safety factor") >= 1, "profile safety factor must be at least one")
    def seconds(profile, n):
        return factor * (profile["setup_seconds"] + n * profile["seconds_per_repetition"]
            + profile["publication_verification_seconds"] * max(1., n / profile["profile_repetitions"]))
    capacities = {name: max((n for n in range(1, count + 1) if seconds(p, n) <= wall), default=0)
                  for name, p in profiles.items()}
    common = min(capacities.values())
    batch = max(1, common)
    def costs(n):
        full, remainder = divmod(n, batch)
        return {name: (full * math.ceil(seconds(p, batch) / 60)
                      + (math.ceil(seconds(p, remainder) / 60) if remainder else 0)) / 60
                for name, p in profiles.items()}
    requested_hours = math.fsum(costs(count).values())
    while count and math.fsum(costs(count).values()) > decision["budget_h100_hours"]:
        count -= 1
    rows = costs(count)
    leaves = len(profiles) * math.ceil(count / batch)
    feasible = count > 0 and common > 0 and leaves <= 40
    return {"repetitions_requested_per_scenario": decision["repetitions_per_scenario"],
        "repetitions_per_scenario": count, "total_repetitions": len(profiles) * count,
        "repetitions_reduced": count < decision["repetitions_per_scenario"],
        "reduction_reason": "uniform reduction to 3,600 H100-hour ceiling" if count < 25 else None,
        "requested_screening_gpu_hours": requested_hours,
        "screening_gpu_hours": math.fsum(rows.values()), "scenario_gpu_hours": rows,
        "repetitions_per_leaf": batch, "common_feasible_repetitions_per_leaf": common,
        "feasible_repetitions_per_leaf": capacities, "projected_leaf_count": leaves,
        "feasible_under_requested_wall_and_leaf_cap": feasible,
        "projected_required_leaf_wall_seconds": max(seconds(p, min(count, batch)) for p in profiles.values()) if count else 0.,
        "limits": {"gpu_hours_per_leaf": 4, "leaves_per_instance": 40, "run_gpu_hours": 3600,
                   "concurrent_gpu_units": 4, "concurrent_cpu_units": 6},
        "concurrency_plan": {"gpu_pool_required": 4, "gpus_per_leaf": 1, "max_gpu_leaves_running": 4,
                             "schedule": "four dependency chains; coordinator gpu pool capped at four",
                             "exempt_partitions": decision["gpu_concurrency_exempt_partitions"]},
        "infeasibility": None if feasible else "No feasible complete-repetition plan within four-hour leaves and forty leaves; continuation or new measurements required."}


def estimate_budget(request):
    request.verify_inputs()
    # Screening requires its owner decision; registered estimates need no allocation.
    config = read_mapping(request.config_path)
    task = json.loads(Path(request.task_path).read_text())
    require(request.stage == task['stage'] == 'campaign-estimate', 'stage mismatch')
    p = task['parameters']
    recipe = p['recipe']
    campaign.validate_recipe(recipe)
    inputs = {}
    for key, cls, expected in [('endpoint_input', PreparedEndpoint, recipe['endpoint_hash']),
                                ('frame_input', CovariateFrame, recipe['frame_hash'])]:
        record = cls.from_json(dependency(request, config, p[key]).read_text())
        require(record.content_hash == expected, 'estimate endpoint/frame hash mismatch')
        inputs[key] = record
    scenarios = campaign.registered_scenarios(inputs['endpoint_input'], inputs['frame_input'])
    require(set(p['profiles']) == {s['name'] for s in scenarios}, 'all registered final profiles required')
    screening = bool(campaign.screening_metadata(recipe))
    decision = campaign.validate_screening(recipe, config.get('approvals', {})) if screening else None
    if screening:
        require('final_repetitions' not in p, 'screening cannot request final cells')
        count = decision['repetitions_per_scenario']
    else:
        count = integer(p['final_repetitions'], 'final repetitions', 1000)
    wall = integer(p['wall_seconds'], 'leaf wall seconds', 1)
    cpus = integer(p['cpus_per_task'], 'CPUs per task', 1)
    gpus = p['gpus']
    require(type(gpus) is int and gpus in (0, 1), 'nested estimator supports zero or one GPU')
    require(not screening or gpus == 1, 'screening requires single-GPU leaves')
    require(gpus == int(campaign.resolve_device().type == 'cuda'), 'estimate device/resource mismatch')
    require(gpus * math.ceil(wall / 60) * 60 <= 14400, 'leaf exceeds four GPU-hours')
    factor = finite(p['profile_safety_factor'], 'profile safety factor')
    require(factor >= 1, 'profile safety factor must be at least one')
    stamps = campaign.fingerprint()
    profiles = {s['name']: campaign._profile(request, config, p['profiles'][s['name']], s, recipe, stamps)
                for s in scenarios}

    if screening:
        return {'schema_version': 1, 'estimate_only': True, 'admitted': False,
            'submission_performed': False, 'certifies_production_coverage': False,
            'recipe_hash': digest(recipe), 'recipe': recipe, **stamps,
            **campaign.screening_metadata(recipe), 'profiles': profiles,
            'requested_leaf_wall_seconds': wall, 'profile_safety_factor': factor,
            'cpus_per_task': cpus, 'gpus': gpus,
            **screening_plan(profiles, decision, wall, factor),
            'limitations': ['Planning estimate; campaign-lock measures locked setup before admission.',
                           'Complete repetitions only; overlong repetitions cannot be split by this adapter.',
                           'No release criterion is certified; final cells require a separate month-2 allocation.']}

    def seconds(profile, n):
        return factor * (profile['setup_seconds'] + n * profile['seconds_per_repetition']
            + profile['publication_verification_seconds'] * max(1., n / profile['profile_repetitions']))

    capacities = {}
    for name, profile in profiles.items():
        # Monotone cost: binary search avoids allocating any repetition list.
        lo, hi = 0, count
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if seconds(profile, mid) <= wall:
                lo = mid
            else:
                hi = mid - 1
        capacities[name] = lo
    common = min(capacities.values())
    rows = {}
    # Even an infeasible four-hour batching proposal must supply the owner the
    # projected cost of the >=1000/scenario instance. Forty leaves require a
    # common batch size of ceil(count / floor(40/scenario_count)). Report its
    # needed wall time explicitly, without admitting the longer leaves.
    require(len(scenarios) <= 40, 'registered family exceeds forty leaves even with one leaf per scenario')
    minimum_batch = math.ceil(count / (40 // len(scenarios)))
    estimate_batch = common if common >= minimum_batch else minimum_batch
    total_seconds, total_leaves = 0., 0
    for name, profile in profiles.items():
        full, remainder = divmod(count, estimate_batch)
        leaves = full + bool(remainder)
        cost = full * seconds(profile, estimate_batch) + (seconds(profile, remainder) if remainder else 0.)
        total_seconds += cost
        total_leaves += leaves
        rows[name] = {**profile, 'feasible_repetitions_per_leaf': capacities[name],
            'projected_leaf_count': leaves, 'projected_cpu_wall_hours': cost / 3600,
            'projected_cpu_core_hours': cpus * cost / 3600,
            'projected_gpu_hours': gpus * cost / 3600,
            'projected_max_leaf_seconds': seconds(profile, min(count, estimate_batch))}
    feasible = common >= minimum_batch
    return {'schema_version': 1, 'estimate_only': True, 'admitted': False,
        'allocation_required_for_final_coverage': True, 'submission_performed': False,
        'certifies_production_coverage': False, 'recipe_hash': digest(recipe),
        'recipe': recipe, **stamps, 'profiles': rows,
        'final_repetitions_per_scenario': count, 'total_repetitions': count * len(scenarios),
        'requested_leaf_wall_seconds': wall, 'profile_safety_factor': factor,
        'cpus_per_task': cpus, 'gpus': gpus,
        'common_feasible_repetitions_per_leaf': common,
        'leaf_count_at_requested_wall': len(scenarios) * math.ceil(count / common) if common else None,
        'feasible_under_requested_wall_and_leaf_cap': feasible,
        'projected_repetitions_per_leaf': estimate_batch, 'projected_leaf_count': total_leaves,
        'projected_required_leaf_wall_seconds': max(r['projected_max_leaf_seconds'] for r in rows.values()),
        'final_coverage_cpu_wall_hours': total_seconds / 3600,
        'final_coverage_cpu_core_hours': cpus * total_seconds / 3600,
        'final_coverage_gpu_hours': gpus * total_seconds / 3600,
        'limits': {'gpu_hours_per_leaf': 4, 'leaves_per_instance': 40, 'run_gpu_hours': 2500,
                   'concurrent_gpu_units': 8, 'concurrent_cpu_units': 6},
        'limitations': ['Planning estimate, not a campaign allocation or admission.',
            'Uses unlocked profile setup; campaign-lock must measure locked setup before admission.',
            'CPU wall hours sum leaf elapsed time; core hours multiply by requested CPUs.',
            'No screening, collector, scheduler overhead or reruns included; no throughput claim.']}


def run_stage(request: StageRequest) -> StageResult:
    try:
        estimate = estimate_budget(request)
        return publish(request, {'budget_estimate.json': estimate}, status='pass',
                       message='Measured budget estimate only; owner allocation and campaign-lock remain required')
    except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
        return StageResult(request_hash=request.content_hash, status='blocked', artifacts=(),
                           message=str(exc) or type(exc).__name__)
