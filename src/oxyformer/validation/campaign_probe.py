"""CPU setup measurements and separate, immutable screening admission receipts.

Measurements size work; only admission.json authorizes a screening lock.
"""
from copy import deepcopy
import math
import os
from pathlib import Path
import time

import numpy as np

from oxyformer.contracts import StageRequest, StageResult
from oxyformer.provenance import ContractError, canonical_json, file_hash, require
from oxyformer.validation.coverage import (
    BUDGET_RECORD, OVERHEAD_PHASES, RESUME_OVERHEAD_FLOOR_SECONDS, dependency,
    digest, finite, publish, read_json, read_leaf_measurements, usable_leaf_compute, output_metadata)

SMALL_OUTPUT_VERIFICATION_SECONDS = 60.
SMALL_OUTPUT_BYTES = 50_000_000
CALIBRATION_HEADROOM = 1.5
PROFILE_INPUTS = ['_execution/task.json', '_execution/request.json', '_execution/result.json',
                  BUDGET_RECORD, 'result.json']
LOCK_OUTPUTS = ['recipe_lock.json', 'expanded_units.json', 'task_manifest.json', 'budget.json',
                'gate.json', 'artifact_manifest.json']


def calibration_inputs(parameters):
    inputs = {ref['dependency']: list(PROFILE_INPUTS) for ref in parameters['profiles'].values()}
    for key in ('endpoint_input', 'frame_input'):
        ref = parameters[key]
        inputs.setdefault(ref['dependency'], []).append(ref['path'])
    return inputs


def input_digests(request, config, inputs):
    """Record the exact declared files, including the publication fingerprints."""
    hashes = dict(zip(request.dependency_paths, request.dependency_hashes))
    result = {}
    for unit, names in inputs.items():
        result[unit] = {}
        for name in names:
            path = dependency(request, config, {'dependency': unit, 'path': name})
            require(file_hash(path) == hashes[str(path)], 'probe input hash mismatch')
            result[unit][name] = hashes[str(path)]
    return result


def read_receipt(request, config, ref):
    from oxyformer.execution.integrity import verify_publication
    path = dependency(request, config, ref)
    root = Path(config['dependencies'][ref['dependency']]).resolve(strict=True)
    result = StageResult.from_json((root / '_execution/result.json').read_text())
    require(result.status == 'pass', 'receipt stage did not pass')
    verify_publication(root, result)
    require(any(a.path == ref['path'] and a.sha256 == file_hash(path) for a in result.artifacts),
            'receipt hash mismatch')
    return read_json(path)


def calibration_statistics(records, profiles):
    """Each record contributes its own last leaf exactly once."""
    require(bool(records), 'empty setup calibration set')
    own = [r['leaves'][-1] for r in records]
    require(len({r['task_id'] for r in own}) == len(own), 'duplicate calibration leaf')
    setups = [r['setup_seconds'] for r in own]
    p90 = float(np.percentile(setups, 90))
    statistics = {k: {'mean': float(np.mean([r[k] for r in own])),
        'p50': float(np.percentile([r[k] for r in own], 50)),
        'p90': float(np.percentile([r[k] for r in own], 90)),
        'min': min(r[k] for r in own), 'max': max(r[k] for r in own)}
        for k in (*OVERHEAD_PHASES, 'compute_seconds')}
    completed = {}
    completed_chains = set()
    for record in sorted(records, key=lambda r: len(r['leaves'])):
        name = record['binding']['scenario']['name']
        require(name in profiles, 'calibration scenario has no profile')
        # Prefix records share their first leaf identity. Take each chain's
        # first completion, then the largest excess across chains of a scenario.
        chain = (name, record['leaves'][0]['task_id'])
        if record['counts']['success'] >= 1 and chain not in completed_chains:
            completed_chains.add(chain)
            leaves = record['leaves']
            delivered = math.fsum(r['compute_seconds'] for r in leaves)
            candidate = {'leaves': len(leaves), 'delivered_seconds': delivered,
                'first_leaf_task_id': chain[1],
                'excess_per_leaf': (delivered - profiles[name]['seconds_per_repetition']) / len(leaves)}
            if name not in completed or candidate['excess_per_leaf'] > completed[name]['excess_per_leaf']:
                completed[name] = candidate
    resume = max(RESUME_OVERHEAD_FLOOR_SECONDS,
                 max((r['excess_per_leaf'] for r in completed.values()), default=0.))
    return {'source': 'sealed_campaign_leaves', 'leaf_count': len(own), 'statistics': statistics,
        'setup_seconds': max(1.25 * p90, max(setups)),
        'resume_overhead_seconds': resume, 'resume_uncalibrated': not bool(completed),
        'resume_source': {'rule': 'max_completed_chain_excess_with_floor',
                          'floor_seconds': RESUME_OVERHEAD_FLOOR_SECONDS, 'completed': completed},
        'realized_compute_per_leaf': statistics['compute_seconds']['mean'],
        'records': [{'record_hash': r['record_hash'], 'task_id': r['leaves'][-1]['task_id'],
                     'budget_generation': r['budget_generation']} for r in records]}


def calibrate(request, config, parameters, profiles):
    refs = parameters.get('setup_calibration')
    if refs is not None:
        require(isinstance(refs, list) and bool(refs), 'setup_calibration must be a nonempty list')
        records = [read_leaf_measurements(dependency(request, config, ref)) for ref in refs]
        calibration = calibration_statistics(records, profiles)
    else:
        refs = parameters.get('calibration_probes', [])
        require(isinstance(refs, list) and 1 <= len(refs) <= 2,
                'one or two pre-lock calibration probe receipts required')
        require(len({(r['dependency'], r['path']) for r in refs}) == len(refs), 'duplicate calibration draw')
        expected = input_digests(request, config, calibration_inputs(parameters))
        draws = []
        for ref in refs:
            receipt = read_receipt(request, config, ref)
            require(receipt['mode'] == 'calibration' and receipt['input_digests'] == expected,
                    'calibration probe verified different inputs')
            seconds = finite(receipt['setup_seconds'], 'calibration probe setup')
            require(seconds >= 0, 'negative probe setup')
            draws.append({'reference': ref, 'setup_seconds': seconds})
        chosen = min(d['setup_seconds'] for d in draws)
        profile_allowance = max(p['leaf_overheads']['setup_seconds'] for p in profiles.values())
        base = max(chosen + SMALL_OUTPUT_VERIFICATION_SECONDS, profile_allowance)
        calibration = {'source': 'pre_lock_probe', 'draws': draws, 'chosen_seconds': chosen,
            'probe_to_profile_ratio': chosen / profile_allowance if profile_allowance else None,
            'setup_without_headroom': base, 'setup_seconds': base * CALIBRATION_HEADROOM,
            'headroom': CALIBRATION_HEADROOM, 'resume_overhead_seconds': RESUME_OVERHEAD_FLOOR_SECONDS,
            'resume_uncalibrated': True, 'resume_source': {'rule': 'uncalibrated_floor',
                'floor_seconds': RESUME_OVERHEAD_FLOOR_SECONDS}}
    result = deepcopy(profiles)
    for profile in result.values():
        overheads = profile['leaf_overheads']
        overheads['setup_seconds'] = calibration['setup_seconds']
        for phase in OVERHEAD_PHASES[1:]:
            if refs is not None and calibration['source'] == 'sealed_campaign_leaves':
                overheads[phase] = max(overheads[phase], calibration['statistics'][phase]['max'])
        profile['resume_overhead_seconds'] = calibration['resume_overhead_seconds']
        profile['resume_source'] = deepcopy(calibration['resume_source'])
        profile['assumed_usable_per_leaf'] = usable_leaf_compute(parameters['wall_seconds'], overheads,
            resume_overhead_seconds=profile['resume_overhead_seconds']) - profile['resume_overhead_seconds']
        profile['calibration_source'] = calibration['source']
    return result, calibration


def probe_setup_seconds():
    """Launcher stamps the monotonic clock before clone and environment startup."""
    started = finite(float(os.environ['OXYFORMER_SETUP_STARTED']), 'probe start')
    elapsed = time.monotonic() - started
    require(elapsed >= 0, 'invalid probe start')
    return elapsed


def probe(request, config, task, *, measurer=None):
    p = task['parameters']
    mode = p['mode']
    require(mode in ('calibration', 'admission'), 'unknown setup probe mode')
    if mode == 'calibration':
        expected = calibration_inputs(p)
        lock_hash = None
    else:
        lock_path = dependency(request, config, p['lock_input'])
        lock_hash = file_hash(lock_path)
        expansion = read_json(dependency(request, config, {
            'dependency': p['lock_input']['dependency'], 'path': 'expanded_units.json'}))
        require(expansion['lock_hash'] == lock_hash, 'probe expansion lock mismatch')
        first = expansion['plans'][0]['tasks'][0]
        expected = deepcopy(first['needs'])
        admission = first.get('admission_receipt')
        if admission:
            names = expected[admission['dependency']]
            names.remove(admission['path'])
            if not names:
                del expected[admission['dependency']]
    require(task['needs'] == expected, 'probe must verify the declared input shape exactly')
    digests = input_digests(request, config, expected)
    seconds = finite((measurer or probe_setup_seconds)(), 'probe setup')
    require(seconds >= 0, 'negative probe setup')
    return {'schema_version': 1, 'mode': mode, 'setup_seconds': seconds,
            'input_digests': digests, 'lock_hash': lock_hash, 'request_hash': request.content_hash}


def admit(request, config, task):
    from oxyformer.execution.campaign import validate_plan
    from oxyformer.validation.campaign import slice_count
    p = task['parameters']
    path = dependency(request, config, p['lock_input'])
    lock = read_json(path)
    lock_hash = file_hash(path)
    require(file_hash(dependency(request, config, lock['estimate_input'])) == lock['estimate_hash'],
            'admission estimate_hash mismatch')
    expansion = read_json(dependency(request, config, {
        'dependency': p['lock_input']['dependency'], 'path': 'expanded_units.json'}))
    require(expansion['lock_hash'] == lock_hash, 'admission expansion lock mismatch')
    plans = expansion['plans']
    require(bool(plans) and all(plan['spec']['inputs'] == plans[0]['spec']['inputs'] for plan in plans),
            'admission requires equal batch-level spec inputs')
    require({plan['spec']['id'] for plan in plans} == set(lock['instances'].values()),
            'admission missing locked instance')
    for plan in plans:
        validate_plan(plan, config['approvals'])
        require(plan['spec']['recipe_lock']['sha256'] == lock_hash, 'admission spec lock mismatch')
    receipt = read_receipt(request, config, p['probe_input'])
    require(receipt['mode'] == 'admission' and receipt['lock_hash'] == lock_hash,
            'admission probe lock_hash mismatch')
    require(receipt['input_digests'] == input_digests(request, config, plans[0]['spec']['inputs']),
            'admission probe verified different inputs')
    setup = finite(receipt['setup_seconds'], 'admission probe setup')
    require(setup >= 0, 'negative probe setup')
    counts = {}
    for batch in lock['batches']:
        profile = deepcopy(lock['budget']['profiles'][batch['scenario']['name']])
        profile['leaf_overheads']['setup_seconds'] = max(profile['leaf_overheads']['setup_seconds'],
            setup + SMALL_OUTPUT_VERIFICATION_SECONDS)
        try:
            counts[batch['batch_id']] = slice_count(profile, len(batch['draws']), batch['wall_seconds'],
                lock['budget']['profile_safety_factor'], enforce_limit=False)
        except ContractError as exc:
            raise ContractError(f'admission not established: measured setup {setup}; counts {counts}; {exc}') from exc
    require(all(counts[b['batch_id']] <= min(40, b['slices']) for b in lock['batches']),
            f'admission not established: measured setup {setup}; recomputed counts {counts}')
    candidate = {**output_metadata(task), 'schema_version': 1, 'admitted': True, 'lock_hash': lock_hash,
        'estimate_hash': lock['estimate_hash'], 'admission_measurements': {
            'probe': p['probe_input'], 'probe_setup_seconds': setup, 'recomputed_counts': counts}}
    # All outputs from both small producers, not merely the two headline files.
    total = len(canonical_json(candidate).encode())
    for unit in (p['lock_input']['dependency'], lock['estimate_input']['dependency']):
        root = Path(config['dependencies'][unit])
        terminal = StageResult.from_json((root / '_execution/result.json').read_text())
        for artifact in terminal.artifacts:
            if not artifact.path.startswith('_execution/'):
                from oxyformer.execution.runner import dependency_file
                total += dependency_file(root, artifact.path).stat().st_size
    require(total <= SMALL_OUTPUT_BYTES,
            f'admission not established: measured setup {setup}; recomputed counts {counts}; small outputs exceed {SMALL_OUTPUT_BYTES} bytes')
    return candidate


def require_admission(request, config, task, lock):
    ref = task.get('admission_receipt')
    require(ref is not None and ref == lock.get('admission_receipt'), 'screening admission receipt required')
    receipt = read_receipt(request, config, ref)
    require(receipt.get('admitted') is True and receipt.get('lock_hash') == digest(lock)
            and receipt.get('estimate_hash') == lock['estimate_hash'], 'screening admission receipt lock mismatch or not admitted')


def run_stage(request: StageRequest) -> StageResult:
    try:
        request.verify_inputs()
        config, task = read_json(request.config_path), read_json(request.task_path)
        require(task['stage'] == request.stage, 'stage mismatch')
        if request.stage == 'campaign-setup-probe':
            value, name = probe(request, config, task), 'setup_probe.json'
        else:
            require(request.stage == 'campaign-admit', 'unknown campaign probe stage')
            value, name = admit(request, config, task), 'admission.json'
        return publish(request, {name: value}, status='pass', message='Setup measured' if name == 'setup_probe.json' else 'Screening lock admitted')
    except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
        return StageResult(request_hash=request.content_hash, status='blocked', artifacts=(), message=str(exc) or type(exc).__name__)
