"""CPU setup measurements and separate, immutable screening admission receipts.

Measurements size work; only admission.json authorizes a screening lock.
"""
from copy import deepcopy
from hashlib import sha256
import math
import os
from pathlib import Path
import shutil
import stat
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
PROFILE_PAYLOADS = [BUDGET_RECORD, 'result.json', 'timing.json', 'runtime.json',
                    'coverage_measurements.json', 'continuation.tar']
PROFILE_SETUP_INPUTS = list(dict.fromkeys([*PROFILE_INPUTS, *PROFILE_PAYLOADS]))
PROBE_RECEIPT_INPUTS = ['setup_probe.json', '_execution/request.json',
                       '_execution/dependencies.json', '_execution/dependency_check.json']


def publication_shape(tree):
    shape = {'archive_bytes': tree['continuation.tar']['size'],
        'record_count': sum(entry.get('type') == stat.S_IFREG for entry in tree.values())}
    require(all(type(value) is int and value > 0 for value in shape.values()), 'invalid predecessor publication shape')
    return shape


def synthetic_publication_shape(tree, producer):
    """Count the locked payload shape without the builder's own receipts."""
    receipts = set(producer['outputs']) - set(PROFILE_PAYLOADS)
    return publication_shape({name: entry for name, entry in tree.items() if name not in receipts})


def compare_publication_shape(expected, observed, *, synthetic=False):
    require(set(expected) == set(observed) == {'archive_bytes', 'record_count'}
        and all(type(value) is int and value > 0 for value in (*expected.values(), *observed.values())),
        'invalid predecessor publication shape')
    deltas = {key: abs(observed[key] - expected[key]) if synthetic else max(0, observed[key] - expected[key])
              for key in expected}
    # Integer comparisons keep the exact 10% and 50% boundaries inclusive.
    exceeds_ten = any(10 * deltas[key] > expected[key] for key in expected)
    exceeds_fifty = any(2 * deltas[key] > expected[key] for key in expected)
    status = ('admission-shape-mismatch' if (exceeds_ten if synthetic else exceeds_fifty)
        else 'admission-shape-exceeded' if exceeds_ten else 'matched')
    return {'expected': expected, 'observed': observed, 'status': status,
        'relative_difference': {key: deltas[key] / expected[key] for key in expected}}


def profile_setup_sources(request, config, refs, scenario, profile, stamps):
    """Authenticate the two early phase-4 profiles before locking their shape."""
    from oxyformer.execution.integrity import FINGERPRINT, verify_published_tree
    from oxyformer.validation.coverage import load_budget
    require(set(refs) == {'1', '2'}, 'two early profile leaves required for resumed admission')
    terminal_unit = profile['source_unit']
    terminal_record = load_budget(request, config, terminal_unit)
    require(terminal_record['record_hash'] == profile['budget_record_hash'], 'terminal profile record mismatch')
    terminal_root = Path(config['dependencies'][terminal_unit]).resolve()
    terminal = StageResult.from_json((terminal_root / '_execution/result.json').read_text())
    terminal_tree = verify_published_tree(terminal_root, terminal)
    producer = StageRequest.from_json((terminal_root / '_execution/request.json').read_text())
    hashes = dict(zip(producer.dependency_paths, producer.dependency_hashes))
    snapshot_name = '_execution/dependencies.json'
    snapshot = {}
    if (snapshot_name in terminal_tree and hashes.get(str(terminal_root / snapshot_name))
            == terminal_tree[snapshot_name]['sha256']):
        snapshot = read_json(terminal_root / snapshot_name)
    require(isinstance(snapshot, dict), 'terminal profile chain snapshot is invalid')
    sources = {}
    for depth, ref in refs.items():
        unit = ref['dependency']
        require(ref['path'] == BUDGET_RECORD, 'profile setup source must be a budget record')
        record = load_budget(request, config, unit)
        require(record['binding']['mode'] == 'profile'
            and record['binding']['scenario'] == scenario
            and record['binding']['draws'] == profile['draws']
            and record['chained'] and terminal_record['chained']
            and record['binding']['chain']['owner'] == terminal_record['binding']['chain']['owner']
            and record['binding']['recipe_hash'] == terminal_record['binding']['recipe_hash']
            and record['leaves'][-1] == profile['leaf_measurements'][int(depth) - 1],
            'resumed admission requires the same scenario profile chain')
        require(all(record['profile'][key] == stamps[key] for key in ('scientific_fingerprint', 'environment_hash')),
            'resumed admission requires same-tree profile leaves')
        root = Path(config['dependencies'][unit]).resolve()
        terminal = StageResult.from_json((root / '_execution/result.json').read_text())
        tree = verify_published_tree(root, terminal)
        # Match an actual publication consumed by the terminal profile, not
        # merely equal leaf measurements or a repeat of the same owner/task ID.
        ancestor = snapshot.get(str(root), {})
        sealed = ancestor.get(FINGERPRINT, {}) if isinstance(ancestor, dict) else {}
        attested_hash = sealed.get('sha256') if isinstance(sealed, dict) else None
        require(tree[FINGERPRINT]['sha256'] in (hashes.get(str(root / FINGERPRINT)), attested_hash),
            'resumed admission requires the same profile chain publication')
        sources[depth] = {'reference': ref, 'publication_hash': tree[FINGERPRINT]['sha256'],
            'archive_hash': tree['continuation.tar']['sha256'], 'shape': publication_shape(tree),
            'repetition_records': sum(name.startswith('repetitions/') and name.endswith('/result.json')
                and entry.get('type') == stat.S_IFREG for name, entry in tree.items()),
            'outputs': read_json(root / '_execution/task.json')['outputs']}
    return sources


def expected_predecessor_shape(profile, outputs, *, draw_count=1):
    sources = profile.get('setup_profile_sources', {})
    if not sources:
        return None  # Locking is possible; admission must refuse missing profiles.
    source = sources['1']
    shape = dict(source['shape'])
    # A fast scenario can represent several repetitions in its first leaf.
    # Include one profile archive per potentially started repetition, bounded
    # by the expansion. This is an admission shape, never a scientific archive.
    copies = min(draw_count, max(1, math.ceil(profile['assumed_usable_per_leaf'] / profile['seconds_per_repetition'])))
    shape['archive_bytes'] *= copies
    shape['record_count'] += len(set(outputs) - set(source['outputs'])) - len(set(source['outputs']) - set(outputs))
    # A completed repetition publishes one individual result. A partial profile
    # can have none yet; the expanded first leaf can finish several.
    shape['record_count'] += copies - source.get('repetition_records', 0)
    require(shape['record_count'] > 0, 'invalid expanded predecessor shape')
    return shape


def first_leaf_inputs(expansion):
    first = expansion['plans'][0]['tasks'][0]
    expected = deepcopy(first['needs'])
    admission = first.get('admission_receipt')
    if admission:
        names = expected[admission['dependency']]
        names.remove(admission['path'])
        if not names:
            del expected[admission['dependency']]
    return first, expected


def locked_probe_inputs(request, config, task):
    p = task['parameters']
    lock_path = dependency(request, config, p['lock_input'])
    lock_hash = file_hash(lock_path)
    lock = read_json(lock_path)
    if lock.get('campaign_mode') == 'diagnostic':
        from oxyformer.validation.campaign import validate_diagnostic
        decision = validate_diagnostic(lock['recipe'], config.get('approvals'))
        require(lock['owner_decision_hash'] == digest(decision), 'month2_followon decision hash drift')
    expansion = read_json(dependency(request, config, {
        'dependency': p['lock_input']['dependency'], 'path': 'expanded_units.json'}))
    require(expansion['lock_hash'] == lock_hash, 'probe expansion lock mismatch')
    first, inputs = first_leaf_inputs(expansion)
    require(file_hash(dependency(request, config, lock['estimate_input'])) == lock['estimate_hash'],
        'probe estimate_hash mismatch')
    return lock, lock_hash, expansion, first, inputs


def two_draws(receipt):
    values = receipt.get('setup_draws')
    require(isinstance(values, list) and len(values) == 2
        and all(finite(value, 'probe draw') >= 0 for value in values), 'two setup measurements required')
    require(receipt['setup_seconds'] == min(values), 'probe minimum mismatch')
    return min(values)


def declared_publication_shape(request, config, unit, *, synthetic=False):
    from oxyformer.execution.integrity import FINGERPRINT, RESULT
    path = dependency(request, config, {'dependency': unit, 'path': FINGERPRINT})
    hashes = dict(zip(request.dependency_paths, request.dependency_hashes))
    require(file_hash(path) == hashes[str(path)], 'predecessor manifest hash mismatch')
    value = read_json(path)
    tree = {**value['entries'], FINGERPRINT: {'type': stat.S_IFREG},
        RESULT: {'type': stat.S_IFREG}}
    if synthetic:
        producer = read_json(dependency(request, config, {'dependency': unit, 'path': '_execution/task.json'}))
        return synthetic_publication_shape(tree, producer)
    return publication_shape(tree)


def resumed_probe_receipt(task, deps, publication):
    """The runner's only synthetic receipt adapter; ordinary leaves cannot use it."""
    p = task['parameters']
    require(type(p['depth']) is int and p['depth'] in (1, 2), 'resumed probe depth must be 1 or 2')
    ref = p['predecessor_input']
    require(ref['path'] == 'setup_probe.json' and ref['dependency'] in deps, 'synthetic predecessor must be declared')
    root = deps[ref['dependency']]
    terminal, tree = publication(root)
    value = read_json(root / ref['path'])
    producer = read_json(root / '_execution/task.json')
    require(producer['stage'] == 'campaign-setup-probe'
        and producer['parameters']['mode'] == value['mode'] == 'admission-predecessor'
        and value['depth'] == producer['parameters']['depth'] == p['depth'], 'invalid synthetic predecessor producer')
    lock_path = deps[p['lock_input']['dependency']] / p['lock_input']['path']
    lock = read_json(lock_path)
    require(value['lock_hash'] == file_hash(lock_path) == task['recipe_lock']['sha256']
        and value['estimate_hash'] == lock['estimate_hash'] and value['scenario'] == p['scenario'],
        'synthetic predecessor lock/scenario mismatch')
    expected = lock['budget']['expected_predecessor_shapes'][p['scenario']]
    require(expected is not None, 'same-tree profile leaves unavailable for resumed admission')
    compared = compare_publication_shape(expected, synthetic_publication_shape(tree, producer), synthetic=True)
    require(compared['status'] == 'matched', 'admission-shape-mismatch: ' + canonical_json(compared))
    payload = value['dependency_receipt']
    if p['depth'] == 2:
        previous = value['grand_predecessor']
        from oxyformer.execution.integrity import FINGERPRINT
        require(payload['snapshot'][previous['attempt']][FINGERPRINT]['sha256'] == previous['publication_hash'],
            'synthetic grand-predecessor missing from receipt')
    return root, payload


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
    chains = {}
    for record in records:
        binding, stamps = record['binding'], record['profile']
        first, last = record['leaves'][0], record['leaves'][-1]
        shape = [first['task_id'], binding['scenario']['name'], binding.get('recipe_hash'),
            binding.get('lock_hash'), binding.get('chain', {}).get('owner') if binding.get('chain') else None,
            stamps['scientific_fingerprint'], stamps['environment_hash'], record['budget_generation']]
        key = digest(shape)
        fresh = finite(first['setup_seconds'], 'calibrated fresh setup')
        chain = chains.setdefault(key, {'lineage_shape': key, 'scenario': binding['scenario']['name'],
            'first_leaf': first['task_id'], 'fresh': fresh, 'resumed': {}})
        require(chain['fresh'] == fresh, 'calibration fresh measurement drift')
        index = last.get('step', len(record['leaves']) - 1) + 1
        require(type(index) is int and index == len(record['leaves']), 'calibration leaf order mismatch')
        if index > 1:
            require(index not in chain['resumed'], 'duplicate calibration chain step')
            chain['resumed'][index] = last['setup_seconds']
    ordered_chains = [{**chain, 'leaf_indices': sorted(chain['resumed']),
        'resumed': [chain['resumed'][index] for index in sorted(chain['resumed'])]}
        for _, chain in sorted(chains.items())]
    return {'source': 'sealed_campaign_leaves', 'leaf_count': len(own), 'statistics': statistics,
        'setup_chains': ordered_chains,
        'setup_seconds': max(1.25 * p90, max(setups)),
        'resume_overhead_seconds': resume, 'resume_uncalibrated': not bool(completed),
        'resume_source': {'rule': 'max_completed_chain_excess_with_floor',
                          'floor_seconds': RESUME_OVERHEAD_FLOOR_SECONDS, 'completed': completed},
        'realized_compute_per_leaf': statistics['compute_seconds']['mean'],
        'records': [{'record_hash': r['record_hash'], 'task_id': r['leaves'][-1]['task_id'],
                     'budget_generation': r['budget_generation'],
                     'scientific_fingerprint': r['profile']['scientific_fingerprint'],
                     'environment_hash': r['profile']['environment_hash']} for r in records]}


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
    require(mode in ('calibration', 'admission', 'admission-resumed'), 'unknown setup probe mode')
    details = {}
    if mode == 'calibration':
        expected = calibration_inputs(p)
        lock_hash = None
    else:
        lock, lock_hash, expansion, first, expected = locked_probe_inputs(request, config, task)
        scenario = first['parameters']['scenario']['name']
        require(p.get('scenario', scenario) == scenario, 'probe scenario differs from expanded first leaf')
        details.update(estimate_hash=lock['estimate_hash'], scenario=scenario,
            depth=p.get('depth', 0) if mode == 'admission-resumed' else 0)
    base_inputs = deepcopy(expected)
    if mode == 'admission-resumed':
        require(type(p['depth']) is int and p['depth'] in (1, 2), 'resumed probe depth must be 1 or 2')
        ref = p['predecessor_input']
        expected[ref['dependency']] = ['setup_probe.json', *PROFILE_PAYLOADS]
        predecessor = read_receipt(request, config, ref)
        require(predecessor['mode'] == 'admission-predecessor' and predecessor['depth'] == p['depth']
            and predecessor['lock_hash'] == lock_hash and predecessor['scenario'] == scenario,
            'resumed probe predecessor mismatch')
        summary = config.get('verification_summary', {})
        require(summary.get('ancestors_reused', 0) > 0 and summary.get('reuse_miss') is False,
            'reuse-miss: resumed probe did not reuse the locked lineage')
        observed = declared_publication_shape(request, config, ref['dependency'], synthetic=True)
        compared = compare_publication_shape(lock['budget']['expected_predecessor_shapes'][scenario], observed, synthetic=True)
        require(compared['status'] == 'matched', 'admission-shape-mismatch: ' + canonical_json(compared))
        details.update(predecessor_shape=compared, verification=summary,
            predecessor_publication_hash=dict(zip(request.dependency_paths, request.dependency_hashes))[
                str(Path(config['dependencies'][ref['dependency']]) / '_execution/fingerprint.json')])
    first_draw = p.get('first_draw')
    if first_draw is not None:
        require(mode != 'calibration' and first_draw['path'] == 'setup_probe.json', 'invalid repeated probe')
        expected[first_draw['dependency']] = ['setup_probe.json']
    require(task['needs'] == expected, 'probe must verify the declared input shape exactly')
    digests = input_digests(request, config, base_inputs)
    input_digests(request, config, expected)
    prior = []
    if first_draw is not None:
        receipt = read_receipt(request, config, first_draw)
        require(receipt['mode'] == mode and receipt['lock_hash'] == lock_hash
            and receipt['input_digests'] == digests
            and all(receipt.get(key) == value for key, value in details.items() if key != 'verification'),
            'repeated probe verified a different shape')
        prior = receipt.get('setup_draws')
        require(isinstance(prior, list) and len(prior) == 1
            and finite(prior[0], 'first probe draw') >= 0 and receipt['setup_seconds'] == prior[0],
            'repeated probe requires one independent first draw')
    seconds = finite((measurer or probe_setup_seconds)(), 'probe setup')
    require(seconds >= 0, 'negative probe setup')
    draws = [*prior, seconds]
    return {'schema_version': 1, 'mode': mode, 'setup_seconds': min(draws), 'setup_draws': draws,
            'input_digests': digests, 'lock_hash': lock_hash, 'request_hash': request.content_hash, **details}


def synthetic_predecessor(request, config, task):
    """Copy same-tree profile payloads; attest the final lineage, never train."""
    from oxyformer.execution.integrity import FINGERPRINT, RESULT, verify_published_tree
    from oxyformer.provenance import ArtifactRecord, ArtifactLineage
    from oxyformer.validation.coverage import load_budget
    p = task['parameters']
    lock, lock_hash, expansion, first, base = locked_probe_inputs(request, config, task)
    depth, scenario = p['depth'], p['scenario']
    require(type(depth) is int and depth in (1, 2)
        and scenario == first['parameters']['scenario']['name'], 'invalid synthetic predecessor depth/scenario')
    profile = lock['budget']['profiles'][scenario]
    expected = lock['budget']['expected_predecessor_shapes'][scenario]
    require(expected is not None and set(profile.get('setup_profile_sources', {})) == {'1', '2'},
        'same-tree profile leaves unavailable for resumed admission')
    source = profile['setup_profile_sources'][str(depth)]
    require(p['profile_input'] == source['reference'], 'synthetic predecessor profile reference mismatch')
    source_unit = p['profile_input']['dependency']
    record = load_budget(request, config, source_unit)
    require(record['binding']['mode'] == 'profile' and record['binding']['scenario'] == first['parameters']['scenario']
        and record['binding']['draws'] == profile['draws']
        and record['leaves'][-1] == profile['leaf_measurements'][depth - 1], 'synthetic predecessor scenario/profile mismatch')
    require(all(record['profile'][key] == lock[key] for key in ('scientific_fingerprint', 'environment_hash')),
        'resumed admission requires same-tree profile leaves')
    profile_root = Path(config['dependencies'][source_unit])
    profile_result = StageResult.from_json((profile_root / RESULT).read_text())
    profile_tree = verify_published_tree(profile_root, profile_result, source['publication_hash'])
    require(profile_tree['continuation.tar']['sha256'] == source['archive_hash'], 'profile continuation changed')
    ref = p['receipt_input']
    receipt = read_receipt(request, config, ref)
    two_draws(receipt)
    require(receipt['mode'] == ('admission' if depth == 1 else 'admission-resumed')
        and receipt['depth'] == depth - 1 and receipt['lock_hash'] == lock_hash
        and receipt['estimate_hash'] == lock['estimate_hash']
        and receipt['input_digests'] == input_digests(request, config, base),
        'synthetic predecessor receipt does not cover final lock lineage')
    expected_needs = {**base, ref['dependency']: list(PROBE_RECEIPT_INPUTS), source_unit: list(PROFILE_SETUP_INPUTS)}
    require(task['needs'] == expected_needs, 'synthetic predecessor input shape mismatch')
    receipt_root = Path(config['dependencies'][ref['dependency']])
    terminal = StageResult.from_json((receipt_root / RESULT).read_text())
    own = verify_published_tree(receipt_root, terminal)
    producer = StageRequest.from_json(dependency(request, config,
        {'dependency': ref['dependency'], 'path': '_execution/request.json'}).read_text())
    require(producer.content_hash == terminal.request_hash, 'source receipt request mismatch')
    snapshot_path = dependency(request, config, {'dependency': ref['dependency'], 'path': '_execution/dependencies.json'})
    require(dict(zip(producer.dependency_paths, producer.dependency_hashes)).get(str(snapshot_path)) == file_hash(snapshot_path),
        'source dependency receipt is unbound')
    snapshot = read_json(snapshot_path)
    check = read_json(dependency(request, config, {'dependency': ref['dependency'], 'path': '_execution/dependency_check.json'}))
    require(check['status'] == 'pass' and set(check['attempts']) == set(snapshot)
        and all(detail['status'] == 'unchanged' and detail['changed_paths'] == [] and detail['tainted_paths'] == []
            and detail['fingerprint'] == snapshot[root] for root, detail in check['attempts'].items()),
        'source dependency receipt is incomplete or mismatched')
    snapshot[str(receipt_root)] = own
    check['attempts'][str(receipt_root)] = {'status': 'unchanged', 'changed_paths': [], 'tainted_paths': [], 'fingerprint': own}
    value = {'schema_version': 1, 'mode': 'admission-predecessor', 'depth': depth,
        'scenario': scenario, 'lock_hash': lock_hash, 'estimate_hash': lock['estimate_hash'],
        'profile_source': source, 'expected_shape': expected,
        'source_receipt': {'attempt': str(receipt_root), 'publication_hash': own[FINGERPRINT]['sha256']},
        'dependency_receipt': {'snapshot': snapshot, 'check': check},
        'scientific_execution': False}
    reference_bytes = profile['setup_profile_sources']['1']['shape']['archive_bytes']
    copies, remainder = divmod(expected['archive_bytes'], reference_bytes)
    require(copies >= 1 and remainder == 0, 'invalid locked profile archive shape')
    value['profile_archive_copies'] = copies
    value['archive_shape_rule'] = 'cycle authenticated source bytes to the locked first-leaf byte count'
    if depth == 2:
        source_task = read_json(receipt_root / '_execution/task.json')
        source_config = read_json(receipt_root / '_execution/config.json')
        grand_unit = source_task['parameters']['predecessor_input']['dependency']
        grand_root = source_config['dependencies'][grand_unit]
        value['grand_predecessor'] = {'attempt': grand_root,
            'publication_hash': snapshot[grand_root][FINGERPRINT]['sha256']}
    lineage = ArtifactLineage(source_hashes=(request.config_hash, request.task_hash), unit_ids=(request.content_hash,),
        parent_hashes=request.dependency_hashes, split_hash=None, config_hash=request.config_hash,
        model_hash=None, environment=(('code_identity', request.code_identity),), seed=None, parameter_count=None)
    artifacts = []
    for name in PROFILE_PAYLOADS:
        path = dependency(request, config, {'dependency': source_unit, 'path': name})
        destination = Path(request.output_dir) / name
        require(not destination.exists(), 'synthetic predecessor payload already exists')
        if name == 'continuation.tar':
            combined = sha256()
            remaining = expected['archive_bytes']
            with destination.open('xb') as output:
                while remaining:
                    copied = sha256()
                    with path.open('rb') as original:
                        while chunk := original.read(1024 * 1024):
                            copied.update(chunk)
                            selected = chunk[:remaining]
                            output.write(selected)
                            combined.update(selected)
                            remaining -= len(selected)
                    require(copied.hexdigest() == profile_tree[name]['sha256'], 'synthetic profile copy mismatch')
            expected_hash = combined.hexdigest()
        else:
            shutil.copyfile(path, destination)
            expected_hash = profile_tree[name]['sha256']
        require(file_hash(destination) == expected_hash, 'synthetic profile copy mismatch')
        artifacts.append(ArtifactRecord(path=name, sha256=expected_hash, kind='synthetic_profile_copy', lineage=lineage))
    # The normal runner contributes its controls after the worker returns.
    # Fill the remaining record slots with profile metadata, never fake science.
    control_names = {str(path.relative_to(request.output_dir))
        for path in (Path(request.output_dir) / '_execution').rglob('*') if path.is_file()}
    control_names.update((FINGERPRINT, RESULT, '_execution/dependency_check.json'))
    base_records = len(control_names) + len(PROFILE_PAYLOADS) + 2  # probe and artifact manifest
    require(base_records <= expected['record_count'], 'synthetic controls exceed locked publication shape')
    profile_record = dependency(request, config, {'dependency': source_unit, 'path': 'result.json'})
    for index in range(expected['record_count'] - base_records):
        name = f'synthetic-records/profile-{index}.json'
        destination = Path(request.output_dir) / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(profile_record, destination)
        require(file_hash(destination) == profile_tree['result.json']['sha256'], 'synthetic record copy mismatch')
        artifacts.append(ArtifactRecord(path=name, sha256=profile_tree['result.json']['sha256'],
            kind='synthetic_profile_copy', lineage=lineage))
    return publish(request, {'setup_probe.json': value}, status='pass', message='Synthetic predecessor; no scientific computation',
        extra_artifacts=tuple(artifacts))


def setup_admission_plan(lock, plans, lock_unit, approvals):
    """Two independent jobs per logical probe; three final receipts feed admit."""
    from oxyformer.execution.campaign import _unit
    lock_hash = digest(lock)
    if lock.get('campaign_mode') == 'diagnostic':
        from oxyformer.validation.campaign import validate_diagnostic
        decision = validate_diagnostic(lock['recipe'], approvals)
        require(plans[0]['tasks'][0]['parameters']['scenario']['name'] == decision['diagnostic_campaign']['scenarios'][0],
                'diagnostic admission scenario drift')
    first, base = first_leaf_inputs({'plans': plans})
    scenario = first['parameters']['scenario']['name']
    profile = lock['budget']['profiles'][scenario]
    lock_ref = {'dependency': lock_unit, 'path': 'recipe_lock.json'}
    recipe_ref = {**lock_ref, 'sha256': lock_hash}
    tasks, receipts = [], {}

    def identifier(label):
        return 'setup-' + digest([lock_hash, label])[:22]

    def append(label, parameters, needs, outputs=None):
        task = {'id': identifier(label), 'stage': 'campaign-setup-probe', 'recipe_lock': recipe_ref,
            'parameters': {'lock_input': lock_ref, 'scenario': scenario, **parameters},
            'needs': deepcopy(needs), 'outputs': outputs or ['setup_probe.json', 'artifact_manifest.json']}
        tasks.append(task)
        return {'dependency': task['id'], 'path': 'setup_probe.json'}

    predecessor = None
    for depth in range(3):
        mode = 'admission' if depth == 0 else 'admission-resumed'
        needs = deepcopy(base)
        parameters = {'mode': mode}
        if depth:
            source = profile.get('setup_profile_sources', {}).get(str(depth))
            source_ref = source['reference'] if source else {
                'dependency': profile['leaf_measurements'][depth - 1]['task_id'], 'path': BUDGET_RECORD}
            previous = receipts['fresh' if depth == 1 else 'resumed_1']
            synthetic_needs = {**base, previous['dependency']: list(PROBE_RECEIPT_INPUTS),
                source_ref['dependency']: list(PROFILE_SETUP_INPUTS)}
            predecessor = append(f'predecessor-{depth}', {'mode': 'admission-predecessor', 'depth': depth,
                'profile_input': source_ref, 'receipt_input': previous}, synthetic_needs,
                ['setup_probe.json', 'artifact_manifest.json', *PROFILE_PAYLOADS])
            parameters.update(depth=depth, predecessor_input=predecessor)
            needs[predecessor['dependency']] = ['setup_probe.json', *PROFILE_PAYLOADS]
        first_draw = append(f'probe-{depth}-draw-1', parameters, needs)
        second = append(f'probe-{depth}-draw-2', {**parameters, 'first_draw': first_draw},
            {**needs, first_draw['dependency']: ['setup_probe.json']})
        receipts['fresh' if depth == 0 else f'resumed_{depth}'] = second
    tasks.append({'id': lock['admission_receipt']['dependency'], 'stage': 'campaign-admit',
        'recipe_lock': recipe_ref, 'parameters': {'lock_input': lock_ref, 'probe_inputs': receipts},
        'needs': {**base, **{ref['dependency']: ['setup_probe.json'] for ref in receipts.values()}},
        'outputs': ['admission.json', 'artifact_manifest.json']})
    if lock.get('campaign_mode') == 'diagnostic':
        from oxyformer.validation.campaign import screening_metadata
        for task in tasks:
            task.update(screening_metadata(lock['recipe']))
    units = [_unit({'id': lock['campaign_id'], 'campaign_mode': 'screening'}, task, [], 0, 14400, 'admission')
             for task in tasks]
    return {'schema_version': 1, 'lock_hash': lock_hash, 'receipt_inputs': receipts, 'tasks': tasks, 'units': units}


def admit(request, config, task):
    from oxyformer.execution.campaign import validate_plan
    from oxyformer.validation.coverage import projected_setup_allowance, require_setup_fits
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
    refs = p.get('probe_inputs', {})
    require(set(refs) == {'fresh', 'resumed_1', 'resumed_2'}
        and len({(ref['dependency'], ref['path']) for ref in refs.values()}) == 3,
        'three admission probe receipts required')
    expected = input_digests(request, config, plans[0]['spec']['inputs'])
    scenario = plans[0]['tasks'][0]['parameters']['scenario']['name']
    minima, draws = {}, {}
    for key, depth in (('fresh', 0), ('resumed_1', 1), ('resumed_2', 2)):
        receipt = read_receipt(request, config, refs[key])
        require(receipt['mode'] == ('admission' if depth == 0 else 'admission-resumed')
            and receipt.get('depth') == depth and receipt['lock_hash'] == lock_hash
            and receipt.get('estimate_hash') == lock['estimate_hash'] and receipt.get('scenario') == scenario,
            'admission probe lock_hash/estimate/depth mismatch')
        require(receipt['input_digests'] == expected, 'admission probe verified different inputs')
        minima[key] = two_draws(receipt)
        draws[key] = receipt['setup_draws']
        if depth:
            require(receipt['verification'].get('reuse_miss') is False
                and receipt['verification'].get('ancestors_reused', 0) > 0, 'reuse-miss: admission probe')
            comparison = receipt['predecessor_shape']
            require(comparison['expected'] == lock['budget']['expected_predecessor_shapes'][scenario]
                and compare_publication_shape(comparison['expected'], comparison['observed'], synthetic=True)['status'] == 'matched',
                'admission-shape-mismatch: probe shape reference')
    longest = max(len(plan['expected_leaves']) for plan in plans)
    allowance = projected_setup_allowance(minima['fresh'], [minima['resumed_1'], minima['resumed_2']],
        longest, extrapolate=False)
    setup = max(minima.values())
    counts, fits = {}, {}
    for batch in lock['batches']:
        profile = lock['budget']['profiles'][batch['scenario']['name']]
        counts[batch['batch_id']] = batch['slices']
        try:
            fits[batch['scenario']['name']] = require_setup_fits(allowance, window=batch['wall_seconds'],
                resume_reserve=profile['resume_overhead_seconds'], usable_compute=profile['assumed_usable_per_leaf'])
        except ContractError as exc:
            raise ContractError(f'admission not established: measured setup {setup}; counts {counts}; {exc}') from exc
    candidate = {**output_metadata(task), 'schema_version': 1, 'admitted': True, 'lock_hash': lock_hash,
        'estimate_hash': lock['estimate_hash'], 'setup_policy': 'three-receipt-minima-v1',
        'admission_measurements': {'probes': refs, 'setup_draws': draws, 'setup_minima': minima,
            'allowance_seconds': allowance['allowance'], 'projection': allowance,
            'locked_counts': counts, 'fits': fits}}
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
    if 'setup_safety_net' in lock['budget']:
        from oxyformer.validation.coverage import projected_setup_allowance
        require(receipt.get('setup_policy') == 'three-receipt-minima-v1', 'three-receipt screening admission required')
        measured = receipt['admission_measurements']
        minima = measured['setup_minima']
        require(set(minima) == {'fresh', 'resumed_1', 'resumed_2'}, 'invalid admission minima')
        expected = projected_setup_allowance(minima['fresh'], [minima['resumed_1'], minima['resumed_2']],
            max(batch['slices'] for batch in lock['batches']), extrapolate=False)
        require(measured['allowance_seconds'] == expected['allowance'], 'admission allowance mismatch')
    return receipt


def run_stage(request: StageRequest) -> StageResult:
    try:
        request.verify_inputs()
        config, task = read_json(request.config_path), read_json(request.task_path)
        require(task['stage'] == request.stage, 'stage mismatch')
        if request.stage == 'campaign-setup-probe':
            if task['parameters']['mode'] == 'admission-predecessor':
                return synthetic_predecessor(request, config, task)
            value, name = probe(request, config, task), 'setup_probe.json'
        else:
            require(request.stage == 'campaign-admit', 'unknown campaign probe stage')
            value, name = admit(request, config, task), 'admission.json'
        return publish(request, {name: value}, status='pass', message='Setup measured' if name == 'setup_probe.json' else 'Screening lock admitted')
    except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
        return StageResult(request_hash=request.content_hash, status='blocked', artifacts=(), message=str(exc) or type(exc).__name__)
