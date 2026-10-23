"""Detect upstream state changes after worker exit; never repair upstream.
Managed outputs are confined; general write prevention is deferred to ARC-1339.
"""
from hashlib import sha256
import errno
import json
import os
from pathlib import Path
import re
import sys
import time
from dataclasses import replace

import yaml

from oxyformer.contracts import StageRequest, StageResult
from oxyformer.provenance import ContractError, canonical_json, relative_artifact_path, require
from .integrity import (DEPENDENCY_CHECK, FINGERPRINT, RESULT, _replace_control, _repair_control_directory,
    post_execution_check, publish_result, record_taints, snapshot_dependency_identities, observe_dependencies,
    directory_path, read_regular, regular_file_stat, regular_file_hash as file_hash,
    fingerprint_tree, publication_receipt, InputChanged, acquisition_read, _acquisition_binding,
    integrity_observation, verification_timings, verification_span, timed_verification, verification_diagnostic,
    acquisition_changed_paths, verify_input_hash, authority_exists,
    verify_inputs, verify_result, verify_published_tree, verify_publication)
from .identity import git_bytes, code_identity, environment_record, scientific_fingerprint, verify_recipe
from .paths import atomic_json, atomic_write, output_path


def allocation_threads(*, membership=Path('/proc/self/cgroup'), mountinfo=Path('/proc/self/mountinfo')):
    """Slurm's allocation, otherwise affinity limited by visible cgroup quotas.

    Discover both the mount point and its hierarchy root from mountinfo. This
    handles nonstandard locations and bind-mounted subtrees without assuming
    controller directory names. Fractional quotas round down to at least one.
    """
    allocated = os.environ.get('SLURM_CPUS_PER_TASK')
    if allocated is not None:
        require(allocated.isdecimal() and int(allocated) > 0,
            'SLURM_CPUS_PER_TASK must be a positive integer')
        return int(allocated)
    count = len(os.sched_getaffinity(0))
    locations = {}
    for line in membership.read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        if controllers == '':
            locations['cgroup2'] = relative
        elif 'cpu' in controllers.split(','):
            locations['cgroup'] = relative

    def unescape(value):
        return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), value)

    for line in mountinfo.read_text().splitlines():
        mount, filesystem = line.split(' - ', 1)
        kind, _, options = filesystem.split()
        if kind not in locations or (kind == 'cgroup' and 'cpu' not in options.split(',')):
            continue
        fields = mount.split()
        hierarchy_root, root = Path(unescape(fields[3])), Path(unescape(fields[4]))
        location = Path('/').joinpath(*[p for p in locations[kind].split('/') if p not in ('', '.', '..')])
        if not location.is_relative_to(hierarchy_root):
            # A visible mount of a sibling cgroup does not constrain this task.
            continue
        path = root / location.relative_to(hierarchy_root)
        quota_name = 'cpu.max' if kind == 'cgroup2' else 'cpu.cfs_quota_us'
        for directory in (path, *path.parents):
            if directory != root and root not in directory.parents:
                break
            quota = directory / quota_name
            if not quota.is_file():
                continue
            if kind == 'cgroup2':
                value, period = quota.read_text().split()
                if value == 'max':
                    continue
            else:
                value = quota.read_text().strip()
                if int(value) < 0:
                    continue
                period = (directory / 'cpu.cfs_period_us').read_text().strip()
            require(int(value) > 0 and int(period) > 0, 'invalid cgroup CPU quota')
            count = min(count, max(1, int(value) // int(period)))
    return max(1, count)


def configure_threads():
    """Bind native libraries and Torch to the same allocation in this process.

    The launcher exports these before Python starts. Setting them here also
    covers direct CLI calls and makes the worker subprocess inherit the limit.
    Fresh Torch imports honor these native limits. Do not load Torch solely for
    bookkeeping in a non-training stage; fit identities record observed counts.
    """
    count = allocation_threads()
    os.environ['OMP_NUM_THREADS'] = str(count)
    os.environ['MKL_NUM_THREADS'] = str(count)
    torch = sys.modules.get('torch')
    if torch is not None:
        torch.set_num_threads(count)
    return {'source': 'SLURM_CPUS_PER_TASK' if 'SLURM_CPUS_PER_TASK' in os.environ else 'cgroup',
        'allocated_cpus': count, 'torch_intraop_limit': count,
        'torch': torch.get_num_threads() if torch is not None else None,
        'torch_interop': torch.get_num_interop_threads() if torch is not None else None,
        'OMP_NUM_THREADS': os.environ['OMP_NUM_THREADS'],
        'MKL_NUM_THREADS': os.environ['MKL_NUM_THREADS']}


def dependency_variable(unit_id):
    require(isinstance(unit_id, str) and bool(unit_id), 'empty dependency ID')
    return 'SWARM_DEP_' + re.sub('[^A-Z0-9]', '_', unit_id.upper())


def resolve_dependencies(ids, environ=None):
    environ = os.environ if environ is None else environ
    variables = [dependency_variable(i) for i in ids]
    require(len(set(variables)) == len(variables), 'dependency normalization collision')
    result = {}
    for unit, name in zip(ids, variables):
        require(name in environ, f'missing dependency variable {name}')
        path = Path(environ[name])
        require(path.is_absolute(), f'dependency must be absolute: {name}')
        result[unit] = directory_path(path)
        require(result[unit].is_dir(), f'dependency must be an attempt directory: {name}')
    return result


class _JsonSafeLoader(yaml.SafeLoader):
    """Keep YAML date spellings JSON-compatible without changing source bytes."""


_JsonSafeLoader.add_constructor('tag:yaml.org,2002:timestamp',
    lambda loader, node: loader.construct_scalar(node))


def read_mapping(path, *, expected_bytes=None):
    path = Path(path)
    raw = read_regular(path)
    require(expected_bytes is None or raw == expected_bytes, f'input differs from HEAD: {path}')
    text = raw.decode('utf-8')
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        require(path.suffix.lower() != '.json', f'invalid JSON: {path}')
        try:
            value = yaml.load(text, Loader=_JsonSafeLoader)
        except yaml.YAMLError as yaml_error:
            raise ContractError(f'invalid YAML: {path}') from yaml_error
    require(isinstance(value, dict), f'expected mapping: {path}')
    return value


def dependency_file(root, relative):
    relative_artifact_path(relative)
    path = Path(root) / relative
    with acquisition_read(path):
        regular_file_stat(path)
        require(path.resolve(strict=True).is_relative_to(root), f'dependency file escapes attempt: {path}')
        return path


def verify_acquisition(root, receipt_name, *, expected_tree=None):
    with integrity_observation(root):
        binding = _acquisition_binding(root)
        return _verify_acquisition(root, receipt_name, expected_tree=expected_tree,
            observed_tree=None if binding is None else binding[2])


def _verify_acquisition(root, receipt_name, *, expected_tree, observed_tree):
    """Bind a complete acquisition to a create-once external tree baseline.

    Acquisition producers predate StageResult. Never write into their attempts;
    the independent publication store retains the first validated tree and the
    same permanent taint markers used for stage publications.
    """
    relative_artifact_path(receipt_name)
    authority = publication_receipt(root, create=True)
    require(not authority_exists(str(authority) + '.tainted'), f'tainted upstream fingerprint: {root}')
    baseline = Path(str(authority) + '.acquisition')
    tree = fingerprint_tree(root)

    def refuse_changes(changed, message):
        if changed:
            record_taints({'attempts': {str(root): {'changed_paths': sorted(set(changed))}}})
            raise ContractError(message + '; changed paths: ' +
                ', '.join(str(Path(root) / name) for name in sorted(set(changed))))

    def compare(expected, message, actual=None):
        actual = tree if actual is None else actual
        refuse_changes(acquisition_changed_paths(expected, actual), message)

    refuse_changes([name for name, entry in tree.items() if entry.get('changed')],
        'acquisition changed during fingerprinting (tainted)')

    if observed_tree is not None:
        compare(observed_tree, 'acquisition fingerprint mismatch against observation binding (tainted)')
    if expected_tree is not None:
        compare(expected_tree, 'acquisition fingerprint differs from consumer baseline (tainted)')
    if authority_exists(baseline):
        compare(read_mapping(baseline)['entries'], f'acquisition fingerprint mismatch (tainted): {root}')
    require(not any('error' in entry for entry in tree.values()), 'acquisition fingerprint unreadable')
    # Establish valid first-time inputs before interpreting later path/type
    # failures as observed changes. A rescan cannot erase such observations.
    for name in (receipt_name, 'payload.tar'):
        require(tree.get(name, {}).get('sha256') is not None,
            f'acquisition input is missing or not regular: {Path(root) / name}')
    observed_name = receipt_name
    try:
        receipt_path = dependency_file(root, receipt_name)
        observed_name = 'payload.tar'
        dependency_file(root, observed_name)
        observed_name = receipt_name
        receipt_bytes = read_regular(receipt_path)
    except InputChanged as exc:
        if exc.content_changed:
            refuse_changes([str(exc.path.relative_to(root))], str(exc))
        raise
    except ContractError as exc:
        # The snapshot above established regular inputs and receipt_name was
        # validated before reading. A helper's path/type refusal is a change.
        refuse_changes([observed_name], str(exc))
        raise
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
            refuse_changes([observed_name], 'acquisition input removed or replaced during verification')
        # Unknown I/O failure: preserve any positive evidence in a rescan,
        # but never mistake failure to read for proof of different bytes.
        compare(tree, 'acquisition changed during verification (tainted)', fingerprint_tree(root))
        raise
    refuse_changes([receipt_name] if tree[receipt_name]['sha256'] != sha256(receipt_bytes).hexdigest() else [],
        'acquisition receipt changed during verification (tainted)')
    receipt = json.loads(receipt_bytes)
    require(isinstance(receipt, dict), 'acquisition receipt must be a mapping')
    require(receipt.get('status') == 'complete', 'acquisition receipt is not complete')
    require(tree['payload.tar']['sha256'] == receipt.get('payload_sha256'), 'acquisition payload hash mismatch')
    require(tree['payload.tar']['size'] == receipt.get('payload_bytes'), 'acquisition payload size mismatch')
    value = {'attempt': str(root), 'receipt': receipt_name, 'entries': tree}
    try:
        atomic_json(baseline.parent, baseline.name, value)
    except FileExistsError:
        original = read_mapping(baseline)
        compare(original['entries'], f'acquisition fingerprint mismatch (tainted): {root}')
        require(original == value, f'acquisition baseline identity mismatch: {root}')
    require(not authority_exists(str(authority) + '.tainted'), f'tainted upstream fingerprint: {root}')
    return tree


def _recorded_acquisition_inputs(current, parent, receipt, hashes):
    """Check a producer's sealed evidence, without observing the acquisition.

    The publication fingerprint authenticates both control files. The request
    binds the admission snapshot; the successful post-exit check binds the same
    snapshot to the bytes the producer used. Paths here are lexical: an archived
    acquisition (even its entire directory) need not exist for a consumer.
    """
    baseline = current / '_execution/dependencies.json'
    require(str(baseline) in hashes, 'acquisition baseline not bound to published request')
    snapshots = read_mapping(baseline)
    require(str(parent) in snapshots, 'acquisition baseline absent from published request')
    entries = snapshots[str(parent)]
    check = read_mapping(current / DEPENDENCY_CHECK)
    observed = check.get('attempts', {}).get(str(parent), {})
    require(check.get('status') == 'pass' and observed.get('status') == 'unchanged'
        and observed.get('changed_paths') == [] and observed.get('tainted_paths') == []
        and observed.get('fingerprint') == entries,
        'acquisition dependency check differs from admitted baseline')
    recorded = {}
    for path, digest in hashes.items():
        path = Path(path)
        if path.is_relative_to(parent):
            entry = entries.get(str(path.relative_to(parent)), {})
            require(entry.get('sha256') == digest,
                f'acquisition input not bound to admitted baseline: {path}')
            recorded[str(path)] = digest
    relative_artifact_path(receipt)
    require(all(str(parent / name) in recorded for name in ('payload.tar', receipt)),
        'acquisition payload and receipt must be bound to published request')
    return recorded


def _verify_publication_result(result, request, recorded_inputs):
    """Verify a historical result with explicitly authenticated input evidence.

    Live stages still use verify_result, including full input verification.
    Publication artifacts and inputs without an attestation keep live checks.
    """
    require(result.request_hash == request.content_hash, 'stage request mismatch')
    for path, digest in zip((request.config_path, request.task_path) + request.dependency_paths,
            (request.config_hash, request.task_hash) + request.dependency_hashes):
        if path in recorded_inputs:
            require(recorded_inputs[path] == digest, 'sealed input baseline hash mismatch')
        else:
            verify_input_hash(path, digest)
    root = directory_path(request.output_dir)
    for artifact in result.artifacts:
        path = (root / artifact.path).resolve(strict=True)
        require(path.is_relative_to(root), 'artifact escapes output directory')
        require(file_hash(root / artifact.path) == artifact.sha256,
            f'artifact hash mismatch: {artifact.path}')


def measurement_dependencies(task):
    refs = task.get('parameters', {}).get('setup_calibration')
    if refs is None:
        return set()
    require(task['stage'] in ('campaign-estimate', 'campaign-lock'),
            'setup_calibration is permitted only on estimate and lock tasks')
    require(isinstance(refs, list) and bool(refs), 'invalid setup_calibration')
    for ref in refs:
        require(set(ref) == {'dependency', 'path'} and ref['path'] == 'budget_record.json',
                'invalid setup_calibration reference')
        require(ref['path'] in task.get('needs', {}).get(ref['dependency'], []),
                'setup_calibration must be a declared input')
    require(len({ref['dependency'] for ref in refs}) == len(refs), 'duplicate calibration dependency')
    return {ref['dependency'] for ref in refs}


def scientific_ancestors(task, deps, edges, acquisitions=()):
    pending = [root for unit, root in deps.items()
        if unit not in measurement_dependencies(task) and unit not in acquisitions]
    selected = set()
    while pending:
        root = pending.pop()
        if root not in selected:
            selected.add(root)
            pending.extend(edges[root])
    return selected


class DependencyReuse:
    """Attest transitive bytes; never substitute evidence for a direct input.

    A receipt is useful only inside its currently verified producer publication.
    Its startup snapshot is separately hash-bound by that producer's request,
    and its passing post-check must agree entry-for-entry with the snapshot.
    Missing or inconsistent entries are verified in full before any pruning.
    """
    def __init__(self, deps):
        self.direct = set(deps.values())
        self.publications = {}
        self.entries = {}
        self.sources = {}
        self.rewalk = {}
        self.measurements = set()
        self.shared = set()
        self.predecessor = None

    def publication(self, root):
        if root not in self.publications:
            terminal = StageResult.from_json(read_regular(root / RESULT))
            require(terminal.status == 'pass', 'chain stopped: allowance-exceeded'
                if 'allowance-exceeded' in terminal.message else 'dependency stage did not pass')
            tree = verify_published_tree(root, terminal)
            request = StageRequest.from_json(read_regular(root / '_execution/request.json'))
            require(request.content_hash == terminal.request_hash and Path(request.output_dir) == root,
                'receipt producer request mismatch')
            self.publications[root] = (terminal, tree)
        return self.publications[root]

    def add_receipt(self, root, *, payload=None):
        terminal, own = self.publication(root)
        fingerprint = next(a.sha256 for a in terminal.artifacts if a.path == FINGERPRINT)
        try:
            if payload is None:
                request = StageRequest.from_json(read_regular(root / '_execution/request.json'))
                snapshot_path = root / '_execution/dependencies.json'
                hashes = dict(zip(request.dependency_paths, request.dependency_hashes))
                require(hashes.get(str(snapshot_path)) == own['_execution/dependencies.json']['sha256'],
                    'dependency receipt snapshot is unbound')
                # Both files were checked against this publication, above.
                require(DEPENDENCY_CHECK in own, 'dependency receipt is unbound')
                snapshot = read_mapping(snapshot_path)
                check = read_mapping(root / DEPENDENCY_CHECK)
            else:
                # Only the registered synthetic-predecessor producer supplies
                # this payload, as a declared, currently checked output.
                snapshot, check = payload['snapshot'], payload['check']
            require(check['status'] == 'pass' and isinstance(snapshot, dict), 'dependency receipt did not pass')
            attempts = check['attempts']
            require(isinstance(attempts, dict), 'invalid dependency receipt')
        except (ContractError, KeyError, TypeError, ValueError):
            return  # An unbound receipt cannot prune any branch.
        for name in set(snapshot) | set(attempts):
            if not isinstance(name, str):
                continue
            path = Path(name)
            before, after = snapshot.get(name), attempts.get(name, {})
            after = after if isinstance(after, dict) else {}
            if not path.is_absolute():
                continue
            if not isinstance(before, dict):
                observed = after.get('fingerprint', {})
                publication = observed.get(FINGERPRINT, {}) if isinstance(observed, dict) else {}
                if isinstance(publication, dict) and publication.get('sha256'):
                    self.rewalk[path] = publication['sha256']
                continue
            publication = before.get(FINGERPRINT, {})
            expected = publication.get('sha256') if isinstance(publication, dict) else None
            # Acquisition evidence is never a reusable publication receipt.
            if expected is None:
                continue
            if (after.get('status') != 'unchanged' or after.get('changed_paths')
                    or after.get('tainted_paths') or after.get('fingerprint') != before):
                self.rewalk[path] = expected
                continue
            if path in self.entries and self.entries[path] != before:
                self.rewalk[path] = expected
                continue
            self.entries[path] = before
            self.sources[path] = {'attempt': str(root), 'publication_hash': fingerprint}

    def reusable(self, root, expected=None, *, measurement_path=None):
        if root in self.direct or root in self.rewalk or root not in self.entries:
            return False
        if measurement_path is False and root in self.measurements and root not in self.shared:
            return False
        if expected is not None and self.entries[root][FINGERPRINT]['sha256'] != expected:
            self.rewalk[root] = expected
            return False
        return True

    def recorded_inputs(self, request, *, measurement_path, measurement_parents):
        records = {}
        roots = sorted(self.entries, key=lambda p: len(p.parts), reverse=True)
        for name, expected in zip(request.dependency_paths, request.dependency_hashes):
            path = Path(name)
            # Direct inputs retain every current-byte check, including a file
            # nested beneath another attested root.
            if any(path.is_relative_to(root) for root in self.direct):
                continue
            for root in roots:
                if path.is_relative_to(root) and self.reusable(root):
                    if root not in self.shared and root in self.measurements and not (
                            measurement_path or any(path.is_relative_to(parent) for parent in measurement_parents)):
                        continue
                    entry = self.entries[root].get(str(path.relative_to(root)), {})
                    if entry.get('sha256') == expected:
                        records[name] = expected
                    else:
                        self.rewalk[root] = self.entries[root][FINGERPRINT]['sha256']
                    break
        return records

    def reused(self, verified):
        return {str(root): tree for root, tree in self.entries.items()
            if root not in verified and self.reusable(root)}

    def annotate(self, timings, verified):
        reused = self.reused(verified)
        walked = set(verified) - self.direct
        timings.update(ancestors_reused=len(reused), ancestors_rewalked=len(walked),
            measurement_ancestors_reused=sum(Path(root) in self.measurements for root in reused),
            predecessor_publications_added=1,
            reuse_miss=len(walked) > 1,
            receipt_sources={root: self.sources[Path(root)] for root in reused})


def dependency_reuse(task, deps, *, output_dir=None):
    chain = task.get('continuation', {})
    resumed = task['stage'] in ('coverage', 'simulation-smoke') and chain.get('predecessor')
    probe = task['stage'] == 'campaign-setup-probe' and task.get('parameters', {}).get('mode') == 'admission-resumed'
    if not (resumed or probe):
        return None
    policy = DependencyReuse(deps)
    if probe:
        from oxyformer.validation.campaign_probe import resumed_probe_receipt
        root, payload = resumed_probe_receipt(task, deps, policy.publication)
        policy.predecessor = root
        policy.add_receipt(root, payload=payload)
    else:
        require(chain['predecessor'] in deps, 'continuation predecessor must be an explicit dependency')
        policy.predecessor = deps[chain['predecessor']]
        policy.add_receipt(policy.predecessor)
    policy.shared = set(policy.entries)
    # The estimate's attestation is additive. Older estimates lack it; their
    # ancestors still need a bound predecessor receipt or the full walk.
    for root in deps.values():
        task_path = root / '_execution/task.json'
        producer = read_mapping(task_path)
        if producer.get('stage') != 'campaign-estimate':
            continue
        terminal, tree = policy.publication(root)
        if 'measurement_only_dependencies.json' not in tree:
            continue
        attestation = read_mapping(root / 'measurement_only_dependencies.json')
        if attestation.get('schema_version') != 2:
            continue
        estimate = read_mapping(root / 'budget_estimate.json')
        records = estimate.get('calibration', {}).get('records', [])
        if not records or attestation.get('records') != records:
            continue
        payload = attestation.get('dependency_receipt', {})
        try:
            units = measurement_dependencies(producer)
            request = StageRequest.from_json(read_regular(root / '_execution/request.json'))
            hashes = dict(zip(request.dependency_paths, request.dependency_hashes))
            config = read_mapping(request.config_path)
            require(set(attestation['measurement_only_dependencies']) == units
                == {record['task_id'] for record in records}, 'measurement attestation identity mismatch')
            for record in records:
                path = Path(config['dependencies'][record['task_id']])
                require(payload['snapshot'][str(path)]['budget_record.json']['sha256']
                    == hashes[str(path / 'budget_record.json')] == record['record_hash'],
                    'measurement attestation hash mismatch')
        except (ContractError, KeyError, TypeError, ValueError):
            continue
        previous = set(policy.entries)
        policy.add_receipt(root, payload=payload)
        policy.measurements.update(set(policy.entries) - previous)
        policy.measurements.update(Path(name) for name in payload.get('snapshot', {}))
    if output_dir is not None:
        # Pruning a branch must not prune its output-isolation check. Include
        # deeper receipt entries that traversal will never visit at all.
        for ancestor in set(policy.entries) | set(policy.rewalk):
            require(not output_dir.is_relative_to(ancestor) and not ancestor.is_relative_to(output_dir),
                f'output overlaps an attested ancestor: {ancestor}')
    return policy


def rewalk_receipt_misses(policy, trees, verified, output_dir, consumer_task, edges, budgets):
    if policy is None:
        return
    for root, expected in policy.rewalk.items():
        # A fresh cache is essential: an earlier pruned visit is not a full
        # verification of this branch. Rechecking a miss is intentionally slow.
        checked = {}
        verify_dependency_result(root, expected_hash=expected, trees=trees, verified=checked,
            output_dir=output_dir, consumer_task=consumer_task, scientific_edges=edges,
            budget_authorities=budgets)
        verified.update(checked)


def verify_dependency_result(root, *, expected_hash=None, trees=None, active=None, verified=None,
    output_dir=None, consumer_task=None, measurement_only_dependencies=None, budget_authorities=None, scientific_edges=None,
    reuse=None):
    with integrity_observation(root):
        return _verify_dependency_result(root, expected_hash=expected_hash, trees=trees,
            active=active, verified=verified, output_dir=output_dir, consumer_task=consumer_task,
            measurement_only_dependencies=measurement_only_dependencies, budget_authorities=budget_authorities,
            scientific_edges=scientific_edges, reuse=reuse)


def _verify_dependency_result(root, *, expected_hash, trees, active, verified, output_dir,
                              consumer_task=None, measurement_only_dependencies=None, budget_authorities=None, scientific_edges=None,
                              reuse=None):
    """Verify all lineage publications; only live stages observe acquisitions."""
    root = directory_path(root)
    active = set() if active is None else active
    verified = {} if verified is None else verified
    entered = set()
    budget_authorities = set() if budget_authorities is None else budget_authorities
    pending = {}
    allowed_measurements = measurement_dependencies(consumer_task) if consumer_task else set()
    stack = [(root, expected_hash, None, False, allowed_measurements, False)]
    def verify_budget_authority(current, request, task, allowed):
        measurement_dependencies(task)
        if request.stage not in ('coverage', 'simulation-smoke'):
            return
        if task['id'] in allowed:
            if measurement_only_dependencies is not None:
                measurement_only_dependencies.add(task['id'])
        elif current not in budget_authorities:
            from oxyformer.validation.coverage import load_budget
            names = ('_execution/request.json', '_execution/task.json', '_execution/result.json',
                'budget_record.json', 'result.json')
            paths = tuple(str(current / name) for name in names)
            consumer = replace(request, dependency_paths=paths,
                dependency_hashes=tuple(file_hash(p) for p in paths))
            load_budget(consumer, {'dependencies': {task['id']: str(current)}}, task['id'])
            budget_authorities.add(current)

    try:
        while stack:
            current, expected, unit, ready, allowed_measurements, measurement_path = stack.pop()
            with verification_span(current, 'lineage_walk'):
                if ready:
                    result, tree = pending.pop(current)
                    if trees is not None:
                        trees[str(current)] = tree
                    verified[current] = result
                    active.remove(current)
                else:
                    if reuse is not None and reuse.reusable(current, expected, measurement_path=measurement_path):
                        if scientific_edges is not None:
                            scientific_edges[current] = ()
                        continue
                    current = directory_path(current)
                    if output_dir is not None:
                        require(not output_dir.is_relative_to(current) and not current.is_relative_to(output_dir),
                            f'output overlaps an upstream attempt: {current}')
                    require(current not in active, 'dependency publication cycle')
                    if current in verified:
                        result = verified[current]
                        require(expected is None or any(a.path == FINGERPRINT and a.sha256 == expected
                                for a in result.artifacts),
                            'dependency published fingerprint identity changed')
                        verify_budget_authority(current,
                            StageRequest.from_json(read_regular(dependency_file(current, '_execution/request.json'))),
                            read_mapping(current / '_execution/task.json'), allowed_measurements)
                    else:
                        active.add(current)
                        entered.add(current)
                        request = StageRequest.from_json(read_regular(dependency_file(current, '_execution/request.json')))
                        result = StageResult.from_json(read_regular(dependency_file(current, RESULT)))
                        require(Path(request.output_dir).resolve() == current, 'dependency attempt owner mismatch')
                        require(result.status == 'pass', 'chain stopped: allowance-exceeded'
                            if 'allowance-exceeded' in result.message else 'dependency stage did not pass')
                        cached = reuse.publications.get(current) if reuse is not None else None
                        if cached is None:
                            tree = verify_published_tree(current, result, expected)
                        else:
                            require(result == cached[0] and (expected is None or
                                cached[1][FINGERPRINT]['sha256'] == expected), 'cached publication identity changed')
                            tree = cached[1]
                        producer_task = read_mapping(current / '_execution/task.json')
                        own_measurements = measurement_dependencies(producer_task)
                        child_measurements = allowed_measurements | own_measurements
                        config = read_mapping(request.config_path)
                        hashes = dict(zip(request.dependency_paths, request.dependency_hashes))
                        parents = []
                        recorded_inputs = {}
                        for parent_unit, parent in config.get('dependencies', {}).items():
                            parent = Path(parent)
                            require(parent.is_absolute(), 'dependency publication path must be absolute')
                            acquisition = config.get('acquisitions', {}).get(parent_unit)
                            if acquisition is not None:
                                if output_dir is not None:
                                    require(not output_dir.is_relative_to(parent) and not parent.is_relative_to(output_dir),
                                        f'output overlaps an upstream attempt: {parent}')
                                recorded_inputs.update(_recorded_acquisition_inputs(
                                    current, parent, acquisition, hashes))
                            else:
                                parent = directory_path(parent)
                                digest = hashes.get(str(parent / FINGERPRINT))
                                require(digest is not None, 'dependency fingerprint absent from published request')
                                parents.append((parent, digest, parent_unit, False, child_measurements,
                                    measurement_path or parent_unit in own_measurements))
                        if scientific_edges is not None:
                            # Edge ownership matters: an ancestor's calibration list
                            # cannot classify an ordinary edge owned by this task.
                            scientific_edges[current] = tuple(parent for parent, _, parent_unit, *_ in parents
                                if parent_unit not in own_measurements)
                        if reuse is not None:
                            recorded_inputs.update(reuse.recorded_inputs(request, measurement_path=measurement_path,
                                measurement_parents=[parent for parent, _, parent_unit, *_ in parents
                                    if parent_unit in own_measurements]))
                        _verify_publication_result(result, request, recorded_inputs)
                        verify_budget_authority(current, request, producer_task, allowed_measurements)
                        pending[current] = (result, tree)
                        stack.append((current, expected, unit, True, allowed_measurements, measurement_path))
                        stack.extend(reversed(parents))
                        continue
                if unit is not None:
                    verify_dependency_id(current, unit)
        return verified[root]
    finally:
        active.difference_update(entered)


def verify_dependency_id(root, unit):
    producer = read_mapping(dependency_file(root, '_execution/task.json'))
    require(producer.get('id') == unit, f'dependency producer identity mismatch: {unit}')


@timed_verification("recipe_drift")
def verify_dependency_recipe(root, lock):
    """Bind locked upstream science to this recipe; preserve pre-lock inputs."""
    request = StageRequest.from_json(read_regular(dependency_file(root, '_execution/request.json')))
    task_path = dependency_file(root, '_execution/task.json')
    require(file_hash(task_path) == request.task_hash, 'dependency task hash mismatch')
    task = read_mapping(task_path)
    if not task.get('recipe_lock'):
        return [task_path]
    identity_path = dependency_file(root, '_execution/identity.json')
    identity = read_mapping(identity_path)
    require(identity.get('head') == request.code_identity, 'dependency code identity mismatch')
    require(identity.get('scientific_fingerprint') == lock['scientific_fingerprint'],
        'dependency scientific code/config drift')
    return [task_path, identity_path]


def verify_continuation(task, deps, *, verification=None):
    chain = task.get('continuation')
    if chain is None:
        return
    require(set(chain) == {'owner', 'step', 'predecessor'}, 'invalid continuation fields')
    require(isinstance(chain['owner'], str) and bool(chain['owner']), 'continuation owner missing')
    require(type(chain['step']) is int and chain['step'] >= 0, 'invalid continuation step')
    predecessor = chain['predecessor']
    if chain['step'] == 0:
        require(predecessor is None, 'initial continuation has predecessor')
        return
    require(predecessor in deps, 'continuation predecessor must be an explicit dependency')
    previous = read_mapping(dependency_file(deps[predecessor], '_execution/task.json'))
    prev_chain = previous.get('continuation', {})
    require(previous.get('id') == predecessor, 'continuation predecessor identity mismatch')
    require(prev_chain.get('owner') == chain['owner'], 'continuation ownership mismatch')
    require(prev_chain.get('step') == chain['step'] - 1, 'continuation step is not consecutive')
    require(previous.get('recipe_lock') == task.get('recipe_lock'), 'continuation recipe mismatch')
    for field in ('stage', 'campaign', 'parameters'):
        require(previous.get(field) == task.get(field), f'continuation {field} mismatch')
    verify_dependency_result(deps[predecessor], **(verification or {}))


def verify_only(attempt, scratch, *, successor=False, full_walk=False):
    """Replay sealed declared inputs without loading a lock or executing a stage.

    Recipe comparisons use the sealed consumer's lock, never this checkout's
    fingerprint. This diagnostic cannot authorize a scientific execution.
    """
    attempt, scratch = Path(attempt).resolve(strict=True), Path(scratch).resolve(strict=False)
    require(not scratch.exists(), 'diagnostic scratch must be new')
    require(not scratch.is_relative_to(attempt), 'diagnostic scratch overlaps attempt')
    with verification_diagnostic():
        terminal = StageResult.from_json(read_regular(attempt / RESULT))
        require(terminal.status == 'pass', 'diagnostic requires a sealed passing attempt')
        own_tree = verify_published_tree(attempt, terminal)
        request = StageRequest.from_json(read_regular(attempt / '_execution/request.json'))
        require(request.content_hash == terminal.request_hash, 'diagnostic request mismatch')
        config, task = read_mapping(request.config_path), read_mapping(request.task_path)
        deps = {unit: Path(root) for unit, root in config.get('dependencies', {}).items()}
        require(not config.get('acquisitions'), 'diagnostic requires stage-publication inputs')
        require(all(not scratch.is_relative_to(root) and not root.is_relative_to(scratch)
            for root in deps.values()), 'diagnostic scratch overlaps dependency')
        hashes = dict(zip(request.dependency_paths, request.dependency_hashes))
        if successor:
            # The sealed lock declares the next leaf even when the campaign was
            # stopped before that leaf could publish. Never invent its task.
            ref = task['recipe_lock']
            expansion_path = deps[ref['dependency']] / 'expanded_units.json'
            verify_input_hash(expansion_path, hashes[str(expansion_path)])
            expansion = read_mapping(expansion_path)
            candidates = [leaf for plan in expansion['plans'] for leaf in plan['tasks']
                if leaf.get('continuation', {}).get('predecessor') == task['id']]
            require(len(candidates) == 1, 'diagnostic requires one lock-declared successor')
            previous_task = task
            task = candidates[0]
            deps = {unit: root for unit, root in deps.items() if unit in task['needs']}
            deps[previous_task['id']] = attempt
            require(set(deps) == set(task['needs']), 'diagnostic successor input set unavailable')
            hashes[str(attempt / FINGERPRINT)] = next(
                a.sha256 for a in terminal.artifacts if a.path == FINGERPRINT)
        trees, verified, edges, budget_authorities = {}, {}, {}, set()
        with verification_timings() as timings:
            reuse = None if full_walk else dependency_reuse(task, deps, output_dir=scratch)
            rewalk_receipt_misses(reuse, trees, verified, scratch, task, edges, budget_authorities)
            for unit, root in deps.items():
                verify_dependency_result(root, expected_hash=hashes[str(root / FINGERPRINT)],
                    trees=trees, verified=verified, output_dir=scratch, consumer_task=task,
                    budget_authorities=budget_authorities, scientific_edges=edges, reuse=reuse)
                verify_dependency_id(root, unit)
            rewalk_receipt_misses(reuse, trees, verified, scratch, task, edges, budget_authorities)
            primary_seconds = time.monotonic()
            verify_continuation(task, deps, verification={'trees': trees, 'verified': verified,
                'budget_authorities': budget_authorities, 'scientific_edges': edges, 'reuse': reuse}
                if reuse is not None else None)
            continuation_seconds = time.monotonic() - primary_seconds
            if task.get('recipe_lock'):
                ref = task['recipe_lock']
                lock_path = dependency_file(deps[ref['dependency']], ref['path'])
                require(file_hash(lock_path) == ref['sha256'], 'diagnostic recipe hash mismatch')
                lock = read_mapping(lock_path)
                for root in scientific_ancestors(task, deps, edges) & set(verified):
                    verify_dependency_recipe(root, lock)
            if successor or reuse is not None:
                for unit, root in deps.items():
                    for name in task['needs'][unit]:
                        path = root / name
                        expected = own_tree[name]['sha256'] if root == attempt else hashes[str(path)]
                        verify_input_hash(path, expected)
            else:
                verify_inputs(request)
        timings['continuation_walk_seconds'] = continuation_seconds
        timings['verified_roots'] = len(verified)
        if reuse is not None:
            reuse.annotate(timings, verified)
        value = {'diagnostic': True, 'successor': successor, 'status': 'pass', 'attempt': str(attempt),
            'task_id': task['id'], 'continuation': task.get('continuation'),
            'request_hash': request.content_hash, 'timings': timings}
        scratch.mkdir(parents=True)
        atomic_json(scratch, DEPENDENCY_CHECK, value)
        return value


def finalize_coverage_budget(request, result, started):
    """Reconcile worker accounting after verification; seal only in publish_result.

    The record write and immutable terminal publication spend an explicit part
    of the existing checkpoint reserve. No budget consumer accepts this record
    until the independent publication receipt binds the terminal StageResult.
    """
    if request.stage not in ('coverage', 'simulation-smoke'):
        return result
    from oxyformer.validation.coverage import (BUDGET_RECORD, MEASUREMENTS, OVERHEAD_PHASES,
        budget_binding, finite, publication_reserve, reserved_wall_limit, remaining_compute, uses_continuation, task_leaf_budget)
    from oxyformer.provenance import ArtifactRecord
    artifacts = [a for a in result.artifacts if a.path == MEASUREMENTS]
    if not artifacts:
        require(result.status != 'pass', 'coverage worker measurements missing')
        return result
    require(len(artifacts) == 1, 'duplicate coverage measurements')
    path = Path(request.output_dir) / MEASUREMENTS
    require(file_hash(path) == artifacts[0].sha256, 'coverage measurements hash mismatch')
    raw = read_mapping(path)
    task = read_mapping(request.task_path)
    timing = raw['timing']
    binding = budget_binding(request, task, timing['recipe_hash'])
    require(raw['binding'] == binding, 'coverage measurement binding mismatch')
    chained = uses_continuation(task)
    history = timing['leaves'][:-1] if chained else []
    if chained:
        leaf = dict(timing['leaves'][-1])
        leaf['leaf_budget'] = task_leaf_budget(task)
        leaf['measured_overheads'] = task['parameters'].get('leaf_budget') is not None
    else:
        leaf = {'task_id': task['id'], 'step': 0, 'compute_seconds': sum(timing['complete_repetition_seconds']),
            'setup_seconds': timing['setup_seconds'], 'import_seconds': 0., 'export_seconds': 0.}
    # Runner preparation precedes the worker clock; charge it once as setup.
    leaf['setup_seconds'] += max(0., raw['started_monotonic'] - started)
    if leaf.get('setup_safety'):
        from oxyformer.validation.coverage import setup_overruns
        leaf['setup_safety']['measured'] = setup_overruns([*history, leaf],
            leaf['setup_safety']['allowance_seconds'])
    check_path = Path(request.output_dir) / DEPENDENCY_CHECK
    verification = read_mapping(check_path).get('timings', {}) if check_path.is_file() else {}
    if 'ancestors_reused' in verification:
        leaf['verification'] = {key: verification[key] for key in
            ('ancestors_reused', 'ancestors_rewalked', 'predecessor_publications_added', 'reuse_miss')}
        if verification['reuse_miss']:
            leaf['verification']['status'] = 'reuse-miss'
    elapsed = time.monotonic() - started
    leaf['wall_seconds'] = elapsed
    leaf['publication_seconds'] = elapsed - sum(leaf[k] for k in
        ('setup_seconds', 'import_seconds', 'compute_seconds', 'export_seconds'))
    wall = finite(binding['wall_seconds'], 'coverage wall')
    leaf['post_check_reserve_seconds'] = publication_reserve(wall)
    exceeded = elapsed > reserved_wall_limit(wall)
    if chained:
        phases = OVERHEAD_PHASES if leaf['measured_overheads'] else ('setup_seconds', 'import_seconds')
        exceeded |= leaf['compute_seconds'] > remaining_compute(leaf['leaf_budget'], **{k: leaf[k] for k in phases})
    require(leaf['publication_seconds'] >= 0, 'coverage accounting exceeds measured runner interval')
    # This is the last wall check. Subsequent record/publication writes are
    # reserved, and no code mutates the terminal result after it is sealed.
    if exceeded:
        result = replace(result, status='fail', message='Coverage final wall/compute budget exceeded')
    counts = raw['counts']
    complete = raw['received'] == len(binding['draws']) and counts['incomplete'] == 0
    if not chained and binding['mode'] == 'profile' and timing['device'].startswith('cuda'):
        if not complete or counts['success'] != len(binding['draws']):
            result = replace(result, status='fail', message='CUDA profile incomplete or unsuccessful; no usable profile')
    value = {'schema_version': 1, 'binding': binding, 'chained': chained,
        'leaves': [*history, leaf], 'counts': counts, 'received': raw['received'],
        'complete': complete, 'success': complete and counts['success'] == len(binding['draws']),
        'profile': {k: timing[k] for k in ('device', 'environment', 'production_equivalent',
            'complete_repetition_seconds', 'scientific_fingerprint', 'environment_hash')},
        'measurement_scope': 'runner_final_wall_check',
        'excluded': ['reserved budget-record and terminal StageResult publication']}
    record_path = atomic_json(request.output_dir, BUDGET_RECORD, value)
    artifact = ArtifactRecord(path=BUDGET_RECORD, kind='coverage_budget', sha256=file_hash(record_path),
        lineage=artifacts[0].lineage)
    require(not any(a.path == BUDGET_RECORD for a in result.artifacts), 'worker supplied runner budget record')
    return replace(result, artifacts=(*result.artifacts, artifact))


def run(stage, out, repo, *, deps_env=False, task_file=None, task_id=None, approvals=None, execute=None):
    started = time.monotonic()
    out = Path(out).absolute()
    require(out.is_dir() and not out.is_symlink(), 'output must be an existing attempt directory')
    out = out.resolve(strict=True)
    if 'SWARM_UNIT_DIR' in os.environ:
        require(out == Path(os.environ['SWARM_UNIT_DIR']).resolve(strict=True),
            '--out must equal SWARM_UNIT_DIR')
    repo = Path(repo).resolve(strict=True)
    require(not out.is_relative_to(repo), 'output may not be inside repository')
    thread_environment = configure_threads()
    head = code_identity(repo, out)
    registry_file = repo / 'configs/execution/stages.yaml'
    registry = read_mapping(registry_file)
    require(registry.get('schema_version') == 1, 'unsupported registry version')
    require(stage in registry['stages'], f'unknown stage: {stage}')
    settings = registry['stages'][stage]
    approvals_file = Path(approvals) if approvals else repo / 'configs/approvals.yaml'
    approvals_value = read_mapping(approvals_file, expected_bytes=git_bytes(repo, 'show', 'HEAD:configs/approvals.yaml'))
    require(approvals_value.get('schema_version') == 1 and approvals_value.get('approved_by'),
        'owner approvals are missing')
    sources = {str(registry_file): file_hash(registry_file),
        str(approvals_file.resolve()): file_hash(approvals_file)}
    if task_file is not None:
        task_file = Path(task_file).absolute()
        document = read_mapping(task_file)
        sources[str(task_file)] = file_hash(task_file)
        if 'tasks' in document:
            tasks = document['tasks']
            ids = [t['id'] for t in tasks]
            require(len(set(ids)) == len(ids), 'duplicate task IDs')
            require(task_id in ids, 'task-id required and must select a concrete task')
            task = dict(tasks[ids.index(task_id)])
        else:
            task = dict(document)
            require(task_id is None or task.get('id') == task_id, 'task-id mismatch')
    else:
        require(task_id is None, '--task-id requires --task')
        task = {'id': settings.get('unit_id', stage), 'stage': stage, 'needs': settings.get('needs', {}),
            'outputs': settings.get('outputs', [])}
    require(task.get('stage') == stage, 'task stage mismatch')
    task.setdefault('needs', settings.get('needs', {}))
    task.setdefault('outputs', settings.get('outputs', []))
    for unit, paths in settings.get('needs', {}).items():
        require(unit in task['needs'] and set(paths) <= set(task['needs'][unit]),
            'task omitted stage-required dependency')
    require(set(settings.get('outputs', [])) <= set(task['outputs']),
        'task omitted stage-required output')
    measurement_dependencies(task)
    measurement_only_dependencies = set()
    needs = task.get('needs', {})
    require(isinstance(needs, dict), 'task needs must map IDs to relative files')
    require(deps_env or not needs, 'dependencies require --deps-env')
    deps = resolve_dependencies(needs) if needs else {}
    files = []
    published_hashes = {}
    acquisition_hashes = {}
    dependency_trees = {}
    verified_dependencies = {}
    scientific_edges = {}
    budget_authorities = set()
    acquisitions = {}
    with verification_timings() as timings:
        reuse = dependency_reuse(task, deps, output_dir=out)
        rewalk_receipt_misses(reuse, dependency_trees, verified_dependencies, out, task,
            scientific_edges, budget_authorities)
        for unit, root in deps.items():
            require(not out.is_relative_to(root) and not root.is_relative_to(out),
                f'output overlaps an upstream attempt: {root}')
            require(isinstance(needs[unit], list) and needs[unit], 'dependency requires explicit files')
            acquisition_receipt = settings.get('acquisition_receipts', {}).get(unit)
            if acquisition_receipt is not None:
                require(acquisition_receipt in needs[unit], 'acquisition receipt must be a declared input')
                require('payload.tar' in needs[unit], 'acquisition payload must be a declared input')
                dependency_trees[str(root)] = verify_acquisition(root, acquisition_receipt)
                acquisitions[unit] = acquisition_receipt
            else:
                # Stopped leaves never create scientific payloads. Report their
                # terminal refusal before checking for those absent files.
                with verification_span(root, 'lineage_walk'):
                    require((root / RESULT).is_file(), f'stage receipt missing or not regular: {root / RESULT}')
                    terminal = StageResult.from_json(read_regular(root / RESULT))
                    require(terminal.status == 'pass', 'chain stopped: allowance-exceeded'
                        if 'allowance-exceeded' in terminal.message else 'dependency stage did not pass')
            for relative in needs[unit]:
                path = dependency_file(root, relative)
                files.append(path)
                if acquisition_receipt is not None:
                    acquisition_hashes[path] = dependency_trees[str(root)][relative]['sha256']
            if acquisition_receipt is not None:
                continue
            result = verify_dependency_result(root, trees=dependency_trees, verified=verified_dependencies,
                output_dir=out, consumer_task=task, measurement_only_dependencies=measurement_only_dependencies,
                budget_authorities=budget_authorities, scientific_edges=scientific_edges, reuse=reuse)
            verify_dependency_id(root, unit)
            published_hashes[root / FINGERPRINT] = next(
                a.sha256 for a in result.artifacts if a.path == FINGERPRINT)
            allowed = {a.path for a in result.artifacts} | {
                '_execution/task.json', '_execution/request.json', '_execution/result.json',
                '_execution/environment.json', '_execution/identity.json',
                DEPENDENCY_CHECK, '_execution/dependencies.json'}
            require(set(needs[unit]) <= allowed, 'dependency file not declared by passing stage')
            producer = StageRequest.from_json(read_regular(root / '_execution/request.json'))
            budget_files = ('_execution/task.json', 'budget_record.json') if producer.stage in ('coverage', 'simulation-smoke') else ()
            for relative in ('_execution/request.json', '_execution/result.json', FINGERPRINT, *budget_files):
                path = dependency_file(root, relative)
                if path not in files:
                    files.append(path)
        rewalk_receipt_misses(reuse, dependency_trees, verified_dependencies, out, task,
            scientific_edges, budget_authorities)
        require(len(set(files)) == len(files), 'duplicate dependency files')
        for relative in task.get('outputs', []):
            require(not relative.startswith('_execution/'), 'reserved execution output')
            path = output_path(out, relative)
            require(not path.is_relative_to(repo), 'output overlaps cloned repository')
            require(not path.exists(), f'output already exists: {relative}')
        output_path(out, '_execution').mkdir()
        verify_continuation(task, deps, verification={'trees': dependency_trees,
            'verified': verified_dependencies, 'budget_authorities': budget_authorities,
            'scientific_edges': scientific_edges, 'reuse': reuse} if reuse is not None else None)
        lock_ref = task.get('recipe_lock')
        if settings.get('requires_recipe', False) or 'campaign' in task:
            require(bool(lock_ref), 'locked stage or campaign task requires a recipe lock')
        if lock_ref:
            require(set(lock_ref) == {'dependency', 'path', 'sha256'}, 'invalid recipe reference')
            require(lock_ref['dependency'] in deps, 'recipe lock dependency missing')
            lock_file = dependency_file(deps[lock_ref['dependency']], lock_ref['path'])
            require(lock_file in files, 'recipe lock must be a declared input')
            require(file_hash(lock_file) == lock_ref['sha256'], 'recipe lock hash mismatch')
            lock = read_mapping(lock_file)
            verify_recipe(repo, lock)
            # Compute reachability afresh for this consumer, across all its roots.
            # A record is scientific if ANY path to it uses only ordinary edges.
            scientific = scientific_ancestors(task, deps, scientific_edges, acquisitions)
            for root in verified_dependencies:
                if root not in scientific:
                    continue
                for path in verify_dependency_recipe(root, lock):
                    if path not in files:
                        files.append(path)
            require(file_hash(approvals_file) == file_hash(repo / 'configs/approvals.yaml'),
                'locked approvals differ from fingerprinted repository config')
    timings['verified_roots'] = len(verified_dependencies)
    if reuse is not None:
        reuse.annotate(timings, verified_dependencies)
    attested = reuse.reused(verified_dependencies) if reuse is not None else {}
    # Persist the full receipt for successors; only currently consumed trees
    # enter the current-byte post-check and publication-authority check.
    receipt_trees = {**attested, **dependency_trees}
    config = {'stage': stage, 'settings': settings, 'approvals': approvals_value,
        'input_sources': sources, 'dependencies': {k: str(v) for k, v in deps.items()},
        'acquisitions': acquisitions, 'yaml_timestamp_policy': 'preserve_scalar_text'}
    if stage == 'campaign-setup-probe':
        config['verification_summary'] = {key: timings[key] for key in
            ('verified_roots', 'ancestors_reused', 'ancestors_rewalked',
             'predecessor_publications_added', 'reuse_miss') if key in timings}
    if stage in ('coverage', 'simulation-smoke'):
        config['coverage_started_monotonic'] = started
    config_path = atomic_json(out, '_execution/config.json', config)
    task_path = atomic_json(out, '_execution/task.json', task)
    tree_path = atomic_json(out, '_execution/dependencies.json', receipt_trees)
    files.append(tree_path)
    request = StageRequest(stage=stage, config_path=str(config_path), config_hash=file_hash(config_path),
        task_path=str(task_path), task_hash=file_hash(task_path),
        dependency_paths=tuple(map(str, files)),
        dependency_hashes=tuple(published_hashes[p] if p in published_hashes
            else verify_input_hash(p, acquisition_hashes[p], hash_file=file_hash)
            if p in acquisition_hashes else file_hash(p) for p in files),
        output_dir=str(out), code_identity=head)
    atomic_write(out, '_execution/request.json', request.to_json())
    environment = environment_record()
    environment['threads'] = thread_environment
    atomic_json(out, '_execution/environment.json', environment)
    atomic_json(out, '_execution/identity.json', {'head': head, 'scientific_fingerprint': scientific_fingerprint(repo)})
    immutable_controls = {str(out / ('_execution/' + name)): file_hash(out / ('_execution/' + name))
        for name in ('request.json', 'environment.json', 'identity.json')}
    with observe_dependencies(out, receipt_trees):
        dependency_identities = snapshot_dependency_identities(dependency_trees)
        try:
            verify_inputs(request)
            module_name = settings.get('module')
            require(isinstance(module_name, str) and module_name.startswith('oxyformer.'),
                'stage module not registered')
            if execute is None:
                from .worker import execute
                module_name = 'oxyformer.execution.integrity'
            result = execute(request, module_name, repo)
            require(isinstance(result, StageResult), 'stage did not return StageResult')
            if measurement_only_dependencies and result.status == 'pass':
                from oxyformer.provenance import ArtifactRecord
                measurement_record = {'measurement_only_dependencies': sorted(measurement_only_dependencies)}
                estimate_path = out / 'budget_estimate.json'
                if stage == 'campaign-estimate' and estimate_path.is_file():
                    scientific = scientific_ancestors(task, deps, scientific_edges, acquisitions)
                    measured = {name: tree for name, tree in dependency_trees.items()
                        if Path(name) not in scientific and FINGERPRINT in tree}
                    measurement_record.update(schema_version=2,
                        records=read_mapping(estimate_path).get('calibration', {}).get('records', []),
                        dependency_receipt={'snapshot': measured, 'check': {'status': 'pass',
                            'attempts': {name: {'status': 'unchanged', 'changed_paths': [],
                                'tainted_paths': [], 'fingerprint': tree} for name, tree in measured.items()}}})
                path = atomic_json(out, 'measurement_only_dependencies.json',
                    measurement_record)
                result = replace(result, artifacts=(*result.artifacts, ArtifactRecord(
                    path=path.name, kind='measurement_only_dependencies', sha256=file_hash(path),
                    lineage=result.artifacts[0].lineage)))
        except BaseException as exc:
            result = StageResult(request_hash=request.content_hash, status='fail', artifacts=(),
                message=str(exc).strip() or type(exc).__name__)
        changed = []
        try:
            check = post_execution_check(dependency_trees, observed_changes=changed, identities=dependency_identities)
            check['timings'] = timings
            # Another consumer may have observed a write that was restored before
            # this final snapshot. Such observed taints remain permanent.
            for root, detail in check['attempts'].items():
                marker = Path(str(publication_receipt(root)) + '.tainted')
                if authority_exists(marker):
                    observed = json.loads(read_regular(marker))
                    changed.extend(str(Path(root) / name) for name in observed)
                    detail['changed_paths'] = sorted(set(detail['changed_paths']) | set(observed))
                    detail['tainted_paths'] = sorted(set(detail['tainted_paths']) | set(observed))
                    detail['status'] = 'tainted'
                    check['status'] = 'fail'
            changed = [str(Path(root) / name) for root, detail in check['attempts'].items()
                for name in detail['changed_paths']]
            for root, tree in attested.items():
                check['attempts'][root] = {'status': 'unchanged', 'changed_paths': [],
                    'tainted_paths': [], 'fingerprint': tree,
                    'attested_by': reuse.sources[Path(root)]}
            record_taints(check)
            control_directory_changed = _repair_control_directory(out)
            collisions = [str(out / name) for name in (DEPENDENCY_CHECK, RESULT, FINGERPRINT)
                if os.path.lexists(out / name)]
            if control_directory_changed:
                collisions.append(str(out / '_execution'))
            _replace_control(out, DEPENDENCY_CHECK, canonical_json(check))
            if check['status'] == 'fail':
                result = StageResult(request_hash=request.content_hash, status='fail', artifacts=(),
                    message=('upstream attempt changed; changed paths: ' + ', '.join(changed)) if changed
                    else 'upstream fingerprint unreadable: ' + ', '.join(root for root, detail in
                        check['attempts'].items() if detail['status'] == 'unreadable'))
            else:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    require(not collisions, 'reserved execution control collision: ' + ', '.join(collisions))
                    for path, digest in immutable_controls.items():
                        require(file_hash(path) == digest, f'execution control changed: {path}')
                    # Trees cover publication bytes; their independent release
                    # receipts must also remain valid after the worker exits.
                    for root, dependency_result in verified_dependencies.items():
                        verify_publication(root, dependency_result)
                    verify_result(result, request)
                    require(all(not (out / a.path).resolve().is_relative_to(repo) for a in result.artifacts),
                        'artifact overlaps cloned repository')
                    require(all(not a.path.startswith('_execution/') for a in result.artifacts), 'reserved execution artifact')
                    declared = set(task.get('outputs', []))
                    if result.status == 'pass':
                        runner_outputs = ({'budget_record.json'}
                            if request.stage in ('coverage', 'simulation-smoke') else set())
                        require(declared - runner_outputs <= {a.path for a in result.artifacts}, 'stage omitted declared outputs')
                    for path, digest in sources.items():
                        require(file_hash(path) == digest, f'input source changed: {path}')
                    observed_head = code_identity(repo, out)
                    require(observed_head == request.code_identity,
                        f'code identity changed: recorded {request.code_identity}, observed {observed_head}')
                except BaseException as exc:
                    result = StageResult(request_hash=request.content_hash, status='fail', artifacts=(),
                        message=str(exc).strip() or type(exc).__name__)
            result = finalize_coverage_budget(request, result, started)
            return publish_result(out, result, owned_controls=True, dependency_roots=dependency_trees,
                dependency_identities=dependency_identities)
        except BaseException as exc:
            message = 'stage finalization failed: ' + (str(exc).strip() or type(exc).__name__)
            if changed:
                message += '; upstream attempt changed; changed paths: ' + ', '.join(changed)
            print('failed: ' + message, file=sys.stderr)
            return StageResult(request_hash=request.content_hash, status='fail', artifacts=(), message=message)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Diagnostic dependency verification only; never executes or seals.')
    parser.add_argument('--verify-only', required=True, metavar='ATTEMPT')
    parser.add_argument('--scratch', required=True, metavar='NEW_DIRECTORY')
    parser.add_argument('--successor', action='store_true', help='Verify the lock-declared next leaf using this sealed predecessor.')
    parser.add_argument('--full-walk', action='store_true', help='Diagnostic comparison: disable resumed receipt reuse.')
    arguments = parser.parse_args()
    report = verify_only(arguments.verify_only, arguments.scratch, successor=arguments.successor,
        full_walk=arguments.full_walk)
    print(canonical_json(report))
