"""Detect upstream state changes after worker exit; never repair upstream.
Managed outputs are confined; general write prevention is deferred to ARC-1339.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys

import yaml

from oxyformer.contracts import StageRequest, StageResult
from oxyformer.provenance import ContractError, canonical_json, relative_artifact_path, require
from .integrity import (DEPENDENCY_CHECK, FINGERPRINT, RESULT, _replace_control, _repair_control_directory,
    post_execution_check, publish_result, record_taints,
    directory_path, read_regular, regular_file_stat, regular_file_hash as file_hash,
    fingerprint_tree, changed_paths, publication_receipt,
    verify_inputs, verify_result, verify_published_tree)
from .identity import git_bytes, code_identity, environment_record, scientific_fingerprint, verify_recipe
from .paths import atomic_json, atomic_write, output_path


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
            value = yaml.safe_load(text)
        except yaml.YAMLError as yaml_error:
            raise ContractError(f'invalid YAML: {path}') from yaml_error
    require(isinstance(value, dict), f'expected mapping: {path}')
    return value


def dependency_file(root, relative):
    relative_artifact_path(relative)
    path = Path(root) / relative
    regular_file_stat(path)
    require(path.resolve(strict=True).is_relative_to(root), f'dependency file escapes attempt: {path}')
    return path


def verify_acquisition(root, receipt_name, *, expected_tree=None):
    """Bind a complete acquisition to a create-once external tree baseline.

    Acquisition producers predate StageResult. Never write into their attempts;
    the independent publication store retains the first validated tree and the
    same permanent taint markers used for stage publications.
    """
    receipt_path = dependency_file(root, receipt_name)
    dependency_file(root, 'payload.tar')
    receipt_bytes = read_regular(receipt_path)
    receipt = json.loads(receipt_bytes)
    require(isinstance(receipt, dict), 'acquisition receipt must be a mapping')
    require(receipt.get('status') == 'complete', 'acquisition receipt is not complete')
    authority = publication_receipt(root, create=True)
    require(not os.path.lexists(str(authority) + '.tainted'), f'tainted upstream fingerprint: {root}')
    tree = fingerprint_tree(root)
    require(not any('error' in entry for entry in tree.values()), 'acquisition fingerprint unreadable')
    require(tree[receipt_name]['sha256'] == sha256(receipt_bytes).hexdigest(), 'acquisition receipt changed during verification')
    require(tree['payload.tar']['sha256'] == receipt.get('payload_sha256'), 'acquisition payload hash mismatch')
    require(tree['payload.tar']['size'] == receipt.get('payload_bytes'), 'acquisition payload size mismatch')
    if expected_tree is not None:
        require(not changed_paths(expected_tree, tree), 'acquisition fingerprint differs from consumer baseline')
    baseline = Path(str(authority) + '.acquisition')
    value = {'attempt': str(root), 'receipt': receipt_name, 'entries': tree}
    try:
        atomic_json(baseline.parent, baseline.name, value)
    except FileExistsError:
        require(read_mapping(baseline) == value, f'acquisition fingerprint mismatch (tainted): {root}')
    return tree


def verify_dependency_result(root, *, expected_hash=None, trees=None, active=None, verified=None,
    output_dir=None):
    """Verify the complete lineage with an explicit postorder traversal."""
    root = directory_path(root)
    active = set() if active is None else active
    verified = {} if verified is None else verified
    entered = set()
    pending = {}
    stack = [(root, expected_hash, None, False)]
    try:
        while stack:
            current, expected, unit, ready = stack.pop()
            if ready:
                result, tree = pending.pop(current)
                if trees is not None:
                    trees[str(current)] = tree
                verified[current] = result
                active.remove(current)
            else:
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
                else:
                    active.add(current)
                    entered.add(current)
                    request = StageRequest.from_json(read_regular(dependency_file(current, '_execution/request.json')))
                    result = StageResult.from_json(read_regular(dependency_file(current, RESULT)))
                    require(Path(request.output_dir).resolve() == current, 'dependency attempt owner mismatch')
                    require(result.status == 'pass', 'dependency stage did not pass')
                    tree = verify_published_tree(current, result, expected)
                    verify_result(result, request)
                    config = read_mapping(request.config_path)
                    hashes = dict(zip(request.dependency_paths, request.dependency_hashes))
                    parents = []
                    for parent_unit, parent in config.get('dependencies', {}).items():
                        parent = Path(parent)
                        require(parent.is_absolute(), 'dependency publication path must be absolute')
                        parent = directory_path(parent)
                        acquisition = config.get('acquisitions', {}).get(parent_unit)
                        if acquisition is not None:
                            if output_dir is not None:
                                require(not output_dir.is_relative_to(parent) and not parent.is_relative_to(output_dir),
                                    f'output overlaps an upstream attempt: {parent}')
                            snapshots = read_mapping(current / '_execution/dependencies.json')
                            require(str(parent) in snapshots, 'acquisition baseline absent from published request')
                            require(str(current / '_execution/dependencies.json') in hashes,
                                'acquisition baseline not bound to published request')
                            acquisition_tree = verify_acquisition(parent, acquisition, expected_tree=snapshots[str(parent)])
                            if trees is not None:
                                trees[str(parent)] = acquisition_tree
                        else:
                            digest = hashes.get(str(parent / FINGERPRINT))
                            require(digest is not None, 'dependency fingerprint absent from published request')
                            parents.append((parent, digest, parent_unit, False))
                    pending[current] = (result, tree)
                    stack.append((current, expected, unit, True))
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


def verify_continuation(task, deps):
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
    verify_dependency_result(deps[predecessor])


def run(stage, out, repo, *, deps_env=False, task_file=None, task_id=None, approvals=None, execute=None):
    out = Path(out).absolute()
    require(out.is_dir() and not out.is_symlink(), 'output must be an existing attempt directory')
    out = out.resolve(strict=True)
    if 'SWARM_UNIT_DIR' in os.environ:
        require(out == Path(os.environ['SWARM_UNIT_DIR']).resolve(strict=True),
            '--out must equal SWARM_UNIT_DIR')
    repo = Path(repo).resolve(strict=True)
    require(not out.is_relative_to(repo), 'output may not be inside repository')
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
    needs = task.get('needs', {})
    require(isinstance(needs, dict), 'task needs must map IDs to relative files')
    require(deps_env or not needs, 'dependencies require --deps-env')
    deps = resolve_dependencies(needs) if needs else {}
    files = []
    published_hashes = {}
    dependency_trees = {}
    verified_dependencies = {}
    acquisitions = {}
    for unit, root in deps.items():
        require(not out.is_relative_to(root) and not root.is_relative_to(out),
            f'output overlaps an upstream attempt: {root}')
        require(isinstance(needs[unit], list) and needs[unit], 'dependency requires explicit files')
        for relative in needs[unit]:
            files.append(dependency_file(root, relative))
        acquisition_receipt = settings.get('acquisition_receipts', {}).get(unit)
        if acquisition_receipt is not None:
            require(acquisition_receipt in needs[unit], 'acquisition receipt must be a declared input')
            require('payload.tar' in needs[unit], 'acquisition payload must be a declared input')
            dependency_trees[str(root)] = verify_acquisition(root, acquisition_receipt)
            acquisitions[unit] = acquisition_receipt
            continue
        require((root / '_execution/result.json').is_file(), f'stage receipt missing or not regular: {root / "_execution/result.json"}')
        result = verify_dependency_result(root, trees=dependency_trees, verified=verified_dependencies,
            output_dir=out)
        verify_dependency_id(root, unit)
        published_hashes[root / FINGERPRINT] = next(
            a.sha256 for a in result.artifacts if a.path == FINGERPRINT)
        allowed = {a.path for a in result.artifacts} | {
            '_execution/task.json', '_execution/request.json', '_execution/result.json',
            '_execution/environment.json', '_execution/identity.json'}
        require(set(needs[unit]) <= allowed, 'dependency file not declared by passing stage')
        for relative in ('_execution/request.json', '_execution/result.json', FINGERPRINT):
            path = dependency_file(root, relative)
            if path not in files:
                files.append(path)
    require(len(set(files)) == len(files), 'duplicate dependency files')
    for relative in task.get('outputs', []):
        require(not relative.startswith('_execution/'), 'reserved execution output')
        path = output_path(out, relative)
        require(not path.is_relative_to(repo), 'output overlaps cloned repository')
        require(not path.exists(), f'output already exists: {relative}')
    output_path(out, '_execution').mkdir()
    verify_continuation(task, deps)
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
        for root in verified_dependencies:
            for path in verify_dependency_recipe(root, lock):
                if path not in files:
                    files.append(path)
        require(file_hash(approvals_file) == file_hash(repo / 'configs/approvals.yaml'),
            'locked approvals differ from fingerprinted repository config')
    config = {'stage': stage, 'settings': settings, 'approvals': approvals_value,
        'input_sources': sources, 'dependencies': {k: str(v) for k, v in deps.items()},
        'acquisitions': acquisitions}
    config_path = atomic_json(out, '_execution/config.json', config)
    task_path = atomic_json(out, '_execution/task.json', task)
    tree_path = atomic_json(out, '_execution/dependencies.json', dependency_trees)
    files.append(tree_path)
    request = StageRequest(stage=stage, config_path=str(config_path), config_hash=file_hash(config_path),
        task_path=str(task_path), task_hash=file_hash(task_path),
        dependency_paths=tuple(map(str, files)),
        dependency_hashes=tuple(published_hashes[p] if p in published_hashes
            else file_hash(p) for p in files),
        output_dir=str(out), code_identity=head)
    atomic_write(out, '_execution/request.json', request.to_json())
    environment = environment_record()
    atomic_json(out, '_execution/environment.json', environment)
    atomic_json(out, '_execution/identity.json', {'head': head, 'scientific_fingerprint': scientific_fingerprint(repo)})
    immutable_controls = {str(out / ('_execution/' + name)): file_hash(out / ('_execution/' + name))
        for name in ('request.json', 'environment.json', 'identity.json')}
    try:
        verify_inputs(request)
        module_name = settings.get('module')
        require(isinstance(module_name, str) and module_name.startswith('oxyformer.'),
            'stage module not registered')
        if execute is None:
            from .worker import execute
        result = execute(request, module_name, repo)
        require(isinstance(result, StageResult), 'stage did not return StageResult')
    except BaseException as exc:
        result = StageResult(request_hash=request.content_hash, status='fail', artifacts=(),
            message=str(exc).strip() or type(exc).__name__)
    changed = []
    try:
        check = post_execution_check(dependency_trees)
        changed = [str(Path(root) / name) for root, detail in check['attempts'].items()
            for name in detail['changed_paths']]
        record_taints(check)
        control_directory_changed = _repair_control_directory(out)
        collisions = [str(out / name) for name in (DEPENDENCY_CHECK, RESULT, FINGERPRINT)
            if os.path.lexists(out / name)]
        if control_directory_changed:
            collisions.append(str(out / '_execution'))
        _replace_control(out, DEPENDENCY_CHECK, canonical_json(check))
        if check['status'] == 'fail':
            result = StageResult(request_hash=request.content_hash, status='fail', artifacts=(),
                message='upstream attempt tainted; changed paths: ' + ', '.join(changed))
        else:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                require(not collisions, 'reserved execution control collision: ' + ', '.join(collisions))
                for path, digest in immutable_controls.items():
                    require(file_hash(path) == digest, f'execution control changed: {path}')
                verify_result(result, request)
                require(all(not (out / a.path).resolve().is_relative_to(repo) for a in result.artifacts),
                    'artifact overlaps cloned repository')
                require(all(not a.path.startswith('_execution/') for a in result.artifacts), 'reserved execution artifact')
                declared = set(task.get('outputs', []))
                if result.status == 'pass':
                    require(declared <= {a.path for a in result.artifacts}, 'stage omitted declared outputs')
                for path, digest in sources.items():
                    require(file_hash(path) == digest, f'input source changed: {path}')
                observed_head = code_identity(repo, out)
                require(observed_head == request.code_identity,
                    f'code identity changed: recorded {request.code_identity}, observed {observed_head}')
            except BaseException as exc:
                result = StageResult(request_hash=request.content_hash, status='fail', artifacts=(),
                    message=str(exc).strip() or type(exc).__name__)
        return publish_result(out, result, owned_controls=True)
    except BaseException as exc:
        message = 'stage finalization failed: ' + (str(exc).strip() or type(exc).__name__)
        if changed:
            message += '; upstream attempt tainted; changed paths: ' + ', '.join(changed)
        print('failed: ' + message, file=sys.stderr)
        return StageResult(request_hash=request.content_hash, status='fail', artifacts=(), message=message)
