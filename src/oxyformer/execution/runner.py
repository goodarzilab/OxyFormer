"""Trusted stage dispatch, with immutable inputs and attempt-owned writes.

This is an execution contract, not a sandbox for hostile Python modules. Stage
implementations must use output_dir for all writes and treat dependency_paths as
read-only. The runner never modifies upstream files or their permissions; input
trees are fingerprinted before execution and rechecked after the stage worker
exits. Fingerprints bind directory entries, types, modes, sizes, content hashes
and symlink targets; timestamps and inode numbers are deliberately excluded.
An identical rewrite or restored input state is accepted, not tracked as an
event. Archives are extracted only via execution.paths.safe_extract into the
consuming attempt. Runner-managed writes, receipts and declared artifacts are
confined to output_dir, and caches are redirected there. General confinement of
faulty stage code writing outside every attempt tree is the explicitly deferred
host-runtime follow-up ARC-1339; such undeclared writes are not detected here.
"""
import json
import os
from pathlib import Path
import re
import sys

import yaml

from oxyformer.contracts import StageRequest, StageResult
from oxyformer.provenance import ContractError, canonical_json, relative_artifact_path, require
from .integrity import (DEPENDENCY_CHECK, FINGERPRINT, RESULT, _replace_control, _repair_control_directory,
                        post_execution_check, publish_result,
                        directory_path, read_regular, regular_file_stat, regular_file_hash as file_hash,
                        verify_inputs, verify_result, verify_published_tree)
from .identity import code_identity, environment_record, scientific_fingerprint, verify_recipe
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
    # JSON is also YAML syntax, but PyYAML's numeric resolver changes 1e-05
    # into a string. Preserve canonical JSON types before considering YAML.
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


def verify_dependency_result(root, *, expected_hash=None, trees=None, active=None, verified=None,
                             output_dir=None):
    root = directory_path(root)
    if output_dir is not None:
        require(not output_dir.is_relative_to(root) and not root.is_relative_to(output_dir),
                f'output overlaps an upstream attempt: {root}')
    active = set() if active is None else active
    verified = {} if verified is None else verified
    require(root not in active, 'dependency publication cycle')
    if root in verified:
        result = verified[root]
        require(expected_hash is None or any(a.path == FINGERPRINT and a.sha256 == expected_hash
                                            for a in result.artifacts),
                'dependency published fingerprint identity changed')
        return result
    active.add(root)
    try:
        result_file = dependency_file(root, '_execution/result.json')
        request = StageRequest.from_json(read_regular(dependency_file(root, '_execution/request.json')))
        result = StageResult.from_json(read_regular(result_file))
        require(Path(request.output_dir).resolve() == root, 'dependency attempt owner mismatch')
        require(result.status == 'pass', 'dependency stage did not pass')
        tree = verify_published_tree(root, result, expected_hash)
        verify_result(result, request)
        # The request binds this dependency map and each parent's fingerprint
        # digest. Verify the entire recorded lineage, not just direct inputs.
        config = read_mapping(request.config_path)
        hashes = dict(zip(request.dependency_paths, request.dependency_hashes))
        for unit, parent in config.get('dependencies', {}).items():
            parent = Path(parent)
            require(parent.is_absolute(), 'dependency publication path must be absolute')
            parent = directory_path(parent)
            expected = hashes.get(str(parent / FINGERPRINT))
            require(expected is not None, 'dependency fingerprint absent from published request')
            verify_dependency_result(parent, expected_hash=expected, trees=trees,
                                     active=active, verified=verified, output_dir=output_dir)
            verify_dependency_id(parent, unit)
        if trees is not None:
            trees[str(root)] = tree
        verified[root] = result
        return result
    finally:
        active.remove(root)


def verify_dependency_id(root, unit):
    producer = read_mapping(dependency_file(root, '_execution/task.json'))
    require(producer.get('id') == unit, f'dependency producer identity mismatch: {unit}')


def verify_dependency_recipe(root, lock):
    """Bind locked upstream science to this recipe; preserve pre-lock inputs."""
    request = StageRequest.from_json(read_regular(dependency_file(root, '_execution/request.json')))
    task_path = dependency_file(root, '_execution/task.json')
    require(file_hash(task_path) == request.task_hash, 'dependency task hash mismatch')
    task = read_mapping(task_path)
    # Acquisition/design/lock producers run before the final recipe exists.
    # Their immutable receipts remain valid across later scientific code merges.
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
    approvals_value = read_mapping(approvals_file)
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
    # Stage-required inputs cannot be removed by a selected shard/task.
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
    for unit, root in deps.items():
        require(not out.is_relative_to(root) and not root.is_relative_to(out),
                f'output overlaps an upstream attempt: {root}')
        require(isinstance(needs[unit], list) and needs[unit], 'dependency requires explicit files')
        for relative in needs[unit]:
            files.append(dependency_file(root, relative))
        # Acquisition commands predate the common stage API and publish source
        # receipts instead. Only the committed registry can declare that format;
        # a task cannot exempt a failed/incomplete stage by omitting its receipt.
        acquisition_receipt = settings.get('acquisition_receipts', {}).get(unit)
        if acquisition_receipt is not None:
            require(acquisition_receipt in needs[unit], 'acquisition receipt must be a declared input')
        # Every producer now seals its own attempt. An acquisition receipt
        # describes source data but cannot replace the publication fingerprint.
        require((root / '_execution/result.json').is_file(), f'stage receipt missing or not regular: {root / "_execution/result.json"}')
        result = verify_dependency_result(root, trees=dependency_trees, verified=verified_dependencies,
                                          output_dir=out)
        # Environment keys are wiring, never producer identity authority.
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
    # Execution directory is a once-only reservation; no in-place attempt resume.
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
              'input_sources': sources, 'dependencies': {k: str(v) for k, v in deps.items()}}
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
                # Compare upstream types before the contract's byte hashing: an
                # upstream regular file replaced by a FIFO must fail, never block.
                # Flush our streams before checking any stage-declared log hash.
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
                code_identity(repo, out)
            except BaseException as exc:
                result = StageResult(request_hash=request.content_hash, status='fail', artifacts=(),
                                     message=str(exc).strip() or type(exc).__name__)
        # No status output after artifact validation: run.log may be an artifact.
        return publish_result(out, result, owned_controls=True)
    except BaseException as exc:
        message = 'stage finalization failed: ' + (str(exc).strip() or type(exc).__name__)
        if changed:
            message += '; upstream attempt tainted; changed paths: ' + ', '.join(changed)
        # Execution already happened. Receipt I/O cannot turn failure into a
        # missing-prerequisite status or hide the paths we detected in memory.
        print('failed: ' + message, file=sys.stderr)
        return StageResult(request_hash=request.content_hash, status='fail', artifacts=(), message=message)
