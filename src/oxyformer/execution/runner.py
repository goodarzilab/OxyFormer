"""Trusted stage dispatch, with immutable inputs and attempt-owned writes.

Stages run in a child with kernel write restrictions and supervised metadata
operations. This protects against faulty stages, not hostile same-user actors.
The runner never changes upstream permissions; hashes are also rechecked before
publishing success. Archives are extracted into the consuming attempt.
"""
import importlib
import json
import os
from pathlib import Path
import re

import yaml

from oxyformer.contracts import StageRequest, StageResult
from oxyformer.provenance import ContractError, file_hash, relative_artifact_path, require
from .identity import code_identity, environment_record, scientific_fingerprint, verify_recipe
from .isolation import isolated_stage, require_parent_ready
from .paths import atomic_json, atomic_write, isolated_caches, output_path
from .publication import ControlRecords


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
        result[unit] = path.resolve(strict=True)
        require(result[unit].is_dir(), f'dependency must be an attempt directory: {name}')
    return result


def read_mapping(path):
    path = Path(path)
    text = path.read_text()
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
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), 'dependency file escapes attempt')
    return path


def verify_dependency_result(root):
    result_file = dependency_file(root, '_execution/result.json')
    request = StageRequest.from_json(dependency_file(root, '_execution/request.json').read_text())
    result = StageResult.from_json(result_file.read_text())
    require(Path(request.output_dir).resolve() == root, 'dependency attempt owner mismatch')
    require(result.status == 'pass', 'dependency stage did not pass')
    result.verify(request)
    return result


def verify_dependency_recipe(root, lock):
    """Bind locked upstream science to this recipe; preserve pre-lock inputs."""
    request = StageRequest.from_json(dependency_file(root, '_execution/request.json').read_text())
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


def run(stage, out, repo, *, deps_env=False, task_file=None, task_id=None, approvals=None):
    require_parent_ready()
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
        task_file = Path(task_file).resolve(strict=True)
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
        task = {'id': stage, 'stage': stage, 'needs': settings.get('needs', {}),
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
    stage_dependencies = []
    for unit, root in deps.items():
        require(not out.is_relative_to(root) and not root.is_relative_to(out),
                'output overlaps an upstream attempt')
        require(isinstance(needs[unit], list) and needs[unit], 'dependency requires explicit files')
        for relative in needs[unit]:
            files.append(dependency_file(root, relative))
        # Acquisition commands predate the common stage API and publish source
        # receipts instead. Only the committed registry can declare that format;
        # a task cannot exempt a failed/incomplete stage by omitting its receipt.
        acquisition_receipt = settings.get('acquisition_receipts', {}).get(unit)
        if acquisition_receipt is not None:
            require(acquisition_receipt in needs[unit], 'acquisition receipt must be a declared input')
        if acquisition_receipt is None or (root / '_execution/result.json').exists():
            require((root / '_execution/result.json').is_file(), f'stage receipt missing: {unit}')
            result = verify_dependency_result(root)
            stage_dependencies.append(root)
            allowed = {a.path for a in result.artifacts} | {
                '_execution/task.json', '_execution/request.json', '_execution/result.json',
                '_execution/environment.json', '_execution/identity.json'}
            require(set(needs[unit]) <= allowed, 'dependency file not declared by passing stage')
            for relative in ('_execution/request.json', '_execution/result.json'):
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
        for root in stage_dependencies:
            for path in verify_dependency_recipe(root, lock):
                if path not in files:
                    files.append(path)
        require(file_hash(approvals_file) == file_hash(repo / 'configs/approvals.yaml'),
                'locked approvals differ from fingerprinted repository config')
    config = {'stage': stage, 'settings': settings, 'approvals': approvals_value,
              'input_sources': sources, 'dependencies': {k: str(v) for k, v in deps.items()}}
    config_path = atomic_json(out, '_execution/config.json', config)
    task_path = atomic_json(out, '_execution/task.json', task)
    request = StageRequest(stage=stage, config_path=str(config_path), config_hash=file_hash(config_path),
                           task_path=str(task_path), task_hash=file_hash(task_path),
                           dependency_paths=tuple(map(str, files)),
                           dependency_hashes=tuple(map(file_hash, files)),
                           output_dir=str(out), code_identity=head)
    atomic_write(out, '_execution/request.json', request.to_json())
    environment = environment_record()
    atomic_json(out, '_execution/environment.json', environment)
    fingerprint = scientific_fingerprint(repo)
    atomic_json(out, '_execution/identity.json', {'head': head, 'scientific_fingerprint': fingerprint})
    controls = ControlRecords(out)
    try:
        request.verify_inputs()
        module_name = settings.get('module')
        require(isinstance(module_name, str) and module_name.startswith('oxyformer.'),
                'stage module not registered')
        def invoke():
            try:
                module = importlib.import_module(module_name)
            except (ImportError, FileNotFoundError) as exc:
                return StageResult(request_hash=request.content_hash, status='blocked', artifacts=(),
                                   message=str(exc).strip() or type(exc).__name__)
            require(callable(getattr(module, 'run_stage', None)), 'stage has no run_stage(StageRequest)')
            return module.run_stage(request)

        with isolated_caches(out):
            result = isolated_stage(request, invoke, tuple(deps.values()))
        controls.verify()
        require(isinstance(result, StageResult), 'stage did not return StageResult')
        result.verify(request)
        require(all(not (out / a.path).resolve().is_relative_to(repo) for a in result.artifacts),
                'artifact overlaps cloned repository')
        require(all(not a.path.startswith('_execution/') for a in result.artifacts), 'reserved execution artifact')
        declared = set(task.get('outputs', []))
        if result.status == 'pass':
            require(declared <= {a.path for a in result.artifacts}, 'stage omitted declared outputs')
        for path, digest in sources.items():
            require(file_hash(path) == digest, f'input source changed: {path}')
        require(code_identity(repo, out) == head, 'code identity changed during execution')
        require(scientific_fingerprint(repo) == fingerprint, 'scientific fingerprint changed during execution')
    except Exception as exc:
        result = StageResult(request_hash=request.content_hash, status='fail', artifacts=(),
                             message=str(exc).strip() or type(exc).__name__)
    except BaseException:
        controls.close()
        raise
    try:
        try:
            controls.verify()
        except Exception as exc:
            message = str(exc).strip() or type(exc).__name__
            controls.reject(message)
            result = StageResult(request_hash=request.content_hash, status='fail', artifacts=(),
                                 message='rejected execution controls: ' + message)
        atomic_write(out, '_execution/result.json', result.to_json())
        return result
    finally:
        controls.close()
