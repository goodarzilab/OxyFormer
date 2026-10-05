"""Expand coordinator units without submission. Retain the spec to rederive
all slices and collector edges. Code prerequisites in needs are merge barriers.
"""
from copy import deepcopy
from decimal import Decimal
from hashlib import sha256
import math
import re
import shlex

from oxyformer.provenance import canonical_json, check_hash, relative_artifact_path, require
from .runner import dependency_variable

PYTHON = '/mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python'
REMOTE = 'https://github.com/goodarzilab/OxyFormer.git'
RESERVED = {'code_commit.txt', 'run.log', 'task.json', 'src', '_execution'}
KINDS = {'screening', 'primary', 'ablation', 'final-coverage', 'anchor', 'refit-audit'}


def concrete_id(value):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,31}', value),
            'IDs must be concrete and at most 32 characters')


def concrete(value):
    if isinstance(value, str):
        require(not any(marker in value for marker in ('{', '}', '<TEMPLATE', '<ID>')),
                'unresolved template')
    elif isinstance(value, dict):
        for key, item in value.items():
            concrete(key)
            concrete(item)
    elif isinstance(value, list):
        for item in value:
            concrete(item)


def outputs_valid(outputs):
    require(isinstance(outputs, list) and outputs and len(set(outputs)) == len(outputs),
            'explicit unique outputs required')
    for name in outputs:
        relative_artifact_path(name)
        require(name.split('/')[0] not in RESERVED, 'reserved output name')
    require(not any(a != b and b.startswith(a + '/') for a in outputs for b in outputs),
            'overlapping outputs')


def resources(gpus, seconds):
    require(type(gpus) is int and 0 <= gpus <= 8, 'invalid GPU count')
    require(type(seconds) is int and seconds > 0, 'positive integer wall_seconds required')
    # Slurm rounds requested time up to whole minutes (sbatch --time).
    # Bind admission, the emitted limit and accounting to the same allocation.
    seconds = ((seconds + 59) // 60) * 60
    gpu_hours = gpus * seconds / 3600
    require(math.isfinite(gpu_hours) and gpu_hours <= 4, 'GPU leaf exceeds four GPU-hours')
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    flags = ['--partition=standard', '--account=root', '--qos=normal',
             '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=16G',
             f'--time={hours:02}:{minutes:02}:{seconds:02}']
    if gpus:
        flags.append(f'--gpus={gpus}')
    return flags, gpu_hours


def stage_command(task):
    """All executable shell is fixed; task bytes and identifiers are shell quoted."""
    concrete_id(task['id'])
    concrete_id(task['stage'])
    encoded = shlex.quote(canonical_json(task))
    return '\n'.join([
        'set -euo pipefail',
        ': "${SWARM_UNIT_DIR:?}"',
        f'git clone --depth 1 --branch dev {REMOTE} "$SWARM_UNIT_DIR/src"',
        'git -C "$SWARM_UNIT_DIR/src" rev-parse HEAD > "$SWARM_UNIT_DIR/code_commit.txt"',
        'export PYTHONDONTWRITEBYTECODE=1',
        'export PYTHONPATH="$SWARM_UNIT_DIR/src/src"',
        'cd "$SWARM_UNIT_DIR/src"',
        f'printf %s {encoded} > "$SWARM_UNIT_DIR/task.json"',
        f'exec {PYTHON} -B -m oxyformer.cli run-stage --stage {shlex.quote(task["stage"])} '
        '--repo "$SWARM_UNIT_DIR/src" --out "$SWARM_UNIT_DIR" --deps-env '
        f'--task "$SWARM_UNIT_DIR/task.json" --task-id {shlex.quote(task["id"])} '
        '--approvals "$SWARM_UNIT_DIR/src/configs/approvals.yaml" > "$SWARM_UNIT_DIR/run.log" 2>&1',
    ])


def _spec_check(spec, approvals):
    require(spec.get('schema_version') == 1, 'unsupported campaign schema')
    concrete(spec)
    concrete_id(spec['id'])
    require(spec['kind'] in KINDS, 'unknown campaign kind')
    require(isinstance(spec['work'], list) and bool(spec['work']), 'campaign work required')
    ids = [w['id'] for w in spec['work']]
    require(len(set(ids)) == len(ids), 'duplicate work IDs')
    for value in ids + spec['prerequisites'] + list(spec['inputs']):
        concrete_id(value)
    require(len(set(spec['prerequisites'])) == len(spec['prerequisites']), 'duplicate prerequisites')
    external = set(spec['prerequisites']) | set(spec['inputs'])
    variables = [dependency_variable(x) for x in external]
    require(len(set(variables)) == len(variables), 'dependency normalization collision')
    lock = spec['recipe_lock']
    require(set(lock) == {'dependency', 'path', 'sha256'}, 'invalid recipe reference')
    check_hash(lock['sha256'])
    require(lock['dependency'] in spec['inputs'] and lock['path'] in spec['inputs'][lock['dependency']],
            'recipe must be an explicit input')
    for paths in spec['inputs'].values():
        require(isinstance(paths, list) and paths and len(set(paths)) == len(paths), 'invalid input files')
        for path in paths:
            relative_artifact_path(path)
    count = sum(len(w['slices']) for w in spec['work'])
    require(0 < count <= 40, 'campaign exceeds forty leaves')
    total_gpu_seconds = 0
    for work in spec['work']:
        concrete_id(work['stage'])
        outputs_valid(work['outputs'])
        require(work['slices'], 'empty continuation chain')
        for segment in work['slices']:
            resources(segment['gpus'], segment['wall_seconds'])
            total_gpu_seconds += segment['gpus'] * ((segment['wall_seconds'] + 59) // 60) * 60
    collector = spec['collector']
    concrete_id(collector['stage'])
    outputs_valid(collector['outputs'])
    resources(0, collector['wall_seconds'])
    if spec['kind'] in {'final-coverage', 'anchor', 'refit-audit'}:
        allocation = approvals.get('owner_decisions', {}).get('campaign_allocations', {}).get(spec['id'])
        require(isinstance(allocation, dict) and allocation.get('kind') == spec['kind'],
                'missing owner campaign allocation')
        amount = allocation.get('gpu_hours')
        require(type(amount) in (int, float) and math.isfinite(amount)
                and Decimal(str(amount)) * 3600 >= total_gpu_seconds,
                'owner allocation does not cover campaign')
    return external


def _id(spec, slot):
    digest = sha256(canonical_json({'campaign': spec, 'slot': slot}).encode()).hexdigest()
    return spec['id'][:8] + '-' + digest[:22]


def _unit(spec, task, prerequisites, gpus, seconds, role):
    flags, hours = resources(gpus, seconds)
    return {'id': task['id'], 'kind': 'slurm', 'title': f'{spec["id"]}: {task["id"]}',
            'needs': sorted(set(prerequisites) | set(task['needs'])), 'runtime': 'oxyformer-env',
            'outputs': ['code_commit.txt', 'run.log', '_execution/result.json', *task['outputs']],
            'inputs': [PYTHON], 'sbatch': flags, 'gpu_hours': hours,
            'pool': 'gpu' if gpus else 'cpu', 'write_scopes': [task['id'] + '/**'],
            'command': stage_command(task), 'max_attempts': 1}


def _build(spec):
    units, tasks, leaves = [], [], []
    for work in spec['work']:
        previous = None
        owner = sha256(canonical_json({'campaign': spec['id'], 'work': work,
                                      'recipe': spec['recipe_lock']}).encode()).hexdigest()
        for step, segment in enumerate(work['slices']):
            unit_id = _id(spec, [work['id'], step])
            needs = deepcopy(spec['inputs'])
            if previous:
                needs[previous] = ['_execution/task.json', '_execution/request.json',
                                   '_execution/result.json', *work['outputs']]
            task = {'id': unit_id, 'stage': work['stage'], 'campaign': spec['id'],
                    'parameters': deepcopy(work.get('parameters', {})), 'needs': needs,
                    'outputs': work['outputs'], 'recipe_lock': spec['recipe_lock'],
                    'continuation': {'owner': owner, 'step': step, 'predecessor': previous}}
            tasks.append(task)
            units.append(_unit(spec, task, spec['prerequisites'], segment['gpus'], segment['wall_seconds'], 'leaf'))
            leaves.append(unit_id)
            previous = unit_id
    needs = deepcopy(spec['inputs'])
    for task in tasks:
        needs[task['id']] = ['_execution/task.json', '_execution/request.json',
                             '_execution/result.json', *task['outputs']]
    task = {'id': _id(spec, ['collector']), 'stage': spec['collector']['stage'],
            'campaign': spec['id'], 'needs': needs, 'outputs': spec['collector']['outputs'],
            'recipe_lock': spec['recipe_lock'], 'expected_leaves': leaves}
    tasks.append(task)
    units.append(_unit(spec, task, spec['prerequisites'], 0, spec['collector']['wall_seconds'], 'collector'))
    return units, tasks, leaves


def validate_plan(plan, approvals):
    spec = plan['spec']
    external = _spec_check(spec, approvals)
    require(plan.get('schema_version') == 1, 'unsupported expansion schema')
    units = plan['units']
    ids = [u['id'] for u in units]
    require(len(set(ids)) == len(ids), 'duplicate unit IDs')
    require(not set(ids) & external, 'generated unit collides with prerequisite')
    for value in ids:
        concrete_id(value)
    variables = [dependency_variable(x) for x in set(ids) | external]
    require(len(set(variables)) == len(variables), 'dependency normalization collision')
    by_id = {u['id']: u for u in units}
    visiting, done = set(), set()

    def visit(unit_id):
        if unit_id in done or unit_id in external:
            return
        require(unit_id in by_id, 'unknown dependency')
        require(unit_id not in visiting, 'dependency cycle')
        visiting.add(unit_id)
        for dependency in by_id[unit_id]['needs']:
            visit(dependency)
        visiting.remove(unit_id)
        done.add(unit_id)

    for unit_id in ids:
        visit(unit_id)
    expected_units, expected_tasks, leaves = _build(spec)
    collector_id = expected_units[-1]['id']
    require(collector_id in by_id and set(leaves) <= set(by_id[collector_id]['needs']),
            'collector omitted required leaf dependency')
    require(set(ids) == {u['id'] for u in expected_units}, 'required campaign leaves omitted or added')
    require(plan['expected_leaves'] == leaves, 'expected leaf manifest drift')
    require(plan['tasks'] == expected_tasks, 'task or continuation ownership drift')
    require({u['id']: u for u in expected_units} == by_id,
            'unit command, resources, outputs or dependency drift')
    return plan


def expand_campaign(spec, approvals):
    """Validate a finite, explicit work list; do not infer tuning or repetitions.

    `work[].slices` must come from profiling/the locked scientific controller.
    Expansion preserves these slices exactly; it never shortens required work.
    New final/anchor/refit allocations must exist in owner_decisions under
    campaign_allocations[spec.id] with matching kind and sufficient gpu_hours.
    """
    spec = deepcopy(spec)
    _spec_check(spec, approvals)
    units, tasks, leaves = _build(spec)
    return validate_plan({'schema_version': 1, 'spec': spec, 'units': units,
                          'tasks': tasks, 'expected_leaves': leaves}, approvals)


UNIT_SCHEMA = {
    'schema_version': 1,
    'unit_fields': ['id', 'kind', 'title', 'needs', 'runtime', 'outputs', 'inputs',
                    'sbatch', 'gpu_hours', 'pool', 'write_scopes', 'command', 'max_attempts'],
    'task_fields': ['id', 'stage', 'campaign', 'needs', 'outputs', 'recipe_lock',
                    'parameters', 'continuation', 'expected_leaves'],
    'spec_required': ['schema_version', 'id', 'kind', 'prerequisites', 'inputs',
                      'recipe_lock', 'work', 'collector'],
    'work_required': ['id', 'stage', 'outputs', 'slices'],
    'slice_required': ['gpus', 'wall_seconds'],
    'limits': {'leaves': 40, 'gpu_hours_per_leaf': 4, 'id_length': 32, 'arrays': False},
    'dependency_environment': 'SWARM_DEP_' + '<uppercase ID; nonalphanumeric replaced by underscore>',
    'stage_receipts': 'Every input attempt requires a passing StageResult binding _execution/fingerprint.json; acquisition_receipts additionally requires the source receipt.',
    'slurm_accounting': 'GPU-hours use whole-minute limits; campaign admission compares integer GPU-seconds to the decimal owner allocation.',
    'concrete_strings': 'String fields may not contain curly braces or template markers; nested JSON values must be mappings/lists.',
    'locked_stages': 'Registry requires_recipe or a campaign field makes recipe_lock mandatory.',
    'merge_barrier': 'code prerequisite needs are satisfied only by coordinator merged receipts',
    'fingerprint': 'tracked-science-v2: every src/ and scripts/ file; other tracked paths except *.md outside docs/plan/; disk checked against HEAD',
    'attempt_fingerprint': 'entries, file types, mode bits, sizes, content hashes, symlink targets; timestamps and inode numbers excluded; compares input states, not write history',
    'validation': 'oxyformer.execution.campaign.validate_plan(expansion, owner_approvals)',
}
