"""Measurement-only recipe arithmetic; no campaign or scientific configuration changes."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Quantity:
    value: float
    basis: frozenset[str]

    def __post_init__(self):
        if not math.isfinite(self.value) or self.value < 0:
            raise ValueError('finite nonnegative quantity required')
        if not self.basis or not self.basis <= {'measured', 'extrapolated', 'uncalibrated'}:
            raise ValueError('quantity requires an explicit basis')

    def times(self, factor):
        return Quantity(self.value * factor, self.basis)

    def with_basis(self, label):
        return Quantity(self.value, self.basis | {label})

    def to_dict(self):
        return {'value': self.value, 'basis': sorted(self.basis)}


def add(*terms):
    return Quantity(math.fsum(t.value for t in terms), frozenset().union(*(t.basis for t in terms)))


def ratio(numerator, denominator):
    if denominator.value <= 0:
        raise ValueError('positive denominator required')
    return Quantity(numerator.value / denominator.value, numerator.basis | denominator.basis)


def epoch_projection(*, required, completed, wall_seconds, boundary_seconds, attempts):
    """Use completed epochs only; partial-epoch wall time is never a rate."""
    if type(required) is not int or required <= 0:
        raise ValueError('positive required epoch count')
    if type(completed) is not int or not 0 <= completed <= required:
        raise ValueError('invalid completed epoch count')
    if attempts not in (1, 2):
        raise ValueError('one or two attempts required')
    if not (math.isfinite(wall_seconds) and math.isfinite(boundary_seconds)
            and 0 <= boundary_seconds <= wall_seconds):
        raise ValueError('invalid boundary time')
    if completed == required:
        return Quantity(wall_seconds, frozenset({'measured'}))
    if attempts != 2:
        raise ValueError('unfinished measurement requires its second attempt')
    if completed == 0:
        return None
    if boundary_seconds <= 0:
        raise ValueError('completed epochs require positive boundary time')
    return Quantity(boundary_seconds * required / completed, frozenset({'extrapolated'}))


def calibration(anchor, receipt, *, measured_anchor, frame_matches, cell_matches, operations_match):
    reasons = []
    if not measured_anchor:
        reasons.append('anchor extrapolated')
    if not frame_matches:
        reasons.append('frame mismatch')
    if not cell_matches:
        reasons.append('cell mismatch')
    if not operations_match:
        reasons.append('operation mismatch')
    result = None if anchor is None else ratio(anchor, receipt)
    if result is None:
        reasons.append('anchor failed_measurement')
    elif not .80 <= result.value <= 1.25:
        reasons.append('ratio outside 0.80–1.25')
    if reasons and result is not None:
        result = result.with_basis('uncalibrated')
    return {'passed': not reasons, 'reasons': reasons,
            'ratio': None if result is None else result.to_dict()}


def feasibility(compute_hours, overhead):
    allocated = Quantity(compute_hours.value * overhead.value, compute_hours.basis | overhead.basis)
    return [dict(repetitions=repetitions, scenarios=scenarios,
                 compute_h100_hours=compute_hours.times(repetitions * scenarios).to_dict(),
                 allocated_h100_hours=allocated.times(repetitions * scenarios).to_dict(),
                 idealised_lower_bound_wall_hours_at_16_gpus=allocated.times(repetitions * scenarios / 16).to_dict())
            for repetitions in (39, 100, 300, 1000) for scenarios in (1, 4)]


BASE_TREE = 'f25e8853ba691883b447e0f89e9ce334f84f56ac'


def operation_counts(*, seed_count, grid_points):
    """Derive counts from the protected source at the recorded launch base."""
    if seed_count not in (1, 3) or grid_points not in (2, 4):
        raise ValueError('unregistered study dimensions')
    return pinned_work(seed_count=seed_count, grid_points=grid_points)['counts']


def project(measurements, counts, *, tree):
    if tree != BASE_TREE or any(m['tree'] != tree for m in measurements.values()):
        raise ValueError('measurement tree mismatch')
    if set(measurements) != set(counts):
        raise ValueError('missing or extra measurement terms')
    if any(m['seconds'] is None for m in measurements.values()):
        return None
    return add(*(measurements[name]['seconds'].times(count)
                 for name, count in counts.items())).times(1 / 3600)


def source_work(nested_source, coverage_source, *, seed_count, grid_points,
                held_inner_folds=(0, 1, 2)):
    """Walk the task-building statements and coverage iteration in pinned source.

    Only the expressions that build the controller task list are evaluated.
    No training, imports or controller functions are executed by this walk.
    The caller supplies target-bearing inner folds from its prepared endpoint.
    """
    import ast
    from collections import Counter
    from types import SimpleNamespace

    nested = ast.parse(nested_source)
    coverage = ast.parse(coverage_source)
    advance = next(n for n in nested.body if isinstance(n, ast.FunctionDef) and n.name == '_advance')
    estimate = next(n for n in coverage.body if isinstance(n, ast.FunctionDef) and n.name == 'estimate_repetition')
    env = {'config': SimpleNamespace(grid=tuple(range(grid_points))),
           'range': range, 'len': len}
    tasks = []

    def expression(node, scope):
        # Sources are trusted repository files at the recorded commit, never
        # measurement inputs. Limit expression evaluation to this small scope.
        return eval(compile(ast.Expression(node), '<counted-work>', 'eval'),
                    {'__builtins__': {}, **scope}, {})

    def mutations(nodes, scope):
        for node in nodes:
            if isinstance(node, ast.For):
                touches = any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                              and n.value.id == 'tasks' for n in ast.walk(node))
                if touches:
                    if not isinstance(node.target, ast.Name):
                        raise ValueError('unsupported task iteration')
                    for value in expression(node.iter, scope):
                        # _advance omits inner partitions without target rows.
                        if value in held_inner_folds:
                            mutations(node.body, {**scope, node.target.id: value})
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == 'tasks':
                    values = expression(call.args[0], scope)
                    if call.func.attr == 'append':
                        tasks.append(values)
                    elif call.func.attr == 'extend':
                        tasks.extend(values)
                    else:
                        raise ValueError('unsupported task mutation')
            elif isinstance(node, ast.While):
                break  # Execution is counted from the constructed task list.
    mutations(advance.body, env)
    loops = [n for n in ast.walk(estimate) if isinstance(n, ast.For)
             and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                     and c.func.attr == 'run_fold' for c in ast.walk(n))]
    if len(loops) != 2:
        raise ValueError('coverage fold/seed iteration changed')
    outer = next(n for n in loops if isinstance(n.target, ast.Name) and n.target.id == 'fold')
    seed_loop = next(n for n in loops if isinstance(n.target, ast.Name) and n.target.id == 'neural_seed')
    if not isinstance(seed_loop.iter, ast.Name) or seed_loop.iter.id != 'seeds':
        raise ValueError('coverage seed iteration changed')
    outer_count = len(expression(outer.iter, env))
    scoring_calls = Counter(c.func.id for c in ast.walk(estimate)
                            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                            and c.func.id in ('one_step', 'cv_tmle', 'spatial_sensitivities'))
    if scoring_calls != Counter(one_step=1, cv_tmle=1, spatial_sensitivities=1):
        raise ValueError('coverage scoring operations changed')
    counts = Counter()
    for kind, fold, grid in tasks:
        if kind == 'ssl':
            counts['ssl'] += 1
        elif kind in ('outcome', 'origin'):
            counts[('refit_' if grid is None else 'inner_') + kind] += 1
        elif kind != 'select':
            raise ValueError('unknown nested operation')
    counts = {name: count * outer_count * seed_count for name, count in counts.items()}
    counts['scoring'] = 1
    return {'counts': counts, 'tasks_per_outer_seed': tasks,
            'outer_folds': outer_count, 'seed_count': seed_count,
            'scoring_operations': dict(scoring_calls)}


def pinned_work(*, seed_count, grid_points, tree=BASE_TREE):
    """Read the actual base source, and reject drift in protected runtime code."""
    from hashlib import sha256
    from pathlib import Path
    import subprocess

    if tree != BASE_TREE:
        raise ValueError('counted-work tree mismatch')
    sources, digests = {}, {}
    for name, path in (
        ('nested', 'src/oxyformer/training/nested_cv.py'),
        ('coverage', 'src/oxyformer/validation/coverage.py'),
    ):
        raw = subprocess.check_output(['git', 'show', tree + ':' + path])
        if Path(path).read_bytes() != raw:
            raise ValueError('protected source differs from counted-work tree')
        sources[name] = raw.decode()
        digests[path] = sha256(raw).hexdigest()
    return {**source_work(sources['nested'], sources['coverage'],
                          seed_count=seed_count, grid_points=grid_points),
            'tree': tree, 'source_sha256': digests}


def clean_identity():
    """A measurement starts only from committed, clean, base-descended code."""
    import subprocess
    if subprocess.check_output(['git', 'status', '--porcelain']).strip():
        raise ValueError('dirty checkout refused')
    subprocess.run(['git', 'merge-base', '--is-ancestor', BASE_TREE, 'HEAD'], check=True)
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()


def atomic_record(path, value):
    import json
    import os
    from pathlib import Path
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def resume_record(path, binding):
    """Resume only this exact measurement; keep the last durable slice pointer."""
    import json
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        return {'binding': binding, 'attempts': [], 'slices': [], 'predecessor': None}
    record = json.loads(path.read_text())
    if record['binding'] != binding:
        raise ValueError('measurement continuation binding mismatch')
    return record


def epoch_timing(deadline, *, finalize=False):
    """Use the controller's existing cooperative phase checks to slice epochs.

    The check before evaluation is not an epoch boundary. Stop at the following
    check, after evaluation has completed. A completed training transaction has
    a second evaluation check before its held-out prediction pass; a finalizing
    call lets that pass finish and stops before the next training task.
    """
    from oxyformer.training.fit import DeadlineTiming

    class EpochTiming(DeadlineTiming):
        def __init__(self):
            super().__init__(deadline)
            self.evaluated = False
            self.stopped = False

        def boundary(self, phase, device='cpu'):
            expired = super().boundary(phase, device)
            stop = self.evaluated and (phase != 'evaluation' if finalize else True)
            self.evaluated = self.evaluated or phase == 'evaluation'
            self.stopped = self.stopped or stop
            return expired or self.stopped

    return EpochTiming()


def guarded_gpu_arguments(output, name):
    """The orchestrator alone releases these held requests after node guarding."""
    return ['--parsable', '--hold', '--partition=standard', '--account=root',
            '--nodes=1', '--ntasks=1', '--cpus-per-task=8', '--mem=64G',
            '--gres=gpu:1', '--time=08:00:00', '--job-name=' + name,
            '--output=' + str(output)]


def require_compute_node(*, gpu=False):
    import os
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise ValueError('Slurm compute node required')
    if gpu:
        import torch
        if not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):
            raise ValueError('one H100 is required')
        if torch.cuda.device_count() != 1:
            raise ValueError('exactly one visible GPU required')


def ssl_timing(deadline):
    """Collect the existing whole-call SSL interval bracketed by the controller."""
    from oxyformer.training.fit import DeadlineTiming

    class SSLTiming(DeadlineTiming):
        def __init__(self):
            super().__init__(deadline)
            self.scope_start = None
            self.intervals = []

        def boundary(self, phase, device='cpu'):
            result = super().boundary(phase, device)
            if self.scope_start is None:
                self.scope_start = self.last
            return result

        def finish_unit(self, device='cpu'):
            start = self.scope_start
            super().finish_unit(device)
            if start is not None:
                self.intervals.append(self.clock() - start)
            self.scope_start = None

    return SSLTiming()


def measure_fit_task(endpoint, recipe, study, directory, *, position, predecessor=None,
                     deadline=None):
    """Run a single controller task, checkpointing after each completed epoch.

    Other tasks must be completed in their own jobs before this call. No model,
    selection result or controller checkpoint is fabricated to skip tuning.
    """
    from dataclasses import replace
    from pathlib import Path
    import os
    import time
    import torch
    from oxyformer.training import nested_cv
    from oxyformer.training.checkpoint import load_checkpoint
    from oxyformer.training.fit import FoldArtifacts
    from oxyformer.provenance import file_hash

    require_compute_node(gpu=True)
    head = clean_identity()
    work = pinned_work(seed_count=study.seed_count, grid_points=study.grid_points)
    tasks = work['tasks_per_outer_seed']
    kind, fold, grid = tasks[position]
    if kind == 'select':
        raise ValueError('selection is performed with the following outer SSL task')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    binding = {'tree': BASE_TREE, 'head': head, 'endpoint': endpoint.content_hash,
               'recipe': recipe, 'study': cell_values(study), 'position': position,
               'incoming': None if predecessor is None else predecessor.checkpoint.sha256}
    journal = root / 'progress.json'
    record = resume_record(journal, binding)
    job = os.environ['SLURM_JOB_ID']
    if job not in record['attempts']:
        if len(record['attempts']) >= 2:
            raise ValueError('two attempts already used')
        record['attempts'].append(job)
    atomic_record(journal, record)
    previous = (FoldArtifacts.from_json(record['predecessor']) if record['predecessor']
                else predecessor)
    deadline = time.monotonic() + 8 * 3600 - 900 if deadline is None else deadline
    options = nested_cv._settings(recipe, endpoint)
    base_config = endpoint.configuration(0, root, **options, study=study, device='cuda:0')
    groups = nested_cv._check_fitting_minima(base_config, endpoint.outer, endpoint.geography)
    parent = base_config.inner.split if fold < 3 else endpoint.outer
    local, fitting, held = nested_cv._partition(base_config, parent, fold if fold < 3 else 0)
    if kind == 'ssl':
        allowed = set(fitting)
        grouped = [tuple(i for i in g if i in allowed) for g in groups if allowed.intersection(g)]
        stopping = min(grouped, key=lambda g: nested_cv.digest([1103, g, 'ssl-stopping']))
        batches = math.ceil((len(fitting) - len(stopping)) / 256) + math.ceil(len(stopping) / 256)
    else:
        model_ids = tuple(i for i in fitting if kind != 'origin' or i in endpoint.outer.target_ids)
        batches = math.ceil(len(model_ids) / base_config.settings.batch_size)
    torch.cuda.reset_peak_memory_stats()
    while time.monotonic() < deadline - 900:
        before = (load_checkpoint(previous.checkpoint, previous.checkpoint.identity)['controller']
                  if previous else None)
        active = None if before is None else before['active']
        finalize = active is not None and active['progress']['epoch'] >= base_config.frozen_epochs
        timing = ssl_timing(deadline) if kind == 'ssl' else epoch_timing(deadline, finalize=finalize)
        slice_root = root / ('slice-' + str(len(record['slices'])) + '-job-' + job)
        # A killed call can leave a private incomplete directory. Never reuse it.
        suffix = 0
        while slice_root.exists():
            suffix += 1
            slice_root = root / ('slice-' + str(len(record['slices'])) + '-job-' + job + '-' + str(suffix))
        config = replace(base_config, output_dir=str(slice_root), predecessor=previous,
                         max_batches=batches if kind == 'ssl' else None,
                         deadline_timing=timing, slice_seconds=min(14400., deadline-time.monotonic()),
                         checkpoint_margin_seconds=timing.margin_seconds)
        started = time.monotonic()
        artifact = nested_cv.run_fold(config, endpoint.outer, 1103, geography=endpoint.geography)
        wall = time.monotonic() - started
        after = load_checkpoint(artifact.checkpoint, artifact.checkpoint.identity)['controller']
        completed_task = after['position'] > position
        if after['position'] > position + 1:
            raise ValueError('measurement crossed a task boundary')
        compute = after['compute']['wall_seconds'] - (before['compute']['wall_seconds'] if before else 0.)
        if kind == 'ssl':
            descriptor = after['initializations'].get(fold) if completed_task else after['ssl_pending']
            from oxyformer.training.checkpoint import CheckpointArtifact
            ssl = CheckpointArtifact.from_json(descriptor)
            progress = load_checkpoint(ssl, ssl.identity)['progress']
            if len(timing.intervals) < 2:
                raise ValueError('missing whole-call SSL timing')
            compute = timing.intervals[1]
            epochs = progress['epoch']
            partial = progress['step'] % max(1, math.ceil((len(fitting)-len(stopping))/256))
        else:
            progress = after['active']['progress'] if after['active'] is not None else None
            epochs = base_config.frozen_epochs if completed_task else progress['epoch']
            partial = 0 if progress is None else progress['step'] % batches
        record['slices'].append({'job': job, 'wall_seconds': wall, 'compute_seconds': compute,
            'epochs_completed': epochs, 'minibatches_partial_epoch': partial,
            'minibatches_per_epoch': batches, 'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(),
            'complete': completed_task, 'checkpoint_sha256': file_hash(artifact.checkpoint.path)})
        record['predecessor'] = artifact.to_json()
        atomic_record(journal, record)
        previous = artifact
        if completed_task:
            return record
    return record


def measure_scoring(endpoint, recipe, study, root):
    """Time both estimators on synthetic, full target-by-seed predictions only."""
    from dataclasses import replace
    from pathlib import Path
    import time
    import torch
    from oxyformer.contracts import OOFNuisances
    from oxyformer.validation.coverage import estimate_repetition

    require_compute_node(gpu=True)
    head = clean_identity()
    names = endpoint.data.manifest.original_ids
    weights = dict(zip(names, (1.,) * len(names) if endpoint.data.manifest.weight_field is None
                       else endpoint.data.column(endpoint.data.manifest.weight_field)))
    parts = []
    for fold in range(5):
        ids = endpoint.outer.evaluation_ids(fold)
        for seed in study.seeds:
            lineage = replace(endpoint.data.manifest.lineage, unit_ids=ids,
                              split_hash=endpoint.outer.content_hash,
                              parent_hashes=(endpoint.data.manifest.content_hash,),
                              model_hash=None, seed=seed, parameter_count=None)
            part = OOFNuisances(spec=endpoint.data.manifest.spec, lineage=lineage,
                original_ids=ids, fold_ids=(fold,) * len(ids), seed_ids=(seed,) * len(ids),
                mu_a=(0.,) * len(ids), mu_d=(0.,) * len(ids),
                r_a=(1.,) * len(ids), r_d=(1.,) * len(ids),
                origin_weights=tuple(weights[i] for i in ids))
            parts.append(part.to_json())
    expected = {(i, s) for i in endpoint.outer.original_ids for s in study.seeds}
    observed = [(i, s) for p in parts for i, s in zip(
        OOFNuisances.from_json(p).original_ids, OOFNuisances.from_json(p).seed_ids)]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError('synthetic scoring shape mismatch')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    continuation = {'parts': parts}
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = estimate_repetition(endpoint, recipe, root, float('inf'),
                                 study=study, continuation=continuation)
    wall = time.monotonic() - started
    if result is None:
        raise ValueError('scoring did not finish')
    # Intentionally discard all scientific values. Synthetic predictions have
    # no inferential meaning; only the call's elapsed time is a measurement.
    del result
    return {'tree': BASE_TREE, 'head': head, 'study': cell_values(study),
            'inputs': 'synthetic inputs, timing only', 'basis': ['measured'],
            'wall_seconds': wall, 'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(),
            'prediction_rows': len(observed)}


def calibration_receipt(path, frame_path):
    """Load the sealed repetition compute target, retaining the exact file hash."""
    from pathlib import Path
    import json
    from oxyformer.provenance import file_hash
    from oxyformer.validation.coverage import read_leaf_measurements
    path = Path(path)
    record = read_leaf_measurements(path)
    profile = record['profile']
    import subprocess
    tree = record['binding']['code_identity']
    study = record['binding']['study']
    receipt_work = source_work(
        subprocess.check_output(['git', 'show', tree + ':src/oxyformer/training/nested_cv.py']).decode(),
        subprocess.check_output(['git', 'show', tree + ':src/oxyformer/validation/coverage.py']).decode(),
        seed_count=study['seed_count'], grid_points=study['grid_points'])
    seconds = profile['complete_repetition_seconds']
    if len(seconds) != 1:
        raise ValueError('calibration requires one complete repetition')
    from oxyformer.contracts import StageRequest
    producer = StageRequest.from_json((path.parent / '_execution/request.json').read_text())
    declared = dict(zip(producer.dependency_paths, producer.dependency_hashes))
    frames = [digest for name, digest in declared.items() if Path(name).name == 'frame.json']
    expected = frames[0] if len(frames) == 1 else None
    if expected is None:
        raise ValueError('receipt does not declare the measurement frame artifact')
    return {'compute_hours': Quantity(seconds[0] / 3600, frozenset({'measured'})),
            'study': record['binding']['study'], 'tree': record['binding']['code_identity'], 'frame_hash': expected,
            'frame_matches': expected == file_hash(frame_path),
            'path': str(path), 'sha256': file_hash(path), 'work': receipt_work}


def leaf_overhead(paths):
    """Use each campaign leaf's own record once; never sum cumulative history."""
    from pathlib import Path
    from oxyformer.validation.coverage import read_leaf_measurements
    from oxyformer.provenance import file_hash

    paths = tuple(map(Path, paths))
    if len(paths) != 140 or len(set(paths)) != 140:
        raise ValueError('exactly 140 unique campaign-D leaf records required')
    own, files = [], []
    for path in paths:
        record = read_leaf_measurements(path)
        leaf = record['leaves'][-1]
        own.append(leaf)
        files.append({'path': str(path), 'sha256': file_hash(path)})
    if len({leaf['task_id'] for leaf in own}) != 140:
        raise ValueError('duplicate campaign leaf')
    allocated = Quantity(math.fsum(leaf['wall_seconds'] for leaf in own) / 3600,
                         frozenset({'measured'}))
    compute = Quantity(math.fsum(leaf['compute_seconds'] for leaf in own) / 3600,
                       frozenset({'measured'}))
    return {'allocated_hours': allocated.to_dict(), 'compute_hours': compute.to_dict(),
            'ratio': ratio(allocated, compute).to_dict(), 'files': files}


def cell_values(study):
    return {name: getattr(study, name) for name in
            ('reference_cap', 'frozen_epochs', 'grid_points', 'seed_count')}


def study_cells():
    from oxyformer.training.fit import RecipeStudy
    return {'a': RecipeStudy(), 'b': RecipeStudy(reference_cap=32),
            **{cell: RecipeStudy(reference_cap=32, frozen_epochs=epochs, grid_points=2, seed_count=1)
               for cell, epochs in (('c', 150), ('d', 60), ('e', 30))}}


def prepare_inputs(profile_root, destination):
    """Bind a null-effect draw to the same real frame used by the sealed profile."""
    import json
    from pathlib import Path
    from oxyformer.contracts import StageRequest
    from oxyformer.provenance import file_hash
    from oxyformer.training.nested_cv import PreparedEndpoint
    from oxyformer.validation.coverage import bind_observations, fixed_exposures
    from oxyformer.validation.generators import generate_suite_a
    from oxyformer.validation.scm import CovariateFrame, SCMConfig

    require_compute_node()
    head = clean_identity()
    profile_root, root = Path(profile_root), Path(destination)
    request = StageRequest.from_json((profile_root / '_execution/request.json').read_text())
    task = json.loads((profile_root / '_execution/task.json').read_text())
    if file_hash(profile_root / '_execution/task.json') != request.task_hash:
        raise ValueError('profile task hash mismatch')
    hashes = dict(zip(request.dependency_paths, request.dependency_hashes))
    paths = {}
    for name in ('frame.json', 'endpoint.json'):
        matches = [Path(p) for p in request.dependency_paths if Path(p).name == name]
        if len(matches) != 1 or file_hash(matches[0]) != hashes[str(matches[0])]:
            raise ValueError('missing or changed profile ' + name)
        paths[name] = matches[0]
    receipt = calibration_receipt(profile_root / 'budget_record.json', paths['frame.json'])
    template = PreparedEndpoint.from_json(paths['endpoint.json'].read_text())
    frame = CovariateFrame.from_json(paths['frame.json'].read_text())
    params = task['parameters']
    if params['scenario']['name'] != 'null_effect':
        raise ValueError('null_effect profile required')
    scenario = SCMConfig(**params['scenario'])
    fixed = fixed_exposures(template, frame)
    sample = generate_suite_a(frame, scenario, template.policy, seed=params['draws'][0]['seed'],
                              **({'fixed_exposures': fixed} if fixed else {}))
    endpoint = bind_observations(template, sample.observations)
    if not sample.integration_uncertainty.converged:
        raise ValueError('null-effect draw integration did not converge')
    root.mkdir(parents=True, exist_ok=True)
    endpoint_path = root / 'measurement-endpoint.json'
    endpoint_path.write_text(endpoint.to_json())
    metadata = {'tree': BASE_TREE, 'prepared_head': head, 'recipe': params['recipe'],
                'endpoint_path': str(endpoint_path), 'endpoint_sha256': file_hash(endpoint_path),
                'frame_path': str(paths['frame.json']), 'frame_sha256': file_hash(paths['frame.json']),
                'profile_path': str(profile_root / 'budget_record.json'),
                'receipt': {**receipt, 'compute_hours': receipt['compute_hours'].to_dict()},
                'work': {c: pinned_work(seed_count=s.seed_count, grid_points=s.grid_points)
                         for c, s in study_cells().items()}}
    atomic_record(root / 'inputs.json', metadata)
    return metadata


def main():
    import argparse
    import json
    from pathlib import Path
    from oxyformer.provenance import file_hash
    from oxyformer.training.nested_cv import PreparedEndpoint
    from oxyformer.training.fit import FoldArtifacts

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'fit', 'score', 'drive'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--cell', choices=tuple('abcde'))
    parser.add_argument('--position', type=int)
    parser.add_argument('--predecessor', type=Path)
    args = parser.parse_args()
    require_compute_node(gpu=args.command in ('fit', 'score'))
    if args.command == 'drive':
        drive_measurements(args.root)
        return
    if args.command == 'prepare':
        if args.profile is None:
            parser.error('--profile is required for preparation')
        prepare_inputs(args.profile, args.root)
        return
    if args.cell is None:
        parser.error('--cell is required')
    metadata = json.loads((args.root / 'inputs.json').read_text())
    if metadata['tree'] != BASE_TREE:
        raise ValueError('measurement tree mismatch')
    path = Path(metadata['endpoint_path'])
    if file_hash(path) != metadata['endpoint_sha256']:
        raise ValueError('measurement endpoint changed')
    endpoint = PreparedEndpoint.from_json(path.read_text())
    study = study_cells()[args.cell]
    destination = args.root / args.cell
    destination.mkdir(exist_ok=True)
    if args.command == 'score':
        result = measure_scoring(endpoint, metadata['recipe'], study, destination / 'scoring-work')
        atomic_record(destination / 'scoring.json', result)
    else:
        if args.position is None:
            parser.error('--position is required for fitting')
        previous = (None if args.predecessor is None else FoldArtifacts.from_json(
            json.loads(args.predecessor.read_text())['predecessor']))
        measure_fit_task(endpoint, metadata['recipe'], study, destination / str(args.position),
                         position=args.position, predecessor=previous)




def fit_quantity(record, required):
    slices = record['slices']
    total = math.fsum(s['compute_seconds'] for s in slices)
    completed = 0
    boundary = 0.
    accumulated = 0.
    for item in slices:
        accumulated += item['compute_seconds']
        if item['epochs_completed'] > completed and item['minibatches_partial_epoch'] == 0:
            completed = item['epochs_completed']
            boundary = accumulated
    return epoch_projection(required=required, completed=completed,
        wall_seconds=total, boundary_seconds=boundary, attempts=len(record['attempts']))


def numbered(value, basis=('measured',)):
    """Give every numerical leaf a basis, including report dimensions."""
    if type(value) in (int, float):
        return {'value': value, 'basis': sorted(basis)}
    if isinstance(value, dict):
        if set(value) == {'value', 'basis'}:
            return value
        return {k: numbered(v, basis) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [numbered(v, basis) for v in value]
    return value


def measurement_report(root, overhead):
    import json
    from pathlib import Path
    root = Path(root)
    inputs = json.loads((root / 'inputs.json').read_text())
    projections, measurements, formulas = {}, {}, {}
    for cell, study in study_cells().items():
        directory = root / cell
        grid = study.grid_points
        outer = 3 * (1 + 2 * grid)
        positions = {'ssl': outer + 1, 'inner_outcome': 1, 'inner_origin': 1 + grid,
                     'refit_outcome': outer + 2, 'refit_origin': outer + 3}
        terms, records = {}, {}
        for term, position in positions.items():
            record = json.loads((directory / str(position) / 'progress.json').read_text())
            required = 30 if term == 'ssl' else study.frozen_epochs or 150
            terms[term] = {'tree': record['binding']['tree'],
                           'seconds': fit_quantity(record, required)}
            records[term] = numbered(record)
        scoring = json.loads((directory / 'scoring.json').read_text())
        terms['scoring'] = {'tree': scoring['tree'],
                            'seconds': Quantity(scoring['wall_seconds'], frozenset({'measured'}))}
        records['scoring'] = numbered(scoring)
        counts = inputs['work'][cell]['counts']
        projections[cell] = project(terms, counts, tree=inputs['tree'])
        formulas[cell] = {'expression': 'sum(count[term] * seconds[term]) / 3600',
                          'counts': numbered(counts),
                          'seconds': {k: None if v['seconds'] is None else v['seconds'].to_dict()
                                      for k, v in terms.items()}}
        measurements[cell] = records
    anchor = projections['e']
    loaded_receipt = calibration_receipt(inputs['profile_path'], inputs['frame_path'])
    receipt = {**loaded_receipt, 'compute_hours': loaded_receipt['compute_hours'].to_dict()}
    receipt['frame_matches'] = receipt['frame_hash'] == inputs['frame_sha256']
    check = calibration(anchor, Quantity(**{**receipt['compute_hours'],
        'basis': frozenset(receipt['compute_hours']['basis'])}),
        measured_anchor=anchor is not None and anchor.basis == {'measured'},
        frame_matches=receipt['frame_matches'],
        cell_matches=receipt['study'] == cell_values(study_cells()['e']),
        operations_match=inputs['work']['e']['counts'] == receipt['work']['counts'])
    cells = {}
    for name, q in projections.items():
        if q is not None and not check['passed']:
            q = q.with_basis('uncalibrated')
        cells[name] = {'study': numbered(cell_values(study_cells()[name])),
                       'compute_h100_hours_per_repetition': None if q is None else q.to_dict(),
                       'status': 'failed_measurement' if q is None else 'projected'}
        projections[name] = q
    overhead_q = Quantity(overhead['ratio']['value'], frozenset(overhead['ratio']['basis']))
    return {'schema_version': numbered(1), 'tree': BASE_TREE, 'cells': cells,
            'measurements': measurements, 'formula': formulas,
            'calibration': {**check, 'receipt': numbered(receipt),
                            'operations': {'projection': numbered(inputs['work']['e']),
                                           'receipt': numbered(receipt['work'])}},
            'overhead': overhead, 'feasibility': {
                cell: [] if q is None else numbered(feasibility(q, overhead_q), q.basis | overhead_q.basis)
                for cell, q in projections.items()},
            'limitations': [
                'Synthetic scoring inputs, timing only; no estimate is reported.',
                'First outer fold and first inner fold timings represent other folds.',
                'Outer SSL timing represents the SSL terms in the projection.',
                'Epoch slicing adds controller reconstruction work to measured compute.',
                'Idealised wall time assumes perfect utilisation of sixteen H100 GPUs.',
                'CPU tests do not validate GPU timing or Slurm placement.',
                'No final recipe or repetition count is recommended.']}




def drive_measurements(root):
    """Serialise each cell's genuine controller prerequisites, with two GPUs.

    The driver runs on a CPU compute allocation. All GPU operations are separate
    held jobs; this process never releases a job or changes node exclusions.
    """
    import fcntl
    import json
    from pathlib import Path
    import shlex
    import subprocess
    import sys
    import time

    require_compute_node()
    clean_identity()
    root = Path(root)
    metadata = json.loads((root / 'inputs.json').read_text())
    lock = (root / 'driver.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger_path = root / 'driver-jobs.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    failures = {}
    while True:
        for item in ledger:
            output = subprocess.check_output(['sacct', '-n', '-X', '-j', item['job_id'],
                '--format=JobIDRaw,Partition,NodeList,Elapsed,State,ExitCode', '-P'], text=True)
            rows = [line.split('|') for line in output.strip().splitlines() if line]
            rows = [row for row in rows if row[0] == item['job_id']]
            if len(rows) != 1:
                raise ValueError('scheduler accounting missing for ' + item['job_id'])
            _, partition, node, elapsed, state, code = rows[0][:6]
            item.update(partition=partition, node=node, elapsed=elapsed, state=state, exit_code=code)
        atomic_record(ledger_path, ledger)
        live = [j for j in ledger if j['state'] in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED')]
        if len(live) > 2:
            raise ValueError('measurement GPU concurrency exceeded')
        pending = []
        for cell in ('e', 'a', 'b', 'c', 'd'):
            if cell in failures or any(j['cell'] == cell for j in live):
                continue
            jobs = [j for j in ledger if j['cell'] == cell]
            if not (root / cell / 'scoring.json').exists():
                operation, previous = 'score', None
            else:
                schedule = metadata['work'][cell]['tasks_per_outer_seed']
                previous = None
                operation = None
                for position, (kind, fold, grid) in enumerate(schedule):
                    if kind == 'select':
                        continue
                    progress = root / cell / str(position) / 'progress.json'
                    record = json.loads(progress.read_text()) if progress.exists() else None
                    if record and record['slices'] and record['slices'][-1]['complete']:
                        previous = progress
                        continue
                    operation = str(position)
                    break
                if operation is None:
                    continue
            attempts = [j for j in jobs if j['operation'] == operation]
            if len(attempts) >= 2:
                failures[cell] = {'status': 'failed_measurement', 'operation': operation,
                                  'reason': 'sub-measurement incomplete after two attempts'}
                continue
            pending.append((cell, operation, previous))
        if not live and not pending:
            atomic_record(root / 'driver-failures.json', failures)
            return
        for cell, operation, previous in pending[:2-len(live)]:
            command = [sys.executable, '-B', '-m', 'oxyformer.validation.cost_probe',
                       'score' if operation == 'score' else 'fit', '--root', str(root), '--cell', cell]
            if operation != 'score':
                command.extend(['--position', operation])
                if previous is not None:
                    command.extend(['--predecessor', str(previous)])
            args = guarded_gpu_arguments(root / 'job-%j.log', 'cost-m2-' + cell + '-' + operation)
            job_id = subprocess.check_output(['sbatch', *args,
                '--wrap', shlex.join(command)], text=True).strip()
            if not job_id.isdigit():
                raise ValueError('invalid sbatch response')
            ledger.append({'job_id': job_id, 'cell': cell, 'operation': operation,
                           'state': 'PENDING', 'partition': 'standard', 'basis': ['measured']})
            atomic_record(ledger_path, ledger)
        print('measurement driver:', len(live), 'active jobs;', len(failures), 'failed cells', flush=True)
        time.sleep(60)


if __name__ == '__main__':
    main()
