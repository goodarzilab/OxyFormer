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


def epoch_projection(*, required, completed, wall_seconds, boundary_seconds, attempts, native_complete=False):
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
    if type(native_complete) is not bool or (native_complete and completed == 0):
        raise ValueError('native completion requires completed epochs')
    if completed == required or native_complete:
        return Quantity(wall_seconds, frozenset({'measured'}))
    if attempts != 2:
        raise ValueError('unfinished measurement requires its second attempt')
    if completed == 0:
        return None
    if boundary_seconds <= 0:
        raise ValueError('completed epochs require positive boundary time')
    return Quantity(boundary_seconds * required / completed, frozenset({'extrapolated'}))


def calibration(anchor, receipt, *, measured_anchor, frame_matches, cell_matches,
                operations_match, sources_match=True):
    reasons = []
    if not measured_anchor:
        reasons.append('anchor extrapolated')
    if not frame_matches:
        reasons.append('frame mismatch')
    if not cell_matches:
        reasons.append('cell mismatch')
    if not operations_match:
        reasons.append('operation mismatch')
    if not sources_match:
        reasons.append('protected source mismatch')
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


BASE_TREE = '57d0a04e419b0bc2c8bffb3adb46b586297746f4'
PROTECTED_SOURCES = (
    ('nested', 'src/oxyformer/training/nested_cv.py'),
    ('coverage', 'src/oxyformer/validation/coverage.py'),
    ('fit', 'src/oxyformer/training/fit.py'),
    ('pretrain', 'src/oxyformer/training/pretrain.py'),
)


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
    for name, path in PROTECTED_SOURCES:
        raw = subprocess.check_output(['git', 'show', tree + ':' + path])
        if Path(path).read_bytes() != raw:
            raise ValueError('protected source differs from counted-work tree')
        sources[name] = raw.decode()
        digests[path] = sha256(raw).hexdigest()
    return {**source_work(sources['nested'], sources['coverage'],
                          seed_count=seed_count, grid_points=grid_points),
            'tree': tree, 'source_sha256': digests}


def clean_identity(*, cwd=None):
    """A measurement starts only from committed, clean code; protected bytes are checked separately."""
    import subprocess
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=cwd).strip():
        raise ValueError('dirty checkout refused')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode().strip()


def require_month2_decision():
    from pathlib import Path
    import yaml
    decisions = yaml.safe_load(Path('configs/approvals.yaml').read_text())
    if not decisions.get('owner_decisions', {}).get('month2_followon'):
        raise ValueError('owner_decisions.month2_followon is absent')


def job_is_live(state):
    # Unknown/transitional states retain their slot; only explicit terminal
    # accounting states allow replacement. CANCELLED may include an actor id.
    terminal = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY',
                'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED', 'REVOKED'}
    return state.split()[0].rstrip('+') not in terminal


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


def epoch_timing(deadline, *, finalize=False, terminal=False):
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
            stop = self.evaluated and (phase != 'evaluation' if finalize and not terminal else True)
            self.evaluated = self.evaluated or phase == 'evaluation'
            self.stopped = self.stopped or stop
            return expired or self.stopped

    return EpochTiming()


def measurement_tasks(work):
    """Measure representative tasks; count the full tuning grid only in R2."""
    tasks = work['tasks_per_outer_seed']
    lookup = {tuple(task): i for i, task in enumerate(tasks)}
    inner_ssl = lookup['ssl', 0, None]
    outer_ssl = lookup['ssl', 3, None]
    return [
        (inner_ssl, None, False),
        (lookup['outcome', 0, 0], inner_ssl, False),
        (lookup['origin', 0, 0], inner_ssl, True),
        (outer_ssl, None, True),
        (lookup['outcome', 3, None], outer_ssl, True),
        (lookup['origin', 3, None], outer_ssl, True),
    ]


def timing_predecessor(config, endpoint, root, *, position, predecessor=None):
    """Set up one timing-only task using a fixed first-grid selection.

    This is not a completed tuning result. It carries no calibration or OOF
    selection claims, and the measurement stops before inferential finalization.
    Trained initialization tensors come only from the appropriate real SSL fit.
    """
    from copy import deepcopy
    from dataclasses import replace
    from pathlib import Path
    from oxyformer.training import nested_cv
    from oxyformer.training.checkpoint import (CheckpointRequest, load_checkpoint,
        save_checkpoint, model_state_hash)

    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    if predecessor is None:
        stop = CheckpointRequest()
        stop.request()
        predecessor = nested_cv.run_fold(replace(config, output_dir=str(root / 'blank'),
            predecessor=None, stop_request=stop), endpoint.outer, 1103,
            geography=endpoint.geography)
    checkpoint = predecessor.checkpoint
    controller = deepcopy(load_checkpoint(checkpoint, checkpoint.identity)['controller'])
    if controller['active'] is not None or controller['ssl_pending'] is not None:
        raise ValueError('timing setup requires a completed SSL task')
    controller.update(position=position, results=[], final={}, calibration=None,
        selection={kind: {'grid': 0, 'epochs': config.frozen_epochs}
                   for kind in ('outcome', 'origin')},
        timing_only='fixed first grid point; no inner selection performed')
    tensors = nested_cv._tensor_state(controller)
    lineage = replace(checkpoint.lineage, model_hash=model_state_hash(tensors),
                      parent_hashes=(checkpoint.content_hash,),
                      parameter_count=sum(t.numel() for t in tensors.values()))
    artifact = save_checkpoint(root, identity=checkpoint.identity, lineage=lineage,
        state={'model': tensors, 'controller': controller,
               'progress': {'epoch': position, 'step': position}},
        complete=False, reason='requested', predecessor=checkpoint)
    return replace(predecessor, checkpoint=artifact)


def guarded_gpu_arguments(output, name):
    """The orchestrator alone releases these held requests after node guarding."""
    return ['--parsable', '--hold', '--no-requeue', '--partition=standard', '--account=root',
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


def partial_epoch_batches(progress, training_batches, *, validation_batch_size=None):
    """Count transactions since the last completed training/validation epoch."""
    partial = progress['step'] - progress['epoch'] * training_batches
    if partial > 0 and validation_batch_size is not None and progress['phase'] == 'validation':
        partial += math.ceil(progress['validation_cursor'] / validation_batch_size)
    if partial < 0:
        raise ValueError('epoch and minibatch counters disagree')
    return partial


def measure_fit_task(endpoint, recipe, study, directory, *, position, predecessor=None,
                     deadline=None, standalone=False):
    """Run a single controller task, checkpointing after each completed epoch.

    Standalone tasks use the explicitly declared first grid point, without
    running a tuning grid. Their initialization still requires a real SSL fit.
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
    if standalone:
        allowed = {p for p, _, fixed in measurement_tasks(work) if fixed}
        if position not in allowed:
            raise ValueError('unsupported standalone timing position')
        binding['harness_head'] = clean_identity(cwd=Path(__file__).resolve().parents[3])
        binding['mode'] = 'fixed first grid point, timing only'
    journal = root / 'progress.json'
    record = resume_record(journal, binding)
    job = os.environ['SLURM_JOB_ID']
    record.setdefault('execution_harness_heads', {})[job] = clean_identity(
        cwd=Path(__file__).resolve().parents[3])
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
    if standalone:
        record['timing_setup'] = {'mode': binding['mode'],
            'epochs': base_config.ssl_epochs if kind == 'ssl' else base_config.frozen_epochs,
            'selection_performed': False}
        if kind != 'ssl':
            record['timing_setup'].update(grid_index=0,
                learning_rate=base_config.grid[0][0], dropout=base_config.grid[0][1])
        if record['predecessor'] is None:
            if kind != 'ssl':
                if predecessor is None:
                    raise ValueError('real SSL predecessor required')
                saved = load_checkpoint(predecessor.checkpoint, predecessor.checkpoint.identity)['controller']
                if fold not in saved['initializations']:
                    raise ValueError('SSL predecessor has the wrong fitting partition')
            setup = root / ('timing-setup-job-' + job)
            suffix = 0
            while setup.exists():
                suffix += 1
                setup = root / ('timing-setup-job-' + job + '-' + str(suffix))
            previous = timing_predecessor(base_config, endpoint, setup,
                position=position, predecessor=predecessor)
            record['predecessor'] = previous.to_json()
            atomic_record(journal, record)
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
        timing = ssl_timing(deadline) if kind == 'ssl' else epoch_timing(deadline,
            finalize=finalize, terminal=standalone and position == len(tasks) - 1)
        slice_batches = batches
        if kind == 'ssl' and before and before['ssl_pending']:
            from oxyformer.training.checkpoint import CheckpointArtifact
            pending = CheckpointArtifact.from_json(before['ssl_pending'])
            progress = load_checkpoint(pending, pending.identity)['progress']
            slice_batches -= partial_epoch_batches(progress,
                max(1, math.ceil((len(fitting)-len(stopping))/256)), validation_batch_size=256)
            if slice_batches <= 0:
                raise ValueError('SSL partial epoch exceeds its transaction count')
        slice_root = root / ('slice-' + str(len(record['slices'])) + '-job-' + job)
        # A killed call can leave a private incomplete directory. Never reuse it.
        suffix = 0
        while slice_root.exists():
            suffix += 1
            slice_root = root / ('slice-' + str(len(record['slices'])) + '-job-' + job + '-' + str(suffix))
        config = replace(base_config, output_dir=str(slice_root), predecessor=previous,
                         max_batches=slice_batches if kind == 'ssl' else None,
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
            partial = partial_epoch_batches(progress,
                max(1, math.ceil((len(fitting)-len(stopping))/256)), validation_batch_size=256)
        else:
            progress = after['active']['progress'] if after['active'] is not None else None
            # The epoch slice before held-out evaluation retains the actual
            # terminal progress, including native stopping below the cap.
            if completed_task and active is None:
                raise ValueError('completed nuisance task lacks observed terminal progress')
            epochs = active['progress']['epoch'] if completed_task else progress['epoch']
            partial = 0 if progress is None else partial_epoch_batches(progress, batches)
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
    pinned_work(seed_count=study.seed_count, grid_points=study.grid_points)
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
    return {'binding': {'endpoint': endpoint.content_hash, 'tree': BASE_TREE,
                        'recipe': recipe, 'study': cell_values(study)},
            'tree': BASE_TREE, 'head': head, 'study': cell_values(study),
            'inputs': 'synthetic inputs, timing only', 'basis': ['measured'],
            'wall_seconds': wall, 'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(),
            'prediction_rows': len(observed)}


def calibration_receipt(path, frame_path):
    """Load the sealed repetition compute target, retaining the exact file hash."""
    from hashlib import sha256
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
    sources, baseline = {}, {}
    for name, source_path in PROTECTED_SOURCES:
        sources[name] = subprocess.check_output(['git', 'show', tree + ':' + source_path])
        baseline[name] = subprocess.check_output(['git', 'show', BASE_TREE + ':' + source_path])
    receipt_work = source_work(
        sources['nested'].decode(), sources['coverage'].decode(),
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
            'sources_match': sources == baseline,
            'source_sha256': {source_path: sha256(sources[name]).hexdigest()
                              for name, source_path in PROTECTED_SOURCES},
            'path': str(path), 'sha256': file_hash(path), 'work': receipt_work}


def leaf_overhead(paths):
    """Sum reserved allocations and each leaf's own compute exactly once."""
    from pathlib import Path
    from oxyformer.validation.coverage import leaf_allocation, read_leaf_measurements
    from oxyformer.provenance import file_hash

    paths = tuple(map(Path, paths))
    if len(paths) != 140 or len(set(paths)) != 140:
        raise ValueError('exactly 140 unique campaign-D leaf records required')
    own, allocations, files = [], [], []
    for path in paths:
        record = read_leaf_measurements(path)
        leaf = record['leaves'][-1]
        own.append(leaf)
        allocation = leaf_allocation(record['binding']['wall_seconds'])['allocation_seconds']
        allocations.append(allocation)
        files.append({'path': str(path), 'sha256': file_hash(path),
                      'allocation_seconds': numbered(allocation),
                      'compute_seconds': numbered(leaf['compute_seconds'])})
    if len({leaf['task_id'] for leaf in own}) != 140:
        raise ValueError('duplicate campaign leaf')
    allocated = Quantity(math.fsum(allocations) / 3600, frozenset({'measured'}))
    runner_wall = Quantity(math.fsum(leaf['wall_seconds'] for leaf in own) / 3600,
                           frozenset({'measured'}))
    compute = Quantity(math.fsum(leaf['compute_seconds'] for leaf in own) / 3600,
                       frozenset({'measured'}))
    return {'allocated_hours': allocated.to_dict(), 'compute_hours': compute.to_dict(),
            'runner_wall_hours': runner_wall.to_dict(),
            'allocation_definition': 'coverage.leaf_allocation(binding.wall_seconds): stage plus setup, rounded to whole minutes',
            'ratio': ratio(allocated, compute).to_dict(), 'files': files}


def cell_values(study):
    return {name: getattr(study, name) for name in
            ('reference_cap', 'frozen_epochs', 'grid_points', 'seed_count')}


def study_cells():
    from oxyformer.training.fit import RecipeStudy
    return {'a': RecipeStudy(), 'b': RecipeStudy(reference_cap=32),
            **{cell: RecipeStudy(reference_cap=32, frozen_epochs=epochs, grid_points=2, seed_count=1)
               for cell, epochs in (('c', 150), ('d', 60), ('e', 30))}}


def lifecycle_identity(root):
    """Read-only validation of root inputs and every reusable artifact.

    File SHA256 authenticates stored bytes; PreparedEndpoint.content_hash binds
    fitting inputs. Legacy scoring has no such evidence and is never inferred
    from a filename or from a fitting journal. Preserve it and refuse reuse.
    """
    import json
    from pathlib import Path
    from oxyformer.provenance import file_hash
    from oxyformer.training.nested_cv import PreparedEndpoint

    root = Path(root)
    inputs = json.loads((root / 'inputs.json').read_text())
    def refuse(detail):
        raise ValueError('measurement lifecycle identity: ' + detail)
    if inputs.get('tree') != BASE_TREE:
        refuse('tree mismatch')
    path = Path(inputs['endpoint_path'])
    if not path.is_absolute():
        refuse('endpoint path must be absolute')
    if file_hash(path) != inputs['endpoint_sha256']:
        refuse('endpoint file changed')
    endpoint_hash = PreparedEndpoint.from_json(path.read_text()).content_hash
    if inputs.get('endpoint_content_hash', endpoint_hash) != endpoint_hash:
        refuse('endpoint content changed')
    if file_hash(inputs['frame_path']) != inputs['frame_sha256']:
        refuse('frame changed')
    for cell, study in study_cells().items():
        work = inputs['work'][cell]
        expected_work = pinned_work(seed_count=study.seed_count, grid_points=study.grid_points)
        # JSON normalisation accounts for task tuples stored as arrays.
        if json.dumps(work, sort_keys=True) != json.dumps(expected_work, sort_keys=True):
            refuse('work mismatch for cell ' + cell)
        expected = {'tree': inputs['tree'], 'endpoint': endpoint_hash,
                    'recipe': inputs['recipe'], 'study': cell_values(study)}
        for artifact in sorted((root / cell).glob('*/progress.json')):
            record = json.loads(artifact.read_text())
            binding = record.get('binding', {})
            try:
                position = int(artifact.parent.name)
                task = work['tasks_per_outer_seed'][position]
                if position < 0 or task[0] == 'select':
                    raise ValueError()
            except (ValueError, IndexError):
                refuse('invalid fit position: ' + str(artifact))
            for key, value in {**expected, 'position': position}.items():
                if binding.get(key) != value:
                    refuse(key + ' mismatch: ' + str(artifact))
        score = root / cell / 'scoring.json'
        if score.exists():
            record = json.loads(score.read_text())
            if 'binding' not in record:
                refuse('legacy scoring excluded: endpoint identity unverifiable; '
                       'matching measurement required: ' + str(score))
            for key, value in expected.items():
                if record['binding'].get(key) != value:
                    refuse(key + ' mismatch: ' + str(score))
            if record.get('tree') != inputs['tree'] or record.get('study') != cell_values(study):
                refuse('scoring dimensions mismatch: ' + str(score))
    return inputs, endpoint_hash


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
    require_month2_decision()
    profile_root, root = Path(profile_root).resolve(), Path(destination).resolve()
    existing = lifecycle_identity(root) if (root / 'inputs.json').exists() else None
    if existing is None and root.exists() and any(root.iterdir()):
        raise ValueError('measurement lifecycle identity: nonempty root lacks inputs.json')
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
    candidate = {'tree': BASE_TREE, 'recipe': params['recipe'],
                 'frame_sha256': file_hash(paths['frame.json']),
                 'endpoint_content_hash': endpoint.content_hash}
    if existing is not None:
        stored, stored_endpoint = existing
        previous = {**stored, 'endpoint_content_hash': stored_endpoint}
        changed = [key for key, value in candidate.items() if previous.get(key) != value]
        if changed:
            raise ValueError('measurement lifecycle identity: preparation changed ' + ', '.join(changed))
        return stored
    root.mkdir(parents=True, exist_ok=True)
    endpoint_path = root / 'measurement-endpoint.json'
    endpoint_path.write_text(endpoint.to_json())
    metadata = {'tree': BASE_TREE, 'prepared_head': head, 'recipe': params['recipe'],
                'endpoint_path': str(endpoint_path), 'endpoint_sha256': file_hash(endpoint_path),
                'endpoint_content_hash': endpoint.content_hash,
                'frame_path': str(paths['frame.json']), 'frame_sha256': file_hash(paths['frame.json']),
                'profile_path': str(profile_root / 'budget_record.json'),
                'receipt': {**receipt, 'compute_hours': receipt['compute_hours'].to_dict()},
                'work': {c: pinned_work(seed_count=s.seed_count, grid_points=s.grid_points)
                         for c, s in study_cells().items()}}
    atomic_record(root / 'inputs.json', metadata)
    return metadata


def main():
    import argparse
    import os
    import json
    from pathlib import Path
    from oxyformer.provenance import file_hash
    from oxyformer.training.nested_cv import PreparedEndpoint
    from oxyformer.training.fit import FoldArtifacts

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'fit', 'score', 'drive', 'report'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--cell', choices=tuple('abcde'))
    parser.add_argument('--position', type=int)
    parser.add_argument('--predecessor', type=Path)
    parser.add_argument('--overhead', type=Path)
    parser.add_argument('--fit-checkout', type=Path)
    parser.add_argument('--harness-checkout', type=Path)
    parser.add_argument('--standalone', action='store_true')
    args = parser.parse_args()
    require_compute_node(gpu=args.command in ('fit', 'score'))
    if args.command == 'report':
        if args.overhead is None:
            parser.error('--overhead is required')
        publish_report(args.root, json.loads(args.overhead.read_text()))
        return
    if args.command == 'drive':
        drive_measurements(args.root, fit_checkout=args.fit_checkout,
                           harness_checkout=args.harness_checkout)
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
        result = measure_scoring(endpoint, metadata['recipe'], study, destination / ('scoring-work-' + os.environ['SLURM_JOB_ID']))
        atomic_record(destination / 'scoring.json', result)
    else:
        if args.position is None:
            parser.error('--position is required for fitting')
        previous = (None if args.predecessor is None else FoldArtifacts.from_json(
            json.loads(args.predecessor.read_text())['predecessor']))
        measure_fit_task(endpoint, metadata['recipe'], study, destination / str(args.position),
                         position=args.position, predecessor=previous, standalone=args.standalone)




def fit_statistics(record):
    """Summarise durable slices without counting partial epochs as boundaries."""
    slices = record['slices']
    compute = wall = boundary_compute = boundary_wall = 0.
    completed = boundary_epochs = 0
    for item in slices:
        compute += item['compute_seconds']
        wall += item['wall_seconds']
        completed = max(completed, item['epochs_completed'])
        if item['epochs_completed'] > boundary_epochs and item['minibatches_partial_epoch'] == 0:
            boundary_epochs = item['epochs_completed']
            boundary_compute, boundary_wall = compute, wall
    last = slices[-1] if slices else {}
    return {'compute_seconds': compute, 'wall_seconds': wall,
            'epochs_completed': completed,
            'epochs_at_last_completed_boundary': boundary_epochs,
            'compute_seconds_at_last_completed_epoch': boundary_compute,
            'wall_seconds_at_last_completed_epoch': boundary_wall,
            'minibatches_partial_epoch': last.get('minibatches_partial_epoch'),
            'minibatches_per_epoch': last.get('minibatches_per_epoch'),
            'peak_gpu_memory_bytes': max((s['peak_gpu_memory_bytes'] for s in slices), default=None),
            'native_complete': bool(last.get('complete', False))}


def fit_quantity(record, required):
    stats = fit_statistics(record)
    if stats['epochs_completed'] >= required and not stats['native_complete']:
        if len(record['attempts']) != 2:
            raise ValueError('unfinished evaluation requires its second attempt')
        return None
    completed = (stats['epochs_completed'] if stats['native_complete']
                 else stats['epochs_at_last_completed_boundary'])
    return epoch_projection(required=required, completed=completed,
        wall_seconds=stats['compute_seconds'],
        boundary_seconds=stats['compute_seconds_at_last_completed_epoch'],
        attempts=len(record['attempts']), native_complete=stats['native_complete'])


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


def public_fit_record(path, record):
    """Link checkpoint journals without copying their opaque model descriptors."""
    from oxyformer.provenance import file_hash
    return numbered({**{k: v for k, v in record.items() if k != 'predecessor'},
                     'summary': fit_statistics(record),
                     'journal_path': str(path), 'journal_sha256': file_hash(path)})


def projection_counts(work):
    """Partition the walked SSL tasks by their actual fitting partition."""
    tasks = work['tasks_per_outer_seed']
    inner = sum(kind == 'ssl' and fold != 3 for kind, fold, grid in tasks)
    outer = sum(kind == 'ssl' and fold == 3 for kind, fold, grid in tasks)
    multiplier = work['outer_folds'] * work['seed_count']
    counts = dict(work['counts'])
    if counts.pop('ssl') != (inner + outer) * multiplier:
        raise ValueError('SSL source counts disagree')
    return {**counts, 'inner_ssl': inner * multiplier, 'refit_ssl': outer * multiplier}


def measurement_report(root, overhead):
    import json
    from pathlib import Path
    root = Path(root)
    inputs, _ = lifecycle_identity(root)
    terminal_path = root / 'driver-failures.json'
    if not terminal_path.exists():
        raise ValueError('measurement driver has not finished')
    failures = json.loads(terminal_path.read_text())
    jobs = json.loads((root / 'driver-jobs.json').read_text())
    if any(job_is_live(j['state']) for j in jobs):
        raise ValueError('measurements are still running')
    for cell, failure in failures.items():
        for item in failure.get('operations', [failure]):
            attempts = [j for j in jobs if j['cell'] == cell and j['operation'] == item['operation']]
            if len(attempts) != 2:
                raise ValueError('failed measurement requires two recorded attempts')
    projections, measurements, formulas = {}, {}, {}
    for cell, study in study_cells().items():
        directory = root / cell
        schedule = inputs['work'][cell]['tasks_per_outer_seed']
        positions = {}
        for position, (kind, fold, grid) in enumerate(schedule):
            if fold == 3 and kind in ('ssl', 'outcome', 'origin'):
                positions['refit_' + kind] = position
            elif fold == 0 and kind == 'ssl':
                positions['inner_ssl'] = position
            elif fold == 0 and grid == 0 and kind in ('outcome', 'origin'):
                positions['inner_' + kind] = position
        if set(positions) != {'inner_ssl', 'refit_ssl', 'inner_outcome', 'inner_origin', 'refit_outcome', 'refit_origin'}:
            raise ValueError('representative operations missing from counted work')
        terms, records = {}, {}
        for term, position in positions.items():
            path = directory / str(position) / 'progress.json'
            required = 30 if term.endswith('ssl') else study.frozen_epochs or 150
            if not path.exists():
                if cell not in failures:
                    raise ValueError('missing measurement: ' + cell + '/' + term)
                terms[term] = {'tree': BASE_TREE, 'seconds': None}
                records[term] = {'status': 'failed_measurement', 'basis': ['failed_measurement'],
                    'reason': 'not performed after prerequisite failure: ' + failures[cell]['reason']}
                continue
            record = json.loads(path.read_text())
            # A killed process may never have appended its job id to the journal.
            # Scheduler attempts remain evidence even when no new slice survived.
            attempts = [j['job_id'] for j in jobs if j['cell'] == cell and j['operation'] == str(position)]
            if attempts:
                record['attempts'] = attempts
            seconds = fit_quantity(record, required)
            terms[term] = {'tree': record['binding']['tree'], 'seconds': seconds}
            records[term] = public_fit_record(path, record)
            records[term]['requested_max_epochs'] = numbered(required)
            records[term]['duration_seconds'] = None if seconds is None else seconds.to_dict()
            if seconds is None:
                stats = fit_statistics(record)
                reason = ('post-fit evaluation incomplete after two attempts'
                          if stats['epochs_completed'] >= required else
                          'no durable completed-epoch timing boundary after two attempts'
                          if stats['epochs_completed'] else 'zero completed epochs after two attempts')
                records[term]['summary'] = numbered(stats, ('failed_measurement',))
                records[term]['requested_max_epochs'] = numbered(required, ('failed_measurement',))
                records[term].update(status='failed_measurement', basis=['failed_measurement'],
                                     reason=reason)
        score_path = directory / 'scoring.json'
        if score_path.exists():
            scoring = json.loads(score_path.read_text())
            terms['scoring'] = {'tree': scoring['tree'],
                'seconds': Quantity(scoring['wall_seconds'], frozenset({'measured'}))}
            records['scoring'] = numbered(scoring)
        elif cell in failures:
            terms['scoring'] = {'tree': BASE_TREE, 'seconds': None}
            records['scoring'] = {'status': 'failed_measurement', 'basis': ['failed_measurement'],
                                 'reason': failures[cell]['reason']}
        else:
            raise ValueError('missing scoring measurement: ' + cell)
        auxiliary = {}
        for position, (kind, fold, grid) in enumerate(schedule):
            path = directory / str(position) / 'progress.json'
            if position not in positions.values() and path.exists():
                auxiliary[str(position)] = public_fit_record(path, json.loads(path.read_text()))
        records['prerequisites'] = auxiliary
        counts = projection_counts(inputs['work'][cell])
        projections[cell] = project(terms, counts, tree=inputs['tree'])
        formulas[cell] = {'expression': 'sum(count[term] * seconds[term]) / 3600',
                          'counts': numbered(counts),
                          'seconds': {k: None if v['seconds'] is None else v['seconds'].to_dict()
                                      for k, v in terms.items()}}
        measurements[cell] = records
    anchor = projections['e']
    loaded_receipt = calibration_receipt(inputs['profile_path'], inputs['frame_path'])
    receipt = {**loaded_receipt, 'compute_hours': loaded_receipt['compute_hours'].to_dict()}
    receipt['frame_matches'] = (receipt['frame_matches'] and
                               receipt['frame_hash'] == inputs['frame_sha256'])
    check = calibration(anchor, loaded_receipt['compute_hours'],
        measured_anchor=anchor is not None and anchor.basis == {'measured'},
        frame_matches=receipt['frame_matches'],
        cell_matches=receipt['study'] == cell_values(study_cells()['e']),
        operations_match=projection_counts(inputs['work']['e']) == projection_counts(receipt['work']),
        sources_match=receipt['sources_match'])
    cells = {}
    for name, q in projections.items():
        if q is not None and not check['passed']:
            q = q.with_basis('uncalibrated')
        cells[name] = {'study': numbered(cell_values(study_cells()[name])),
                       'compute_h100_hours_per_repetition': None if q is None else q.to_dict(),
                       'status': 'failed_measurement' if q is None else 'projected'}
        if q is None:
            cells[name]['reason'] = failures.get(name, {}).get('reason', 'a required term failed measurement')
        projections[name] = q
    overhead_q = Quantity(overhead['ratio']['value'], frozenset(overhead['ratio']['basis']))
    return {'schema_version': numbered(1), 'tree': BASE_TREE, 'cells': cells,
            'measurements': measurements, 'formula': formulas,
            'calibration': {**check, 'receipt': numbered(receipt),
                            'operations': {'projection': numbered({**inputs['work']['e'],
                                'counts': projection_counts(inputs['work']['e'])}),
                                'receipt': numbered({**receipt['work'],
                                    'counts': projection_counts(receipt['work'])})}},
            'overhead': overhead, 'feasibility': {
                cell: [] if q is None else numbered(feasibility(q, overhead_q), q.basis | overhead_q.basis)
                for cell, q in projections.items()},
            'limitations': [
                'Synthetic scoring inputs, timing only; no estimate is reported.',
                'First outer fold and first inner fold timings represent other folds.',
                'Registered seed 1103 represents the other registered neural seeds.',
                'The first hyperparameter grid point represents other grid points in each inner-fit timing term.',
                'Outer refits use the recorded fixed first grid point at the cell epoch count, without running inner selection. Selection workload enters the projection through the counted inner fits.',
                'Inner and outer SSL have separate measured timings and source-derived counts.',
                'The frozen plan specifies maximum thirty SSL epochs with patience five; native patience completion is measured, with its actual epoch count.',
                'Epoch slicing adds controller reconstruction work to measured compute.',
                'Fit summaries sum returned durable slices; whole-job wall time, including unreturned interrupted calls, is recorded separately in slurm_jobs.json.',
                'Peak GPU memory is the largest recorded PyTorch allocator peak; CUDA context, non-PyTorch allocations and any unreturned interrupted call are excluded. Missing observations are null.',
                'The overhead numerator is reserved allocation from sealed stage budgets, not elapsed runner wall; failed scheduler attempts are outside the sealed-leaf ratio.',
                'Idealised wall time assumes perfect utilisation of sixteen H100 GPUs.',
                'CPU tests do not validate GPU timing or Slurm placement.',
                'No final recipe or repetition count is recommended.']}


def recover_submission(item):
    """Resolve an accepted submission whose reply was lost; never guess/retry."""
    import getpass
    import subprocess
    name = item['submission_name']
    outputs = [
        subprocess.check_output(['sacct', '-n', '-X', '--user', getpass.getuser(),
            '--name', name, '--starttime', item['submission_day'],
            '--format=JobIDRaw,JobName%100', '-P'], text=True),
        subprocess.check_output(['squeue', '--me', '--noheader', '--format=%i|%j'], text=True),
    ]
    found = set()
    for output in outputs:
        for line in output.splitlines():
            fields = line.strip().split('|')
            if len(fields) >= 2 and fields[1].strip() == name and fields[0].strip().isdigit():
                found.add(fields[0].strip())
    if len(found) != 1:
        raise ValueError('submission outcome unresolved; reconcile unique job name ' + name)
    item['job_id'] = found.pop()
    item['state'] = 'PENDING'


def submit_measurement(ledger_path, ledger, *, root, cell, operation, command, checkout):
    """Persist intent before sbatch, then bind its reply to that same intent."""
    from datetime import datetime, timezone
    import shlex
    import subprocess
    import uuid
    name = 'cost-m2-' + cell + '-' + operation + '-' + uuid.uuid4().hex[:16]
    item = {'job_id': None, 'cell': cell, 'operation': operation,
            'state': 'SUBMITTING', 'partition': 'standard', 'basis': ['measured'],
            'submission_name': name, 'submission_day': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            'checkout': str(checkout)}
    ledger.append(item)
    atomic_record(ledger_path, ledger)
    args = guarded_gpu_arguments(root / 'job-%j.log', name)
    job_id = subprocess.check_output(['sbatch', *args, '--chdir', str(checkout),
        '--wrap', shlex.join(['env', 'PYTHONPATH=src', *command])], text=True).strip()
    if not job_id.isdigit():
        raise ValueError('invalid sbatch response; submission intent retained')
    item.update(job_id=job_id, state='PENDING')
    atomic_record(ledger_path, ledger)


def next_measurement(directory, work, jobs):
    """Continue independent terms after an exhausted measurement attempt pair."""
    import json
    from pathlib import Path
    directory = Path(directory)
    failures = []

    def exhausted(operation):
        if sum(j['operation'] == operation for j in jobs) < 2:
            return False
        failures.append({'operation': operation,
                         'reason': 'sub-measurement ' + operation + ' incomplete after two attempts'})
        return True

    def completed(position):
        path = directory / str(position) / 'progress.json'
        record = json.loads(path.read_text()) if path.exists() else None
        return bool(record and record['slices'] and record['slices'][-1]['complete'])

    if not (directory / 'scoring.json').exists() and not exhausted('score'):
        return ('score', None, False), failures
    for position, source, fixed in measurement_tasks(work):
        operation = str(position)
        if completed(position) or exhausted(operation):
            continue
        if source is not None and not completed(source):
            # A real SSL initialization is required. Other independent terms
            # still run, and missing dependent terms remain explicit failures.
            continue
        previous = None if source is None else directory / str(source) / 'progress.json'
        return (operation, previous, fixed), failures
    return None, failures


def drive_measurements(root, *, fit_checkout=None, harness_checkout=None):
    """Schedule representative fits and their SSL prerequisites, with two GPUs.

    The driver runs on a CPU compute allocation. All GPU operations are separate
    held jobs; this process never releases a job or changes node exclusions.
    """
    import fcntl
    import json
    from pathlib import Path
    import subprocess
    import sys
    import time

    require_compute_node()
    clean_identity()
    require_month2_decision()
    fitting_checkout = Path.cwd() if fit_checkout is None else Path(fit_checkout).resolve()
    harness_checkout = Path.cwd() if harness_checkout is None else Path(harness_checkout).resolve()
    if fit_checkout is not None:
        clean_identity(cwd=fitting_checkout)
    if harness_checkout != Path.cwd():
        clean_identity(cwd=harness_checkout)
    root = Path(root).resolve()
    metadata, _ = lifecycle_identity(root)
    lock = (root / 'driver.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger_path = root / 'driver-jobs.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    failures = {}
    while True:
        for item in ledger:
            if item['job_id'] is None:
                recover_submission(item)
                atomic_record(ledger_path, ledger)
            output = subprocess.check_output(['sacct', '-n', '-X', '-j', item['job_id'],
                '--format=JobIDRaw,Partition,NodeList,Elapsed,State,ExitCode', '-P'], text=True)
            rows = [line.split('|') for line in output.strip().splitlines() if line]
            rows = [row for row in rows if row[0] == item['job_id']]
            if len(rows) != 1:
                raise ValueError('scheduler accounting missing for ' + item['job_id'])
            _, partition, node, elapsed, state, code = rows[0][:6]
            item.update(partition=partition, node=node, elapsed=elapsed, state=state, exit_code=code)
        atomic_record(ledger_path, ledger)
        live = [j for j in ledger if job_is_live(j['state'])]
        if len(live) > 2:
            raise ValueError('measurement GPU concurrency exceeded')
        pending = []
        for cell in ('e', 'a', 'b', 'c', 'd'):
            if any(j['cell'] == cell for j in live):
                continue
            jobs = [j for j in ledger if j['cell'] == cell]
            next_job, issues = next_measurement(root / cell, metadata['work'].get(cell), jobs)
            if issues:
                failures[cell] = {'status': 'failed_measurement',
                    'operation': issues[0]['operation'], 'operations': issues,
                    'reason': '; '.join(item['reason'] for item in issues)}
            else:
                failures.pop(cell, None)
            if next_job is not None:
                pending.append((cell, *next_job))
        if not live and not pending:
            atomic_record(root / 'driver-failures.json', failures)
            return
        for cell, operation, previous, standalone in pending[:2-len(live)]:
            entry = ([str(harness_checkout / 'src/oxyformer/validation/cost_probe.py')]
                     if operation != 'score' else ['-m', 'oxyformer.validation.cost_probe'])
            command = [sys.executable, '-B', *entry,
                       'score' if operation == 'score' else 'fit', '--root', str(root), '--cell', cell]
            if operation != 'score':
                command.extend(['--position', operation])
                if previous is not None:
                    command.extend(['--predecessor', str(previous)])
                if standalone:
                    command.append('--standalone')
            submit_measurement(ledger_path, ledger, root=root, cell=cell,
                operation=operation, command=command,
                checkout=Path.cwd() if operation == 'score' else fitting_checkout)
        print('measurement driver:', len(live), 'active jobs;', len(failures), 'failed cells', flush=True)
        time.sleep(60)




def publish_report(root, overhead):
    """Publish only after every required measurement has a terminal record."""
    from pathlib import Path
    require_compute_node()
    clean_identity()
    root = Path(root)
    report = measurement_report(root, overhead)
    destination = root.parent
    import json
    jobs_path = destination / 'slurm_jobs.json'
    existing = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    jobs = {item['job_id']: item for item in existing}
    for item in json.loads((root / 'driver-jobs.json').read_text()):
        days, clock = item['elapsed'].split('-') if '-' in item['elapsed'] else ('0', item['elapsed'])
        hours, minutes, seconds = map(int, clock.split(':'))
        wall_seconds = int(days) * 86400 + hours * 3600 + minutes * 60 + seconds
        jobs[item['job_id']] = {**jobs.get(item['job_id'], {}),
            **{key: item[key] for key in ('job_id', 'partition', 'node', 'elapsed', 'state', 'exit_code')},
            'sub_measurement': 'cell ' + item['cell'] + ', controller operation ' + item['operation'],
            'wall_seconds': numbered(wall_seconds),
            'basis': ['measured']}
    atomic_record(jobs_path, list(jobs.values()))
    atomic_record(destination / 'recipe_cost_m2.json', report)

    def display(quantity):
        if quantity is None:
            return 'failed_measurement; no projection'
        return f"{quantity['value']:.6g} [" + ', '.join(quantity['basis']) + ']'

    check = report['calibration']
    lines = ['# Recipe cost study', '',
             'Calibration: ' + ('passed' if check['passed'] else '**UNCALIBRATED**'),
             'Calibration ratio: ' + display(check['ratio']), '']
    if check['reasons']:
        lines.extend(['Reasons: ' + '; '.join(check['reasons']) + '.',
            'Potential contributors include representative-fold differences, reconstruction work',
            'from epoch slicing, and the receipt identity differences named above.', ''])
    lines.extend(['Counted-work tree: ' + report['tree'], '',
                  'Every displayed numeric value has its propagated basis in brackets.', '',
                  '| Cell | Compute H100-hours per repetition |', '| --- | --- |'])
    for cell, value in report['cells'].items():
        lines.append('| ' + cell + ' | ' + display(value['compute_h100_hours_per_repetition']) + ' |')
    failure_lines = ['Failure reason for cell ' + cell + ': ' + value['reason']
                     for cell, value in report['cells'].items() if value.get('reason')]
    if failure_lines:
        lines.extend(['', *failure_lines])
    lines.extend(['', 'Formula: sum(count per repetition × seconds per fit or pass), converted to H100-hours.', '',
                  '| Cell | Term | Count per repetition | Seconds per fit or pass |',
                  '| --- | --- | --- | --- |'])
    for cell, formula in report['formula'].items():
        for term, count in formula['counts'].items():
            lines.append('| ' + cell + ' | ' + term + ' | ' + display(count) + ' | ' +
                         display(formula['seconds'][term]) + ' |')
    lines.extend(['',
        'Refit timing uses the fixed first grid point at the cell epoch count, as clarified by the orchestrator.',
        'Inner selection is not run for this timing: its workload enters the projection through the counted inner fits.', '',
        '| Cell | Refit | Grid index | Learning rate | Dropout | Epoch bound |',
        '| --- | --- | --- | --- | --- | --- |'])
    for cell, records in report['measurements'].items():
        for term in ('refit_outcome', 'refit_origin'):
            setup = records[term].get('timing_setup')
            if setup is not None:
                lines.append('| ' + cell + ' | ' + term + ' | ' + ' | '.join(
                    display(setup[k]) for k in ('grid_index', 'learning_rate', 'dropout', 'epochs')) + ' |')
    lines.extend(['', 'Campaign allocated/compute ratio: ' + display(overhead['ratio']),
        'Reserved allocation GPU-hours: ' + display(overhead['allocated_hours']),
        'Measured compute GPU-hours: ' + display(overhead['compute_hours']), '',
        '| Cell | Repetitions | Scenarios | Compute H100-hours | Allocated H100-hours | Idealised wall hours at sixteen GPUs |',
        '| --- | --- | --- | --- | --- | --- |'])
    for cell, rows in report['feasibility'].items():
        for row in rows:
            lines.append('| ' + cell + ' | ' + ' | '.join(display(row[k]) for k in
                ('repetitions', 'scenarios', 'compute_h100_hours', 'allocated_h100_hours',
                 'idealised_lower_bound_wall_hours_at_16_gpus')) + ' |')
    lines.extend(['', *report['limitations'], '',
                  'Full validation, mutation and review evidence are recorded separately.',
                  'This measurement report alone is not a completion or merge authorization.'])
    (destination / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
