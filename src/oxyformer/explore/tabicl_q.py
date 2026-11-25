"""Exploratory A6-Q: frozen TabICLv2 outcome-only Q for saved simulation chains.

No weight changes: per outer fold the pinned regressor is conditioned in context
on an outcome-blind sample of the Q-only outer fit's training rows and queried
per held-out target at the factual and policy dose, in resumable batches. Only
the registered comparator's checkpoint identities are read.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module, metadata
import math
import os
from pathlib import Path
import platform
from shutil import copyfile
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch

from oxyformer.models.tabicl_comparator import (CHECKPOINT_IDENTITY_FIELDS, PACKAGE_PINS,
    REGISTERED_CHECKPOINTS, _matrix, _unit_weights)
from oxyformer.provenance import ContractError, canonical_json, file_hash, require
from oxyformer.training import fit as fitting
from oxyformer.training import nested_cv

VARIANTS = ('A6-Q',)
SEED = 1103
MAX_CONTEXT_ROWS = 30_000
MAX_COLUMNS = 500
QUERY_BATCH = 32
SALT = 'oxyformer.explore.tabicl_q.context.v1'
# The registered regressor; verification also requires it in REGISTERED_CHECKPOINTS.
REGRESSOR = dict(package='tabicl', package_version='2.2.0', repository='jingang/TabICL',
    revision='4dcd344ece2c00be9e831fdd35bed57b5ad83e19', filename='tabicl-regressor-v2-20260212.ckpt',
    sha256='0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a')
ESTIMATOR = dict(n_estimators=8, norm_methods=['none', 'power'], feat_shuffle_method='latin',
    outlier_threshold=4.0, batch_size=1, kv_cache=True, use_amp=False, use_fa3=False,
    offload_mode=False, random_state=SEED, n_jobs=1, verbose=False)
NUMERICS = dict(deterministic_algorithms=True, fp32_precision='ieee', autocast=False, grad=False)
RUNTIME_PACKAGES = ('torch', 'numpy', 'scipy', 'scikit-learn')
clock = time.monotonic


class TabICLBackend:
    """Imports tabicl only when a fit runs; CI injects a substitute."""

    def version(self):
        try:
            return metadata.version('tabicl')
        except metadata.PackageNotFoundError as exc:
            raise ContractError('missing optional dependency tabicl; A6-Q blocked') from exc

    def estimator(self, **kwargs):
        require(self.version() == PACKAGE_PINS['tabicl'], 'tabicl version differs from the approved pin')
        try:
            module = import_module('tabicl')
        except ImportError as exc:
            raise ContractError('missing optional dependency tabicl; A6-Q blocked') from exc
        return module.TabICLRegressor(**kwargs)


BACKEND = TabICLBackend()


def check_rows(rows):
    require(type(rows) is int and 2 <= rows <= MAX_CONTEXT_ROWS,
            f'context rows must be an integer in [2, {MAX_CONTEXT_ROWS}]')
    return rows


def registered_identity():
    identity = tuple(REGRESSOR[field] for field in CHECKPOINT_IDENTITY_FIELDS)
    require(set(REGRESSOR) == set(CHECKPOINT_IDENTITY_FIELDS) and identity in REGISTERED_CHECKPOINTS
            and 'regressor' in REGRESSOR['filename'], 'unregistered checkpoint identity; A6-Q blocked')
    return dict(REGRESSOR)


def candidate_settings(rows):
    """Candidate identity: checkpoint by registered fields and SHA-256, never by path."""
    return dict(checkpoint=registered_identity(), context_rows=check_rows(rows), salt=SALT, seed=SEED,
                query_batch=QUERY_BATCH, estimator=ESTIMATOR, numerics=NUMERICS)


def check_checkpoint(checkpoint):
    """Admission for a fold that still has work; advance() verifies the hash on a private copy."""
    require(checkpoint is not None, 'A6-Q requires --checkpoint PATH')
    identity = registered_identity()
    path = Path(checkpoint)
    require(path.name == identity['filename'], 'checkpoint filename differs from the registered regressor')
    require(path.is_file(), 'local checkpoint missing; automatic downloads are forbidden')


def verified_checkpoint(path):
    """Copy, then hash the private copy; only that copy ever reaches the backend."""
    identity = registered_identity()
    path = Path(path)
    require(path.name == identity['filename'], 'checkpoint filename differs from the registered regressor')
    require(path.is_file(), 'local checkpoint missing; automatic downloads are forbidden')
    storage = TemporaryDirectory(prefix='oxyformer-a6q-')
    copy = Path(storage.name) / identity['filename']
    try:
        copyfile(path, copy)
        require(file_hash(copy) == identity['sha256'], 'checkpoint hash mismatch')
        copy.chmod(0o400)
    except BaseException:
        storage.cleanup()
        raise
    return storage, str(copy)


def rank(salt, fold, original_id):
    return sha256(canonical_json([salt, fold, original_id]).encode()).hexdigest()


def select_context(candidates, fold, rows, salt=SALT):
    """Outcome-blind: reads IDs only. Keeps the candidates' order."""
    check_rows(rows)
    candidates = tuple(candidates)
    require(len(set(candidates)) == len(candidates), 'duplicate context candidates')
    if len(candidates) <= rows:
        return candidates
    chosen = set(sorted(candidates, key=lambda oid: (rank(salt, fold, oid), oid))[:rows])
    return tuple(oid for oid in candidates if oid in chosen)


def digest(value):
    return sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class FoldContext:
    ids: tuple
    matrix: np.ndarray
    labels: np.ndarray
    targets: tuple
    factual: np.ndarray
    shifted: np.ndarray
    moved: tuple
    record: dict


def build_context(endpoint, recipe, fold, rows, directory, *, salt=SALT):
    """Context rows come from the same partition the Q-only outer outcome fit uses."""
    check_rows(rows)
    names = tuple(name for name, _ in endpoint.feature_kinds)
    require(all(kind == 'numeric' for _, kind in endpoint.feature_kinds), 'A6-Q supports numeric covariates only')
    require(1 + len(names) <= MAX_COLUMNS, f'more than {MAX_COLUMNS} columns including treatment')
    require(endpoint.family == 'identity', 'Q-only requires an identity-link endpoint')
    config = endpoint.configuration(fold, str(Path(directory).resolve()), **nested_cv._settings(recipe, endpoint))
    manifest = endpoint.data.manifest
    allowed = nested_cv._validate_fitting(manifest.spec, endpoint.outer, manifest, config, SEED)
    _, candidates, targets = nested_cv._partition(config, endpoint.outer, fold)
    require(candidates == allowed == endpoint.outer.training_ids(fold), 'context candidates differ from outer training')
    require(targets == endpoint.outer.evaluation_ids(fold) and bool(targets), 'held-out targets differ')
    # Every candidate needs a unit weight, sampled or not.
    pool = fitting._inputs(config, candidates)
    _unit_weights(pool.origin_weights, len(candidates), 'unit')
    ids = select_context(candidates, fold, rows, salt)
    require(set(ids) <= set(candidates) and set(ids).isdisjoint(targets), 'held-out target in context')
    view = endpoint.data.covariates(names)
    context = fitting._take_inputs(pool, ids)
    treatment = np.asarray(context.a_mmhg, dtype=float)
    labels = np.asarray(fitting._values(config, manifest.outcome_field, ids), dtype=float)
    require(np.isfinite(treatment).all() and np.isfinite(labels).all(), 'invalid context treatment or outcome')
    matrix = np.column_stack((treatment, _matrix(fitting.subset(view, ids))))
    # Queries are formed exactly as nested_cv._predict forms them.
    inputs = fitting._inputs(config, targets)
    action = endpoint.policy.apply(inputs.a_mmhg, inputs.policy_covariates)
    require(action.a_mmhg == inputs.a_mmhg, 'policy factual dose differs')
    x = _matrix(fitting.subset(view, targets))
    factual = np.column_stack((np.asarray(action.a_mmhg, dtype=float), x))
    shifted = np.column_stack((np.asarray(action.d_mmhg, dtype=float), x))
    require(np.isfinite(factual[:, 0]).all() and np.isfinite(shifted[:, 0]).all(), 'invalid query dose')
    columns = [manifest.exposure_field, *names]
    selected = digest(list(ids))
    record = dict(fold=fold, candidates=len(candidates), rows=rows, selected=len(ids), salt=salt,
        columns=columns, selected_ids_sha256=selected, targets=len(targets), query_batch=QUERY_BATCH,
        context_sha256=digest(dict(ids=selected, columns=columns,
            matrix=sha256(matrix.tobytes()).hexdigest(), labels=sha256(labels.tobytes()).hexdigest())))
    moved = tuple(bool(d != a) for a, d in zip(action.a_mmhg, action.d_mmhg))
    return FoldContext(ids=ids, matrix=matrix, labels=labels, targets=targets, factual=factual,
                       shifted=shifted, moved=moved, record=record)


def runtime_fingerprint(backend, device):
    device = torch.device(device)
    values = dict(python=platform.python_version(), tabicl=backend.version(), device=str(device),
                  **{name: metadata.version(name) for name in RUNTIME_PACKAGES})
    if device.type == 'cuda':
        values.update(device_name=torch.cuda.get_device_name(device),
                      capability='.'.join(map(str, torch.cuda.get_device_capability(device))))
    require(values['tabicl'] == PACKAGE_PINS['tabicl'], 'tabicl version differs from the approved pin')
    return values


@contextmanager
def numerics(device):
    """IEEE FP32, deterministic kernels, no autocast and no autograd."""
    device = torch.device(device)
    with ExitStack() as stack:
        if device.type == 'cuda':
            require(os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') != '1', 'forced TF32 refused')
            if os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None:
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                stack.callback(os.environ.pop, 'CUBLAS_WORKSPACE_CONFIG', None)
            require(os.environ['CUBLAS_WORKSPACE_CONFIG'] in (':4096:8', ':16:8'),
                    'CUDA determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8')
            stack.enter_context(torch.cuda.device(device))
            stack.callback(torch.use_deterministic_algorithms, torch.are_deterministic_algorithms_enabled(),
                           warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
            torch.use_deterministic_algorithms(True)
            settings = ((torch.backends, 'fp32_precision', 'ieee'),
                        (torch.backends.cuda.matmul, 'fp32_precision', 'ieee'),
                        (torch.backends.cudnn, 'fp32_precision', 'ieee'),
                        (torch.backends.cudnn, 'deterministic', True),
                        (torch.backends.cudnn, 'benchmark', False))
            for owner, name, old in [(o, n, getattr(o, n)) for o, n, _ in settings]:
                stack.callback(setattr, owner, name, old)
            for owner, name, value in settings:
                setattr(owner, name, value)
            stack.enter_context(torch.autocast('cuda', enabled=False))
        stack.enter_context(torch.no_grad())
        yield


def build_estimator(backend, checkpoint_path, device):
    """Construction only; the estimator receives the private copy."""
    settings = dict(ESTIMATOR, norm_methods=list(ESTIMATOR['norm_methods']))
    return backend.estimator(**settings, model_path=checkpoint_path, allow_auto_download=False,
                             checkpoint_version=REGRESSOR['filename'], device=str(device))


def condition(estimator, context, device):
    """In-context conditioning only: one synchronous TabICL call, not interruptible mid-call."""
    with numerics(device):
        estimator.fit(context.matrix.copy(), context.labels.copy())
    return estimator


def fit_estimator(backend, checkpoint_path, context, device):
    return condition(build_estimator(backend, checkpoint_path, device), context, device)


def predict_rows(estimator, rows, device):
    """One query row per call, so no other target can enter its prediction."""
    values = []
    with numerics(device):
        for row in rows:
            value = np.asarray(estimator.predict(row[None, :].copy(), output_type='mean'), dtype=float).reshape(-1)
            require(value.shape == (1,) and math.isfinite(value[0]), 'invalid foundation prediction')
            values.append(float(value[0]))
    return values


def predict_targets(estimator, context, indices, device):
    """mu_a at action.a_mmhg; mu_d at action.d_mmhg, equal to mu_a for non-movers."""
    indices = list(indices)
    mu_a = predict_rows(estimator, context.factual[indices], device)
    movers = [i for i in indices if context.moved[i]]
    shifted = dict(zip(movers, predict_rows(estimator, context.shifted[movers], device)))
    mu_d = [shifted[i] if context.moved[i] else value for i, value in zip(indices, mu_a)]
    return mu_a, mu_d


def bounds(count):
    return [(start, min(start + QUERY_BATCH, count)) for start in range(0, count, QUERY_BATCH)]


def out_of_time(now, deadline, longest, margin):
    return now + longest > deadline - margin


def recorded_fit_seconds(record):
    """The last earlier slice's context-fit seconds, or 0 if no slice has conditioned yet."""
    fits = [event['fit_seconds'] for event in record['slices'] if event.get('conditioned')]
    return float(fits[-1]) if fits else 0.


def advance(record, endpoint, recipe, directory, *, settings, checkpoint, slice_seconds, device, save,
            request, backend=None):
    """One slice of one fold; save() persists record atomically after every batch."""
    backend = BACKEND if backend is None else backend
    started = clock()
    deadline, margin = started + slice_seconds, min(60., slice_seconds / 10)
    # The verified private copy exists before the backend is consulted at all.
    storage, path = verified_checkpoint(checkpoint)
    try:
        verified = clock()
        runtime = runtime_fingerprint(backend, device)
        # Saved batches resume only against the provenance they were predicted under.
        require(not record.get('batches') or ('runtime' in record and 'context' in record),
                'journal batches lack runtime or context provenance; resume refused')
        require(record.setdefault('runtime', runtime) == runtime, 'runtime fingerprint differs; resume refused')
        context = build_context(endpoint, recipe, record['fold'], settings['context_rows'], directory,
                                salt=settings['salt'])
        require(record.setdefault('context', context.record) == context.record,
                'context digest differs; resume refused')
        batches = record.setdefault('batches', [])
        plan = bounds(len(context.targets))
        for index, batch in enumerate(batches):
            require(batch['index'] == index and [batch['start'], batch['stop']] == list(plan[index])
                    and len(batch['mu_a']) == len(batch['mu_d']) == batch['stop'] - batch['start'],
                    'journal batches differ from the batch plan')
        built = constructed = fitted = clock()
        times = record.setdefault('phase_seconds', dict(checkpoint=0., context=0., fit=0., prediction=0.))
        times['checkpoint'] += verified - started
        times['context'] += built - verified
        slice_index, reason, conditioned = len(record['slices']), 'complete', False
        # Only this slice's batches estimate the next (zero before its first),
        # so one past outlier cannot block later slices. The fit cannot be
        # interrupted, so its admission is checked before and after construction.
        longest = 0.
        need = recorded_fit_seconds(record) + longest
        if request.requested or out_of_time(built, deadline, need, margin):
            reason = 'signal' if request.requested else 'deadline'
        else:
            estimator = build_estimator(backend, path, device)
            constructed = fitted = clock()
            if request.requested or out_of_time(constructed, deadline, need, margin):
                reason = 'signal' if request.requested else 'deadline'
            else:
                condition(estimator, context, device)
                fitted, conditioned = clock(), True
        times['fit'] += fitted - built
        plan_left = range(len(batches), len(plan)) if conditioned else []
        # Journal this slice's timings before any batch, so a killed slice keeps them.
        event = dict(wall_seconds=fitted - started, complete=False, stop_reason='running',
            checkpoint_seconds=verified - started, context_seconds=built - verified,
            estimator_seconds=constructed - built, fit_seconds=fitted - constructed, conditioned=conditioned,
            admission_seconds=need, batches_done=len(batches), batches_total=len(plan),
            context_rows=context.record['selected'], device=str(device))
        record['slices'].append(event)
        save()
        for index in plan_left:
            if request.requested or out_of_time(clock(), deadline, longest, margin):
                reason = 'signal' if request.requested else 'deadline'
                break
            before = clock()
            start, stop = plan[index]
            mu_a, mu_d = predict_targets(estimator, context, range(start, stop), device)
            seconds = clock() - before
            longest = max(longest, seconds)
            times['prediction'] += seconds
            batches.append(dict(index=index, start=start, stop=stop, mu_a=mu_a, mu_d=mu_d,
                                wall_seconds=seconds, slice=slice_index))
            save()
    finally:
        storage.cleanup()
    complete = len(batches) == len(plan)
    if complete:
        record['parts'] = [dict(fold=record['fold'], seed=SEED, original_ids=list(context.targets),
            mu_a=[v for b in batches for v in b['mu_a']], mu_d=[v for b in batches for v in b['mu_d']])]
        record['status'] = 'complete'
    event.update(wall_seconds=clock() - started, complete=complete, stop_reason=reason, batches_done=len(batches))
    save()
    return event
