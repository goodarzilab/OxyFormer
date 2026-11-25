"""Exploratory Q-only refits of saved simulation draws; no release claim."""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import tarfile
import time
import uuid

import torch

from oxyformer.contracts import OOFNuisances, StageRequest
from oxyformer.explore import tabicl_q
from oxyformer.explore.real_fit import (atomic_record, checkout_identity, logger_for,
    state_lock, write_directory)
from oxyformer.provenance import ContractError, file_hash, read_artifact, require
from oxyformer.training import nested_cv
from oxyformer.training.checkpoint import CheckpointRequest, load_checkpoint
from oxyformer.training.fit import FoldArtifacts, RecipeStudy, resolve_device, subset
from oxyformer.training.pretrain import environment_identity
from oxyformer.validation import coverage
from oxyformer.validation.generators import ObservedRecords

STUDIES = {'cell-d': dict(reference_cap=32, frozen_epochs=60, grid_points=2, seed_count=1),
           'cell-e': dict(reference_cap=32, frozen_epochs=30, grid_points=2, seed_count=1)}
PART_KEYS = {'fold', 'seed', 'original_ids', 'mu_a', 'mu_d'}
# A6-Q is a frozen TabICLv2 adapter, not a nested_cv network variant.
VARIANTS = nested_cv.Q_VARIANTS + tabicl_q.VARIANTS


def load_chain(path):
    """Same digest/dependency checks as attribution_audit.load, with two modes."""
    path = Path(path).resolve()
    result = json.loads((path / 'result.json').read_text())
    require(result['terminal'] and result['campaign_mode'] in ('diagnostic', 'm2_pilot_d'),
            'terminal diagnostic or m2_pilot_d required')
    with tarfile.open(path / 'continuation.tar', 'r:') as archive:
        members = archive.getmembers()
        require(len({m.name for m in members}) == len(members) and all(m.isfile() for m in members),
                'invalid tar members')
        header = json.load(archive.extractfile('coverage.json'))
        require(len(header['states']) == 1, 'one saved state required')

        def blob(key):
            require(len(key) == 64 and all(c in '0123456789abcdef' for c in key), 'invalid blob key')
            value = archive.extractfile('blobs/' + key).read()
            require(sha256(value).hexdigest() == key, 'blob digest mismatch')
            return value.decode()

        states = [{**state, 'parts': [OOFNuisances.from_json(blob(k)) for k in state['parts']],
                   'sample': {k: blob(state['sample'][k]) for k in
                              ('observations', 'structural_causal_truth')}} for state in header['states']]
    request = StageRequest.from_json((path / '_execution/request.json').read_text())

    def resolve_dependency(name):
        try:
            return Path(name).resolve()
        except RuntimeError as exc:
            raise ContractError('unreadable dependency: ' + str(name) + ': ' + str(exc)) from exc

    def dependency(expected, cls=None):
        for name, hashed in zip(request.dependency_paths, request.dependency_hashes):
            if hashed == expected:
                source = Path(name)
                resolved = resolve_dependency(source)
                anchor = resolve_dependency(request.output_dir)
                if not source.exists() and resolved.is_relative_to(anchor):
                    target = resolve_dependency(path / resolved.relative_to(anchor))
                    require(target.is_relative_to(resolve_dependency(path)), 'external dependency: ' + str(source))
                    source = target
                raw = source.read_bytes()
                require(sha256(raw).hexdigest() == expected, 'dependency digest mismatch')
                return cls.from_json(raw.decode()) if cls else json.loads(raw)
        raise ContractError('missing dependency: ' + expected)

    endpoint = dependency(header['binding']['endpoint_hash'], nested_cv.PreparedEndpoint)
    recipe = dependency(header['binding']['lock_hash'])['recipe']
    require(coverage.digest(recipe) == header['binding']['recipe_hash'], 'recipe digest mismatch')
    require(header['binding']['frame_hash'] == states[0]['binding']['frame_hash'], 'frame binding mismatch')
    return result, states, endpoint, recipe


def bind(chain_dir, out):
    chain = Path(chain_dir).resolve()
    destination = write_directory(out)
    require(not destination.exists(), 'bound output already exists')
    archive_hash = file_hash(chain / 'continuation.tar')
    _, states, base, recipe = load_chain(chain)
    require(file_hash(chain / 'continuation.tar') == archive_hash, 'archive changed during bind')
    state = states[0]
    endpoint = coverage.bind_observations(base, ObservedRecords.from_json(state['sample']['observations']))
    require(endpoint.family == 'identity', 'Q-only requires an identity-link endpoint')
    parts = state['parts']
    require(len(parts) == 5, 'five saved parts required')
    expected = []
    for fold, part in enumerate(parts):
        ids = endpoint.outer.evaluation_ids(fold)
        require(part.original_ids == ids and set(part.fold_ids) == {fold}
                and set(part.seed_ids) == {1103}, 'saved fold/seed/order mismatch')
        expected.append(dict(fold=fold, seed=1103, original_ids=list(ids)))
    destination.mkdir(parents=True)
    (destination / 'endpoint.json').write_text(endpoint.to_json())
    atomic_record(destination / 'recipe.json', recipe)
    binding = dict(chain_dir=str(chain), archive_hash=archive_hash,
        frame_hash=state['binding']['frame_hash'], endpoint_hash=endpoint.content_hash,
        recipe_hash=coverage.digest(recipe), expected_parts=expected, exploratory=True,
        certifies_release_criteria=False)
    atomic_record(destination / 'binding.json', binding)
    return binding


def load_bound(out):
    root = Path(out)
    binding = json.loads((root / 'binding.json').read_text())
    endpoint = read_artifact(root / 'endpoint.json', nested_cv.PreparedEndpoint, binding['endpoint_hash'])
    recipe = json.loads((root / 'recipe.json').read_text())
    require(coverage.digest(recipe) == binding['recipe_hash'], 'bound recipe digest mismatch')
    return endpoint, recipe, binding


def fit(out, fold, variant, study, slice_seconds, run_logging, *, device='cuda:0', checkpoint=None,
        context_rows=None):
    require(type(fold) is int and fold in range(5), 'unsupported fold')
    require(variant in VARIANTS and study in STUDIES, 'unsupported variant or study')
    require(math.isfinite(slice_seconds) and 0 < slice_seconds <= 10800, 'slice must be in (0, 10800] seconds')
    require(run_logging in ('wandb', 'none'), 'explicit run logging mode required')
    tabicl = variant in tabicl_q.VARIANTS
    if tabicl:
        settings = tabicl_q.candidate_settings(tabicl_q.MAX_CONTEXT_ROWS if context_rows is None else context_rows)
    else:
        require(checkpoint is None and context_rows is None, '--checkpoint and --context-rows apply only to A6-Q')
    endpoint, recipe, binding = load_bound(out)
    synthetic = endpoint.data.manifest.spec.endpoint.startswith('synthetic')
    require(run_logging != 'none' or synthetic, 'W&B logging is required for real endpoints')
    device = str(resolve_device(device))
    require(synthetic or torch.device(device).type == 'cuda', 'one GPU required for a saved endpoint fit')
    root = write_directory(out)
    identity = dict(chain_dir=binding['chain_dir'], frame_hash=binding['frame_hash'],
        archive_hash=binding['archive_hash'], endpoint_hash=endpoint.content_hash,
        recipe_hash=binding['recipe_hash'], variant=variant, study=STUDIES[study],
        code_commit=checkout_identity(), environment=dict(environment_identity(torch.device(device))),
        run_logging=run_logging)
    if tabicl:
        identity['tabicl'] = settings

    def matches(saved):
        # A complete A6-Q fold may be reread without --context-rows.
        if tabicl and context_rows is None and isinstance(saved.get('tabicl'), dict):
            return saved == dict(identity, tabicl=dict(settings, context_rows=saved['tabicl'].get('context_rows')))
        return saved == identity
    # Serialize admissions and collect against all folds of this candidate.
    with state_lock(root):
        candidate = root / 'candidate.json'
        saved = json.loads(candidate.read_text()) if candidate.exists() else None
        require(saved is None or matches(saved), 'candidate identity differs')
        directory = root / f'fold-{fold}'
        path = directory / 'progress.json'
        record = json.loads(path.read_text()) if path.exists() else dict(identity=identity,
            fold=fold, seed=1103, status='incomplete', predecessor=None, parts=[], slices=[])
        require(matches(record['identity']) and record['fold'] == fold, 'journal identity differs')
        if saved is None:
            atomic_record(candidate, record['identity'])
        # A complete fold returns its journal without loading weights or predicting.
        if record['status'] == 'complete':
            return record
        require(saved in (None, identity), 'candidate identity differs')
        require(record['identity'] == identity, 'journal identity differs')
        if tabicl:
            tabicl_q.check_checkpoint(checkpoint)
        logger = logger_for(run_logging, dict(identity, fold=fold, exploratory=True), directory)
        try:
            atomic_record(path, record)
            if tabicl:
                request = CheckpointRequest()
                with request.signals():
                    event = tabicl_q.advance(record, endpoint, recipe, directory, settings=settings,
                        checkpoint=checkpoint, slice_seconds=slice_seconds, device=device,
                        save=lambda: atomic_record(path, record), request=request)
                if logger is not None:
                    logger.log(event)
                return record
            destination = directory / ('slice-' + uuid.uuid4().hex)
            request = CheckpointRequest()
            config = endpoint.configuration(fold, destination, **nested_cv._settings(recipe, endpoint),
                study=RecipeStudy(**STUDIES[study]), device=device, slice_seconds=slice_seconds,
                checkpoint_margin_seconds=min(60., slice_seconds / 10), stop_request=request,
                predecessor=None if record['predecessor'] is None else FoldArtifacts.from_json(record['predecessor']))
            started = time.monotonic()
            with request.signals():
                artifact = nested_cv.run_fold(config, endpoint.outer, 1103,
                    geography=endpoint.geography, variant=variant, q_only=True)
                record['predecessor'] = artifact.to_json()
                controller = load_checkpoint(artifact.checkpoint, artifact.checkpoint.identity)['controller']
                record['phase_seconds'] = dict(controller['phase_seconds'])
                record['counts'] = controller['counts']
                if artifact.complete:
                    view = subset(endpoint.data.covariates(tuple(n for n, _ in endpoint.feature_kinds)),
                                  artifact.prediction_inputs.original_ids)
                    before = time.monotonic()
                    part = nested_cv.predict_q(artifact, view, endpoint.policy)
                    record['phase_seconds']['prediction'] = time.monotonic() - before
                    record['parts'] = [dict(fold=fold, seed=1103, **part)]
                    record['status'] = 'complete'
                event = dict(wall_seconds=time.monotonic() - started, complete=artifact.complete,
                    checkpoint_position=artifact.checkpoint.epoch, counts=dict(controller['counts']),
                    stop_reason=artifact.checkpoint.reason, output_dir=str(destination))
                record['slices'].append(event)
                atomic_record(path, record)
            if logger is not None:
                logger.log(event)
            return record
        finally:
            if logger is not None:
                logger.finish()


def collect(out):
    root = write_directory(out)
    endpoint, _, binding = load_bound(root)
    with state_lock(root):
        identity = json.loads((root / 'candidate.json').read_text())
        require(identity['endpoint_hash'] == endpoint.content_hash and
                identity['frame_hash'] == binding['frame_hash'] and
                identity['chain_dir'] == binding['chain_dir'], 'candidate binding differs')
        parts = []
        for fold in range(5):
            record = json.loads((root / f'fold-{fold}' / 'progress.json').read_text())
            require(record['identity'] == identity and record['status'] == 'complete'
                    and record['fold'] == fold and record['seed'] == 1103, 'incomplete or mismatched fold')
            require(len(record['parts']) == 1, 'one part per fold required')
            parts.extend(record['parts'])
        require([p['fold'] for p in parts] == list(range(5)), 'parts must be in fold order')
        for fold, part in enumerate(parts):
            require(set(part) == PART_KEYS, f'fold {fold}: part keys differ')
            require(type(part['fold']) is int and type(part['seed']) is int and part['seed'] == 1103,
                    f'fold {fold}: fold/seed differs')
            require(part['original_ids'] == list(endpoint.outer.evaluation_ids(fold)),
                    f'fold {fold}: original_ids differ')
            for key in ('mu_a', 'mu_d'):
                values = part[key]
                require(len(values) == len(part['original_ids']) and
                        all(type(v) in (int, float) and math.isfinite(v) for v in values),
                        f'fold {fold}: invalid {key}')
        result = dict(schema='oxyformer.q_parts.v1', chain_dir=binding['chain_dir'],
            frame_hash=binding['frame_hash'], variant=identity['variant'], study=identity['study'],
            code_commit=identity['code_commit'], parts=parts)
        atomic_record(root / 'q_parts.json', result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('bind')
    p.add_argument('chain_dir')
    p.add_argument('out')
    p = commands.add_parser('fit')
    p.add_argument('out')
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--study', choices=STUDIES, required=True)
    p.add_argument('--slice-seconds', type=float, required=True)
    p.add_argument('--run-logging', choices=('wandb', 'none'), required=True)
    p.add_argument('--checkpoint', help='A6-Q only: local registered TabICLv2 regressor checkpoint')
    p.add_argument('--context-rows', type=int, help='A6-Q only: context sample size, 2..30000 (default 30000)')
    p = commands.add_parser('collect')
    p.add_argument('out')
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    result = {'bind': bind, 'fit': fit, 'collect': collect}[command](**args)
    return 75 if command == 'fit' and result['status'] != 'complete' else 0


if __name__ == '__main__':
    raise SystemExit(main())
