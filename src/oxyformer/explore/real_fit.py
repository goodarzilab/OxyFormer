"""Direct exploratory USALEEP fits; no campaign admission or release claim."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import csv
import fcntl
import importlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
import uuid

import numpy as np
import torch

from oxyformer.contracts import CovariateView, DataManifest, StageResult
from oxyformer.data.adapters.usaleep import primary_outcome_flags
from oxyformer.data.entity_graph import EntityGraph
from oxyformer.data.loaders import load_records
from oxyformer.design import gate
from oxyformer.design.eligibility import GeographyTable, county_screen, usable
from oxyformer.design.policies import PolicyCovariates
from oxyformer.design.splits import Reservation, build_scenario, dependence_groups
from oxyformer.design.support import SupportRecipe, freeze_support, supported_ids
from oxyformer.execution.runner import read_mapping
from oxyformer.models.treatment_query import TreatmentDesign
from oxyformer.provenance import canonical_json, file_hash, read_artifact, require
from oxyformer.training import nested_cv
from oxyformer.training.checkpoint import CheckpointArtifact, CheckpointRequest, load_checkpoint
from oxyformer.training.fit import FoldArtifacts, RecipeStudy, subset, resolve_device
from oxyformer.training.pretrain import environment_identity
from oxyformer.validation import coverage

ROOT = Path(__file__).resolve().parents[3]
ENDPOINT = 'usaleep_life_expectancy'
HEADER = ['Tract ID', 'STATE2KX', 'CNTY2KX', 'TRACT2KX', 'e(0)', 'se(e(0))',
          'Abridged life table flag']
STUDY = dict(reference_cap=32, frozen_epochs=60, grid_points=2, seed_count=1)
LABELS = dict(exploratory=True, sealing='none (owner decision 2026-11-19)',
              effect_release_authorized=False, certifies_release_criteria=False)


def atomic_record(path, value):
    """Replace the journal only after a fully flushed private write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.journal-')
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_outcomes(payload, receipts):
    """Authenticate bytes before parsing the exact acquisition grammar."""
    receipt = json.loads(Path(receipts).read_text())
    entries = [r for r in receipt['resources'] if r['id'] == 'usaleep_data']
    require(len(entries) == 1, 'unique usaleep_data receipt required')
    entry = entries[0]
    with tarfile.open(payload, 'r:*') as archive:
        members = [m for m in archive if m.name == entry['destination']]
        require(len(members) == 1 and members[0].isfile(), 'unique regular USALEEP member required')
        require(members[0].size == entry['bytes'], 'USALEEP byte count mismatch')
        with archive.extractfile(members[0]) as stream:
            raw = stream.read()
    require(len(raw) == entry['bytes'] and sha256(raw).hexdigest() == entry['sha256'],
            'USALEEP sha256 mismatch')
    reader = csv.reader(io.StringIO(raw.decode('utf-8')))
    require(next(reader, None) == HEADER, 'USALEEP header mismatch')
    result = {}
    for row in reader:
        require(len(row) == len(HEADER), 'USALEEP row width mismatch')
        oid = row[0]
        require(oid.isascii() and oid.isdigit() and len(oid) in (10, 11), 'invalid Tract ID')
        oid = oid.zfill(11)
        require(oid not in result, 'duplicate Tract ID')
        try:
            y, flag = float(row[4]), int(row[6])
        except ValueError as exc:
            raise ValueError('unparseable USALEEP outcome or flag') from exc
        require(math.isfinite(y) and flag in (1, 2, 3), 'invalid USALEEP outcome or flag')
        result[oid] = (y, flag)
    require(bool(result), 'empty USALEEP input')
    return result, entry['sha256']


def support_settings(design, approvals, delta_mmhg=2.0, support_caliper_iqr=None):
    """Validate exploratory bounds without mutating the approved design."""
    approved = SupportRecipe(**design['support'])
    caliper = approved.raw_x_radius if support_caliper_iqr is None else support_caliper_iqr
    require(math.isfinite(delta_mmhg), 'delta_mmhg_allowed: finite delta required')
    require(math.isfinite(caliper), 'support_caliper_iqr: finite caliper required')
    if delta_mmhg != 2.0 or caliper != approved.raw_x_radius:
        block = approvals.get('owner_decisions', {}).get('exploratory_unsealed', {}).get(
            'policy_support_sensitivity')
        require(isinstance(block, dict), 'policy_support_sensitivity approval required for non-default delta/caliper')
        require(delta_mmhg in block['delta_mmhg_allowed'], 'delta_mmhg_allowed bound exceeded')
        require(approved.raw_x_radius <= caliper <= block['raw_x_radius_iqr_max'],
                'support_caliper_iqr bound: approved raw_x_radius <= caliper <= raw_x_radius_iqr_max')
        for name in ('bin_width_mmhg', 'min_bin_tracts'):
            require(getattr(approved, name) == block['unchanged_support_settings'][name],
                    f'{name} must remain unchanged')
    return replace(approved, raw_x_radius=caliper)


def target_policy(endpoint):
    ids = endpoint.outer.original_ids
    positions = {oid: j for j, oid in enumerate(endpoint.data.manifest.original_ids)}
    policy_positions = {oid: j for j, oid in enumerate(endpoint.policy_covariates.original_ids)}
    a = endpoint.data.column(endpoint.data.manifest.exposure_field)
    cov = endpoint.policy_covariates
    selected = PolicyCovariates(original_ids=ids,
        geography_ids=tuple(cov.geography_ids[policy_positions[i]] for i in ids),
        support_keys=tuple(cov.support_keys[policy_positions[i]] for i in ids))
    return endpoint.policy.apply([a[positions[i]] for i in ids], selected)


def policy_readout(endpoint, caliper):
    result = target_policy(endpoint)
    n = len(result.moved)
    moved = sum(result.moved)
    fraction = moved / n
    return dict(delta_mmhg=endpoint.policy.delta_mmhg, support_caliper_iqr=caliper,
        n_targets=n, n_moved_targets=moved, shifted_fraction=fraction,
        achieved_mean_shift_mmhg=fraction * endpoint.policy.delta_mmhg)


def policy_diagnostics(endpoint, readout, estimates, nuisances):
    """Descriptive rescalings and shifted-ratio concentration among movers."""
    fraction, shift = readout['shifted_fraction'], readout['achieved_mean_shift_mmhg']
    def rescale(divisor):
        return {name: dict(value=v['value'] / divisor, se=v['se'] / divisor)
                for name, v in estimates.items()} if divisor else None
    moved = dict(zip(endpoint.outer.original_ids, target_policy(endpoint).moved))
    ratios = np.asarray([r for oid, r in zip(nuisances.original_ids, nuisances.r_d) if moved[oid]],
                        dtype=float)
    total = float(ratios.sum())
    return dict(per_mover=rescale(fraction), per_achieved_mmhg=rescale(shift),
        mover_r_d_kish_ess=float(total ** 2 / np.dot(ratios, ratios)) if total else None,
        mover_r_d_max_share=float(ratios.max() / total) if total else None,
        mover_r_d_p99=float(np.quantile(ratios, .99)) if len(ratios) else None)


def build_endpoint(values, design, outcomes, *, delta_mmhg=2.0, support_caliper_iqr=None):
    """Use the real-frame design sequence with an explicitly empty reserve."""
    support_recipe = support_settings(design, values['approvals'], delta_mmhg, support_caliper_iqr)
    manifest, covariates = values['data_manifest'], values['covariates']
    geography, atlas, graph = values['geography'], values['atlas'], values['entity_graph']
    require(manifest.spec.endpoint == ENDPOINT, 'wrong endpoint')
    flags = primary_outcome_flags(values['approvals'])
    require(set(manifest.original_ids) <= outcomes.keys(), 'missing frame outcomes')
    # Verify rather than silently changing the upstream eligibility metadata.
    require(all(r.outcome_flag == outcomes[r.original_id][1] and
                r.label_available == (outcomes[r.original_id][1] in flags)
                for r in geography.rows), 'outcome flag/availability metadata differs')
    gate._validate_data(manifest, covariates, geography, atlas, graph)
    gate.validate_approvals(design, values['approvals'], manifest, covariates, geography, atlas)
    atlas_coverage = gate.atlas_coverage(values)
    require(atlas_coverage['complete'], 'incomplete atlas coverage')
    missing = set(atlas_coverage['accounted_missing_dem_tract_ids'])
    rows, a = geography.rows, {r.tract_id: r for r in atlas.rows}
    reservation = Reservation(design_ids=(), selected_subblocks=(), sealed_subblocks=(),
        total_subblocks=len({(r.county, r.subblock) for r in rows}),
        groups=dependence_groups(rows, graph))
    screens = county_screen(rows, a)
    frozen = freeze_support(rows, reservation.design_ids, a, covariates,
                            recipe=support_recipe, delta_mmhg=delta_mmhg)
    supported = supported_ids(rows, a, frozen)
    reasons = {}
    for row in rows:
        if row.original_id in reservation.design_ids:
            continue
        if row.tract_id in missing:
            reasons[row.original_id] = 'atlas_missing_dem_coverage'
        elif not usable(row, a):
            reasons[row.original_id] = 'flag_label_or_allocation_unusable'
        elif screens[row.county]['reasons']:
            reasons[row.original_id] = 'initial_county_screen'
        elif row.original_id not in supported:
            reasons[row.original_id] = 'outside_frozen_conditional_support'
    candidates = set(manifest.original_ids) - set(reservation.design_ids) - set(reasons)
    scenario = build_scenario(rows, graph, manifest, reservation, candidates, reasons, radius_km=0,
        policy_id=frozen.policy.policy_id, atlas=a, covariates=covariates, support=frozen)
    require(scenario.status == 'pass', 'no tract target passes primary geographic, support and split minima')
    template = scenario.data_manifest
    # Missing atlas exposures stay nullable, exactly as in the simulation adapter.
    schema = tuple(replace(col, nullable=True) if (col.name == manifest.outcome_field or
                   missing and col.name == manifest.exposure_field) else col for col in template.schema)
    template = replace(template, schema=schema)
    outer = replace(scenario.outer, lineage=replace(scenario.outer.lineage,
                    parent_hashes=(template.content_hash,)))
    inners = []
    for inner in scenario.inner:
        local = replace(inner.data_manifest, schema=schema,
            lineage=replace(inner.data_manifest.lineage, parent_hashes=(template.content_hash,)))
        split = replace(inner.split, lineage=replace(inner.split.lineage, parent_hashes=(local.content_hash,)))
        inners.append(replace(inner, data_manifest=local, split=split))
    x = dict(zip(covariates.original_ids, covariates.values))
    by_id = {r.original_id: r for r in rows}
    records = []
    for oid in template.original_ids:
        record = dict(zip(covariates.columns, x[oid]))
        record.update({manifest.id_field: oid,
            manifest.outcome_field: outcomes[oid][0] if outcomes[oid][1] in flags else None,
            manifest.exposure_field: None if by_id[oid].tract_id in missing else a[by_id[oid].tract_id].exposure_mmhg,
            geography.county_field: by_id[oid].county})
        records.append(record)
    data = load_records(records, template, template.spec, template.schema_hash)
    pool_a = [a[by_id[oid].tract_id].exposure_mmhg for oid in frozen.pool_ids]
    center = sum(pool_a) / len(pool_a)
    scale = (sum((v - center) ** 2 for v in pool_a) / len(pool_a)) ** .5
    require(scale > 0, 'zero exposure scale')
    families = values['approvals']['owner_decisions']['endpoint_covariates'][ENDPOINT]['masking_families']
    endpoint = nested_cv.PreparedEndpoint(data=data, entity_graph=graph,
        geography=replace(geography, data_manifest_hash=template.content_hash), outer=outer, inner=tuple(inners),
        policy=frozen.policy, policy_covariates=PolicyCovariates(original_ids=manifest.original_ids,
            geography_ids=tuple(r.assignment_geography for r in rows),
            support_keys=tuple(r.assignment_geography for r in rows)),
        treatment_design=TreatmentDesign(center=center, scale=scale, knots=frozen.spline_knots,
                                        design_hash=frozen.content_hash),
        feature_kinds=tuple((n, 'numeric') for n in covariates.columns),
        families=tuple(tuple(n for n in names if n in covariates.columns)
                       for names in families.values() if set(names).intersection(covariates.columns)),
        county_field=geography.county_field, exposure_assignment_level='tract')
    for fold in range(5):
        endpoint.validate(fold)
    audit = dict(frame_rows=len(rows), in_frame_by_flag=dict(Counter(str(outcomes[i][1]) for i in manifest.original_ids)),
        labelled=sum(outcomes[i][1] in flags for i in manifest.original_ids),
        eligible=sum(usable(r, a) for r in rows), support_pool=len(frozen.pool_ids),
        support_candidates=len(candidates), targets=len(outer.original_ids),
        exclusions_by_reason=dict(Counter(reason for _, reason in scenario.exclusions)),
        design_ids=list(reservation.design_ids), sealed_subblocks=list(reservation.sealed_subblocks),
        layer_counts={name: len(ids) for name, ids in scenario.layer_ids}, **LABELS)
    audit.update(policy_readout(endpoint, support_recipe.raw_x_radius))
    return endpoint, audit


def recipe_for(endpoint, approvals):
    study = approvals['owner_decisions']['m2_pilot_d']['cells']['d']
    require(study == STUDY, 'cell d recipe drift')
    require(approvals['owner_decisions']['exploratory_unsealed']['outcome_reserve_withheld'] is False,
            'exploratory unsealed owner decision required')
    targets = set(endpoint.outer.original_ids)
    locations = {}
    for county in sorted({r.county for r in endpoint.geography.rows if r.original_id in targets}):
        points = [(r.latitude, r.longitude) for r in endpoint.geography.rows
                  if r.county == county and r.original_id in targets]
        locations[county] = [sum(p[j] for p in points) / len(points) for j in range(2)]
    return dict(cell='d', seed=1103, study={key: study[key] for key in STUDY},
        # The registered base stays 150; RecipeStudy supplies the effective cell-d 60.
        nested_cv=dict(ssl_epochs=30, frozen_epochs=150, batch_size=256, precision='fp32'),
        inference=dict(primary_bandwidth_km=100, county_locations=locations))


def recipe_study(recipe):
    study = recipe['study']
    return RecipeStudy(reference_cap=study['reference_cap'], frozen_epochs=study['frozen_epochs'],
                       grid_points=study['grid_points'], seed_count=study['seed_count'])


def prepare(tract_inputs, atlas, usaleep_tar, usaleep_receipts, out, *,
            delta_mmhg=2.0, support_caliper_iqr=None):
    out = Path(out)
    require(not out.exists(), 'prepared output already exists')
    approvals = read_mapping(gate.OWNER_APPROVALS)
    design = read_mapping(gate.DESIGN_CONFIG)
    support_settings(design, approvals, delta_mmhg, support_caliper_iqr)
    paths = {}
    result_path = Path(tract_inputs) / '_execution/result.json'
    publication = StageResult.from_json(result_path.read_text())
    require(publication.status == 'pass', 'tract inputs did not pass')
    hashes = {a.path: a.sha256 for a in publication.artifacts}
    values = {}
    for role, cls in [('data_manifest', DataManifest), ('covariates', CovariateView),
                      ('geography', GeographyTable), ('entity_graph', EntityGraph)]:
        path = Path(tract_inputs) / (role + '.json')
        values[role] = read_artifact(path, cls, hashes[path.name])
        paths[str(path)] = file_hash(path)
    atlas_paths = {('atlas-collect', name): Path(atlas) / name for name in
                   ('atlas.parquet', 'quality.json', 'artifact_manifest.json')}
    values.update(gate.collected_atlas_inputs(atlas_paths, values['data_manifest'].original_ids))
    values['approvals'] = approvals
    outcomes, member_hash = read_outcomes(usaleep_tar, usaleep_receipts)
    if delta_mmhg == 2.0 and support_caliper_iqr is None:
        endpoint, audit = build_endpoint(values, design, outcomes)
    else:
        endpoint, audit = build_endpoint(values, design, outcomes, delta_mmhg=delta_mmhg,
                                         support_caliper_iqr=support_caliper_iqr)
    recipe = recipe_for(endpoint, values['approvals'])
    for path in [*atlas_paths.values(), result_path,
                 Path(usaleep_tar), Path(usaleep_receipts), gate.OWNER_APPROVALS, gate.DESIGN_CONFIG]:
        paths[str(path)] = file_hash(path)
    audit.update(input_sha256=paths, usaleep_member_sha256=member_hash, rows_read=len(outcomes),
                 all_rows_by_flag=dict(Counter(str(v[1]) for v in outcomes.values())),
                 prepared_hash=endpoint.content_hash)
    out.mkdir(parents=True)
    (out / 'endpoint.json').write_text(endpoint.to_json())
    atomic_record(out / 'recipe.json', recipe)
    atomic_record(out / 'prepare_audit.json', audit)
    return audit


def load_prepared(directory):
    root = Path(directory)
    audit = json.loads((root / 'prepare_audit.json').read_text())
    endpoint = read_artifact(root / 'endpoint.json', nested_cv.PreparedEndpoint, audit['prepared_hash'])
    recipe = json.loads((root / 'recipe.json').read_text())
    require(recipe == recipe_for(endpoint, read_mapping(gate.OWNER_APPROVALS)), 'prepared recipe drift')
    return endpoint, recipe, audit


def write_directory(path):
    """Reject output that would dirty the checkout before creating any files."""
    destination, checkout = Path(path).resolve(), ROOT.resolve()
    if destination.is_relative_to(checkout):
        ignored = subprocess.run(['git', 'check-ignore', '--quiet', '--', str(destination) + os.sep],
                                 cwd=checkout, check=False)
        require(ignored.returncode == 0,
                f'{destination}: unignored write path inside checkout; '
                'use a state directory outside the checkout or an already ignored directory')
    return destination


def checkpoint_progress(checkpoint):
    """Distinguish controller position from an active model's training epoch."""
    progress = dict(checkpoint_position=checkpoint.epoch)
    controller = load_checkpoint(checkpoint, checkpoint.identity)['controller']
    if controller['active'] is not None:
        metadata = controller['active'].get('progress')
        epoch = metadata.get('epoch') if isinstance(metadata, dict) else None
        if type(epoch) is int:
            progress['train_epoch'] = epoch
    elif controller['ssl_pending'] is not None:
        progress['train_epoch'] = CheckpointArtifact.from_json(controller['ssl_pending']).epoch
    return progress


def checkout_identity():
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip(), 'dirty checkout refused')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()


def logger_for(mode, config, directory=None):
    require(mode in ('wandb', 'none'), 'explicit run logging mode required')
    if mode == 'none':
        return None
    try:
        wandb = importlib.import_module('wandb')
    except ImportError as exc:
        raise ValueError('wandb requested but package is absent; no training started') from exc
    require(directory is not None, 'W&B output directory required')
    directory = write_directory(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Credentials are deliberately not inspected or included in config.
    return wandb.init(entity='goodarzilab', project='oxyformer-v2', config=config, dir=str(directory))


@contextmanager
def state_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('fold state is already in use') from exc
        yield


def fit(prepared, fold, seed, state, slice_seconds, run_logging, *, device='cuda:0'):
    require(type(fold) is int and fold in range(5) and seed == 1103, 'unsupported fold or seed')
    require(math.isfinite(slice_seconds) and 0 < slice_seconds <= 10800, 'slice must be in (0, 10800] seconds')
    root = write_directory(state)
    endpoint, recipe, _ = load_prepared(prepared)
    require(run_logging != 'none' or endpoint.data.manifest.spec.endpoint.startswith('synthetic'),
            'W&B logging is required for real endpoints; use --run-logging wandb')
    device = str(resolve_device(device))
    identity = dict(prepared_hash=endpoint.content_hash, recipe=recipe, checkout=checkout_identity(),
                    environment=dict(environment_identity(torch.device(device))), fold=fold, seed=seed,
                    run_logging=run_logging)
    with state_lock(root):
        path = root / 'progress.json'
        record = json.loads(path.read_text()) if path.exists() else dict(identity=identity,
            status='incomplete', predecessor=None, parts=[], slices=[])
        require(record['identity'] == identity, 'journal identity differs')
        if record['status'] == 'complete':
            return record
        logger = logger_for(run_logging, identity, root)
        try:
            atomic_record(path, record)
            destination = root / ('slice-' + uuid.uuid4().hex)
            request = CheckpointRequest()
            config = endpoint.configuration(fold, destination, **nested_cv._settings(recipe, endpoint),
                study=recipe_study(recipe), device=device, slice_seconds=slice_seconds,
                checkpoint_margin_seconds=min(60., slice_seconds / 10), stop_request=request,
                predecessor=None if record['predecessor'] is None else FoldArtifacts.from_json(record['predecessor']))
            started = time.monotonic()
            # Keep signal handlers active through prediction and journal publication.
            with request.signals():
                artifact = nested_cv.run_fold(config, endpoint.outer, seed, geography=endpoint.geography)
                record['predecessor'] = artifact.to_json()
                if artifact.complete:
                    view = subset(endpoint.data.covariates(tuple(n for n, _ in endpoint.feature_kinds)),
                                  artifact.prediction_inputs.original_ids)
                    part = nested_cv.predict(artifact, view, endpoint.policy)
                    record['parts'] = [part.to_json()]
                    record['status'] = 'complete'
                event = dict(fold=fold, **checkpoint_progress(artifact.checkpoint), stop_reason=artifact.checkpoint.reason,
                             wall_seconds=time.monotonic() - started, complete=artifact.complete,
                             output_dir=str(destination))
                record['slices'].append(event)
                atomic_record(path, record)
            if logger is not None:
                logger.log(event)
            return record
        finally:
            if logger is not None:
                logger.finish()


def score(prepared, states, out, run_logging):
    destination = write_directory(out)
    endpoint, recipe, audit = load_prepared(prepared)
    records = [json.loads((Path(s) / 'progress.json').read_text()) for s in states]
    require(len(records) == 5 and all(r['status'] == 'complete' for r in records), 'five complete folds required')
    records.sort(key=lambda r: r['identity']['fold'])
    require([r['identity']['fold'] for r in records] == list(range(5)), 'one state per fold required')
    identities = [{k: v for k, v in r['identity'].items() if k != 'fold'} for r in records]
    require(all(i == identities[0] for i in identities), 'fold identities differ')
    require(identities[0]['prepared_hash'] == endpoint.content_hash and identities[0]['recipe'] == recipe
            and identities[0]['seed'] == 1103, 'score prepared identity differs')
    parts = []
    for fold, record in enumerate(records):
        require(len(record['parts']) == 1, 'one prediction part per fold required')
        part = coverage.OOFNuisances.from_json(record['parts'][0])
        require(set(part.fold_ids) == {fold} and set(part.seed_ids) == {1103}, 'prediction fold/seed differs')
        parts.append(part.to_json())
    log_config = dict(prepared_hash=endpoint.content_hash, recipe=recipe, tree=identities[0]['checkout'], **LABELS)
    logger = logger_for(run_logging, log_config, destination)
    try:
        destination.mkdir(parents=True, exist_ok=True)
        # All prediction parts exist, so estimate_repetition performs no fitting.
        estimates = coverage.estimate_repetition(endpoint, recipe, destination, time.monotonic(),
            study=recipe_study(recipe), continuation=dict(parts=parts))
        require(estimates is not None, 'scoring incomplete')
        readout = policy_readout(endpoint, audit.get('support_caliper_iqr', 2.0))
        nuisances = coverage.OOFNuisances.from_json((destination / 'nuisances.json').read_text())
        diagnostics = policy_diagnostics(endpoint, readout, estimates, nuisances)
        result = dict(estimates=estimates, **readout, policy_diagnostics=diagnostics, n={k: audit[k] for k in
            ('rows_read', 'frame_rows', 'labelled', 'eligible', 'support_pool', 'support_candidates', 'targets',
             'in_frame_by_flag', 'exclusions_by_reason', 'layer_counts')},
            run_logging=run_logging, **log_config)
        destination.mkdir(parents=True, exist_ok=True)
        atomic_record(destination / 'result.json', result)
        lines = ['# Exploratory USALEEP estimate', '', LABELS['sealing'], '',
                 'Certifies no release criterion. Effect release authorized: false.', '',
                 '| Estimator | Estimate | Primary SE |', '| --- | ---: | ---: |']
        lines += [f"| {name} | {v['value']:.8g} | {v['se']:.8g} |" for name, v in estimates.items()]
        lines += ['', 'Full readout (including every SE definition and identity):', '', '```json',
                  json.dumps(result, indent=2), '```']
        (destination / 'result.md').write_text('\n'.join(lines) + '\n')
        if logger is not None:
            logger.log(dict(final_estimates=estimates))
        return result
    finally:
        if logger is not None:
            logger.finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    for name in ('tract-inputs', 'atlas', 'usaleep-tar', 'usaleep-receipts', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--delta-mmhg', type=float, default=2.0)
    p.add_argument('--support-caliper-iqr', type=float, default=None)
    p = commands.add_parser('fit')
    for name in ('prepared', 'state'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--fold', required=True, type=int)
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--slice-seconds', required=True, type=float)
    p.add_argument('--run-logging', required=True, choices=('wandb', 'none'))
    p = commands.add_parser('score')
    p.add_argument('--prepared', required=True)
    p.add_argument('--state', required=True, action='append')
    p.add_argument('--out', required=True)
    p.add_argument('--run-logging', choices=('wandb', 'none'), default='wandb')
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    if command == 'score':
        args['states'] = args.pop('state')
    {'prepare': prepare, 'fit': fit, 'score': score}[command](**args)


if __name__ == '__main__':
    main()
