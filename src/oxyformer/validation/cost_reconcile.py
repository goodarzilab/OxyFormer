"""CPU accounting of sealed screening repetitions, without fitting or unpickling."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tarfile

TREE = '626d4dd603ce399b68b93bee644fe27cf452377f'
REPORT_TREE = 'f25e8853ba691883b447e0f89e9ce334f84f56ac'
TERMS = ('inner_ssl', 'inner_outcome', 'inner_origin', 'refit_ssl',
         'refit_outcome', 'refit_origin', 'scoring')
PHASES = ('compute_seconds', 'setup_seconds', 'import_seconds',
          'export_seconds', 'publication_seconds')
LIMIT = ('Terminal continuation archives retain completed OOF prediction blobs, not '
         'the controller task/fit timing history. Cumulative fold wall time cannot '
         'identify individual operations. Real per-term durations, counts, excess '
         'shares and transferable correction factors are unavailable; no per-term '
         'values are estimated. Decomposition is per leaf and per repetition only.')


def q(value, basis='derived'):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('finite numeric quantity required')
    return {'value': value, 'basis': '[' + basis + ']'}


def number(value):
    """Read the harness quantity wrapper; labels describe its timing, not accuracy."""
    result = value['value']
    if type(result) not in (int, float) or not math.isfinite(result) or result < 0:
        raise ValueError('finite nonnegative report quantity required')
    return result


def measured(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('finite nonnegative measurement required')
    return q(value, 'measured')


def load(path):
    return json.loads(Path(path).read_text())


def identity(path):
    return {'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}



def identifier_copy(value):
    """Keep task schedule indices as identifiers, not measured quantities."""
    if isinstance(value, dict):
        if 'value' in value and 'basis' in value:
            return str(number(value))
        return {k: identifier_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [identifier_copy(v) for v in value]
    return str(value) if type(value) in (int, float) else value


def relabel(value, *, count=False):
    """Copy report evidence with explicit labels on every numerical leaf."""
    if isinstance(value, dict):
        if 'value' in value and 'basis' in value:
            basis = value['basis']
            labels = set(basis.strip('[]').split(', ')) if isinstance(basis, str) else set(basis)
            if count:
                labels.add('derived')
            return q(number(value), ', '.join(sorted(labels)))
        return {k: identifier_copy(v) if k in ('tasks_per_outer_seed', 'tasks_by_outer') else
                relabel(v, count=(count or k in ('work', 'counts', 'harness_count',
                        'outer_folds', 'seed_count', 'scoring_operations')) and k != 'seconds')
                for k, v in value.items()}
    if isinstance(value, list):
        return [relabel(v, count=count) for v in value]
    if type(value) in (int, float):
        return q(value) if count else measured(value)
    return value


def read_chain(entry, manifest):
    root = Path(entry['terminal_leaf_attempt_dir'])
    if not root.is_absolute():
        root = manifest.parent / root
    budget_path = root / 'budget_record.json'
    budget = load(budget_path)
    binding = budget['binding']
    if (binding['code_identity'] != TREE or not budget['complete'] or not budget['success']
            or entry['terminal_leaf_state'] != 'DONE'):
        raise ValueError('requires successful complete pinned-tree chain')
    if binding['scenario']['name'] != 'sign_changing':
        raise ValueError('requires sign_changing diagnostic chain')
    screening = dict(frozen_epochs=30, grid_points=2, reference_cap=32, seed_count=1)
    if any(type(binding['study'].get(k)) not in (int, float) or binding['study'][k] != v
           for k, v in screening.items()):
        raise ValueError('requires screening study')
    times = budget['profile']['complete_repetition_seconds']
    if len(times) != 1 or times[0] <= 0:
        raise ValueError('requires one positive complete repetition')
    repetition = measured(times[0])
    leaves = []
    for index, leaf in enumerate(budget['leaves']):
        if leaf['step'] != index:
            raise ValueError('nonconsecutive leaf history')
        phases = {key: measured(leaf[key]) for key in PHASES}
        wall = measured(leaf['wall_seconds'])
        if not math.isclose(math.fsum(v['value'] for v in phases.values()),
                            wall['value'], rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError('leaf phases do not conserve wall time')
        leaves.append({'task': leaf['task_id'], 'step': str(index),
                       'wall_seconds': wall, **phases})
    if not leaves or len({leaf['task'] for leaf in leaves}) != len(leaves):
        raise ValueError('missing or duplicate leaf history')
    total = math.fsum(leaf['compute_seconds']['value'] for leaf in leaves)
    # The leaf timer encloses the per-draw timer, CUDA synchronisation,
    # result writes and calls made after the repetition has already finished.
    # Its excess is accounting evidence, not a timing error or a fit estimate.
    if total < times[0] and not math.isclose(total, times[0], rel_tol=1e-6, abs_tol=1e-3):
        raise ValueError('leaf compute cannot be less than enclosed repetition time')
    archive = root / 'continuation.tar'
    with tarfile.open(archive) as tar:
        members = tar.getnames()
        coverage = json.load(tar.extractfile('coverage.json'))
        if coverage['binding']['code_identity'] != TREE or len(coverage['states']) != 1:
            raise ValueError('archive binding or repetition mismatch')
        state = coverage['states'][0]
        if state['pending'] is not None or state['record']['status'] != 'success':
            raise ValueError('archive repetition not complete')
        if state['binding']['draw'] != binding['draws'][0]:
            raise ValueError('archive draw mismatch')
        if not math.isclose(state['compute_seconds'], times[0], rel_tol=1e-6, abs_tol=1e-3):
            raise ValueError(f"archive compute mismatch: archive={state['compute_seconds']}, budget={times[0]}")
        # Completed parts are the only retained fold artifacts. Do not load
        # observations or infer task time from epoch counts or total wall time.
        part_types = []
        for part in state['parts']:
            blob = json.load(tar.extractfile('blobs/' + part))
            part_types.append(blob['type'])
            if blob['type'] != 'OOFNuisances':
                raise ValueError('unexpected completed part type ' + blob['type'] +
                                 '; inspect its timing schema')
    return {'chain': entry['chain'], 'repetition': binding['draws'][0]['repetition_id'],
            'unrecognized_study_keys': sorted(set(binding['study']) - set(screening)),
            'archive_budget_discrepancy_seconds': measured(abs(state['compute_seconds'] - times[0])),
            'compute_seconds': repetition, 'compute_hours': q(times[0] / 3600),
            'leaf_compute_seconds': q(total), 'leaf_wrapper_seconds': q(max(0., total - times[0])),
            'leaf_timer_residual_seconds': q(total - times[0]),
            'leaf_wrapper_limit': 'Wrapper estimate is the timer residual clipped at zero within rounding tolerance; signed residual is reported separately.',
            'compute_scope': 'Repetition timer measures execute_draw calls; leaf compute also includes wrapper work and already-complete calls.',
            'leaves': leaves,
            'overhead_seconds': {key: q(math.fsum(x[key]['value'] for x in leaves))
                                 for key in PHASES if key != 'compute_seconds'},
            'archive_members': members, 'completed_part_types': part_types,
            'operation_counters': None, 'limit': LIMIT,
            'budget': identity(budget_path), 'archive': identity(archive)}


def reconcile(manifest_path, report_path):
    manifest_path, report_path = Path(manifest_path), Path(report_path)
    manifest, harness = load(manifest_path), load(report_path)
    if manifest['tree'] != TREE or manifest['campaign'] != 'm2-diag-A':
        raise ValueError('unsupported manifest')
    if harness['tree'] != REPORT_TREE:
        raise ValueError('unsupported report tree')
    chains = [read_chain(entry, manifest_path) for entry in manifest['chains']]
    if not chains or len({c['repetition'] for c in chains}) != len(chains):
        raise ValueError('requires distinct repetitions')
    hours = [c['compute_hours']['value'] for c in chains]
    mean = statistics.mean(hours)
    formula = harness['formula']
    projections, terms = {}, {}
    for cell in ('a', 'b', 'c', 'd', 'e'):
        counts, seconds = formula[cell]['counts'], formula[cell]['seconds']
        if set(counts) != set(TERMS) or set(seconds) != set(TERMS):
            raise ValueError('missing or extra harness terms')
        projections[cell] = math.fsum(number(counts[t]) * number(seconds[t]) for t in TERMS) / 3600
        # Stored cell hours may be rounded.
        if not math.isclose(projections[cell], number(harness['cells'][cell]['compute_h100_hours_per_repetition']), rel_tol=1e-6):
            raise ValueError('harness formula and projection disagree')
    anchor = projections['e']
    if anchor <= 0:
        raise ValueError('positive anchor required')
    excess = anchor - mean
    for term in TERMS:
        count, seconds = number(formula['e']['counts'][term]), number(formula['e']['seconds'][term])
        terms[term] = {'harness_count': relabel(formula['e']['counts'][term], count=True),
            'harness_seconds_per_operation': relabel(formula['e']['seconds'][term]),
            'harness_total_hours': q(count * seconds / 3600),
            'real_count': None, 'real_seconds': None, 'excess_share': None,
            'correction_factor': None, 'reason': LIMIT}
    receipt = harness['calibration']['receipt']
    scenario_ratio = 31.8 / 29.0
    return {'manifest': identity(manifest_path), 'harness': {**identity(report_path), 'tree': REPORT_TREE},
        'chain_tree': TREE, 'chains': chains, 'granularity': 'leaf and repetition', 'limit': LIMIT,
        'summary': {'chain_count': q(len(chains)), 'mean_compute_hours': q(mean),
            'min_compute_hours': q(min(hours)), 'max_compute_hours': q(max(hours)),
            'sample_sd_hours': q(statistics.stdev(hours)) if len(hours) > 1 else None,
            'mean_standard_error_hours': q(statistics.stdev(hours) / math.sqrt(len(hours))) if len(hours) > 1 else None,
            'harness_cell_e_hours': q(anchor), 'excess_hours': q(excess),
            'harness_to_measured_ratio': q(anchor / mean),
            'cell_e_correction_factor': q(mean / anchor),
            'factor_observed_min': q(min(hours) / anchor), 'factor_observed_max': q(max(hours) / anchor),
            'uncertainty': 'Observed range and sample spread across collected repetitions; not a confidence interval or hardware transfer guarantee.'},
        'terms': terms, 'harness_formula': relabel(formula, count=True), 'harness_cells': relabel(harness['cells']),
        'calibration': harness['calibration'],
        'receipt': relabel(receipt), 'harness_overhead': relabel(harness['overhead']),
        'scenario_adjustment': {'ratio': q(scenario_ratio),
            'basis_description': 'Only available scenario adjustment: month-one sign_changing profile / null_effect profile.',
            'sign_changing_profile_hours': q(31.8, 'measured'), 'null_effect_profile_hours': q(29.0, 'measured'),
            'receipt_adjusted_hours': q(number(receipt['compute_hours']) * scenario_ratio, 'extrapolated'),
            'chain_mean_null_effect_equivalent_hours': q(mean / scenario_ratio, 'extrapolated'),
            'limit': 'Scenario ratio is descriptive; its transfer uncertainty is not measured.'},
        'candidates': {
            'representative-fold differences': {
                'verdict': 'not assessable',
                'missing_input': 'Per-operation durations by representative fold/grid/seed.'},
            'epoch-slicing reconstruction': {
                'verdict': 'not assessable',
                'missing_input': 'Timing history separating controller reconstruction from fitting.'},
            'receipt identity': receipt_assessment(hours, receipt['compute_hours'], scenario_ratio)},
        'corrected_projection_hours': {'e': q(mean, 'extrapolated'), 'd': None, 'c': None, 'a': None},
        'transfer_limit': 'No term-level correction factor transfers on the retained evidence; other cells are not recalibrated.',
        'limits': ['CPU accounting only; no new GPU timing or training.',
                   'Timings apply only to the H100 hardware of the collected chains.',
                   'Sealed budget excludes reserved final record/publication; failed attempts outside sealed history are not counted.',
                   'No final cell or repetition count is recommended.']}



def receipt_assessment(hours, receipt_quantity, scenario_ratio):
    """Describe scale agreement, without attributing the harness excess."""
    mean = statistics.mean(hours)
    sd = statistics.stdev(hours) if len(hours) > 1 else None

    def assess(value, quantity):
        distance = abs(value - mean)
        if sd is None:
            return {'hours': quantity, 'distance_sample_sds': None,
                    'verdict': 'not assessable', 'missing_input': 'At least two complete repetitions.'}
        # Zero spread admits only an exact match; avoid non-finite JSON.
        return {'hours': quantity,
                'absolute_distance_hours': q(distance),
                'distance_sample_sds': q(distance / sd) if sd > 0 else None,
                'verdict': 'consistent' if distance <= 2 * sd else 'inconsistent'}

    return {'rule': 'Consistent within two sample SDs of the chain mean (inclusive); inconsistent beyond. '
                    'With zero SD only exact equality is consistent; SD distance is unavailable.',
            'adjustment_direction': 'Multiply receipt hours by 31.8/29.0: null_effect to sign_changing.',
            'chain_mean_hours': q(mean), 'sample_sd_hours': q(sd) if sd is not None else None,
            'unadjusted': assess(number(receipt_quantity), relabel(receipt_quantity)),
            'scenario_adjusted': assess(number(receipt_quantity) * scenario_ratio,
                                       q(number(receipt_quantity) * scenario_ratio, 'extrapolated')),
            'limit': 'Scenario transfer uncertainty is unmeasured; scale agreement does not identify a cause.'}


def markdown(report):
    """Render the full accounting, including all numerical leaves and their bases."""
    lines = ['# Screening cost reconciliation', '', report['limit'], '',
             'Aggregate screening factor: ' + str(report['summary']['cell_e_correction_factor']['value']) + ' [derived].',
             'Source calibration passed: ' + str(report['calibration'].get('passed')) + '.',
             'Source calibration (this factor does not override it):',
             json.dumps({k: report['calibration'].get(k) for k in ('ratio', 'reasons')}),
             json.dumps({k: report['calibration']['receipt'].get(k)
                         for k in ('tree', 'frame_matches', 'sources_match')}),
             'Scenario adjustment: 31.8/29.0 [derived]; transfer uncertainty is not measured.', '',
             'All times are seconds unless the field says hours. Null means unavailable.', '']

    def walk(value, label, depth=0):
        if isinstance(value, dict) and set(value) == {'value', 'basis'}:
            basis = value['basis']
            if isinstance(basis, list):
                basis = '[' + ', '.join(basis) + ']'
            lines.append(f'- {label}: {value["value"]:.10g} {basis}')
        elif isinstance(value, dict):
            lines.extend(['', '**' + label + '**', ''])
            for key, item in value.items():
                walk(item, key, depth + 1)
        elif isinstance(value, list):
            for item in value:
                walk(item, label, depth + 1)
        else:
            lines.append(f'- {label}: {value if value is not None else "unavailable"}')
    walk(report, 'Accounting')
    return '\n'.join(lines) + '\n'



def referenced_paths(value, parent):
    """Collect artifact paths named anywhere in the registered harness report."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'path' and isinstance(item, str):
                path = Path(item)
                yield path if path.is_absolute() else parent / path
            else:
                yield from referenced_paths(item, parent)
    elif isinstance(value, list):
        for item in value:
            yield from referenced_paths(item, parent)


def write_outputs(destinations, sources, contents):
    """Reserve both fresh destinations before writing either payload."""
    resolved = [p.resolve() for p in destinations]
    a, b = destinations
    if resolved[0] == resolved[1] or (a.exists() and b.exists() and a.samefile(b)):
        raise ValueError('output destinations alias')
    protected = {p.resolve() for p in sources}
    if any(p in protected for p in resolved):
        raise ValueError('output collides with input')
    # lstat also sees dangling symlinks. O_EXCL repeats the check at creation.
    for path in destinations:
        if os.path.lexists(path):
            raise FileExistsError('output must be fresh: ' + str(path))
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    opened = []
    try:
        for path in destinations:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            opened.append((path, fd))
    except BaseException:
        for path, fd in opened:
            os.close(fd)
            path.unlink()
        raise
    try:
        for (_, fd), content in zip(opened, contents):
            # Keep ownership of descriptors here, including if writing fails.
            with os.fdopen(fd, 'w', closefd=False) as stream:
                stream.write(content)
    except BaseException:
        for path, _ in opened:
            path.unlink()
        raise
    finally:
        for _, fd in opened:
            os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    report = reconcile(args.manifest, args.report)
    sources = [args.manifest, args.report] + [
        Path(c[k]['path']) for c in report['chains'] for k in ('budget', 'archive')]
    destinations = [args.out / name for name in ('reconciliation.json', 'reconciliation.md')]
    sources.extend(referenced_paths(load(args.report), args.report.parent))
    contents = [json.dumps(report, indent=2, allow_nan=False) + '\n', markdown(report)]
    write_outputs(destinations, sources, contents)


if __name__ == '__main__':
    main()
