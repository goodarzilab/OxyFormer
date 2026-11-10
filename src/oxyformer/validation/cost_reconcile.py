"""CPU accounting of sealed screening repetitions, without fitting or unpickling."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def relabel(value, *, count=False):
    """Copy report evidence with explicit labels on every numerical leaf."""
    if isinstance(value, dict):
        if 'value' in value and 'basis' in value:
            basis = value['basis']
            labels = set(basis.strip('[]').split(', ')) if isinstance(basis, str) else set(basis)
            if count:
                labels.add('derived')
            return q(number(value), ', '.join(sorted(labels)))
        return {k: relabel(v, count=count or k in ('counts', 'harness_count')) for k, v in value.items()}
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
    if binding['study'] != dict(frozen_epochs=30, grid_points=2, reference_cap=32, seed_count=1):
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
        leaves.append({'task': leaf['task_id'], 'step': measured(index),
                       'wall_seconds': wall, **phases})
    if not leaves or len({leaf['task'] for leaf in leaves}) != len(leaves):
        raise ValueError('missing or duplicate leaf history')
    total = math.fsum(leaf['compute_seconds']['value'] for leaf in leaves)
    # The leaf timer encloses the per-draw timer, CUDA synchronisation,
    # result writes and calls made after the repetition has already finished.
    # Its excess is accounting evidence, not a timing error or a fit estimate.
    if total < times[0] - 1e-6:
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
        if not math.isclose(state['compute_seconds'], times[0], rel_tol=1e-9):
            raise ValueError('archive compute mismatch')
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
            'compute_seconds': repetition, 'compute_hours': q(times[0] / 3600),
            'leaf_compute_seconds': q(total), 'leaf_wrapper_seconds': q(total - times[0]),
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
        'terms': terms, 'harness_formula': relabel(formula), 'harness_cells': relabel(harness['cells']),
        'calibration': harness['calibration'],
        'receipt': relabel(receipt), 'harness_overhead': relabel(harness['overhead']),
        'scenario_adjustment': {'ratio': q(scenario_ratio),
            'basis_description': 'Only available scenario adjustment: month-one sign_changing profile / null_effect profile.',
            'sign_changing_profile_hours': q(31.8, 'measured'), 'null_effect_profile_hours': q(29.0, 'measured'),
            'receipt_adjusted_hours': q(number(receipt['compute_hours']) * scenario_ratio, 'extrapolated'),
            'chain_mean_null_effect_equivalent_hours': q(mean / scenario_ratio, 'extrapolated'),
            'limit': 'Scenario ratio is descriptive; its transfer uncertainty is not measured.'},
        'candidates': {
            'representative-fold differences': 'Unresolved: no retained per-operation durations; representative fold/grid/seed sampling may contribute, but cannot be isolated.',
            'epoch-slicing reconstruction': 'Compatible with excess, not established as its cause: sliced harness calls reconstruct controllers; archives lack timing history separating reconstruction from fitting.',
            'receipt identity': 'Different scenario and code identity; not an exact replay. Independent complete repetitions corroborate the receipt scale. Frame/source matching flags are harness evidence only; receipt mismatch alone is not established as the cause.'},
        'corrected_projection_hours': {'e': q(mean, 'extrapolated'), 'd': None, 'c': None, 'a': None},
        'transfer_limit': 'No term-level correction factor transfers on the retained evidence; other cells are not recalibrated.',
        'limits': ['CPU accounting only; no new GPU timing or training.',
                   'Timings apply only to the H100 hardware of the collected chains.',
                   'Sealed budget excludes reserved final record/publication; failed attempts outside sealed history are not counted.',
                   'No final cell or repetition count is recommended.']}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    report = reconcile(args.manifest, args.report)
    sources = [args.manifest, args.report] + [
        Path(c[k]['path']) for c in report['chains'] for k in ('budget', 'archive')]
    for name in ('reconciliation.json', 'reconciliation.md'):
        dest = args.out / name
        if dest.exists() and any(dest.samefile(src) for src in sources):
            raise ValueError('output collides with input')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'reconciliation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (args.out / 'reconciliation.md').write_text(markdown(report))


if __name__ == '__main__':
    main()
