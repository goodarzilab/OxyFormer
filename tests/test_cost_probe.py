"""Synthetic arithmetic checks; GPU and scheduler evidence comes from jobs."""
import pytest

from oxyformer.validation.cost_probe import (
    BASE_TREE, Quantity, calibration, epoch_projection, feasibility,
    operation_counts, project, ratio,
)


@pytest.fixture
def counted_fixture(monkeypatch):
    """Arithmetic tests use a source fixture, including in shallow CI clones."""
    from pathlib import Path
    from oxyformer.validation import cost_probe
    nested = Path('src/oxyformer/training/nested_cv.py').read_text()
    coverage = Path('src/oxyformer/validation/coverage.py').read_text()

    def fixture_work(*, seed_count, grid_points, tree=BASE_TREE):
        return {**cost_probe.source_work(nested, coverage, seed_count=seed_count,
                                        grid_points=grid_points), 'tree': tree}

    monkeypatch.setattr(cost_probe, 'pinned_work', fixture_work)


def measured(x):
    return Quantity(x, frozenset({'measured'}))


def test_partial_epoch_is_not_used_as_rate():
    q = epoch_projection(required=30, completed=2, wall_seconds=90,
                         boundary_seconds=40, attempts=2)
    assert q == Quantity(600, frozenset({'extrapolated'}))
    assert epoch_projection(required=30, completed=0, wall_seconds=90,
                            boundary_seconds=0, attempts=2) is None
    with pytest.raises(ValueError, match='second attempt'):
        epoch_projection(required=30, completed=2, wall_seconds=90,
                         boundary_seconds=40, attempts=1)


def test_finished_fit_is_measured():
    assert epoch_projection(required=30, completed=30, wall_seconds=90,
                            boundary_seconds=90, attempts=2) == measured(90)


@pytest.mark.parametrize('seeds', [1, 3])
@pytest.mark.parametrize('grid', [2, 4])
def test_counted_controller_work(seeds, grid, counted_fixture):
    # Fixture traversal of the nested task ordering, including both refits.
    from collections import Counter
    tasks = []
    for outer in range(5):
        for seed in range(seeds):
            for inner in range(3):
                tasks.append('ssl')
                for kind in ('outcome', 'origin'):
                    for point in range(grid):
                        tasks.append('inner_' + kind)
            tasks.extend(('ssl', 'refit_outcome', 'refit_origin'))
    tasks.append('scoring')
    assert operation_counts(seed_count=seeds, grid_points=grid) == dict(Counter(tasks))


def test_projection_includes_scoring_and_refit_work(counted_fixture):
    counts = operation_counts(seed_count=1, grid_points=2)
    measurements = {name: {'tree': BASE_TREE, 'seconds': measured(i * 3600)}
                    for i, name in enumerate(counts, 1)}
    q = project(measurements, counts, tree=BASE_TREE)
    assert q.value == sum(i * count for i, count in enumerate(counts.values(), 1))
    measurements['scoring']['tree'] = 'different'
    with pytest.raises(ValueError, match='tree mismatch'):
        project(measurements, counts, tree=BASE_TREE)


@pytest.mark.parametrize('value,passed', [(.799, False), (.8, True), (1.25, True), (1.251, False)])
def test_calibration_window(value, passed):
    result = calibration(measured(value), measured(1), measured_anchor=True,
                         frame_matches=True, cell_matches=True, operations_match=True)
    assert result['passed'] is passed
    assert ('uncalibrated' in result['ratio']['basis']) is (not passed)


@pytest.mark.parametrize('field,reason', [
    ('measured_anchor', 'anchor extrapolated'), ('frame_matches', 'frame mismatch'),
    ('cell_matches', 'cell mismatch'), ('operations_match', 'operation mismatch'),
    ('sources_match', 'protected source mismatch'),
])
def test_calibration_identity_and_anchor_rules(field, reason):
    args = dict(measured_anchor=True, frame_matches=True, cell_matches=True, operations_match=True)
    args[field] = False
    result = calibration(measured(1), measured(1), **args)
    assert not result['passed'] and reason in result['reasons']
    assert 'uncalibrated' in result['ratio']['basis']


def test_labels_propagate_into_ratios_projections_and_table(counted_fixture):
    q = Quantity(1, frozenset({'extrapolated', 'uncalibrated'}))
    counts = operation_counts(seed_count=1, grid_points=2)
    measurements = {name: {'tree': BASE_TREE, 'seconds': measured(1)} for name in counts}
    measurements['inner_origin']['seconds'] = q
    projected = project(measurements, counts, tree=BASE_TREE)
    labels = {'measured', 'extrapolated', 'uncalibrated'}
    assert projected.basis == labels
    assert ratio(q, measured(1)).basis == labels
    for row in feasibility(projected, measured(2)):
        for key, item in row.items():
            if isinstance(item, dict):
                assert set(item['basis']) == labels


def test_count_walk_responds_to_actual_controller_source():
    from pathlib import Path
    from oxyformer.validation.cost_probe import source_work
    nested = Path('src/oxyformer/training/nested_cv.py').read_text()
    coverage = Path('src/oxyformer/validation/coverage.py').read_text()
    ordinary = source_work(nested, coverage, seed_count=1, grid_points=2)
    changed = source_work(nested.replace('for fold in range(3):', 'for fold in range(2):'),
                          coverage, seed_count=1, grid_points=2)
    assert ordinary['counts']['inner_outcome'] == 30
    assert changed['counts']['inner_outcome'] == 20
    assert changed['counts']['ssl'] == 15
    assert changed['counts']['scoring'] == 1


def test_source_walk_omits_inner_partitions_without_targets():
    from pathlib import Path
    from oxyformer.validation.cost_probe import source_work
    walked = source_work(Path('src/oxyformer/training/nested_cv.py').read_text(),
                         Path('src/oxyformer/validation/coverage.py').read_text(),
                         seed_count=1, grid_points=2, held_inner_folds=(0, 2))
    assert walked['counts']['inner_origin'] == 20
    assert walked['counts']['ssl'] == 15


def test_source_walk_refuses_changed_scoring_operations():
    from pathlib import Path
    from oxyformer.validation.cost_probe import source_work
    with pytest.raises(ValueError, match='scoring operations'):
        source_work(Path('src/oxyformer/training/nested_cv.py').read_text(),
                    Path('src/oxyformer/validation/coverage.py').read_text().replace(
                        'cv_tmle(nuisances', 'different_estimator(nuisances'),
                    seed_count=1, grid_points=2)


def test_harness_resume_binding_and_durable_slice(tmp_path):
    from oxyformer.validation.cost_probe import atomic_record, resume_record
    path = tmp_path / 'progress.json'
    binding = {'tree': BASE_TREE, 'head': 'head', 'frame': 'frame', 'cell': 'e'}
    first = resume_record(path, binding)
    first['attempts'].append('first-job')
    first['slices'].append({'completed': 1, 'seconds': 12})
    first['predecessor'] = 'checkpoint-descriptor'
    atomic_record(path, first)
    second = resume_record(path, binding)
    second['attempts'].append('second-job')
    atomic_record(path, second)
    assert resume_record(path, binding)['slices'] == first['slices']
    assert resume_record(path, binding)['predecessor'] == 'checkpoint-descriptor'
    with pytest.raises(ValueError, match='binding mismatch'):
        resume_record(path, {**binding, 'frame': 'another-frame'})


def test_epoch_slice_stops_after_evaluation():
    from oxyformer.validation.cost_probe import epoch_timing
    timing = epoch_timing(float('inf'))
    assert not timing.boundary('nuisance')
    assert not timing.boundary('evaluation')
    assert timing.boundary('nuisance')


def test_finalizing_slice_preserves_held_out_evaluation():
    from oxyformer.validation.cost_probe import epoch_timing
    timing = epoch_timing(float('inf'), finalize=True)
    assert not timing.boundary('nuisance')
    assert not timing.boundary('evaluation')
    assert not timing.boundary('evaluation')
    assert timing.boundary('nuisance')


def test_real_controller_epoch_slices_and_resume(tmp_path):
    """Exercise the existing controller, not a mirror of its progress state."""
    from dataclasses import replace
    from test_nested_cv import endpoint, tiny_config, state
    from oxyformer.training import nested_cv
    from oxyformer.validation.cost_probe import epoch_timing

    prepared = endpoint(tmp_path / 'data')
    initial = tiny_config(prepared, tmp_path / 'ssl', max_batches=2)
    # The synthetic SSL fixture has one fitting and one validation batch.
    ssl = nested_cv.run_fold(initial, prepared.outer, 1103, geography=prepared.geography)
    assert state(ssl)['position'] == 1
    assert state(ssl)['active'] is None
    config = replace(initial, max_batches=None,
                     settings=replace(initial.settings, frozen_epochs=2))
    # Runtime budgets may change across continuations, scientific settings may
    # not: build the initial checkpoint under the same two-epoch configuration.
    initial = replace(config, output_dir=str(tmp_path / 'ssl-two'), max_batches=2)
    ssl = nested_cv.run_fold(initial, prepared.outer, 1103, geography=prepared.geography)
    first = nested_cv.run_fold(replace(config, output_dir=str(tmp_path / 'epoch-one'),
        predecessor=ssl, deadline_timing=epoch_timing(float('inf'))),
        prepared.outer, 1103, geography=prepared.geography)
    assert state(first)['position'] == 1
    assert state(first)['active']['progress']['epoch'] == 1
    second = nested_cv.run_fold(replace(config, output_dir=str(tmp_path / 'epoch-two'),
        predecessor=first, deadline_timing=epoch_timing(float('inf'))),
        prepared.outer, 1103, geography=prepared.geography)
    assert state(second)['active']['progress']['epoch'] == 2
    final = nested_cv.run_fold(replace(config, output_dir=str(tmp_path / 'finalize'),
        predecessor=second, deadline_timing=epoch_timing(float('inf'), finalize=True)),
        prepared.outer, 1103, geography=prepared.geography)
    assert state(final)['position'] == 2
    assert state(final)['active'] is None
    assert state(final)['results'][0]['kind'] == 'outcome'


def test_measurement_harness_resumes_real_controller_across_attempts(tmp_path, monkeypatch, counted_fixture):
    from dataclasses import replace
    import time
    import torch
    from test_nested_cv import endpoint, tiny_config
    from oxyformer.training import nested_cv
    from oxyformer.training.fit import RecipeStudy
    from oxyformer.validation import cost_probe

    prepared = endpoint(tmp_path / 'data')
    initial = tiny_config(prepared, tmp_path / 'initial', max_batches=2)
    initial = replace(initial, settings=replace(initial.settings, frozen_epochs=2))
    predecessor = nested_cv.run_fold(initial, prepared.outer, 1103, geography=prepared.geography)
    original_configuration = type(prepared).configuration

    def cpu_configuration(self, *args, **kwargs):
        kwargs['device'] = 'cpu'
        return original_configuration(self, *args, **kwargs)

    monkeypatch.setattr(type(prepared), 'configuration', cpu_configuration)
    monkeypatch.setattr(cost_probe, 'require_compute_node', lambda **kwargs: None)
    monkeypatch.setattr(cost_probe, 'clean_identity', lambda **kwargs: 'same-clean-head')
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    original_clock = time.monotonic
    offset = [0.]
    monkeypatch.setattr(time, 'monotonic', lambda: original_clock() + offset[0])
    original_write = cost_probe.atomic_record

    def stop_after_durable_epoch(path, record):
        original_write(path, record)
        if record['attempts'] == ['job-one'] and len(record['slices']) == 1:
            offset[0] = 10000.

    monkeypatch.setattr(cost_probe, 'atomic_record', stop_after_durable_epoch)
    recipe = {'nested_cv': {'ssl_epochs': 1, 'frozen_epochs': 2, 'synthetic': True}}
    monkeypatch.setenv('SLURM_JOB_ID', 'job-one')
    first = cost_probe.measure_fit_task(prepared, recipe, RecipeStudy(), tmp_path / 'measurement',
        position=1, predecessor=predecessor, deadline=time.monotonic() + 10000.)
    assert first['slices'][-1]['epochs_completed'] == 1
    assert not first['slices'][-1]['complete']
    monkeypatch.setenv('SLURM_JOB_ID', 'job-two')
    second = cost_probe.measure_fit_task(prepared, recipe, RecipeStudy(), tmp_path / 'measurement',
        position=1, predecessor=predecessor, deadline=time.monotonic() + 10000.)
    assert second['attempts'] == ['job-one', 'job-two']
    assert second['slices'][-1]['epochs_completed'] == 2
    assert second['slices'][-1]['complete']
    assert cost_probe.fit_quantity(second, 2).basis == {'measured'}


def test_pinned_work_checks_base_source_identity_without_git_history(monkeypatch):
    from pathlib import Path
    import subprocess
    from oxyformer.validation.cost_probe import pinned_work
    calls = []

    def source(command):
        assert command[:2] == ['git', 'show']
        tree, path = command[2].split(':', 1)
        assert tree == BASE_TREE
        calls.append(path)
        return Path(path).read_bytes()

    monkeypatch.setattr(subprocess, 'check_output', source)
    counted = pinned_work(seed_count=1, grid_points=2)
    assert counted['counts']['inner_outcome'] == 30
    assert set(calls) == {'src/oxyformer/training/nested_cv.py',
                          'src/oxyformer/validation/coverage.py',
                          'src/oxyformer/training/fit.py',
                          'src/oxyformer/training/pretrain.py'}
    monkeypatch.setattr(subprocess, 'check_output', lambda command: source(command) + b'\n# drift\n')
    with pytest.raises(ValueError, match='protected source differs'):
        pinned_work(seed_count=1, grid_points=2)


def fit_record(*, epochs=7, complete=True):
    return {'attempts': ['one'], 'binding': {'tree': BASE_TREE},
            'predecessor': 'opaque private checkpoint descriptor',
            'slices': [{'compute_seconds': 70., 'wall_seconds': 100.,
                        'epochs_completed': epochs, 'minibatches_partial_epoch': 0,
                        'minibatches_per_epoch': 5, 'peak_gpu_memory_bytes': 123,
                        'complete': complete}]}


def test_native_patience_completion_is_measured():
    from oxyformer.validation.cost_probe import fit_quantity
    assert fit_quantity(fit_record(), 30) == measured(70.)


def test_terminal_failed_cells_publish_reasons_without_projections(tmp_path, monkeypatch, counted_fixture):
    import json
    from oxyformer.validation import cost_probe as cp
    monkeypatch.setattr(cp, 'lifecycle_identity', lambda root: (__import__('json').loads(
        (root / 'inputs.json').read_text()), 'synthetic-endpoint'))
    work = {c: cp.pinned_work(seed_count=s.seed_count, grid_points=s.grid_points)
            for c, s in cp.study_cells().items()}
    metadata = {'tree': BASE_TREE, 'work': work, 'profile_path': 'sealed-profile',
                'frame_path': 'frame', 'frame_sha256': 'same'}
    (tmp_path / 'inputs.json').write_text(json.dumps(metadata))
    failures = {c: {'status': 'failed_measurement', 'operation': '0',
                   'reason': 'sub-measurement incomplete after two attempts'}
                for c in cp.study_cells()}
    (tmp_path / 'driver-failures.json').write_text(json.dumps(failures))
    jobs = [{'job_id': c + str(n), 'cell': c, 'operation': '0', 'state': 'FAILED'}
            for c in failures for n in (1, 2)]
    (tmp_path / 'driver-jobs.json').write_text(json.dumps(jobs))
    receipt = {'compute_hours': measured(1.), 'study': cp.cell_values(cp.study_cells()['e']),
               'frame_hash': 'same', 'frame_matches': True, 'sources_match': True, 'work': work['e']}
    monkeypatch.setattr(cp, 'calibration_receipt', lambda *args: receipt)
    report = cp.measurement_report(tmp_path, {'ratio': measured(1.).to_dict()})
    for cell in failures:
        assert report['cells'][cell]['status'] == 'failed_measurement'
        assert report['cells'][cell]['reason']
        assert report['cells'][cell]['compute_h100_hours_per_repetition'] is None
        assert report['feasibility'][cell] == []
    assert not report['calibration']['passed']


@pytest.mark.parametrize('sources_match', [True, False])
@pytest.mark.parametrize('frame_matches', [True, False])
@pytest.mark.parametrize('partial_inner', [False, True])
def test_report_preserves_measurements_and_calibration_labels(tmp_path, monkeypatch, counted_fixture, frame_matches, partial_inner, sources_match):
    import json
    from oxyformer.validation import cost_probe as cp
    monkeypatch.setattr(cp, 'lifecycle_identity', lambda root: (__import__('json').loads(
        (root / 'inputs.json').read_text()), 'synthetic-endpoint'))
    work = {c: cp.pinned_work(seed_count=s.seed_count, grid_points=s.grid_points)
            for c, s in cp.study_cells().items()}
    metadata = {'tree': BASE_TREE, 'work': work, 'profile_path': 'sealed-profile',
                'frame_path': 'frame', 'frame_sha256': 'same'}
    cp.atomic_record(tmp_path / 'inputs.json', metadata)
    cp.atomic_record(tmp_path / 'driver-failures.json', {})
    cp.atomic_record(tmp_path / 'driver-jobs.json', [])
    for cell in cp.study_cells():
        for position, (kind, fold, grid) in enumerate(work[cell]['tasks_per_outer_seed']):
            if kind != 'select':
                cp.atomic_record(tmp_path / cell / str(position) / 'progress.json', fit_record())
        cp.atomic_record(tmp_path / cell / 'scoring.json',
                         {'tree': BASE_TREE, 'wall_seconds': 70., 'inputs': 'synthetic inputs, timing only'})
    if partial_inner:
        cp.atomic_record(tmp_path / 'a/1/progress.json', fit_record(epochs=2, complete=False))
        cp.atomic_record(tmp_path / 'driver-jobs.json', [
            {'job_id': str(i), 'cell': 'a', 'operation': '1', 'state': 'TIMEOUT'} for i in (1, 2)])
        cp.atomic_record(tmp_path / 'driver-failures.json', {
            'a': {'operation': '1', 'reason': 'inner fit incomplete after two attempts'}})
    receipt = {'compute_hours': measured(sum(work['e']['counts'].values()) * 70 / 3600),
               'study': cp.cell_values(cp.study_cells()['e']),
               'frame_hash': 'same', 'frame_matches': frame_matches,
               'sources_match': sources_match, 'work': work['e']}
    monkeypatch.setattr(cp, 'calibration_receipt', lambda *args: receipt)
    report = cp.measurement_report(tmp_path, {'ratio': measured(1.).to_dict()})
    assert report['calibration']['passed'] is (frame_matches and sources_match)
    assert 'opaque private' not in json.dumps(report)
    for cell in cp.study_cells():
        basis = set(report['cells'][cell]['compute_h100_hours_per_repetition']['basis'])
        assert ('extrapolated' in basis) is (cell == 'a' and partial_inner)
        assert ('uncalibrated' in basis) is (not (frame_matches and sources_match))
        assert report['measurements'][cell]['refit_ssl']['summary']['epochs_completed']['value'] == 7
        assert report['measurements'][cell]['refit_ssl']['summary']['wall_seconds_at_last_completed_epoch']['value'] == 100
        for row in report['feasibility'][cell]:
            assert all(set(quantity['basis']) == basis for quantity in row.values())
    def labelled_numbers(value):
        if isinstance(value, dict):
            if set(value) == {'value', 'basis'}:
                assert value['basis']
            else:
                for child in value.values():
                    labelled_numbers(child)
        elif isinstance(value, list):
            for child in value:
                labelled_numbers(child)
        else:
            assert type(value) not in (int, float)
    labelled_numbers(report)


def test_report_refuses_unfinished_driver(tmp_path, counted_fixture, monkeypatch):
    from oxyformer.validation import cost_probe as cp
    monkeypatch.setattr(cp, 'lifecycle_identity', lambda root: ({}, 'synthetic-endpoint'))
    cp.atomic_record(tmp_path / 'inputs.json', {})
    with pytest.raises(ValueError, match='driver has not finished'):
        cp.measurement_report(tmp_path, {})


def test_partial_epoch_counts_include_final_training_and_validation_batches():
    from oxyformer.validation.cost_probe import partial_epoch_batches
    # Epoch two has trained all four batches, but has not yet been evaluated.
    progress = {'epoch': 1, 'step': 8, 'phase': 'evaluate'}
    assert partial_epoch_batches(progress, 4) == 4
    # SSL also charges completed validation batches before its epoch closes.
    progress.update(phase='validation', validation_cursor=257)
    assert partial_epoch_batches(progress, 4, validation_batch_size=256) == 6
    # A native completion retains the final validation cursor; it is no longer partial.
    progress['epoch'] = 2
    assert partial_epoch_batches(progress, 4, validation_batch_size=256) == 0


def test_allocated_overhead_uses_reserved_allocation_not_elapsed_wall(monkeypatch):
    from pathlib import Path
    from oxyformer.validation.cost_probe import leaf_overhead
    from oxyformer.validation import coverage
    from oxyformer import provenance
    paths = [Path('/synthetic') / (str(i) + '.json') for i in range(140)]

    def record(path):
        return {'binding': {'wall_seconds': 3300.},
                'leaves': [{'task_id': path.stem, 'wall_seconds': 1000., 'compute_seconds': 900.}]}

    monkeypatch.setattr(coverage, 'read_leaf_measurements', record)
    monkeypatch.setattr(provenance, 'file_hash', lambda path: path.stem)
    result = leaf_overhead(paths)
    assert result['allocated_hours']['value'] == 140.
    assert result['compute_hours']['value'] == 35.
    assert result['ratio']['value'] == 4.
    assert result['runner_wall_hours']['value'] == pytest.approx(140 * 1000 / 3600)


def test_no_returned_slice_does_not_claim_zero_gpu_memory_or_partial_batches():
    from oxyformer.validation.cost_probe import fit_statistics
    stats = fit_statistics({'attempts': ['first', 'second'], 'slices': []})
    assert stats['epochs_completed'] == 0
    assert stats['peak_gpu_memory_bytes'] is None
    assert stats['minibatches_partial_epoch'] is None


def test_driver_records_intent_before_a_submission_reply_can_be_lost(tmp_path, monkeypatch):
    import json
    import subprocess
    import pytest
    from oxyformer.validation import cost_probe as cp
    monkeypatch.setattr(cp, 'lifecycle_identity', lambda root: ({'work': {}}, 'fixture'))
    cp.atomic_record(tmp_path / 'inputs.json', {'work': {}})
    monkeypatch.setattr(cp, 'require_compute_node', lambda: None)
    monkeypatch.setattr(cp, 'clean_identity', lambda: 'clean-head')

    def lost_reply(command, **kwargs):
        assert command[0] == 'sbatch'
        assert '--no-requeue' in command
        # The scheduler accepted this request, but the client died before
        # receiving or durably recording the returned job identifier.
        raise RuntimeError('submission reply lost')

    monkeypatch.setattr(subprocess, 'check_output', lost_reply)
    with pytest.raises(RuntimeError, match='reply lost'):
        cp.drive_measurements(tmp_path)
    ledger = json.loads((tmp_path / 'driver-jobs.json').read_text())
    assert len(ledger) == 1
    assert ledger[0]['state'] == 'SUBMITTING'
    assert ledger[0]['job_id'] is None
    assert ledger[0]['submission_name']


def test_lost_submission_reply_is_recovered_from_the_scheduler(monkeypatch):
    import subprocess
    from oxyformer.validation.cost_probe import recover_submission
    item = {'job_id': None, 'state': 'SUBMITTING', 'submission_name': 'unique-intent',
            'submission_day': '2026-11-01'}

    def query(command, **kwargs):
        assert command[0] in ('sacct', 'squeue')
        return '123|unique-intent\n' if command[0] == 'squeue' else ''

    monkeypatch.setattr(subprocess, 'check_output', query)
    recover_submission(item)
    assert item['job_id'] == '123'
    assert item['state'] == 'PENDING'


def test_unresolved_submission_refuses_instead_of_resubmitting(monkeypatch):
    import subprocess
    import pytest
    from oxyformer.validation.cost_probe import recover_submission
    item = {'job_id': None, 'state': 'SUBMITTING', 'submission_name': 'unknown-intent',
            'submission_day': '2026-11-01'}

    def query(command, **kwargs):
        assert command[0] in ('sacct', 'squeue')
        return ''

    monkeypatch.setattr(subprocess, 'check_output', query)
    with pytest.raises(ValueError, match='submission outcome unresolved'):
        recover_submission(item)
    assert item['job_id'] is None


@pytest.mark.parametrize('active_state', ['RUNNING', 'REQUEUED', 'REQUEUE_HOLD', 'RESIZING'])
def test_driver_recovers_two_live_jobs_without_another_submission(tmp_path, monkeypatch, active_state):
    import json
    import subprocess
    import time
    import pytest
    from oxyformer.validation import cost_probe as cp
    monkeypatch.setattr(cp, 'lifecycle_identity', lambda root: (__import__('json').loads(
        (root / 'inputs.json').read_text()), 'synthetic-endpoint'))
    cp.atomic_record(tmp_path / 'inputs.json', {'work': {}})
    cp.atomic_record(tmp_path / 'driver-jobs.json', [
        {'job_id': None, 'cell': 'e', 'operation': 'score', 'state': 'SUBMITTING',
         'submission_name': 'lost-intent', 'submission_day': '2026-11-01'},
        {'job_id': '222', 'cell': 'a', 'operation': 'score', 'state': 'RUNNING'}])
    monkeypatch.setattr(cp, 'require_compute_node', lambda: None)
    monkeypatch.setattr(cp, 'clean_identity', lambda: 'clean-head')

    def query(command, **kwargs):
        if command[0] == 'squeue':
            return '111|lost-intent\n'
        assert command[0] == 'sacct', 'recovery must not submit a third job'
        if '--name' in command:
            return ''
        job = command[command.index('-j') + 1]
        return job + '|standard|node|00:01:00|' + active_state + '|0:0\n'

    def stop_at_poll(seconds):
        raise RuntimeError('poll boundary')

    monkeypatch.setattr(subprocess, 'check_output', query)
    monkeypatch.setattr(time, 'sleep', stop_at_poll)
    with pytest.raises(RuntimeError, match='poll boundary'):
        cp.drive_measurements(tmp_path)
    ledger = json.loads((tmp_path / 'driver-jobs.json').read_text())
    assert {job['job_id'] for job in ledger} == {'111', '222'}
    assert len(ledger) == 2


def test_measurement_schedule_does_not_run_the_tuning_grid(counted_fixture):
    from oxyformer.validation import cost_probe as cp
    work = cp.pinned_work(seed_count=3, grid_points=4)
    schedule = cp.measurement_tasks(work)
    tasks = work['tasks_per_outer_seed']
    assert [tasks[position] for position, _, _ in schedule] == [
        ('ssl', 0, None), ('outcome', 0, 0), ('origin', 0, 0),
        ('ssl', 3, None), ('outcome', 3, None), ('origin', 3, None)]
    assert len(schedule) == 6
    # Measurement shortcuts must not shrink the repetition's counted workload.
    assert work['counts']['inner_outcome'] == 180
    assert work['counts']['inner_origin'] == 180
    assert schedule[-1][1] == schedule[-2][1] == schedule[-3][0]


def test_fixed_grid_refits_use_real_outer_ssl_without_inner_selection(tmp_path, monkeypatch, counted_fixture):
    from pathlib import Path
    import time
    import torch
    from test_nested_cv import endpoint, state
    from oxyformer.training.fit import FoldArtifacts, RecipeStudy
    from oxyformer.validation import cost_probe as cp

    prepared = endpoint(tmp_path / 'data')
    configure = type(prepared).configuration

    def cpu_configuration(self, *args, **kwargs):
        return configure(self, *args, **{**kwargs, 'device': 'cpu'})

    monkeypatch.setattr(type(prepared), 'configuration', cpu_configuration)
    monkeypatch.setattr(cp, 'require_compute_node', lambda **kwargs: None)
    monkeypatch.setattr(cp, 'clean_identity', lambda **kwargs: 'clean-head')
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setenv('SLURM_JOB_ID', 'first-attempt')
    recipe = {'nested_cv': {'ssl_epochs': 1, 'frozen_epochs': 2, 'synthetic': True}}
    work = cp.pinned_work(seed_count=3, grid_points=4)
    schedule = cp.measurement_tasks(work)
    records = {}
    # Inner SSL and origin exercise skipping the other inner grid points.
    # Outer SSL starts independently of any inner selection artifacts.
    for position, source, fixed in (schedule[0], schedule[2], schedule[3], schedule[4]):
        predecessor = None if source is None else FoldArtifacts.from_json(records[source]['predecessor'])
        record = cp.measure_fit_task(prepared, recipe, RecipeStudy(), tmp_path / str(position),
            position=position, predecessor=predecessor, standalone=fixed)
        assert record['slices'][-1]['complete']
        records[position] = record
        controller = state(FoldArtifacts.from_json(record['predecessor']))
        if fixed:
            assert not record['timing_setup']['selection_performed']
            if position != schedule[3][0]:
                assert record['timing_setup']['grid_index'] == 0
            else:
                assert 'learning_rate' not in record['timing_setup']
            assert controller['selection']['outcome'] == {'grid': 0, 'epochs': 2}
        if position == schedule[4][0]:
            assert not controller['results']
            assert controller['final']['outcome']['learning_rate'] == .0003
            assert controller['final']['outcome']['dropout'] == 0.
            assert controller['active'] is None
    # Resume the last outer-origin task after one durable epoch. Its completion
    # must stop before calibration-transfer code, which needs real tuning OOFs.
    position, source, fixed = schedule[-1]
    predecessor = FoldArtifacts.from_json(records[source]['predecessor'])
    original_clock = time.monotonic
    offset = [0.]
    monkeypatch.setattr(time, 'monotonic', lambda: original_clock() + offset[0])
    original_write = cp.atomic_record

    def stop_after_epoch(path, record):
        original_write(path, record)
        if record['attempts'] == ['first-attempt'] and len(record['slices']) == 1:
            offset[0] = 10000.

    monkeypatch.setattr(cp, 'atomic_record', stop_after_epoch)
    first = cp.measure_fit_task(prepared, recipe, RecipeStudy(), tmp_path / str(position),
        position=position, predecessor=predecessor, standalone=True,
        deadline=time.monotonic() + 10000.)
    assert first['slices'][-1]['epochs_completed'] == 1
    monkeypatch.setenv('SLURM_JOB_ID', 'second-attempt')
    final = cp.measure_fit_task(prepared, recipe, RecipeStudy(), tmp_path / str(position),
        position=position, predecessor=predecessor, standalone=True,
        deadline=time.monotonic() + 10000.)
    assert final['slices'][-1]['complete']
    assert final['slices'][-1]['epochs_completed'] == 2
    artifact = FoldArtifacts.from_json(final['predecessor'])
    controller = state(artifact)
    assert controller['position'] == len(work['tasks_per_outer_seed'])
    assert not artifact.complete
    assert not controller['results'] and controller['calibration'] is None
    assert 'transfer_diagnostics' not in controller
    assert set(controller['final']) == {'origin'}
    assert controller['final']['origin']['learning_rate'] == .0003
    assert controller['final']['origin']['dropout'] == 0.


def test_exhausted_inner_fit_does_not_block_independent_refit_measurements(tmp_path, counted_fixture):
    from oxyformer.validation import cost_probe as cp
    work = cp.pinned_work(seed_count=1, grid_points=2)
    plan = cp.measurement_tasks(work)
    cp.atomic_record(tmp_path / 'scoring.json', {})
    cp.atomic_record(tmp_path / str(plan[0][0]) / 'progress.json', fit_record())
    cp.atomic_record(tmp_path / str(plan[1][0]) / 'progress.json', fit_record(epochs=2, complete=False))
    jobs = [{'operation': str(plan[1][0])} for _ in range(2)]
    next_job, issues = cp.next_measurement(tmp_path, work, jobs)
    assert next_job == (str(plan[2][0]), tmp_path / str(plan[0][0]) / 'progress.json', True)
    assert [item['operation'] for item in issues] == [str(plan[1][0])]
    cp.atomic_record(tmp_path / str(plan[2][0]) / 'progress.json', fit_record())
    next_job, issues = cp.next_measurement(tmp_path, work, jobs)
    assert next_job == (str(plan[3][0]), None, True)


def test_failed_inner_ssl_still_allows_independent_outer_ssl(tmp_path, counted_fixture):
    from oxyformer.validation import cost_probe as cp
    work = cp.pinned_work(seed_count=1, grid_points=2)
    plan = cp.measurement_tasks(work)
    cp.atomic_record(tmp_path / 'scoring.json', {})
    jobs = [{'operation': str(plan[0][0])} for _ in range(2)]
    next_job, issues = cp.next_measurement(tmp_path, work, jobs)
    assert next_job == (str(plan[3][0]), None, True)
    assert [item['operation'] for item in issues] == [str(plan[0][0])]


def test_full_epoch_count_still_requires_completed_fit_evaluation():
    from oxyformer.validation.cost_probe import fit_quantity
    record = fit_record(epochs=150, complete=False)
    record['attempts'] = ['first', 'second']
    assert fit_quantity(record, 150) is None


def test_partial_ssl_history_reports_observed_epochs_without_invented_boundaries():
    from oxyformer.validation.cost_probe import fit_quantity, fit_statistics
    record = fit_record(epochs=0, complete=False)
    record['attempts'] = ['first', 'second']
    original = record['slices'][0]
    record['slices'] = [{**original, 'epochs_completed': epoch,
                         'minibatches_partial_epoch': 1} for epoch in (0, 1, 2)]
    assert fit_statistics(record)['epochs_completed'] == 2
    assert fit_statistics(record)['epochs_at_last_completed_boundary'] == 0
    assert fit_quantity(record, 30) is None


def test_scoring_checks_pinned_source_before_computation(tmp_path, monkeypatch, cpu_measurement):
    from oxyformer.training.fit import RecipeStudy
    from oxyformer.validation import cost_probe as cp, coverage
    monkeypatch.setattr(cp, 'require_compute_node', lambda **kwargs: None)
    monkeypatch.setattr(cp, 'clean_identity', lambda: 'clean-descendant')

    def changed_source(**kwargs):
        raise ValueError('protected source differs from counted-work tree')

    monkeypatch.setattr(cp, 'pinned_work', changed_source)
    monkeypatch.setattr(coverage, 'estimate_repetition', lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match='protected source differs'):
        cp.measure_scoring(cpu_measurement, {}, RecipeStudy(), tmp_path / 'score')


def test_driver_refuses_missing_owner_decision_before_submission(tmp_path, monkeypatch):
    import subprocess
    import yaml
    from oxyformer.validation import cost_probe as cp
    cp.atomic_record(tmp_path / 'inputs.json', {'work': {}})
    monkeypatch.setattr(cp, 'require_compute_node', lambda: None)
    monkeypatch.setattr(cp, 'clean_identity', lambda: 'clean-descendant')
    monkeypatch.setattr(yaml, 'safe_load', lambda text: {'owner_decisions': {}})
    def submission(command, **kwargs):
        pytest.fail('submitted without owner_decisions.month2_followon')
    monkeypatch.setattr(subprocess, 'check_output', submission)
    with pytest.raises(ValueError, match='month2_followon'):
        cp.drive_measurements(tmp_path)


def test_report_uses_distinct_inner_and_outer_ssl_durations(tmp_path, monkeypatch, counted_fixture):
    from oxyformer.validation import cost_probe as cp
    monkeypatch.setattr(cp, 'lifecycle_identity', lambda root: (__import__('json').loads(
        (root / 'inputs.json').read_text()), 'synthetic-endpoint'))
    work = {c: cp.pinned_work(seed_count=s.seed_count, grid_points=s.grid_points)
            for c, s in cp.study_cells().items()}
    cp.atomic_record(tmp_path / 'inputs.json', {'tree': BASE_TREE, 'work': work,
        'profile_path': 'sealed', 'frame_path': 'frame', 'frame_sha256': 'same'})
    cp.atomic_record(tmp_path / 'driver-failures.json', {})
    cp.atomic_record(tmp_path / 'driver-jobs.json', [])
    for cell in cp.study_cells():
        for position, (kind, fold, grid) in enumerate(work[cell]['tasks_per_outer_seed']):
            if kind == 'select':
                continue
            record = fit_record()
            seconds = (1200. if fold == 3 else 600.) if kind == 'ssl' else 0.
            record['slices'][0].update(compute_seconds=seconds, wall_seconds=seconds)
            cp.atomic_record(tmp_path / cell / str(position) / 'progress.json', record)
        cp.atomic_record(tmp_path / cell / 'scoring.json', {'tree': BASE_TREE, 'wall_seconds': 0.})
    monkeypatch.setattr(cp, 'calibration_receipt', lambda *args: {
        'compute_hours': measured(15000. / 3600), 'study': cp.cell_values(cp.study_cells()['e']),
        'frame_hash': 'same', 'frame_matches': True, 'sources_match': True, 'work': work['e']})
    report = cp.measurement_report(tmp_path, {'ratio': measured(1.).to_dict()})
    assert report['cells']['e']['compute_h100_hours_per_repetition']['value'] == pytest.approx(15000. / 3600)
    assert report['formula']['e']['counts']['inner_ssl']['value'] == 15
    assert report['formula']['e']['counts']['refit_ssl']['value'] == 5


def test_publish_report_creates_scheduler_evidence_and_preserves_cpu_jobs(tmp_path, monkeypatch):
    import json
    from oxyformer.validation import cost_probe as cp
    root = tmp_path / 'measurements'
    cp.atomic_record(root / 'driver-jobs.json', [{'job_id': '123', 'cell': 'e',
        'operation': '16', 'partition': 'standard', 'node': 'compute',
        'elapsed': '01:02:03', 'state': 'COMPLETED', 'exit_code': '0:0'}])
    cp.atomic_record(tmp_path / 'slurm_jobs.json', [
        {'job_id': '100', 'sub_measurement': 'CPU tests'},
        {'job_id': '123', 'wall_seconds': measured(1.).to_dict()}])
    report = {'calibration': {'passed': True, 'ratio': measured(1.).to_dict(), 'reasons': []},
        'tree': BASE_TREE, 'cells': {}, 'formula': {}, 'measurements': {},
        'feasibility': {}, 'limitations': []}
    monkeypatch.setattr(cp, 'measurement_report', lambda *args: report)
    monkeypatch.setattr(cp, 'require_compute_node', lambda: None)
    monkeypatch.setattr(cp, 'clean_identity', lambda: 'clean')
    cp.publish_report(root, {k: measured(1.).to_dict() for k in ('ratio', 'allocated_hours', 'compute_hours')})
    jobs = json.loads((tmp_path / 'slurm_jobs.json').read_text())
    assert {j['job_id'] for j in jobs} == {'100', '123'}
    gpu = next(j for j in jobs if j['job_id'] == '123')
    assert gpu['node'] == 'compute' and gpu['elapsed'] == '01:02:03'
    assert gpu['wall_seconds'] == measured(3723.).to_dict()
    assert gpu['sub_measurement'] == 'cell e, controller operation 16'
    assert gpu['basis'] == ['measured']


@pytest.fixture
def cpu_measurement(tmp_path, monkeypatch, counted_fixture):
    import torch
    from test_nested_cv import endpoint
    from oxyformer.validation import cost_probe as cp
    prepared = endpoint(tmp_path / 'data')
    configure = type(prepared).configuration
    monkeypatch.setattr(type(prepared), 'configuration',
        lambda self, *args, **kwargs: configure(self, *args, **{**kwargs, 'device': 'cpu'}))
    monkeypatch.setattr(cp, 'require_compute_node', lambda **kwargs: None)
    monkeypatch.setattr(cp, 'clean_identity', lambda **kwargs: 'same-clean-head')
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setenv('SLURM_JOB_ID', 'first')
    return prepared


def test_ssl_resume_finishes_the_partial_epoch_before_another(tmp_path, monkeypatch, cpu_measurement):
    from dataclasses import replace
    import time
    from oxyformer.training import nested_cv
    from oxyformer.training.fit import RecipeStudy
    from oxyformer.validation import cost_probe as cp
    real_run = nested_cv.run_fold
    calls = []

    def interrupted(config, *args, **kwargs):
        calls.append(config.max_batches)
        if len(calls) == 1:
            config = replace(config, max_batches=1)
        return real_run(config, *args, **kwargs)

    monkeypatch.setattr(nested_cv, 'run_fold', interrupted)
    clock = time.monotonic
    offset = [0.]
    monkeypatch.setattr(time, 'monotonic', lambda: clock() + offset[0])
    write = cp.atomic_record

    def end_attempt(path, record):
        write(path, record)
        if len(record['slices']) in (1, 3):
            offset[0] += 10000.

    monkeypatch.setattr(cp, 'atomic_record', end_attempt)
    recipe = {'nested_cv': {'ssl_epochs': 5, 'frozen_epochs': 2, 'synthetic': True}}
    root = tmp_path / 'ssl-measurement'
    first = cp.measure_fit_task(cpu_measurement, recipe, RecipeStudy(), root,
        position=0, deadline=time.monotonic() + 10000.)
    assert first['slices'][-1]['minibatches_partial_epoch'] == 1
    monkeypatch.setenv('SLURM_JOB_ID', 'second')
    # The attempt-registration write also sees one prior slice; stop only when
    # a new checkpoint is written during this second attempt.
    def end_second(path, record):
        write(path, record)
        if len(record['slices']) == 3:
            offset[0] += 10000.
    monkeypatch.setattr(cp, 'atomic_record', end_second)
    second = cp.measure_fit_task(cpu_measurement, recipe, RecipeStudy(), root,
        position=0, deadline=time.monotonic() + 10000.)
    assert second['slices'][-1]['epochs_completed'] == 2
    assert second['slices'][-1]['minibatches_partial_epoch'] == 0
    assert cp.fit_statistics(second)['epochs_completed'] == 2
    assert cp.fit_quantity(second, 5).basis == {'extrapolated'}


def test_native_nuisance_completion_records_observed_epochs(tmp_path, monkeypatch, cpu_measurement):
    from oxyformer.training import nested_cv
    from oxyformer.training.fit import FoldArtifacts, RecipeStudy
    from oxyformer.validation import cost_probe as cp
    recipe = {'nested_cv': {'ssl_epochs': 1, 'frozen_epochs': 3, 'synthetic': True}}
    ssl = cp.measure_fit_task(cpu_measurement, recipe, RecipeStudy(), tmp_path / 'ssl', position=0)
    transaction = nested_cv._train_transaction

    def native_finish(*args, **kwargs):
        # Exercise the native completion interface with a real one-epoch fit.
        # The registered study has no nuisance stopping partition; this fixture
        # checks the returned progress instead of inventing an epoch count.
        args = list(args)
        args[5] = 1
        return transaction(*args, **kwargs)

    monkeypatch.setattr(nested_cv, '_train_transaction', native_finish)
    result = cp.measure_fit_task(cpu_measurement, recipe, RecipeStudy(), tmp_path / 'inner',
        position=1, predecessor=FoldArtifacts.from_json(ssl['predecessor']))
    assert result['slices'][-1]['complete']
    assert result['slices'][-1]['epochs_completed'] == 1


@pytest.mark.parametrize('changed', ['training/fit.py', 'training/pretrain.py'])
def test_pinned_work_rejects_fit_and_ssl_source_drift(monkeypatch, changed):
    from pathlib import Path
    import subprocess
    from oxyformer.validation import cost_probe as cp
    paths = ['src/oxyformer/' + path for path in (
        'training/nested_cv.py', 'validation/coverage.py',
        'training/fit.py', 'training/pretrain.py')]
    sources = {path: Path(path).read_bytes() for path in paths}
    monkeypatch.setattr(subprocess, 'check_output', lambda command: sources[command[-1].split(':', 1)[1]])
    read_bytes = Path.read_bytes
    def runtime(path):
        raw = read_bytes(path)
        return raw + b'\n# different fitting implementation\n' if str(path) == 'src/oxyformer/' + changed else raw
    monkeypatch.setattr(Path, 'read_bytes', runtime)
    with pytest.raises(ValueError, match='protected source differs'):
        cp.pinned_work(seed_count=1, grid_points=2)


@pytest.fixture
def synthetic_profile(tmp_path):
    import json
    from oxyformer.contracts import StageRequest
    from oxyformer.provenance import file_hash
    from test_coverage import synthetic_endpoint, smoke_recipe
    prepared, frame = synthetic_endpoint(tmp_path / 'data')
    profile = tmp_path / 'profile'
    execution = profile / '_execution'
    execution.mkdir(parents=True)
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    frame_path, endpoint_path = inputs / 'frame.json', inputs / 'endpoint.json'
    frame_path.write_text(frame.to_json())
    endpoint_path.write_text(prepared.to_json())
    config, task_path = execution / 'config.json', execution / 'task.json'
    config.write_text('{}')
    task_path.write_text(json.dumps({'parameters': {
        'scenario': {'name': 'null_effect', 'active_mechanisms': ['null'], 'effect': 'null'},
        'draws': [{'seed': 1103}], 'recipe': smoke_recipe(prepared, frame)}}))
    request = StageRequest(stage='synthetic-profile', config_path=str(config), config_hash=file_hash(config),
        task_path=str(task_path), task_hash=file_hash(task_path),
        dependency_paths=(str(frame_path), str(endpoint_path)),
        dependency_hashes=(file_hash(frame_path), file_hash(endpoint_path)),
        output_dir=str(profile), code_identity=BASE_TREE)
    (execution / 'request.json').write_text(request.to_json())
    (profile / 'budget_record.json').write_text('{}')
    return profile, request


@pytest.mark.parametrize('changed', ['training/fit.py', 'training/pretrain.py'])
def test_calibration_receipt_detects_changed_fitting_source(synthetic_profile, monkeypatch, changed):
    from pathlib import Path
    import subprocess
    from oxyformer.validation import cost_probe as cp, coverage
    profile, request = synthetic_profile
    receipt_tree = 'b' * 40
    record = {'binding': {'code_identity': receipt_tree, 'study': cp.cell_values(cp.study_cells()['e'])},
              'profile': {'complete_repetition_seconds': [3600.]}}
    monkeypatch.setattr(coverage, 'read_leaf_measurements', lambda path: record)
    paths = ['src/oxyformer/' + path for path in (
        'training/nested_cv.py', 'validation/coverage.py',
        'training/fit.py', 'training/pretrain.py')]
    sources = {path: Path(path).read_bytes() for path in paths}
    def tree_source(command):
        tree, path = command[-1].split(':', 1)
        raw = sources[path]
        return raw + b'\n# different fitting implementation\n' if tree == receipt_tree and path == 'src/oxyformer/' + changed else raw
    monkeypatch.setattr(subprocess, 'check_output', tree_source)
    receipt = cp.calibration_receipt(profile / 'budget_record.json', request.dependency_paths[0])
    assert receipt.get('sources_match', True) is False


def test_scoring_uses_only_registered_targets_with_extra_frame_rows(tmp_path, monkeypatch, counted_fixture):
    import torch
    from oxyformer.contracts import OOFNuisances
    from oxyformer.training.fit import RecipeStudy
    from oxyformer.validation import cost_probe as cp, coverage
    from test_coverage import role_endpoint
    endpoint, _ = role_endpoint(tmp_path / 'data')
    assert set(endpoint.outer.target_ids) < set(endpoint.outer.frame_original_ids)
    assert endpoint.outer.target_ids == endpoint.outer.original_ids
    monkeypatch.setattr(cp, 'require_compute_node', lambda **kwargs: None)
    monkeypatch.setattr(cp, 'clean_identity', lambda: 'clean-head')
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    observed = []
    def score(*args, continuation, **kwargs):
        for value in continuation['parts']:
            part = OOFNuisances.from_json(value)
            observed.extend(zip(part.original_ids, part.seed_ids))
        return {}
    monkeypatch.setattr(coverage, 'estimate_repetition', score)
    study = RecipeStudy()
    result = cp.measure_scoring(endpoint, {}, study, tmp_path / 'score')
    expected = {(i, seed) for i in endpoint.outer.target_ids for seed in study.seeds}
    assert len(observed) == len(expected) == result['prediction_rows']
    assert set(observed) == expected


def test_relative_sealed_dependencies_are_refused_before_preparation(synthetic_profile, monkeypatch, tmp_path):
    import json
    from oxyformer.provenance import ContractError
    from oxyformer.validation import cost_probe as cp
    profile, request = synthetic_profile
    payload = json.loads(request.to_json())
    payload['payload']['dependency_paths'] = ['frame.json', 'endpoint.json']
    (profile / '_execution/request.json').write_text(json.dumps(payload))
    monkeypatch.setattr(cp, 'require_compute_node', lambda: None)
    monkeypatch.setattr(cp, 'clean_identity', lambda: 'clean-head')
    def receipt(*args):
        pytest.fail('read receipt after an invalid request')
    monkeypatch.setattr(cp, 'calibration_receipt', receipt)
    destination = tmp_path / 'measurement'
    with pytest.raises(ContractError, match='request paths must be absolute'):
        cp.prepare_inputs(profile, destination)
    assert not destination.exists()


def test_prepared_paths_survive_a_different_runtime_checkout(synthetic_profile, tmp_path, monkeypatch, counted_fixture):
    from pathlib import Path
    from oxyformer.provenance import file_hash
    from oxyformer.training.nested_cv import PreparedEndpoint
    from oxyformer.validation import cost_probe as cp
    profile, request = synthetic_profile
    monkeypatch.setattr(cp, 'require_compute_node', lambda: None)
    monkeypatch.setattr(cp, 'clean_identity', lambda: 'clean-head')
    monkeypatch.setattr(cp, 'calibration_receipt', lambda *args: {'compute_hours': measured(1.)})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cp, 'require_month2_decision', lambda: None)
    metadata = cp.prepare_inputs(Path('profile'), Path('measurements'))
    runtime = tmp_path / 'frozen' / 'runtime'
    runtime.mkdir(parents=True)
    monkeypatch.chdir(runtime)
    for name in ('endpoint_path', 'frame_path', 'profile_path'):
        assert Path(metadata[name]).is_absolute(), name
        assert Path(metadata[name]).exists(), name
    endpoint_path = Path(metadata['endpoint_path'])
    assert file_hash(endpoint_path) == metadata['endpoint_sha256']
    assert PreparedEndpoint.from_json(endpoint_path.read_text()).outer.target_ids


def test_driver_makes_relative_measurement_root_absolute(tmp_path, monkeypatch):
    from pathlib import Path
    from oxyformer.validation import cost_probe as cp
    monkeypatch.setattr(cp, 'lifecycle_identity', lambda root: (__import__('json').loads(
        (root / 'inputs.json').read_text()), 'synthetic-endpoint'))
    monkeypatch.setattr(cp, 'require_compute_node', lambda: None)
    monkeypatch.setattr(cp, 'clean_identity', lambda **kwargs: 'clean-head')
    monkeypatch.setattr(cp, 'require_month2_decision', lambda: None)
    root = tmp_path / 'measurements'
    cp.atomic_record(root / 'inputs.json', {'work': {}})
    monkeypatch.chdir(tmp_path)
    def submit(ledger_path, ledger, **kwargs):
        assert kwargs['root'].is_absolute()
        assert kwargs['checkout'] == tmp_path / 'frozen'
        assert kwargs['command'][2].endswith('src/oxyformer/validation/cost_probe.py')
        assert '-m' not in kwargs['command']
        assert Path(kwargs['command'][kwargs['command'].index('--root') + 1]) == root
        raise RuntimeError('submission inspected')
    monkeypatch.setattr(cp, 'submit_measurement', submit)
    with pytest.raises(RuntimeError, match='submission inspected'):
        cp.drive_measurements(Path('measurements'), fit_checkout=tmp_path / 'frozen')


@pytest.fixture
def lifecycle_root(synthetic_profile, tmp_path, monkeypatch, counted_fixture):
    """Real preparation, fit controller and scoring; synthetic CPU boundaries."""
    import json
    import torch
    from pathlib import Path
    from oxyformer.training.fit import RecipeStudy, FoldArtifacts
    from oxyformer.training.nested_cv import PreparedEndpoint
    from oxyformer.validation import cost_probe as cp
    profile, _ = synthetic_profile
    monkeypatch.setattr(cp, 'require_compute_node', lambda **kwargs: None)
    monkeypatch.setattr(cp, 'clean_identity', lambda **kwargs: 'fixture-head')
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    configure = PreparedEndpoint.configuration
    monkeypatch.setattr(PreparedEndpoint, 'configuration',
        lambda self, *args, **kwargs: configure(self, *args, **{**kwargs, 'device': 'cpu'}))
    study = RecipeStudy(reference_cap=32, grid_points=2, seed_count=1)
    monkeypatch.setattr(cp, 'study_cells', lambda: {'e': study})
    work = cp.pinned_work(seed_count=1, grid_points=2)
    monkeypatch.setattr(cp, 'calibration_receipt', lambda *args: {
        'compute_hours': measured(1.), 'study': cp.cell_values(study),
        'frame_hash': cp_file_hash(), 'frame_matches': True, 'sources_match': True, 'work': work})
    from oxyformer.provenance import file_hash
    def cp_file_hash():
        return file_hash(tmp_path / 'inputs/frame.json')
    root = tmp_path / 'measurements'
    metadata = cp.prepare_inputs(profile, root)
    endpoint = PreparedEndpoint.from_json(Path(metadata['endpoint_path']).read_text())
    records, jobs = {}, []
    for position, source, standalone in cp.measurement_tasks(work):
        job = 'fit-' + str(position)
        monkeypatch.setenv('SLURM_JOB_ID', job)
        previous = None if source is None else FoldArtifacts.from_json(records[source]['predecessor'])
        records[position] = cp.measure_fit_task(endpoint, metadata['recipe'], study,
            root / 'e' / str(position), position=position, predecessor=previous, standalone=standalone)
        assert records[position]['slices'][-1]['complete']
        jobs.append({'job_id': job, 'cell': 'e', 'operation': str(position), 'state': 'COMPLETED'})
    cp.atomic_record(root / 'e/scoring.json', cp.measure_scoring(endpoint, metadata['recipe'], study, root / 'score-work'))
    cp.atomic_record(root / 'driver-jobs.json', jobs)
    cp.atomic_record(root / 'driver-failures.json', {})
    return root, profile, metadata


def test_produced_cell_lifecycle_and_idempotent_preparation(lifecycle_root, monkeypatch):
    from oxyformer.validation import cost_probe as cp
    root, profile, metadata = lifecycle_root
    report = cp.measurement_report(root, {'ratio': measured(1.).to_dict()})
    assert report['cells']['e']['compute_h100_hours_per_repetition'] is not None
    snapshot = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(cp, 'clean_identity', lambda **kwargs: 'later-harness-head')
    assert cp.prepare_inputs(profile, root) == metadata
    assert snapshot == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in snapshot}
    # Change the draw seed through the ordinary sealed-request interface.
    import json
    from dataclasses import replace
    from oxyformer.contracts import StageRequest
    from oxyformer.provenance import file_hash
    task_path = profile / '_execution/task.json'
    task = json.loads(task_path.read_text())
    task['parameters']['draws'][0]['seed'] += 1
    task_path.write_text(json.dumps(task))
    request_path = profile / '_execution/request.json'
    request = StageRequest.from_json(request_path.read_text())
    request_path.write_text(replace(request, task_hash=file_hash(task_path)).to_json())
    with pytest.raises(ValueError, match='preparation changed endpoint_content_hash'):
        cp.prepare_inputs(profile, root)
    assert snapshot == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in snapshot}


@pytest.mark.parametrize('field', ['endpoint', 'tree', 'study', 'position', 'recipe'])
def test_stale_artifacts_refused_before_driver_or_report(lifecycle_root, monkeypatch, field):
    import json
    from oxyformer.validation import cost_probe as cp
    root, _, _ = lifecycle_root
    path = root / 'e/0/progress.json'
    record = json.loads(path.read_text())
    record['binding'][field] = 'old-preparation'
    cp.atomic_record(path, record)
    monkeypatch.setattr(cp, 'submit_measurement', lambda *a, **k: pytest.fail('submitted stale root'))
    snapshot = path.read_bytes()
    for consume in (lambda: cp.drive_measurements(root),
                    lambda: cp.measurement_report(root, {'ratio': measured(1.).to_dict()})):
        with pytest.raises(ValueError, match='lifecycle identity: ' + field + ' mismatch'):
            consume()
    assert path.read_bytes() == snapshot


def test_legacy_scoring_is_preserved_but_excluded(lifecycle_root):
    import json
    from oxyformer.validation import cost_probe as cp
    root, _, _ = lifecycle_root
    path = root / 'e/scoring.json'
    record = json.loads(path.read_text())
    del record['binding']
    cp.atomic_record(path, record)
    snapshot = path.read_bytes()
    for consume in (lambda: cp.drive_measurements(root),
                    lambda: cp.measurement_report(root, {'ratio': measured(1.).to_dict()})):
        with pytest.raises(ValueError, match='legacy scoring excluded: endpoint identity unverifiable'):
            consume()
    assert path.read_bytes() == snapshot


def test_failed_term_summary_basis_and_null_projection(lifecycle_root):
    import json
    from oxyformer.validation import cost_probe as cp
    root, _, _ = lifecycle_root
    path = root / 'e/0/progress.json'
    record = json.loads(path.read_text())
    record.update(slices=[], attempts=['first', 'second'])
    cp.atomic_record(path, record)
    cp.atomic_record(root / 'driver-jobs.json', [
        {'job_id': job, 'cell': 'e', 'operation': '0', 'state': 'TIMEOUT'} for job in record['attempts']])
    cp.atomic_record(root / 'driver-failures.json', {'e': {'operation': '0', 'reason': 'zero epochs'}})
    report = cp.measurement_report(root, {'ratio': measured(1.).to_dict()})
    term = report['measurements']['e']['inner_ssl']
    assert term['status'] == 'failed_measurement'
    assert term['duration_seconds'] is None
    assert report['cells']['e']['compute_h100_hours_per_repetition'] is None
    for value in term['summary'].values():
        if isinstance(value, dict):
            assert value['basis'] == ['failed_measurement']


def test_source_identical_alternate_history_is_accepted(tmp_path, monkeypatch):
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    from oxyformer.validation import cost_probe as cp
    REPO = Path.cwd()
    # HEAD is available even in CI's shallow clone; protected bytes are the fixture.
    baseline = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    monkeypatch.setattr(cp, 'BASE_TREE', baseline)
    alternate = tmp_path / 'alternate'
    alternate.mkdir()
    subprocess.run(['git', 'init', str(alternate)], check=True, capture_output=True)
    objects = Path(subprocess.check_output(['git', 'rev-parse', '--git-path', 'objects'], text=True).strip()).resolve()
    (alternate / '.git/objects/info/alternates').write_text(str(objects) + '\n')
    for path in [p for _, p in cp.PROTECTED_SOURCES] + ['src/oxyformer/validation/cost_probe.py']:
        destination = alternate / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / path, destination)
    subprocess.run(['git', 'add', 'src'], cwd=alternate, check=True)
    tree = subprocess.check_output(['git', 'write-tree'], cwd=alternate, text=True).strip()
    environment = {**os.environ, 'GIT_AUTHOR_NAME': 'Synthetic fixture',
        'GIT_AUTHOR_EMAIL': 'fixture@example.invalid', 'GIT_COMMITTER_NAME': 'Synthetic fixture',
        'GIT_COMMITTER_EMAIL': 'fixture@example.invalid'}
    head = subprocess.check_output(['git', 'commit-tree', tree], cwd=alternate,
        env=environment, input='Synthetic unrelated history\n', text=True).strip()
    subprocess.run(['git', 'update-ref', '--no-deref', 'HEAD', head], cwd=alternate, check=True)
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=alternate)
    for _, path in cp.PROTECTED_SOURCES:
        assert (alternate / path).read_bytes() == subprocess.check_output(
            ['git', 'show', cp.BASE_TREE + ':' + path], cwd=alternate)
    assert cp.clean_identity(cwd=alternate) == head
    monkeypatch.chdir(alternate)
    assert cp.pinned_work(seed_count=1, grid_points=2, tree=baseline)['tree'] == baseline


def test_separate_harness_imports_the_fitting_checkout(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    from oxyformer.validation import cost_probe as cp
    REPO = Path.cwd()
    fit_root, harness_root = tmp_path / 'fit-tree', tmp_path / 'harness-tree'
    for destination in (fit_root, harness_root):
        shutil.copytree(REPO / 'src', destination / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shadow = harness_root / 'src/oxyformer/training/fit.py'
    with shadow.open('a') as stream:
        stream.write("\nraise RuntimeError('HARNESS_FIT_IMPORTED')\n")
    script = harness_root / 'src/oxyformer/validation/cost_probe.py'
    text = script.read_text().replace('from __future__ import annotations',
        "from __future__ import annotations\nimport sys\nprint('PROBE_SCRIPT_PATH0=' + sys.path[0])", 1)
    script.write_text(text)
    environment = {**os.environ, 'PYTHONPATH': 'src', 'CUDA_VISIBLE_DEVICES': ''}
    result = subprocess.run([sys.executable, '-B', '-v', str(script), '--help'],
        cwd=fit_root, env=environment, text=True, capture_output=True, timeout=120)
    actual = next(line.split('=', 1)[1] for line in result.stdout.splitlines()
                  if line.startswith('PROBE_SCRIPT_PATH0='))
    expected_runtime = fit_root / 'src/oxyformer/training/fit.py'
    assert result.returncode == 0, result.stderr[-4000:]
    assert actual == str(script.parent)
    assert str(expected_runtime) in result.stderr
    assert str(shadow) not in result.stderr
    assert 'HARNESS_FIT_IMPORTED' not in result.stderr


def test_previous_root_artifacts_refused_after_endpoint_replacement(lifecycle_root, monkeypatch):
    import json
    from dataclasses import replace
    from pathlib import Path
    from oxyformer.provenance import file_hash
    from oxyformer.training.nested_cv import PreparedEndpoint
    from oxyformer.validation import cost_probe as cp
    root, _, metadata = lifecycle_root
    path = Path(metadata['endpoint_path'])
    endpoint = PreparedEndpoint.from_json(path.read_text())
    # Reproduce the old preparer's overwrite while keeping the old artifacts.
    field = endpoint.data.manifest.outcome_field
    columns = [c.name for c in endpoint.data.manifest.schema]
    rows = tuple(tuple(value + 1. if name == field and value is not None else value
                       for name, value in zip(columns, row)) for row in endpoint.data.rows)
    changed = replace(endpoint, data=replace(endpoint.data, rows=rows))
    assert changed.content_hash != endpoint.content_hash
    path.write_text(changed.to_json())
    metadata.update(endpoint_sha256=file_hash(path), endpoint_content_hash=changed.content_hash)
    cp.atomic_record(root / 'inputs.json', metadata)
    monkeypatch.setattr(cp, 'submit_measurement', lambda *a, **k: pytest.fail('stale submission'))
    for consume in (lambda: cp.drive_measurements(root),
                    lambda: cp.measurement_report(root, {'ratio': measured(1.).to_dict()})):
        with pytest.raises(ValueError, match='lifecycle identity: endpoint mismatch'):
            consume()


def test_legacy_work_provenance_is_checked_without_invalidating_good_fits(lifecycle_root, monkeypatch):
    from oxyformer.validation import cost_probe as cp
    root, _, metadata = lifecycle_root
    original = cp.pinned_work
    paths = [path for _, path in cp.PROTECTED_SOURCES]
    declared = {path: 'fixture-' + str(i) for i, path in enumerate(paths)}
    monkeypatch.setattr(cp, 'pinned_work', lambda **kwargs: {
        **original(**kwargs), 'source_sha256': declared})
    metadata['work']['e']['source_sha256'] = {path: declared[path] for path in paths[:2]}
    cp.atomic_record(root / 'inputs.json', metadata)
    assert cp.lifecycle_identity(root)[0] == metadata
    metadata['work']['e']['source_sha256'][paths[0]] = 'changed-source'
    cp.atomic_record(root / 'inputs.json', metadata)
    with pytest.raises(ValueError, match='source digest mismatch'):
        cp.lifecycle_identity(root)
