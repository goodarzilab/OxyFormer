"""Synthetic arithmetic checks; GPU and scheduler evidence comes from jobs."""
import pytest

from oxyformer.validation.cost_probe import (
    BASE_TREE, Quantity, calibration, epoch_projection, feasibility,
    operation_counts, project, ratio,
)


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
def test_counted_controller_work(seeds, grid):
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


def test_projection_includes_scoring_and_refit_work():
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
])
def test_calibration_identity_and_anchor_rules(field, reason):
    args = dict(measured_anchor=True, frame_matches=True, cell_matches=True, operations_match=True)
    args[field] = False
    result = calibration(measured(1), measured(1), **args)
    assert not result['passed'] and reason in result['reasons']
    assert 'uncalibrated' in result['ratio']['basis']


def test_labels_propagate_into_ratios_projections_and_table():
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


def test_measurement_harness_resumes_real_controller_across_attempts(tmp_path, monkeypatch):
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
    monkeypatch.setattr(cost_probe, 'clean_identity', lambda: 'same-clean-head')
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
