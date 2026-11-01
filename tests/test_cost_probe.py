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
