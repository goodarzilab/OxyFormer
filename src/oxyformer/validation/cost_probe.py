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
    """Count the launch-base coverage loop and nested controller task list.

    coverage.estimate_repetition: five outer folds, each selected seed, then
    one scoring pass. nested_cv._advance: three inner partitions, each with
    SSL followed by outcome/origin for each grid point, then outer SSL and
    one selected refit for each nuisance kind.
    """
    if seed_count not in (1, 3) or grid_points not in (2, 4):
        raise ValueError('unregistered study dimensions')
    outer_seeds = 5 * seed_count
    return {'ssl': outer_seeds * 4,
            'inner_outcome': outer_seeds * 3 * grid_points,
            'inner_origin': outer_seeds * 3 * grid_points,
            'refit_outcome': outer_seeds, 'refit_origin': outer_seeds,
            'scoring': 1}


def project(measurements, counts, *, tree):
    if tree != BASE_TREE or any(m['tree'] != tree for m in measurements.values()):
        raise ValueError('measurement tree mismatch')
    if set(measurements) != set(counts):
        raise ValueError('missing or extra measurement terms')
    if any(m['seconds'] is None for m in measurements.values()):
        return None
    return add(*(measurements[name]['seconds'].times(count)
                 for name, count in counts.items())).times(1 / 3600)
