"""Population accounting shared by exposure construction and collection."""
import math
from fractions import Fraction
import numpy as np
from oxyformer.provenance import require

QUANTILE_INTERPRETATION = 'placement-dependent; not observed habitation'
REASONS = ('nodata', 'outside_coverage', 'outside_physical_domain')


def weighted_quantiles(values, weights):
    """Left inverse CDF of the represented positive placement masses.

    Binary floating-point weights are exact dyadic rationals. Compare their
    integer numerators at a common denominator, so cumulative rounding cannot
    change a boundary and no tolerance can erase a genuine near-boundary mass.
    """
    positive = weights > 0
    values, weights = values[positive], weights[positive]
    if not len(values):
        return [None, None, None]
    order = np.argsort(values, kind='stable')
    ratios = [float(w).as_integer_ratio() for w in weights[order]]
    denominator = max(d for _, d in ratios)  # all denominators are powers of two
    masses = [n * (denominator // d) for n, d in ratios]
    total = sum(masses)
    result, cumulative = [], 0
    numerators = (1, 5, 9)  # exact p10, p50, p90 thresholds, denominator ten
    for index, mass in zip(order, masses):
        cumulative += mass
        while len(result) < 3 and 10 * cumulative >= numerators[len(result)] * total:
            result.append(float(values[index]))
        if len(result) == 3:
            break
    return result


def placement_quantiles(blocks):
    """Exact left CDF of population * placement area / full block area.

    Each input is (integer population, full area, elevations, placement areas).
    Areas are the represented binary64 geometry, not measured habitation. Keep
    integer area prefixes and one rational scale per block; never construct a
    heterogeneous-denominator rational object for every placement. CDF total
    uses the same exact weights (area conservation is checked by the caller).
    """
    distributions = []
    for population, block_area, values, areas in blocks:
        order = np.argsort(values, kind='stable')
        values, areas = values[order], areas[order]
        require(population > 0 and math.isfinite(block_area) and block_area > 0 and
                np.isfinite(values).all() and np.isfinite(areas).all() and (areas > 0).all(),
                'invalid quantile placement inputs')
        if not len(values):
            continue
        denominator = max(float(a).as_integer_ratio()[1] for a in areas)
        prefix = [0]
        for area in areas:
            numerator, divisor = float(area).as_integer_ratio()
            prefix.append(prefix[-1] + numerator * (denominator // divisor))
        scale = Fraction(int(population)) / Fraction(float(block_area)) / denominator
        distributions.append((values, prefix, scale))
    if not distributions:
        return [None, None, None]
    candidates = np.unique(np.concatenate([d[0] for d in distributions]))
    candidates[candidates == 0] = 0.0  # canonical positive zero in output
    total = sum(prefix[-1] * scale for _, prefix, scale in distributions)
    result = []
    for numerator in (1, 5, 9):
        low, high = 0, len(candidates) - 1
        while low < high:
            mid = (low + high) // 2
            cumulative = sum(prefix[np.searchsorted(values, candidates[mid], side='right')] * scale
                             for values, prefix, scale in distributions)
            if 10 * cumulative >= numerator * total:
                high = mid
            else:
                low = mid + 1
        result.append(float(candidates[low]))
    return result


def validate_accounting(exposure, quality):
    """Every block and population unit occurs once per scenario, even if unusable."""
    seen = set()
    totals = {}
    for row in quality['blocks']:
        key = (row['block_id'], row['scenario'])
        require(key not in seen, 'overlapping block accounting')
        seen.add(key)
        require(row['block_id'][:11] == row['tract_id'], 'QC tract membership mismatch')
        require(row['population'] >= 0 and row['covered_population'] >= 0 and
                all(row[r] >= 0 for r in REASONS), 'negative population accounting')
        missing = math.fsum(row[r] for r in REASONS)
        require(math.isclose(row['population'], row['covered_population'] + missing,
                            rel_tol=1e-10, abs_tol=1e-8), 'population conservation failure')
        tract_key = (row['tract_id'], row['scenario'])
        total = totals.setdefault(tract_key, [0, 0.0, 0.0, 0])
        total[0] += row['population']
        total[1] += row['covered_population']
        total[2] += missing
        total[3] += 1
    require(not exposure.duplicated(['tract_id', 'scenario']).any(), 'overlapping tract rows')
    require(set(totals) == set(zip(exposure.tract_id, exposure.scenario)), 'tract omission accounting mismatch')
    for row in exposure.itertuples():
        total, covered, missing, count = totals[(row.tract_id, row.scenario)]
        require(row.population == total and row.block_count == count and
                math.isclose(row.covered_population, covered, rel_tol=1e-10, abs_tol=1e-8) and
                math.isclose(row.missing_population, missing, rel_tol=1e-10, abs_tol=1e-8),
                'tract/block accounting mismatch')
        should_exist = total > 0 and missing == 0
        for value in (row.pressure_mmhg, row.oxygen_deficit_mmhg, row.elevation_p10_m,
                      row.elevation_p50_m, row.elevation_p90_m):
            require(bool(np.isfinite(value)) == should_exist, 'missing coverage cannot disappear')
    scenarios = quality['allocation']['scenarios']
    block_sets = [{r['block_id'] for r in quality['blocks'] if r['scenario'] == s} for s in scenarios]
    require(block_sets and all(b == block_sets[0] for b in block_sets), 'scenario block omissions')
    require({r['scenario'] for r in quality['blocks']} == set(scenarios), 'unexpected scenario')
    require(quality['block_count'] == len(block_sets[0]), 'QC block count mismatch')
    populations = [{r['block_id']: r['population'] for r in quality['blocks'] if r['scenario'] == s}
                   for s in scenarios]
    require(all(p == populations[0] for p in populations) and
            sum(populations[0].values()) == quality['population'], 'QC total population mismatch')
