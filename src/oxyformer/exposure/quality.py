"""Population accounting shared by exposure construction and collection."""
import math
import numpy as np
from oxyformer.provenance import require

QUANTILE_INTERPRETATION = 'placement-dependent; not observed habitation'
REASONS = ('nodata', 'outside_coverage', 'outside_physical_domain')


def weighted_quantiles(values, weights):
    """Inverse weighted empirical CDF, restricted to positive placement mass."""
    positive = weights > 0
    values, weights = values[positive], weights[positive]
    if not len(values):
        return [None, None, None]
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    indices = np.searchsorted(cumulative, np.array([0.1, 0.5, 0.9]) * cumulative[-1], side='left')
    return values[order][indices].tolist()


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
