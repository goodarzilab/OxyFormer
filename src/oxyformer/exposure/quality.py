"""Population accounting and placement-dependent empirical quantiles."""
import math

import numpy as np

from oxyformer.provenance import require

QUANTILE_LABEL = 'placement-dependent; not observed habitation'


def weighted_quantiles(values, weights, probabilities=(0.1, 0.5, 0.9)):
    """Inverse empirical CDF; equal values are coalesced before accumulation."""
    masses = {}
    for value, weight in zip(values, weights):
        if weight > 0:
            masses.setdefault(float(value), []).append(float(weight))
    values = sorted(masses)
    weights = [math.fsum(masses[v]) for v in values]
    require(bool(values), 'quantiles require positive population')
    cumulative = np.cumsum(weights)
    return [float(values[min(np.searchsorted(cumulative, p * cumulative[-1], side='left'),
                            len(values) - 1)]) for p in probabilities]


def check_conservation(population, covered, missing):
    require(population >= 0 and covered >= 0 and missing >= 0, 'negative population mass')
    require(math.isclose(population, covered + missing, rel_tol=1e-12, abs_tol=1e-8),
            'population not conserved')
