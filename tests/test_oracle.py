import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from test_scm import frame, config, policy
from oxyformer.validation.generators import generate_suite_a
from oxyformer.validation.oracle import ObservedLawOracle
from oxyformer.validation.scm import NUMERIC_DOMAIN

@pytest.mark.parametrize('delta', [0., .02, 2., 200.])
@pytest.mark.parametrize('bounds', [(-10010., -10000.), (0., 10.), (10000., 10010.)])
def test_uniform_edges(delta, bounds):
    p = policy((bounds,), delta)
    c = config(beta=.001, noise_sd=0.)
    f = replace(frame(2), weights=(NUMERIC_DOMAIN['weight'][0],)*2)
    sample = generate_suite_a(f, c, p)
    oracle = ObservedLawOracle(sample.observations, sample.structural_causal_truth, p)
    lo, hi = bounds
    for a in [lo, lo+(hi-lo)/4, hi]:
        q, r = oracle.evaluate('o0', a)
        incoming = lo <= a-delta <= hi-delta and delta <= hi-lo and delta != 0
        moved = lo <= a <= hi-delta and delta != 0
        expected = 1. if p.is_identity else float(incoming)+float(not moved)
        assert float(q) == pytest.approx(50+.001*a, abs=1e-9)
        assert float(r) == pytest.approx(expected, abs=1e-9)

@pytest.mark.parametrize('measured', [False, True])
def test_selected_mixture(measured):
    c = config(local_confounding='measured' if measured else 'omitted', exposure_error=.4,
               selected_outcome=True, migration=.2)
    p = policy()
    sample = generate_suite_a(frame(2), c, p)
    o = ObservedLawOracle(sample.observations, sample.structural_causal_truth, p)
    observed_u = dict(zip(sample.observations.measured_columns, sample.observations.measured_x[0])).get('local_confounder')
    def joint(a):
        den, num = 0., 0.
        for u in (-1., 1.):
            if measured and u != observed_u:
                continue
            for error in (-.4, .4):
                for illness, prob in ((0., .7), (1., .3)):
                    rate = .3*u-.25*illness
                    density = np.exp(rate*(a-error))*rate/np.expm1(10*rate) if 0 <= a-error <= 10 else 0.
                    mass = .25*prob*density*expit(1-.2*a-1.2*illness)
                    den += mass
                    num += mass*(50+a-error-.2*illness+2*u+2*illness)
        return den, num
    for a in (.5, 4., 9.5):
        den, num = joint(a)
        q, r = o.evaluate('o0', a)
        expected_ratio = (joint(a-2)[0] if 2 <= a <= 10 else 0.)/den + (1 if a > 8 else 0)
        assert float(q) == pytest.approx(num/den, abs=1e-9)
        assert float(r) == pytest.approx(expected_ratio, abs=1e-9)

def test_no_fitting_imports():
    tree = ast.parse(Path('src/oxyformer/validation/oracle.py').read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(any(word in name for word in ('training', 'sklearn', 'targeting', 'models')) for name in imports)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'fit' for n in ast.walk(tree))

def test_truth_binding():
    from oxyformer.provenance import ContractError
    sample = generate_suite_a(frame(2), config(), policy())
    with pytest.raises(ContractError, match='observation hash'):
        ObservedLawOracle(sample.observations, replace(sample.structural_causal_truth, observation_hash='0'*64), policy())
