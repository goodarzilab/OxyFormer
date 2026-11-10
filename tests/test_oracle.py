import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.special import expit

from test_scm import frame, config, policy
from oxyformer.validation.generators import generate_suite_a
from oxyformer.validation.oracle import ObservedLawOracle
from oxyformer.validation.scm import NUMERIC_DOMAIN

REGISTERED = [(p.stem, yaml.safe_load(p.read_text())) for p in sorted(Path('configs/policies').glob('*.yaml'))]
assert REGISTERED and all(r['policy'] == 'ShiftOrStayPolicy' and r['boundary_action'] == 'stay' for _, r in REGISTERED)

@pytest.mark.parametrize('weight', NUMERIC_DOMAIN['weight'])
@pytest.mark.parametrize('beta', [-200., 200.])
@pytest.mark.parametrize('delta,law', [(d, 'continuous') for d in (0., .02, 200.)]+[pytest.param(r['delta_mmhg'], r['exposure_law'], id=name) for name,r in REGISTERED])
@pytest.mark.parametrize('bounds', [(-10010., -10000.), (0., 10.), (10000., 10010.)])
def test_uniform_edges(delta, law, bounds, beta, weight):
    p = replace(policy((bounds,), delta), exposure_law=law)
    c = config(beta=beta, noise_sd=0.)
    f = replace(frame(2), weights=(weight,)*2)
    sample = generate_suite_a(f, c, p)
    oracle = ObservedLawOracle(sample.observations, sample.structural_causal_truth, p)
    lo, hi = bounds
    for a in [lo, lo+(hi-lo)/4, hi]:
        q, r = oracle.evaluate('o0', a)
        incoming = lo <= a-delta <= hi-delta and delta <= hi-lo and delta != 0
        moved = lo <= a <= hi-delta and delta != 0
        expected = 1. if p.is_identity else float(incoming)+float(not moved)
        assert float(q) == pytest.approx(50+beta*a, abs=1e-9)
        assert float(r) == pytest.approx(expected, abs=1e-9)

@pytest.mark.parametrize('measured', [False, True])
def test_mixture(measured):
    c = config(local_confounding='measured' if measured else 'omitted', exposure_error=.4,
               selected_outcome=True, migration=.2)
    p = policy()
    sample = generate_suite_a(frame(2), c, p)
    o = ObservedLawOracle(sample.observations, sample.structural_causal_truth, p)
    observed_u = sample.observations.measured_x[0][0] if measured else None
    def joint(a, observed=None):
        den, num = 0., 0.
        for u in (-1., 1.):
            if measured and u != observed_u:
                continue
            for error in (-.4, .4):
                for illness, prob in ((0., .7), (1., .3)):
                    rate = .3*u-.25*illness
                    density = np.exp(rate*(a-error))*rate/np.expm1(10*rate) if 0 <= a-error <= 10 else 0.
                    mass = .25*prob*density*expit(1-.2*(a if observed is None else observed)-1.2*illness)
                    den += mass
                    num += mass*(50+a-error-.2*illness+2*u+2*illness)
        return den, num
    for a in (.5, 4., 9.5):
        den, num = joint(a)
        q, r = o.evaluate('o0', a)
        expected_ratio = (joint(a-2, observed=a)[0] if 2 <= a <= 10 else 0.)/den + (1 if a > 8 else 0)
        assert float(q) == pytest.approx(num/den, abs=1e-9)
        assert float(r) == pytest.approx(expected_ratio, abs=1e-9)

def test_no_fitting():
    tree = ast.parse(Path('src/oxyformer/validation/oracle.py').read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(any(word in name for word in ('training', 'sklearn', 'targeting', 'models')) for name in imports)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'fit' for n in ast.walk(tree))

def test_binding():
    from oxyformer.provenance import ContractError
    sample = generate_suite_a(frame(2), config(), policy())
    with pytest.raises(ContractError, match='observation hash'):
        ObservedLawOracle(sample.observations, replace(sample.structural_causal_truth, observation_hash='0'*64), policy())

@pytest.mark.parametrize('mechanism', ['selected_outcome', 'survey_inclusion', 'missing_biomarkers'])
def test_selection_at_observed_exposure(mechanism):
    c = config(**{mechanism: True})
    p = policy()
    sample = generate_suite_a(frame(2), c, p)
    o = ObservedLawOracle(sample.observations, sample.structural_causal_truth, p)
    def density(at, observed):
        value = 0.
        for illness, mass in ((0., .7), (1., .3)):
            rate = -.25*illness if mechanism != 'survey_inclusion' else 0.
            g = np.exp(rate*at)*rate/np.expm1(10*rate) if rate else .1
            prob = {'selected_outcome': expit(1-.2*observed-1.2*illness),
                    'survey_inclusion': expit(.7-.12*observed),
                    'missing_biomarkers': expit(1-.1*observed-.8*illness)}[mechanism]
            value += mass*g*prob
        return value
    for a in (1., 4., 9.):
        ratio = (density(a-2, a)/density(a, a) if a >= 2 else 0.)+(a > 8)
        assert float(o.evaluate('o0', a)[1]) == pytest.approx(ratio, abs=1e-9)

def test_saved_fixed_training_row(tmp_path):
    from oxyformer.validation.generators import ObservedRecords, TruthArtifact
    f = replace(frame(3, 1), support_keys=('s', 's', 'fixed'), weights=(1., 1., 0.))
    p = replace(policy(), components_by_key=(('s', ((0., 10.),)), ('fixed', ())))
    sample = generate_suite_a(f, config(), p, fixed_exposures={'o2': 7.})
    obs, truth = tmp_path/'obs.json', tmp_path/'truth.json'
    obs.write_text(sample.observations.to_json())
    truth.write_text(sample.structural_causal_truth.to_json())
    o = ObservedLawOracle(ObservedRecords.from_json(obs.read_text()),
                         TruthArtifact.from_json(truth.read_text()), p)
    n = o.nuisances(f.original_ids)
    assert n.original_ids == ('o0', 'o1')
    assert o.excluded_fixed_exposure_rows == 1
