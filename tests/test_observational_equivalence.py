"""Identical observations cannot select the correct causal world."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from oxyformer.validation.generators import observational_equivalence_pair, run_estimator, write_sample


def test_paired_world_observations_and_estimator_inputs_are_bit_identical(tmp_path):
    pair = observational_equivalence_pair(seed=73, c=.7, tau=1e12)
    assert pair.m0.observations.to_json().encode() == pair.mtau.observations.to_json().encode()
    assert_array_equal(pair.m0.observations.a, pair.mtau.observations.a)
    assert_array_equal(pair.m0.observations.y, pair.mtau.observations.y)
    seen = []
    def estimator(records):
        seen.append(records.to_json().encode())
        a, y = np.array(records.a), np.array(records.y)
        # A deterministic statistical fit sees the same data, regardless of the
        # unknown decomposition between oxygen and location effects.
        return np.linalg.lstsq(np.column_stack([np.ones(len(a)), a]), y, rcond=None)[0][1]
    assert run_estimator(estimator, pair.m0.observations) == run_estimator(estimator, pair.mtau.observations)
    assert seen[0] == seen[1]
    assert pair.m0.observed_law_truth.value == pair.mtau.observed_law_truth.value == pytest.approx(1.12)
    for name, sample in (("m0", pair.m0), ("mtau", pair.mtau)):
        write_sample(sample, observations_dir=tmp_path/name/"observed", truth_dir=tmp_path/name/"private")
    assert (tmp_path/"m0/observed/observations.json").read_bytes() == (tmp_path/"mtau/observed/observations.json").read_bytes()


def test_distinct_structural_interventions_and_known_policy_truth():
    tau = 3.25
    pair = observational_equivalence_pair(seed=23, c=1.5, tau=tau)
    a = np.array(pair.m0.observations.a)
    shifted = a+2*(a <= 8)
    change0 = pair.world0.intervene(shifted)-pair.world0.intervene(a)
    changetau = pair.worldtau.intervene(shifted)-pair.worldtau.intervene(a)
    assert_array_equal(change0, np.zeros(len(a)))
    assert_allclose(changetau, tau*(shifted-a), atol=3e-14)
    assert pair.world0.structural_effect == 0.
    assert pair.worldtau.structural_effect == tau
    assert pair.m0.structural_causal_truth.value == 0.
    assert pair.mtau.structural_causal_truth.value == pytest.approx(1.6*tau)
    assert pair.m0.structural_causal_truth.value != pair.mtau.structural_causal_truth.value
    # Equal observational fit is compatible with both, not evidence favoring one.
    assert pair.m0.observed_law_truth.value == pair.mtau.observed_law_truth.value
    assert_allclose(pair.world0.intervene(a), pair.m0.observations.y, atol=1e-14)
    assert_allclose(pair.worldtau.intervene(a), pair.mtau.observations.y, atol=1e-14)


def test_pair_preserves_cluster_assignment_and_reproducibility():
    first = observational_equivalence_pair(n_geographies=7, cluster_size=4, seed=8)
    second = observational_equivalence_pair(n_geographies=7, cluster_size=4, seed=8)
    assert first == second
    assert len(first.m0.observations.a) == 28
    assert all(len(set(first.m0.observations.a[i:i+4])) == 1 for i in range(0,28,4))
    assert len(set(first.m0.observations.a)) == 7


def test_location_only_diagnostic_is_not_an_identification_certificate():
    pair = observational_equivalence_pair(seed=41)
    def diagnostic(records):
        # Deliberate section-6.3 red-team callback, not an approved nuisance
        # builder: it bypasses frame.x and uses diagnostic location instead.
        location = np.asarray(records.frame.coordinates)[:,0]
        fit = np.linalg.lstsq(np.column_stack([np.ones(len(location)),location]),
                              np.asarray(records.y),rcond=None)[0]
        fixed_location_prediction = fit[0]+fit[1]*location
        return float(np.mean(fixed_location_prediction-fixed_location_prediction))
    assert run_estimator(diagnostic,pair.m0.observations) == run_estimator(diagnostic,pair.mtau.observations) == 0.
    assert pair.m0.structural_causal_truth.value != pair.mtau.structural_causal_truth.value


@pytest.mark.parametrize("tau,c", [(2.,1.5),(float(2**60),1.5),(-float(2**60),1.5),
                                  (1e300,1e-200),(1e300,1e100)])
def test_large_structural_effect_preserves_factual_anchor_and_serialized_world(tau,c):
    from fractions import Fraction
    from oxyformer.validation.generators import PairedWorld
    pair = observational_equivalence_pair(n_geographies=1,cluster_size=1,seed=0,c=c,tau=tau,noise_sd=0.)
    factual = pair.mtau.observations.a
    for world in (pair.world0,pair.worldtau):
        restored = PairedWorld.from_json(world.to_json())
        assert_array_equal(restored.intervene(factual),pair.mtau.observations.y)
        # Adjacent representable interventions retain their small displacement
        # before multiplying by tau; direct tau*A + (c-tau)*S cannot do so.
        doses = np.nextafter(np.asarray(factual),np.inf)
        expected = [float(Fraction(y)+Fraction(world.structural_effect)*(Fraction(float(a))-Fraction(s)))
                    for y,a,s in zip(pair.mtau.observations.y,doses,factual)]
        assert_array_equal(restored.intervene(doses),expected)
        assert restored.structural_effect == world.structural_effect
        assert restored.factual_location_effect == c
        assert "location_effect" not in restored.to_dict()["payload"]
        # Only the final output is rounded. Coefficients are not replaced by
        # rounded c-tau. The original factual rounding residual stays fixed.
        original_factual = (Fraction(world.baseline[0])+Fraction(c)*Fraction(world.h_s[0])
                            +Fraction(world.epsilon[0]))
        residual = Fraction(pair.mtau.observations.y[0])-original_factual
        original_intervention = (Fraction(world.baseline[0])+Fraction(world.structural_effect)*Fraction(float(doses[0]))
                                 +(Fraction(c)-Fraction(world.structural_effect))*Fraction(world.h_s[0])
                                 +Fraction(world.epsilon[0]))
        assert float(original_intervention+residual) == expected[0]
    assert_array_equal(pair.world0.intervene([1e308]),pair.m0.observations.y)


def test_paired_world_nonrepresentable_response_is_explicit():
    from oxyformer.provenance import ContractError
    pair = observational_equivalence_pair(n_geographies=1,cluster_size=1,tau=1e300)
    with pytest.raises(ContractError,match="nonfinite intervention response"):
        pair.worldtau.intervene([1e308])



def test_paired_response_retains_finite_result_across_intermediate_overflow():
    from fractions import Fraction
    pair = observational_equivalence_pair(n_geographies=1,cluster_size=1,c=-1e307,tau=2.,noise_sd=0.)
    expected = float(Fraction(pair.mtau.observations.y[0])+Fraction(2.)*(Fraction(1e308)-Fraction(pair.mtau.observations.a[0])))
    assert np.isfinite(expected)
    assert pair.worldtau.intervene([1e308])[0] == expected


@pytest.mark.parametrize("parameters", [{"c": 1.1e307}, {"tau": 1.1e300}, {"noise_sd": 101.}])
def test_paired_scenario_domain_is_checked_before_latent_draws(parameters, monkeypatch):
    from oxyformer.provenance import ContractError
    monkeypatch.setattr(np.random, "default_rng", lambda *args: pytest.fail("latent draws preceded validation"))
    with pytest.raises(ContractError, match="outside supported numeric domain"):
        observational_equivalence_pair(**parameters)


def test_paired_intervention_refuses_unsupported_dose_even_in_null_world():
    from oxyformer.provenance import ContractError
    pair = observational_equivalence_pair(n_geographies=1, cluster_size=1)
    for world in (pair.world0, pair.worldtau):
        with pytest.raises(ContractError, match="intervention doses outside supported numeric domain"):
            world.intervene([1.1e308])


@pytest.mark.parametrize("direction", [-1, 1])
def test_continuation_paired_intervention_retains_extended_precision(direction):
    from fractions import Fraction
    pair = observational_equivalence_pair(n_geographies=1, cluster_size=1, tau=1e20, seed=0)
    world = pair.worldtau
    center = np.longdouble(world.h_s[0])
    for dose in (center+direction*np.longdouble(2)**-60,
                 np.nextafter(center, np.longdouble(direction*np.inf))):
        expected = float(Fraction(world.factual_y[0])+Fraction(world.structural_effect)
                         *(Fraction(*dose.as_integer_ratio())-Fraction(world.h_s[0])))
        assert world.intervene(np.array([dose]))[0] == expected


@pytest.mark.parametrize("dose", [2**64-1,2**64,2**64+1,10**100])
def test_continuation_round2_integer_intervention(dose):
    from fractions import Fraction
    pair = observational_equivalence_pair(n_geographies=1,cluster_size=1,seed=0)
    for world in (pair.world0,pair.worldtau):
        expected = float(Fraction(world.factual_y[0])+Fraction(world.structural_effect)
                         *(Fraction(dose)-Fraction(world.h_s[0])))
        assert world.intervene([dose])[0] == expected


def test_continuation_round2_integer_intervention_domain_edge():
    from oxyformer.provenance import ContractError
    pair = observational_equivalence_pair(n_geographies=1,cluster_size=1)
    edge = int(1e308)
    for sign in (-1,1):
        for dose in (sign*(edge-1),sign*edge):
            assert pair.world0.intervene([dose])[0] == pair.world0.factual_y[0]
        with pytest.raises(ContractError,match='outside supported numeric domain'):
            pair.world0.intervene([sign*(edge+1)])
