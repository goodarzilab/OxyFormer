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
