"""Synthetic analytic pushforward checks; no downloads or clinical data."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad

from oxyformer.design.policies import (
    PolicyCovariates, ShiftOrStayPolicy, paired_records, policy_diagnostics,
)
from oxyformer.estimation.mtp import pushforward_density, pushforward_ratio
from oxyformer.provenance import ContractError
from oxyformer.validation.analytic_truth import UniformShiftTruth


def policy(components=((0.0, 10.0),), delta=2.0):
    return ShiftOrStayPolicy(support_design_hash="a" * 64,
                             components_by_key=(("s", components),), delta_mmhg=delta)


def covariates(n):
    ids = tuple(f"o{i}" for i in range(n))
    return PolicyCovariates(original_ids=ids, geography_ids=ids, support_keys=("s",) * n)


def ratio(p, a, density=None):
    return pushforward_ratio(p, a, ("s",) * len(a), density or UniformShiftTruth().density,
                             exposure_law="continuous")


def test_uniform_ratio_includes_unchanged_branch():
    a = np.array([0.1, 1.9, 2.1, 7.9, 8.1, 9.9])
    actual = ratio(policy(), a)
    assert_array_equal(actual, [0, 0, 1, 1, 2, 2])
    assert_array_equal(actual, UniformShiftTruth().ratio(a))


@pytest.mark.parametrize("delta", [0.0, 2.0, 6.0, 10.0, 11.0])
def test_uniform_general_shift_mass_and_linear_truth(delta):
    p = policy(delta=delta)
    truth = UniformShiftTruth(delta=delta)
    grid = np.linspace(0, 10, 1000, endpoint=False) + 0.005
    result = p.apply(grid, covariates(len(grid)))
    assert_allclose(ratio(p, grid), truth.ratio(grid))
    assert np.mean(np.asarray(result.d_mmhg) - grid) == pytest.approx(truth.linear_contrast(1))
    mass = quad(lambda b: pushforward_density(p, [b], ["s"], truth.density,
                                              exposure_law="continuous")[0],
                0, 10, points=sorted({min(delta, 10), max(10 - delta, 0)}))[0]
    assert mass == pytest.approx(1)
    if delta == 2:
        assert truth.linear_contrast(3) == pytest.approx(4.8)
        assert truth.linear_contrast(3) != 6


def test_narrow_and_disconnected_support_do_not_bridge_or_clip():
    p = policy(((0, 1), (2, 6), (8, 9)))
    a = [0, 1, 2, 4, 4.01, 6, 7, 8, 9, 10]
    result = p.apply(a, covariates(len(a)))
    assert_array_equal(result.d_mmhg, [0, 1, 4, 6, 4.01, 6, 7, 8, 9, 10])
    assert_array_equal(result.moved, [False, False, True, True, False, False, False, False, False, False])
    narrow = policy(((0, 1), (3, 4)))
    assert narrow.is_identity
    assert_array_equal(narrow.apply(a, covariates(len(a))).d_mmhg, a)
    # A destination across a gap cannot receive an origin from another component.
    assert_array_equal(ratio(p, [2.5, 4.5, 5.5, 8.5]), [0, 2, 2, 1])


def test_same_assignment_geography_requires_same_exposure_and_frozen_support():
    p = replace(policy(), components_by_key=(("s", ((0, 10),)), ("t", ((0, 1),))))
    cov = PolicyCovariates(original_ids=("person1", "person2"),
                           geography_ids=("tract", "tract"), support_keys=("s", "s"))
    assert_array_equal(p.apply([5, 5], cov).d_mmhg, [7, 7])
    with pytest.raises(ContractError, match="assignment geography"):
        p.apply([5, 6], cov)
    with pytest.raises(ContractError, match="assignment geography"):
        p.apply([0.5, 0.5], replace(cov, support_keys=("s", "t")))


def test_origin_weights_survive_transformation_and_class_prior_balances():
    result = policy().apply([1, 5, 9], covariates(3))
    weights = [2, 6, 10]  # w(a) = 1 + a; transformed weight must not be w(d(a)).
    pairs = paired_records(result, weights, weight_id="synthetic-w(a)")
    weights[0] = 99
    assert pairs.origin_weights == (2, 6, 10, 2, 6, 10)
    assert pairs.a_mmhg == (1, 5, 9, 3, 7, 9)
    assert pairs.original_ids[:3] == pairs.original_ids[3:]
    assert pairs.geography_ids[:3] == pairs.geography_ids[3:]
    assert sum(pairs.origin_weights[3:]) / sum(pairs.origin_weights) == 0.5


def test_exposure_weighted_pushforward_normalizes_and_preserves_moments():
    p = policy()
    def weighted_density(a, keys):
        return np.where((a >= 0) & (a <= 10), (1 + a) / 60, 0)
    assert_allclose(ratio(p, [1, 5, 9], weighted_density), [0, 4 / 6, 1 + 8 / 10])
    def integrate(function):
        return quad(function, 0, 10, points=[2, 8], epsabs=1e-11)[0]
    def pushed(b):
        return pushforward_density(p, [b], ["s"], weighted_density, exposure_law="continuous")[0]
    assert integrate(pushed) == pytest.approx(1, abs=1e-12)
    for power in (1, 2, 3):
        pushed_moment = integrate(lambda b: b**power * pushed(b))
        origin_moment = integrate(lambda a: (a + 2 if a <= 8 else a)**power * (1 + a) / 60)
        assert pushed_moment == pytest.approx(origin_moment)
    # Analytic weighted movement: 2 * integral_0^8 (1+a)/60 da = 4/3.
    assert integrate(lambda b: b * pushed(b)) - integrate(lambda a: a * (1 + a) / 60) == pytest.approx(4 / 3)


def test_conditional_support_keys_use_their_own_target_laws():
    p = replace(policy(), components_by_key=(("s", ((0, 10),)), ("t", ((10, 14),))))
    def density(a, keys):
        return np.array([0.1 if key == "s" else 0.25 for key in keys])
    assert_array_equal(pushforward_ratio(p, [1, 9, 11, 13], ["s", "s", "t", "t"],
                                        density, exposure_law="continuous"), [0, 2, 0, 2])


def test_moved_and_affected_are_separate_diagnostics():
    a = [1, 5, 9]
    result = policy().apply(a, covariates(3))
    diagnostics = policy_diagnostics(result, ratio(policy(), a), [1, 2, 3])
    assert diagnostics.moved == (True, True, False)
    assert diagnostics.affected == (True, True, True)
    assert diagnostics.moved_fraction == pytest.approx(0.5)
    assert diagnostics.affected_fraction == 1
    assert diagnostics.average_shift_mmhg == 1
    identity = policy(delta=0).apply(a, covariates(3))
    diagnostics = policy_diagnostics(identity, [0, 2, 9], [1, 2, 3])
    assert diagnostics.moved_fraction == diagnostics.affected_fraction == diagnostics.average_shift_mmhg == 0


@pytest.mark.parametrize("law", ["mixed", "discrete"])
def test_reject_unsupported_measures(law):
    with pytest.raises(ContractError, match="measure derivation"):
        replace(policy(), exposure_law=law)
    with pytest.raises(ContractError, match="measure derivation"):
        pushforward_density(policy(), [1], ["s"], UniformShiftTruth().density, exposure_law=law)


@pytest.mark.parametrize("components", [((1, 1),), ((2, 1),), ((0, 3), (2, 4)), ((0, 2), (2, 3))])
def test_reject_atoms_and_noncomponent_intervals(components):
    with pytest.raises(ContractError):
        policy(components)


def test_frozen_support_hash_and_serialization():
    components = [["s", [[0, 10]]]]
    p = ShiftOrStayPolicy(support_design_hash="a" * 64, components_by_key=components)
    components[0][1][0][1] = 100
    assert p.components_by_key == (("s", ((0, 10),)),)
    assert ShiftOrStayPolicy.from_json(p.to_json()) == p
    assert replace(p, delta_mmhg=1).policy_id != p.policy_id
    assert replace(p, support_design_hash="b" * 64).policy_id != p.policy_id
    with pytest.raises(FrozenInstanceError):
        p.delta_mmhg = 4
    with pytest.raises(ContractError, match="unknown frozen support"):
        p.apply([1], replace(covariates(1), support_keys=("unknown",)))


def test_undefined_density_ratio_is_rejected():
    with pytest.raises(ContractError, match="positive target density"):
        ratio(policy(), [11])
    with pytest.raises(ContractError, match="negative target density"):
        ratio(policy(), [9], lambda a, keys: -np.ones(len(a)))


def test_tract_recipe_preserves_unresolved_support():
    path = Path(__file__).parents[1] / "configs/policies/usaleep_shift2.yaml"
    config = yaml.safe_load(path.read_text())
    assert config["delta_mmhg"] == 2
    assert config["components_by_key"] is config["policy_id"] is config["support_design_hash"] is None
    assert config["transformed_weights"] == "origin"
    assert config["refit_support_in_outcome_folds"] is False


def test_width_equal_to_shift_retains_frozen_closed_boundary():
    # Plan 4.1: S=[L,U-delta], empty only if width < delta. At equality
    # S={L}; the map is identity a.s. under a continuous law, not pointwise.
    p = policy(((0, 2),))
    assert not p.is_identity
    result = p.apply([0, 1, 2], covariates(3))
    assert result.d_mmhg == (2, 1, 2)
    assert result.moved == (True, False, False)
    assert not result.identity
    assert_array_equal(ratio(p, [0.5, 1.5], UniformShiftTruth(upper=2).density), [1, 1])
