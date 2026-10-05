"""Offline synthetic fixtures and independently integrated population references."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import expit

from oxyformer.design.policies import PolicyCovariates, ShiftOrStayPolicy
from oxyformer.estimation.mtp import pushforward_ratio
from oxyformer.provenance import ContractError, read_artifact
from oxyformer.validation.analytic_truth import UniformShiftTruth
from oxyformer.validation.generators import (
    ObservedRecords, generate_suite_a, load_suite_a, run_estimator, write_sample,
)
from oxyformer.validation.scm import AssignmentLaw, CovariateFrame, LatentState, SCMConfig


def frame(n=6, cluster_size=2):
    return CovariateFrame(original_ids=tuple(f"o{i}" for i in range(n)),
        geography_ids=tuple(f"g{i//cluster_size}" for i in range(n)),
        region_ids=tuple(f"r{i//(2*cluster_size)}" for i in range(n)),
        cluster_ids=tuple(f"g{i//cluster_size}" for i in range(n)),
        coordinates=tuple((float(i//cluster_size)/10, 1.) for i in range(n)),
        columns=("x", "z"), x=((0., None),)*n, support_keys=("s",)*n,
        weights=(1.,)*n, outcome_available=(True,)*n, biomarker_available=(True,)*n)


def policy(components=((0., 10.),), delta=2.):
    return ShiftOrStayPolicy(support_design_hash="a"*64, components_by_key=(("s", components),), delta_mmhg=delta)


def config(effect="linear", **kwargs):
    active = [effect]
    for scale in ("local", "regional"):
        if kwargs.get(scale + "_confounding", "none") != "none":
            active.append(scale + "_" + kwargs[scale + "_confounding"])
    if kwargs.get("assignment", "continuous") != "continuous":
        active.append(kwargs["assignment"])
    for key in ("extreme_ratios", "support_gaps", "heterogeneous_eligibility", "exposure_error", "migration",
                "selected_outcome", "survey_inclusion", "missing_biomarkers", "denominator_error"):
        if kwargs.get(key):
            active.append(key)
    if kwargs.get("registration_probability", 1.) != 1:
        active.append("under_registration")
    return SCMConfig(name="synthetic", effect=effect, active_mechanisms=tuple(active), **kwargs)


@pytest.mark.parametrize("effect", ["null", "linear", "nonlinear", "sign_changing"])
def test_effect_truth_against_independent_quad(effect):
    c = config(effect, beta=1.7)
    sample = generate_suite_a(frame(), c, policy())
    functions = {"null": lambda a: 0., "linear": lambda a: 1.7*a,
                 "nonlinear": lambda a: 1.7*np.sin(a/2),
                 "sign_changing": lambda a: 1.7*(a-5)**2/10}
    f = functions[effect]
    expected = quad(lambda a: (f(a+2)-f(a))/10, 0, 8, epsabs=1e-12)[0]
    assert sample.observed_law_truth.value == pytest.approx(expected, abs=1e-11)
    assert sample.structural_causal_truth.value == pytest.approx(expected, abs=1e-11)
    assert sample.integration_uncertainty.converged
    assert sample.integration_uncertainty.observed_absolute_difference < 1e-10
    if effect == "linear":
        assert expected == pytest.approx(UniformShiftTruth().linear_contrast(1.7))
    if effect == "sign_changing":
        assert f(2)-f(0) < 0 < f(10)-f(8)


@pytest.mark.parametrize("delta", [0., 11.])
def test_identity_truth_is_exactly_zero(delta):
    sample = generate_suite_a(frame(), config(local_confounding="omitted"), policy(delta=delta))
    assert sample.observed_law_truth.value == sample.structural_causal_truth.value == 0.


def _confounding_reference(measured):
    # Independent two-state posterior; no generator integration helpers used.
    def g(a, u):
        rate = .3*u
        return np.exp(rate*a)*rate/np.expm1(10*rate)
    def mu(a):
        return sum(g(a,u)*(50+a+2*u) for u in (-1,1))/sum(g(a,u) for u in (-1,1))
    causal = sum(.5*quad(lambda a: 2*g(a,u), 0, 8, epsabs=1e-12)[0] for u in (-1,1))
    observed = causal if measured else quad(lambda a: .5*(g(a,-1)+g(a,1))*(mu(a+2)-mu(a)), 0, 8, epsabs=1e-12)[0]
    return observed, causal


@pytest.mark.parametrize("kind", ["measured", "omitted"])
def test_confounding_truth_is_statistical_or_structural_not_interchanged(kind):
    sample = generate_suite_a(frame(2), config(local_confounding=kind), policy())
    expected = _confounding_reference(kind == "measured")
    assert_allclose([sample.observed_law_truth.value, sample.structural_causal_truth.value], expected, atol=1e-11)
    assert ("local_confounder" in sample.observations.measured_columns) == (kind == "measured")
    if kind == "omitted":
        assert abs(expected[0]-expected[1]) > .1


def test_geometry_covariates_missingness_clusters_and_origin_weights_preserved():
    original = replace(frame(), x=((1., None), (2., 3.), (None, None), (2., 3.), (5., None), (0., 1.)),
                       weights=(1., 2., 3., 4., 5., 6.),
                       outcome_available=(True, False, True, True, True, True),
                       biomarker_available=(True, True, False, True, True, True))
    c = config(local_confounding="measured", regional_confounding="measured")
    observed = generate_suite_a(original, c, policy()).observations
    assert observed.frame is original
    assert observed.frame.to_json() == original.to_json()
    assert observed.y[1] is observed.y[2] is None
    for i in range(0, 6, 2):
        assert observed.a[i] == observed.a[i+1]
        assert observed.measured_x[i] == observed.measured_x[i+1]
    assert observed.measured_x[0][1] == observed.measured_x[2][1]
    p = policy().apply(observed.a, PolicyCovariates(original_ids=original.original_ids,
        geography_ids=original.geography_ids, support_keys=original.support_keys))
    assert p.d_mmhg[0] == p.d_mmhg[1]


def test_shared_noise_preserves_dependence_not_only_ids():
    # Repeated synthetic populations identify within-cluster covariance.
    from oxyformer.validation.generators import _sample_observations
    f = frame(4)
    draws = np.array([_sample_observations(f, config("null"), policy(), seed).y for seed in range(500)])
    covariance = np.cov(draws, rowvar=False)
    assert covariance[0, 1] > .45
    assert abs(covariance[0, 2]) < .15


def test_support_gaps_and_heterogeneous_eligibility():
    gaps = ((0., 3.), (7., 10.))
    sample = generate_suite_a(frame(80), config(support_gaps=True), policy(gaps))
    assert all(a <= 3 or a >= 7 for a in sample.observations.a)
    assert sample.structural_causal_truth.value == pytest.approx(2/3)
    shifted = policy(gaps).shift_mask(sample.observations.a, ("s",)*80)
    assert all(a+2 <= 3 or 7 <= a+2 <= 10 for a, m in zip(sample.observations.a, shifted) if m)
    f = replace(frame(4), support_keys=("s", "s", "narrow", "narrow"), weights=(1., 2., 3., 4.))
    p = replace(policy(), components_by_key=(("s", ((0.,10.),)), ("narrow", ((0.,1.),))))
    result = generate_suite_a(f, config(heterogeneous_eligibility=True), p)
    assert result.structural_causal_truth.value == pytest.approx(1.6*3/10)


def test_near_deterministic_location_and_extreme_ratio_laws():
    f = frame(100)
    near = config(assignment="near_deterministic")
    # Sample directly here so this checks geometry rather than integrating 50 unique laws.
    from oxyformer.validation.generators import _sample_observations
    obs = _sample_observations(f, near, policy(), 32)
    centers = [10*expit(x[0]) for x in f.coordinates]
    assert np.mean(np.abs(np.array(obs.a)-centers)) < .08
    truth = generate_suite_a(frame(2), near, policy())
    expected = quad(lambda a: 2*np.exp(-abs(a-5)/.05)/(.1*(1-np.exp(-100))), 0, 8, points=[5], epsabs=1e-11)[0]
    assert truth.structural_causal_truth.value == pytest.approx(expected, abs=1e-9)
    extreme = config(extreme_ratios=True)
    law = AssignmentLaw(f, 0, LatentState(), extreme, ((0., 10.),))
    ratios = pushforward_ratio(policy(), [3., 9.], ("s", "s"), lambda a, k: law.density(a), exposure_law="continuous")
    assert ratios[0] == pytest.approx(np.exp(8))
    assert ratios[1] == pytest.approx(np.exp(8)+1)
    truth = generate_suite_a(frame(2), extreme, policy())
    assert truth.structural_causal_truth.value == pytest.approx(2*(-np.expm1(-32))/(-np.expm1(-40)), abs=1e-10)


def test_atoms_are_exact_and_expected_design_rejection_not_jitter():
    sample = generate_suite_a(frame(200), config(assignment="atoms"), policy(), seed=7)
    assert sample.observations.a.count(0.) >= 50
    assert any(a > 0 for a in sample.observations.a)
    assert sample.observed_law_truth.value is sample.structural_causal_truth.value is None
    assert sample.observed_law_truth.status == sample.structural_causal_truth.status == "design_rejected"
    assert "measure derivation" in sample.observed_law_truth.reason
    assert sample.integration_uncertainty.order == 0


def test_missing_selection_survey_and_biomarkers_have_correct_truth_and_masks():
    c = config(selected_outcome=True, survey_inclusion=True, missing_biomarkers=True)
    sample = generate_suite_a(frame(200), c, policy(), seed=40)
    obs = sample.observations
    assert 0 < sum(y is not None for y in obs.y) < 100
    assert not all(obs.flag_available) and not all(obs.survey_included) and not all(obs.biomarker_available)
    for i in range(0, 200, 2):
        assert obs.flag_available[i] == obs.flag_available[i+1]
        assert obs.survey_included[i] == obs.survey_included[i+1]
    # Illness shifts assignment and all observation mechanisms. Independent
    # scalar quadrature of the selected posterior tests both targets.
    def joint(a, illness):
        g = .1 if illness == 0 else -.25*np.exp(-.25*a)/np.expm1(-2.5)
        return (.7 if illness == 0 else .3)*g*expit(1-.2*a-1.2*illness)*expit(.7-.12*a)*expit(1-.1*a-.8*illness)
    def density(a):
        return sum(joint(a,u) for u in (0,1))
    def mu(a):
        return sum(joint(a,u)*(50+a+2*u) for u in (0,1))/density(a)
    normalizer = quad(density, 0, 10, epsabs=1e-12)[0]
    statistical = quad(lambda a: density(a)*(mu(a+2)-mu(a)), 0, 8, epsabs=1e-12)[0]/normalizer
    causal = quad(lambda a: 2*density(a), 0, 8, epsabs=1e-12)[0]/normalizer
    assert sample.observed_law_truth.value == pytest.approx(statistical, abs=1e-10)
    assert sample.structural_causal_truth.value == pytest.approx(causal, abs=1e-10)


def test_exposure_error_and_illness_migration_against_independent_mixture_quad():
    c = config("nonlinear", exposure_error=.4, migration=2., selected_outcome=True)
    sample = generate_suite_a(frame(2), c, policy())
    states = [(u,e) for u in (0,1) for e in (-.4,.4)]
    def joint(a,u,e):
        true = a-e
        if not 0 <= true <= 10:
            return 0.
        g = .1 if u == 0 else -.25*np.exp(-.25*true)/np.expm1(-2.5)
        return .5*(.7 if u == 0 else .3)*g*expit(1-.2*a-1.2*u)
    def mean(a,u,e):
        return 50+np.sin((a-e-2*u)/2)+2*u
    def density(a):
        return sum(joint(a,u,e) for u,e in states)
    def mu(a):
        return sum(joint(a,u,e)*mean(a,u,e) for u,e in states)/density(a)
    breaks = [-.4, 0., .4, 8., 9.6, 10., 10.4]
    norm = sum(quad(density, l,h,epsabs=1e-11)[0] for l,h in zip(breaks[:-1],breaks[1:]))
    stat = quad(lambda a: density(a)*(mu(a+2)-mu(a)), 0,8,points=[.4,7.6],epsabs=1e-11)[0]/norm
    causal = quad(lambda a: sum(joint(a,u,e)*(mean(a+2,u,e)-mean(a,u,e)) for u,e in states), 0,8,points=[.4],epsabs=1e-11)[0]/norm
    assert sample.observed_law_truth.value == pytest.approx(stat, abs=1e-9)
    assert sample.structural_causal_truth.value == pytest.approx(causal, abs=1e-9)
    assert "held fixed" in sample.structural_causal_truth.target


def test_under_registration_noisy_denominator_uses_same_endpoint_scale():
    c = config(registration_probability=.65, denominator_error=.2)
    sample = generate_suite_a(frame(100), c, policy(), seed=4)
    expected = 1.6*.65*(1/.8+1/1.2)/2
    assert sample.observed_law_truth.value == pytest.approx(expected)
    assert sample.structural_causal_truth.value == pytest.approx(expected)
    obs = sample.observations
    assert set(obs.observed_denominator) == {80., 120.}
    for y, count, denominator in zip(obs.y, obs.registered_events, obs.observed_denominator):
        assert y == count/denominator
    assert np.mean(obs.y) < 45  # mean true rate is about 55


def test_truth_is_separate_serializable_and_inaccessible_to_estimator(tmp_path):
    sample = generate_suite_a(frame(), config(local_confounding="omitted"), policy())
    obsdir, private = tmp_path/"estimator", tmp_path/"evaluator"
    write_sample(sample, observations_dir=obsdir, truth_dir=private)
    assert {p.name for p in obsdir.iterdir()} == {"observations.json"}
    assert {p.name for p in private.iterdir()} == {"observed_law.json", "structural_causal.json", "integration_uncertainty.json"}
    observed = read_artifact(obsdir/"observations.json", ObservedRecords, sample.observations.content_hash)
    assert observed == sample.observations
    def estimator(records):
        assert type(records) is ObservedRecords
        assert not hasattr(records, "structural_causal_truth")
        assert not hasattr(records, "config")
        assert "LatentState" not in records.to_json()
        return np.mean(records.y)
    assert run_estimator(estimator, observed) == pytest.approx(np.mean(observed.y))
    with pytest.raises(ContractError, match="ObservedRecords only"):
        run_estimator(estimator, sample)
    with pytest.raises(ContractError, match="separate"):
        write_sample(sample, observations_dir=obsdir, truth_dir=obsdir/"truth")
    with pytest.raises(FileExistsError):
        write_sample(sample, observations_dir=obsdir, truth_dir=private)


def test_explicit_bounded_recipe_and_reproducibility():
    configs = load_suite_a(Path(__file__).parents[1]/"configs/validation/suite_a.yaml")
    assert len(configs) == 14
    assert set().union(*(set(c.active_mechanisms) for c in configs)) >= {
        "null", "linear", "nonlinear", "sign_changing", "local_measured", "regional_measured",
        "local_omitted", "regional_omitted", "near_deterministic", "support_gaps", "atoms", "extreme_ratios",
        "heterogeneous_eligibility", "exposure_error", "migration", "selected_outcome", "survey_inclusion",
        "missing_biomarkers", "under_registration", "denominator_error"}
    first = generate_suite_a(frame(), configs[1], policy(), seed=1)
    again = generate_suite_a(frame(), configs[1], policy(), seed=1)
    other = generate_suite_a(frame(), configs[1], policy(), seed=2)
    assert first == again
    assert first.observations != other.observations
    assert first.observed_law_truth.value == other.observed_law_truth.value
    with pytest.raises(ContractError, match="exactly"):
        SCMConfig(name="silent_mechanism", active_mechanisms=("linear",), exposure_error=1.)
    with pytest.raises(ContractError, match="support_gaps"):
        generate_suite_a(frame(), config(), policy(((0.,3.), (7.,10.))))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_invalid_count_law_rejected_even_if_realized_dose_would_be_valid(seed):
    with pytest.raises(ContractError, match="negative event rate on"):
        generate_suite_a(frame(2), config(beta=-10., registration_probability=.5), policy(), seed=seed)


def test_nonlinear_near_deterministic_truth_at_policy_threshold():
    f = replace(frame(2), coordinates=((float(np.log(4)), 0.),)*2)
    c = config("nonlinear", assignment="near_deterministic")
    result = generate_suite_a(f, c, policy())
    from scipy.stats import laplace
    normalization = laplace.cdf(10, 8, .05)-laplace.cdf(0, 8, .05)
    expected = quad(lambda a: (np.sin((a+2)/2)-np.sin(a/2))*laplace.pdf(a,8,.05)/normalization,
                    0,8,epsabs=1e-12,points=[7.5,7.9])[0]
    assert result.observed_law_truth.value == pytest.approx(expected, abs=1e-10)
    assert result.structural_causal_truth.value == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("scenario", load_suite_a(Path(__file__).parents[1]/"configs/validation/suite_a.yaml"), ids=lambda c: c.name)
def test_every_declared_scenario_runs_with_matching_frozen_support(scenario):
    f, p = frame(4), policy()
    if scenario.support_gaps:
        p = policy(((0.,3.), (7.,10.)))
    if scenario.heterogeneous_eligibility:
        f = replace(f, support_keys=("s", "s", "n", "n"))
        p = replace(p, components_by_key=(("s", ((0.,10.),)), ("n", ((0.,1.),))))
    result = generate_suite_a(f, scenario, p)
    assert result.observed_law_truth.status == ("design_rejected" if scenario.assignment == "atoms" else "integrated")
    assert SCMConfig.from_json(result.observed_law_truth.generator_configuration) == scenario


def test_count_validation_handles_error_shift_across_components():
    # true dose 9.5, error -7 => recorded 2.5 => recorded shift 4.5
    # => intervened true dose 11.5, with invalid Poisson rate 50 - 4.9*11.5.
    # The validator already subtracts the error to obtain the true-dose preimage.
    with pytest.raises(ContractError, match="negative event rate on"):
        generate_suite_a(frame(2), config(beta=-4.9, exposure_error=7.,
                         registration_probability=.5, support_gaps=True),
                         policy(((0.,5.), (7.,10.))))


@pytest.mark.parametrize("missing", ["outcomes", "biomarkers", "positive_weight_rows"])
def test_empty_selected_population_preserves_observations_without_inventing_truth(missing, tmp_path):
    f = frame(2)
    if missing == "outcomes":
        f = replace(f, outcome_available=(False, False))
    elif missing == "biomarkers":
        f = replace(f, biomarker_available=(False, False))
    else:
        f = replace(f, weights=(0., 1.), outcome_available=(True, False))
    result = generate_suite_a(f, config(), policy())
    assert result.observations.frame == f
    assert len(result.observations.a) == 2
    if missing != "positive_weight_rows":
        assert result.observations.y == (None, None)
    for truth in (result.observed_law_truth, result.structural_causal_truth):
        assert truth.status == "empty_target"
        assert truth.value is None
        assert truth.reason == "selected population has zero target mass"
    assert not result.integration_uncertainty.converged
    assert result.integration_uncertainty.observed_absolute_difference is None
    write_sample(result, observations_dir=tmp_path/"observed", truth_dir=tmp_path/"private")


def test_dependence_clusters_can_cross_exposure_geographies():
    from oxyformer.validation.generators import _sample_observations
    f = replace(frame(4), cluster_ids=("household0", "household1", "household0", "household1"))
    draws = np.array([_sample_observations(f, config("null"), policy(), seed).y for seed in range(500)])
    covariance = np.cov(draws, rowvar=False)
    assert covariance[0, 2] > .2  # shared household, distinct exposure geography
    assert abs(covariance[0, 3]) < .15
    sample = generate_suite_a(f, config(), policy())
    assert sample.observations.frame.cluster_ids == f.cluster_ids
    assert sample.observations.a[0] == sample.observations.a[1]
    assert sample.observations.a[2] == sample.observations.a[3]


def test_unknown_effect_is_rejected_by_inherited_contract_validation(tmp_path):
    # Immutable.__post_init__ validates Literal types before comparing switches.
    with pytest.raises(ContractError, match="invalid enum value"):
        SCMConfig(name="unsupported", effect="quadratic", active_mechanisms=("quadratic",))
    recipe = tmp_path/"unsupported.yaml"
    recipe.write_text('suite: A\nscenario_expansion: explicit_only\nscenarios:\n'
                      '  - name: unsupported\n    effect: quadratic\n    active_mechanisms: [quadratic]\n')
    with pytest.raises(ContractError, match="invalid enum value"):
        load_suite_a(recipe)


@pytest.mark.parametrize("scale", [.05, 1e-3, 1e-6, 1e-8, 1e-20, 1e-100, 1e-320, float(np.nextafter(0.,1.))])
def test_concentrated_assignment_retains_mass_and_shifted_conditional_mean(scale):
    sample = generate_suite_a(frame(2), config(assignment="near_deterministic", near_scale=scale), policy())
    from scipy.stats import laplace
    with np.errstate(over="ignore"):  # Infinite standardized tails have exact limiting CDFs.
        expected = 2*(laplace.cdf(8,5,scale)-laplace.cdf(0,5,scale))/(laplace.cdf(10,5,scale)-laplace.cdf(0,5,scale))
    assert sample.observed_law_truth.value == pytest.approx(expected, abs=1e-10)
    assert sample.structural_causal_truth.value == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("coordinate", [0., -800., 800.])
def test_concentrated_gap_and_endpoint_assignments_are_continuous(coordinate):
    f = replace(frame(2), coordinates=((coordinate,0.),)*2)
    c = config(assignment="near_deterministic", near_scale=1e-6, support_gaps=True)
    sample = generate_suite_a(f, c, policy(((0.,3.),(7.,10.))))
    expected = {0.:1., -800.:2., 800.:0.}[coordinate]
    assert sample.observed_law_truth.value == pytest.approx(expected, abs=1e-10)
    assert sample.structural_causal_truth.value == pytest.approx(expected, abs=1e-10)
    assert all(0 < a < 3 or 7 < a < 10 for a in sample.observations.a)


@pytest.mark.parametrize("scale", [1e-3, 1e-6, 1e-8])
@pytest.mark.parametrize("offset", [-1., 0., 1.])
def test_concentrated_nonlinear_policy_threshold_has_independent_scaled_reference(scale, offset):
    requested_center = 8+offset*scale
    coordinate = float(np.log(requested_center/(10-requested_center)))
    center = 10*expit(coordinate)  # Reference the actual floating-point SCM center.
    f = replace(frame(2), coordinates=((coordinate,0.),)*2)
    result = generate_suite_a(f, config("nonlinear",assignment="near_deterministic",near_scale=scale), policy())
    boundary = (8-center)/scale
    normalization = 1-.5*np.exp(-center/scale)-.5*np.exp(-(10-center)/scale)
    def integrand(t):
        a = center+scale*t
        return .5*np.exp(-abs(t))*(np.sin((a+2)/2)-np.sin(a/2))/normalization
    # Tail omitted beyond 50 scale lengths contributes < 2e-22.
    expected = quad(integrand,-50,min(boundary,0.),epsabs=1e-12)[0]
    if boundary > 0:
        expected += quad(integrand,0,boundary,epsabs=1e-12)[0]
    assert result.observed_law_truth.value == pytest.approx(expected, abs=1e-9)
    assert result.structural_causal_truth.value == pytest.approx(expected, abs=1e-9)
    u = result.integration_uncertainty
    assert u.assignment_mass_error < 1e-10
    assert u.selected_mass_fraction == pytest.approx(1., abs=1e-10)


@pytest.mark.parametrize("scale", [.05, 1e-6])
@pytest.mark.parametrize("measured", [False, True])
def test_narrow_posterior_transition_at_shifted_assignment_peak(scale, measured):
    # Positive local cause at center 2; negative local cause at center 6.
    # Shifting the first peak lands exactly on the posterior's sharp crossing.
    coordinates = (float(np.log(2/8)-.2), float(np.log(6/4)+.2))
    f = replace(frame(2,cluster_size=1), coordinates=tuple((v,0.) for v in coordinates), weights=(1.,1.))
    c = config("nonlinear", assignment="near_deterministic",near_scale=scale,
               local_confounding="measured" if measured else "omitted")
    result = generate_suite_a(f,c,policy())
    states = [(10*expit(coord+.2*u),u) for coord in coordinates for u in (-1,1)]
    def response(a,u):
        return 50+np.sin(a/2)+2*u
    def posterior(a,known_u):
        terms = [(center,u) for center,u in states if not measured or u == known_u]
        logs = np.array([-abs(np.longdouble(a)-np.longdouble(center))/scale
                         -np.log1p(-.5*np.exp(-center/scale)-.5*np.exp(-(10-center)/scale))
                         for center,u in terms],dtype=np.longdouble)
        weights = np.exp(logs-logs.max())
        return float(sum(w*response(a,u) for w,(_,u) in zip(weights,terms))/weights.sum())
    observed = causal = 0.
    for center,u in states:
        norm = 1-.5*np.exp(-center/scale)-.5*np.exp(-(10-center)/scale)
        lower,upper = max(-50.,-center/scale),min(50.,(8-center)/scale)
        def integrand(t,statistical):
            a = center+scale*t
            difference = posterior(a+2,u)-posterior(a,u) if statistical else response(a+2,u)-response(a,u)
            return .5*np.exp(-abs(t))*difference/norm
        if lower < upper:
            # Independently split at all pairwise log-density crossings and
            # their shift preimages, including the one at the origin peak.
            cuts = [0.]
            for ci,_ in states:
                for cj,_ in states:
                    cuts.extend([(ci+cj-2*center)/(2*scale), (ci+cj-4-2*center)/(2*scale)])
            cuts = sorted({v for v in cuts if lower < v < upper})
            observed += .25*quad(lambda t: integrand(t,True),lower,upper,points=cuts,epsabs=2e-10,limit=300)[0]
            causal += .25*quad(lambda t: integrand(t,False),lower,upper,points=[v for v in [0.] if lower<v<upper],epsabs=1e-12)[0]
    assert result.observed_law_truth.value == pytest.approx(observed,abs=1e-8)
    assert result.structural_causal_truth.value == pytest.approx(causal,abs=1e-10)
    assert result.integration_uncertainty.selected_mass_fraction == pytest.approx(1.,abs=1e-10)


def test_mass_certificate_rejects_false_convergence(monkeypatch):
    original = AssignmentLaw.quadrature
    def missing_half_the_mass(self, order, breakpoints):
        return tuple(replace(rule,log_weights=rule.log_weights-np.log(2.))
                     for rule in original(self,order,breakpoints))
    monkeypatch.setattr(AssignmentLaw,"quadrature",missing_half_the_mass)
    # Both normalized contrasts and successive orders agree, but mass is wrong.
    with pytest.raises(ContractError,match="did not converge"):
        generate_suite_a(frame(2),config(),policy(),max_order=32)


@pytest.mark.parametrize("delta", [0.,11.])
def test_concentrated_identity_is_exactly_zero(delta):
    result = generate_suite_a(frame(2),config(assignment="near_deterministic",near_scale=1e-6,
                                             local_confounding="omitted"),policy(delta=delta))
    assert result.observed_law_truth.value == result.structural_causal_truth.value == 0.


def test_gap_law_sampler_preserves_component_mass_without_density_underflow():
    c = config(assignment="near_deterministic",near_scale=1e-6,support_gaps=True)
    law = AssignmentLaw(frame(2),0,LatentState(),c,((0.,3.),(7.,10.)))
    rng = np.random.default_rng(718)
    a = np.array([law.sample(rng) for _ in range(3000)])
    assert np.all((a > 0) & (a < 10) & ((a < 3) | (a > 7)))
    assert .47 < np.mean(a < 3) < .53
    assert abs(np.mean(3-a[a<3])-1e-6) < 1e-7
    assert abs(np.mean(a[a>7]-7)-1e-6) < 1e-7
    assert np.isfinite(law.log_density([1.,9.])).all()
    assert (law.density([1.,9.]) == 0).all()  # Underflow is not lack of support.
    assert law.contains([1.,9.]).all()
    assert not law.contains([5.])[0]


def test_concentrated_selection_measurement_migration_and_count_scale_reference():
    scale = 1e-6
    c = config(assignment="near_deterministic",near_scale=scale,exposure_error=.4,migration=2.,
               selected_outcome=True,survey_inclusion=True,missing_biomarkers=True,
               registration_probability=.7,denominator_error=.2)
    f = replace(frame(2),weights=(1.,3.))
    result = generate_suite_a(f,c,policy())
    selected_mass = selected_error = 0.
    for illness in (0,1):
        center = 10*expit(-.2*illness)
        for error in (-.4,.4):
            def integrand(t):
                a = center+scale*t+error
                selection = expit(1-.2*a-1.2*illness)*expit(.7-.12*a)*expit(1-.1*a-.8*illness)
                return .5*np.exp(-abs(t))*selection
            mass = .5*(.7 if illness == 0 else .3)*quad(integrand,-50,50,points=[0.],epsabs=1e-13)[0]
            selected_mass += mass
            selected_error += error*mass
    multiplier = .7*(1/.8+1/1.2)/2
    # Illness terms cancel in the linear response (migration=2, illness effect=2).
    # All origins shift. At shifted doses the positive-error component dominates
    # the posterior up to exponentially negligible exp(-hundreds of thousands).
    expected_observed = multiplier*(2-.4+selected_error/selected_mass)
    assert result.observed_law_truth.value == pytest.approx(expected_observed,abs=1e-9)
    assert result.structural_causal_truth.value == pytest.approx(2*multiplier,abs=1e-10)
    assert result.integration_uncertainty.selected_mass_fraction == pytest.approx(selected_mass,abs=1e-11)


def test_geometric_support_gate_still_rejects_an_actual_gap():
    from oxyformer.validation.generators import _Term, _posterior_mean
    f,c = frame(2),config(support_gaps=True)
    law = AssignmentLaw(f,0,LatentState(),c,((0.,3.),(7.,10.)))
    with pytest.raises(ContractError,match="leaves conditional observed-law support"):
        _posterior_mean(np.array([5.]),[_Term(0,LatentState(),1.,law)],f,c)


@pytest.mark.parametrize("scale", [.05, .25, .5])
def test_extreme_ratio_tilt_composes_with_near_deterministic_assignment(scale):
    from oxyformer.validation.scm import AssignmentLaw, LatentState
    f = replace(frame(2), coordinates=((0.,0.),)*2)
    c = config("nonlinear", assignment="near_deterministic", near_scale=scale, extreme_ratios=True)
    law = AssignmentLaw(f,0,LatentState(),c,((0.,10.),))
    # Independent tilted-Laplace density, including the exactly flat left
    # piece at scale=.25 and the monotone density at scale=.5.
    def unnormalized(a):
        return np.exp(-abs(a-5)/scale-4*a)
    normalizer = quad(unnormalized,0,10,points=[5],epsabs=1e-25,epsrel=1e-11)[0]
    at = np.array([1.,4.,8.])
    assert_allclose(law.density(at),unnormalized(at)/normalizer,rtol=1e-11,atol=0)
    expected = quad(lambda a: unnormalized(a)/normalizer*(np.sin((a+2)/2)-np.sin(a/2)),
                    0,8,points=[5],epsabs=1e-11)[0]
    result = generate_suite_a(f,c,policy())
    assert result.observed_law_truth.value == pytest.approx(expected,abs=1e-9)
    assert result.structural_causal_truth.value == pytest.approx(expected,abs=1e-9)


@pytest.mark.parametrize("scale", [1e-12,1e-20,1e-100,1e-320,float(np.nextafter(0.,1.))])
def test_sub_ulp_assignment_mass_is_classified_before_policy_cutoff_rounding(scale):
    f = replace(frame(2),coordinates=((float(np.log(4)),0.),)*2)
    c = config(assignment="near_deterministic",near_scale=scale)
    result = generate_suite_a(f,c,policy())
    # Center is exactly 8: the continuous symmetric law puts half its mass
    # below the cutoff, even when all stored draws round to the same float.
    assert result.observed_law_truth.value == pytest.approx(1.,abs=1e-10)
    assert result.structural_causal_truth.value == pytest.approx(1.,abs=1e-10)
    assert result.integration_uncertainty.assignment_mass_error < 1e-10
    assert result.integration_uncertainty.selected_mass_fraction == pytest.approx(1.,abs=1e-10)


@pytest.mark.parametrize("scale", [1e-6,1e-20,1e-100,float(np.nextafter(0.,1.))])
def test_endpoint_error_posterior_keeps_local_offsets(scale):
    f = replace(frame(2,cluster_size=1),coordinates=((-800.,0.),(800.,0.)),
                region_ids=("r0","r0"),columns=("x",),x=((0.,),(0.,)))
    result = generate_suite_a(f,config(assignment="near_deterministic",near_scale=scale,
                                      exposure_error=1.),policy(delta=4))
    # Only the low geography's +1 error component shifts (population mass 1/4).
    # At its shifted query, P(error=-1 | A)=expit(2t), t~Exp(1).
    # Integral exp(-t)*expit(2t) dt = pi/4, without production helpers.
    assert result.observed_law_truth.value == pytest.approx(1+np.pi/8,abs=1e-10)
    assert result.structural_causal_truth.value == pytest.approx(1.,abs=1e-10)


@pytest.mark.parametrize("weight", [float(np.nextafter(0.,1.)),1.,1.7e308])
def test_origin_mass_is_invariant_to_common_weight_scale(weight):
    f = replace(frame(2),weights=(weight,weight))
    result = generate_suite_a(f,config(denominator_error=.2),policy())
    assert result.observed_law_truth.value == pytest.approx(5/3,abs=1e-10)
    assert result.structural_causal_truth.value == pytest.approx(5/3,abs=1e-10)
    assert result.integration_uncertainty.selected_mass_fraction == pytest.approx(1.,abs=1e-10)


@pytest.mark.parametrize("scale", [1e-6,1e-100,float(np.nextafter(0.,1.))])
def test_local_posterior_matches_independent_logistic_odds(scale):
    from oxyformer.validation.scm import LocalCoordinates, exact
    from oxyformer.validation.generators import _groups, _posterior_mean
    f = replace(frame(2,cluster_size=1),coordinates=((-800.,0.),(800.,0.)),region_ids=("r0","r0"))
    c = config(assignment="near_deterministic",near_scale=scale,exposure_error=1.)
    terms = next(iter(_groups(f,c,policy(delta=4)).values()))
    t = np.array([0.,.25,1.,4.,16.])
    at = LocalCoordinates(exact(5.),exact(scale),t)
    assert_allclose(_posterior_mean(at,terms,f,c),54+2*expit(2*t)+scale*t,atol=1e-12,rtol=0)
    assert_allclose(_posterior_mean(at,list(reversed(terms)),f,c),54+2*expit(2*t)+scale*t,atol=1e-12,rtol=0)


@pytest.mark.parametrize("scale", [1e-6,1e-100])
def test_tiny_tilted_assignment_retains_unequal_prior_odds(scale):
    ratio = 3*np.exp(-32.)  # Offsets the finite tilt in the shifted posterior.
    f = replace(frame(2,cluster_size=1),coordinates=((-800.,0.),(800.,0.)),
                region_ids=("r0","r0"),weights=(1.,float(ratio)))
    c = config(assignment="near_deterministic",near_scale=scale,exposure_error=1.,extreme_ratios=True)
    result = generate_suite_a(f,c,policy(delta=4))
    log_odds = np.log(ratio)+32+np.log1p(-4*scale)-np.log1p(4*scale)
    posterior = quad(lambda t: np.exp(-t)*expit(log_odds+2*t/(1+4*scale)),0,50,epsabs=1e-12)[0]
    share = 1/(2*(1+ratio))
    # The omitted t>50 contribution is bounded by 6*exp(-50).
    assert result.observed_law_truth.value == pytest.approx(share*(4+2*posterior),abs=1e-10)
    assert result.structural_causal_truth.value == pytest.approx(4*share,abs=1e-10)


def test_tiny_posterior_with_factual_survey_selection():
    f = replace(frame(2,cluster_size=1),coordinates=((-800.,0.),(800.,0.)),region_ids=("r0","r0"))
    result = generate_suite_a(f,config(assignment="near_deterministic",near_scale=1e-100,
                                      exposure_error=1.,survey_inclusion=True),policy(delta=4))
    probabilities = expit(.7-.12*np.array([-1.,1.,9.,11.]))
    mass = probabilities.mean()
    share = probabilities[1]/probabilities.sum()
    assert result.observed_law_truth.value == pytest.approx(share*(4+np.pi/2),abs=1e-10)
    assert result.structural_causal_truth.value == pytest.approx(4*share,abs=1e-10)
    assert result.integration_uncertainty.selected_mass_fraction == pytest.approx(mass,abs=1e-10)


def test_positive_selected_mass_below_float_range_has_log_diagnostic():
    result = generate_suite_a(frame(2),config(assignment="near_deterministic",near_scale=1e-100,
                                             survey_inclusion=True),policy(((10000.,10010.),)))
    assert result.observed_law_truth.status == "integrated"
    assert result.observed_law_truth.value == pytest.approx(2.,abs=1e-10)
    assert result.structural_causal_truth.value == pytest.approx(2.,abs=1e-10)
    assert result.integration_uncertainty.selected_mass_fraction == 0.  # Display underflow only.
    assert result.integration_uncertainty.selected_log_mass_fraction == pytest.approx(.7-.12*10005,abs=1e-10)


def test_tiny_tilted_gap_normalization_keeps_finite_tilt():
    result = generate_suite_a(frame(2),config(assignment="near_deterministic",near_scale=1e-100,
                                             extreme_ratios=True,support_gaps=True),policy(((0.,3.),(7.,10.))))
    expected = 2*expit(-16.)  # Only the component peaked at 7 moves.
    assert result.observed_law_truth.value == pytest.approx(expected,abs=1e-12)
    assert result.structural_causal_truth.value == pytest.approx(expected,abs=1e-12)


@pytest.mark.parametrize("weight", [5e-324,1.,5e307])
def test_unequal_origin_weights_and_row_permutation(weight):
    f = replace(frame(2,cluster_size=1),weights=(weight,2*weight),support_keys=("s","stay"),
                region_ids=("r0","r0"))
    p = replace(policy(),components_by_key=(("s",((0.,10.),)),("stay",((0.,1.),))))
    c = config(heterogeneous_eligibility=True)
    result = generate_suite_a(f,c,p)
    assert result.observed_law_truth.value == pytest.approx(8/15,abs=1e-10)
    row_names = ("original_ids","geography_ids","region_ids","cluster_ids","coordinates","x",
                 "support_keys","weights","outcome_available","biomarker_available")
    permuted = replace(f,**{name:tuple(reversed(getattr(f,name))) for name in row_names})
    other = generate_suite_a(permuted,c,p)
    assert other.observed_law_truth.value == pytest.approx(result.observed_law_truth.value,abs=1e-12)
    split = replace(frame(3,cluster_size=1),weights=(weight,weight,weight),support_keys=("s","stay","stay"),
                    region_ids=("r0",)*3,coordinates=((0.,1.),)*3)
    assert generate_suite_a(split,c,p).observed_law_truth.value == pytest.approx(8/15,abs=1e-10)


def test_grouped_mass_loss_is_detected_independently(monkeypatch):
    import oxyformer.validation.generators as generators
    original = generators._groups
    def lose_half(*args):
        groups = original(*args)
        for terms in groups.values():
            for term in terms:
                term.log_weight -= np.log(2.)
        return groups
    monkeypatch.setattr(generators,"_groups",lose_half)
    with pytest.raises(ContractError,match="grouped origin/latent mass"):
        generate_suite_a(frame(2),config(),policy())


def test_local_support_and_inverse_quantiles_survive_rounding():
    from oxyformer.validation.scm import LocalCoordinates, exact
    scale = float(np.nextafter(0.,1.))
    points = LocalCoordinates(exact(1.),exact(scale),np.array([-1.,1.]))
    assert points.inside(1.,2.).tolist() == [False,True]
    assert points.shifted(2.).inside(3.,4.).tolist() == [False,True]
    law = AssignmentLaw(replace(frame(2),coordinates=((-800.,0.),)*2),0,LatentState(),
                        config(assignment="near_deterministic",near_scale=scale),((0.,10.),))
    u = np.array([.25,.5,.75])
    local = law.quantile_coordinates(0,u)
    assert_allclose(local.values,-np.log1p(-u),atol=1e-14)
    assert (local.values > 0).all()


def test_nearly_flat_laplace_law_against_independent_quad():
    scale = 1e3
    normalizer = quad(lambda a: np.exp(-abs(a-5)/scale),0,10,points=[5],epsabs=1e-12)[0]
    expected = quad(lambda a: np.exp(-abs(a-5)/scale)*(np.sin((a+2)/2)-np.sin(a/2))/normalizer,
                    0,8,points=[5],epsabs=1e-12)[0]
    result = generate_suite_a(frame(2),config("nonlinear",assignment="near_deterministic",near_scale=scale),policy())
    assert result.observed_law_truth.value == pytest.approx(expected,abs=1e-10)
    assert result.structural_causal_truth.value == pytest.approx(expected,abs=1e-10)


def _nonbinary_cutoff_cases():
    from fractions import Fraction
    # A finite set of synthetic boundary cases, not a mechanism campaign grid.
    cases = []
    for lo,hi,coordinate in [(0.,1.,float(np.log(9.))), (9.8,10.,0.),
                             (0.,10.,float(np.log(99.)))]:
        cutoff = Fraction(hi)-Fraction(.1)
        sides = {}
        for direction in (-np.inf,np.inf):
            point = coordinate
            for _ in range(32):
                center = lo+(hi-lo)*float(expit(point))
                side = Fraction(center) < cutoff
                sides.setdefault(side,point)
                point = float(np.nextafter(point,direction))
        # The narrow [9.8,10] center needs a finite logistic displacement:
        # changing coordinate by a subnormal cannot change its rounded center.
        for point in (-1e-12,1e-12) if lo == 9.8 else ():
            sides.setdefault(Fraction(lo+(hi-lo)*float(expit(point))) < cutoff,point)
        assert set(sides) == {False,True}
        for side in (True,False):
            for scale in (1e-16,1e-20,1e-100,float(np.nextafter(0.,1.))):
                cases.append((lo,hi,sides[side],scale))
    return cases


@pytest.mark.parametrize("lo,hi,coordinate,scale", _nonbinary_cutoff_cases())
def test_nonbinary_policy_cutoff_matches_exact_continuous_reference(lo,hi,coordinate,scale):
    from decimal import Decimal, localcontext, MAX_EMAX, MIN_EMIN
    from fractions import Fraction
    # Independent truncated-Laplace CDF. Inputs are the actual declared floats;
    # exact subtraction precedes conversion. Decimal tails can underflow to zero
    # only far below the absolute truth tolerance, never at the near-mode cut.
    center = lo+(hi-lo)*float(expit(coordinate))
    exact_center, unit = Fraction(center), Fraction(scale)
    with localcontext() as ctx:
        ctx.prec,ctx.Emax,ctx.Emin = 80,MAX_EMAX,MIN_EMIN
        def cdf(value):
            ratio = (value-exact_center)/unit
            distance = Decimal(ratio.numerator)/Decimal(ratio.denominator)
            return (-abs(distance)).exp()/2 if distance <= 0 else 1-(-distance).exp()/2
        cutoff = Fraction(hi)-Fraction(.1)
        moved = (cdf(cutoff)-cdf(Fraction(lo)))/(cdf(Fraction(hi))-cdf(Fraction(lo)))
        expected = float(Decimal.from_float(.1)*moved)
    f = replace(frame(1,1),columns=("x",),x=((0.,),),coordinates=((coordinate,0.),))
    sample = generate_suite_a(f,config(assignment="near_deterministic",near_scale=scale,noise_sd=0.),
                              policy(((lo,hi),),delta=.1))
    assert sample.observed_law_truth.value == pytest.approx(expected,abs=1e-11)
    assert sample.structural_causal_truth.value == pytest.approx(expected,abs=1e-11)
    assert sample.integration_uncertainty.converged
    assert sample.integration_uncertainty.assignment_mass_error <= 1e-10


def test_review_round2_exact_narrow_support_null_reproduction():
    f = replace(frame(1,1),columns=("x",),x=((0.,),),coordinates=((0.,0.),))
    sample = generate_suite_a(f,config("null",assignment="near_deterministic",near_scale=1e-100,noise_sd=0.),
                              policy(((9.8,10.),),delta=.1))
    assert sample.observed_law_truth.value == sample.structural_causal_truth.value == 0.


@pytest.mark.parametrize("lo,hi,coordinate,scale", _nonbinary_cutoff_cases()[::4])
def test_moved_quadrature_nodes_have_exact_supported_destinations(lo,hi,coordinate,scale):
    from fractions import Fraction
    from oxyformer.validation.scm import exact_shift_intervals
    f = replace(frame(1,1),coordinates=((coordinate,0.),))
    c = config(assignment="near_deterministic",near_scale=scale)
    law = AssignmentLaw(f,0,LatentState(),c,((lo,hi),))
    intervals = exact_shift_intervals(((lo,hi),),.1)
    for rule in law.quadrature(32,[bound for pair in intervals for bound in pair]):
        at = rule.coordinates
        for lower,upper in intervals:
            selected = at.inside(lower,upper)
            for offset in at.values[selected]:
                numerator,denominator = offset.as_integer_ratio()
                origin = at.anchor+at.unit*Fraction(numerator,denominator)
                assert Fraction(lo) <= origin <= Fraction(hi)
                assert Fraction(lo) <= origin+Fraction(.1) <= Fraction(hi)


def test_exact_width_controls_heterogeneous_eligibility_and_closed_endpoint():
    from fractions import Fraction
    from oxyformer.validation.scm import exact_shift_intervals
    assert exact_shift_intervals(((.1,1.),),.9) == ()
    assert exact_shift_intervals(((0.,1.),),1.) == ((Fraction(0),Fraction(0)),)
    endpoint = replace(frame(1,1),coordinates=((-800.,0.),))
    zero_mass = generate_suite_a(endpoint,config(assignment="near_deterministic",near_scale=1e-100),
                                 policy(((0.,1.),),delta=1.))
    assert zero_mass.observed_law_truth.value == zero_mass.structural_causal_truth.value == 0.
    f = replace(frame(2,1),support_keys=("s","wide"))
    p = replace(policy(delta=.9),components_by_key=(("s",((.1,1.),)),("wide",((0.,1.),))))
    result = generate_suite_a(f,config(heterogeneous_eligibility=True),p)
    expected = .5*.9*float(Fraction(1)-Fraction(.9))
    assert result.structural_causal_truth.value == pytest.approx(expected,abs=1e-12)


def test_count_reachable_intervals_use_exact_measurement_error_preimages(monkeypatch):
    from fractions import Fraction
    import oxyformer.validation.scm as scm
    components = ((.1,1.),(1.5,2.4))
    cfg = config(support_gaps=True,exposure_error=.3,registration_probability=.8)
    calls = []
    def record(a,f,row,state,c):
        calls.append(tuple(np.asarray(a)))
        return np.ones(len(a))
    monkeypatch.setattr(scm,"structural_mean",record)
    scm.validate_count_rates(frame(1,1),cfg,policy(components,delta=.9))
    expected = []
    delta = Fraction(.9)
    for error in (Fraction(-.3),Fraction(.3)):
        intervals = [(Fraction(lo),Fraction(hi)) for lo,hi in components]
        for lo,hi in components:
            for p_lo,p_hi in components:
                lower,upper = Fraction(p_lo),Fraction(p_hi)-delta
                if lower > upper:
                    continue
                start,end = max(Fraction(lo),lower-error),min(Fraction(hi),upper-error)
                if start <= end:
                    intervals.append((start+delta,end+delta))
        expected.extend(intervals)
    assert len(calls) == len(expected)
    for actual,reference in zip(calls,expected):
        # Comparison occurs only at the smooth-rate evaluation boundary.
        values = tuple(np.longdouble(str(v.numerator))/np.longdouble(str(v.denominator)) for v in reference)
        assert actual == values


@pytest.mark.parametrize("case", ["amplified_sign_change", "overflowing_null_support"])
def test_final_review_counterexamples_are_refused_before_computation(case, monkeypatch):
    import oxyformer.validation.generators as generators
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported inputs reached simulator computation")
    monkeypatch.setattr(generators, "_sample_observations", forbidden)
    monkeypatch.setattr(generators, "exact_shift_intervals", forbidden)
    f = replace(frame(1, 1), coordinates=((0., 0.),), columns=("x",), x=((0.,),))
    with pytest.raises(ContractError, match="outside supported numeric domain"):
        if case == "amplified_sign_change":
            generate_suite_a(f, config("sign_changing", beta=1e34,
                             assignment="near_deterministic", near_scale=1e-100, noise_sd=0.),
                             policy(delta=1e-16))
        else:
            generate_suite_a(f, config("null", noise_sd=0.), policy(((-1e308, 1e308),)))


@pytest.mark.parametrize("center", [5., 10000., -10000.])
@pytest.mark.parametrize("beta", [-200., 200.])
def test_supported_sign_changing_boundary_against_exact_reference(center, beta):
    from fractions import Fraction
    f = replace(frame(1, 1), coordinates=((0., 0.),), columns=("x",), x=((0.,),))
    delta = .02
    c = config("sign_changing", beta=beta, assignment="near_deterministic",
               near_scale=1e-100, noise_sd=0.)
    sample = generate_suite_a(f, c, policy(((center-5, center+5),), delta=delta))
    # The symmetric Laplace's mean displacement is zero; excluded tails are
    # exp(-5e100), far below this absolute tolerance. Independent exact algebra.
    expected = float(Fraction(beta)*(2*(Fraction(center)-5)*Fraction(delta)
                                     +Fraction(delta)**2)/10)
    assert sample.observed_law_truth.value == pytest.approx(expected, abs=1e-8, rel=0)
    assert sample.structural_causal_truth.value == pytest.approx(expected, abs=1e-8, rel=0)


def test_supported_null_support_boundary_returns_zero_truths():
    f = replace(frame(1, 1), coordinates=((0., 0.),))
    sample = generate_suite_a(f, config("null", noise_sd=0.), policy(((-10010., 10010.),)))
    assert sample.observed_law_truth.value == sample.structural_causal_truth.value == 0.


@pytest.mark.parametrize("name,value", [
    ("beta", 201.), ("local_strength", -201.), ("regional_strength", 201.),
    ("near_scale", 1001.), ("noise_sd", 101.), ("noise_sd", .001),
    ("exposure_error", 41.), ("exposure_error", .001),
    ("migration", 201.), ("migration", .001),
    ("registration_probability", .001), ("denominator_error", .001),
])
def test_each_mechanism_numeric_input_is_validated(name, value):
    with pytest.raises(ContractError, match="outside supported numeric domain"):
        config(**{name:value})


@pytest.mark.parametrize("field,value", [
    ("coordinates", ((1001., 0.),)), ("x", ((101., None),)),
    ("weights", (1.71e308,)),
])
def test_fixed_frame_numeric_inputs_are_validated(field, value):
    with pytest.raises(ContractError, match="outside supported numeric domain"):
        replace(frame(1, 1), **{field:value})


@pytest.mark.parametrize("delta", [1e-16, .019, 201.])
def test_policy_shift_domain_is_checked_before_sampling(delta, monkeypatch):
    import oxyformer.validation.generators as generators
    monkeypatch.setattr(generators, "_sample_observations",
                        lambda *args: pytest.fail("sampling preceded domain validation"))
    with pytest.raises(ContractError, match="delta outside supported numeric domain"):
        generate_suite_a(frame(), config(), policy(delta=delta))


def test_standalone_assignment_rejects_unsupported_support_before_center():
    with pytest.raises(ContractError, match="support endpoints outside supported numeric domain"):
        AssignmentLaw(frame(1, 1), 0, LatentState(), config("null"), ((-1e308, 1e308),))


def test_recipe_domain_matches_enforced_box_and_has_at_least_100x_margin(tmp_path):
    import yaml
    from oxyformer.validation.scm import REGISTERED_NUMERIC_BOX, NUMERIC_DOMAIN, SIGNED_QUANTITIES
    path = Path(__file__).parents[1]/"configs/validation/suite_a.yaml"
    recipe = yaml.safe_load(path.read_text())
    assert recipe["numeric_domain"]["registered_box"] == {k:list(v) for k,v in REGISTERED_NUMERIC_BOX.items()}
    for name, (lo, hi) in REGISTERED_NUMERIC_BOX.items():
        lower, upper = NUMERIC_DOMAIN[name]
        if name in SIGNED_QUANTITIES:
            assert lower <= -100*max(abs(lo),abs(hi))
            assert upper >= 100*max(abs(lo),abs(hi))
        else:
            assert 0 < lower <= lo/100
            assert upper >= hi*100
    # A caller cannot silently widen/disable the enforced bounds in a recipe.
    recipe["numeric_domain"]["supported_box"]["coefficient"] = [-1e100, 1e100]
    altered = tmp_path/"altered.yaml"
    altered.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ContractError, match="differs from enforced domain"):
        load_suite_a(altered)


@pytest.mark.parametrize("seed", [-1, 1.5, 2**64])
def test_seed_rejected_before_random_sampling(seed):
    with pytest.raises(ContractError, match="seed must be"):
        generate_suite_a(frame(), config(), policy(), seed=seed)


def test_wide_uniform_selected_truth_matches_logistic_antiderivative():
    intercept, slope = .7, .12
    f = replace(frame(2, 1), support_keys=("s", "stay"), columns=("x",), x=((0.,), (0.,)),
                coordinates=((0., 0.), (0., 0.)), region_ids=("r0", "r0"))
    p = replace(policy(), components_by_key=(("s", ((-10010., 10010.),)), ("stay", ((0., 1.),))))
    c = config(survey_inclusion=True, heterogeneous_eligibility=True)
    def selected_mass(lo, hi):
        return (np.logaddexp(0., intercept-slope*lo)
                -np.logaddexp(0., intercept-slope*hi))/(slope*(hi-lo))
    mass_wide = selected_mass(-10010., 10010.)
    mass_stay = selected_mass(0., 1.)
    moved_mass = (np.logaddexp(0., intercept+slope*10010.)
                  -np.logaddexp(0., intercept-slope*10008.))/(slope*20020.)
    expected = 2*moved_mass/(mass_wide+mass_stay)
    result = generate_suite_a(f, c, p)
    assert result.observed_law_truth.value == pytest.approx(expected, abs=1e-8, rel=0)
    assert result.structural_causal_truth.value == pytest.approx(expected, abs=1e-8, rel=0)
    assert result.integration_uncertainty.selected_mass_fraction == pytest.approx(
        (mass_wide+mass_stay)/2, abs=1e-10, rel=0)


def test_positive_count_rate_survives_exact_covariate_baseline_cancellation():
    from fractions import Fraction
    from oxyformer.validation.scm import structural_mean
    f = replace(frame(1, 1), columns=("x", "z", "tiny"), x=((-100., -100., 1e-14),),
                coordinates=((0., 0.),))
    c = config("null", local_confounding="measured", local_strength=1e-15,
               registration_probability=.65)
    for local in (-1., 1.):
        expected = float((Fraction(1e-14)/4+Fraction(local)*Fraction(1e-15))*Fraction(.65))
        mean = structural_mean([0., 10.], f, 0, LatentState(local=local), c)
        assert_allclose(mean, expected, rtol=1e-15, atol=0)
        assert (mean > 0).all()
    result = generate_suite_a(f, c, policy())
    assert result.observed_law_truth.value == result.structural_causal_truth.value == 0.


def test_round2_positive_count_law_survives_quadratic_cross_term_cancellation():
    from fractions import Fraction
    square = (Fraction(-.1)-5)**2
    residual = 4*square-100
    parts = []
    for _ in range(3):
        parts.append(float(residual))
        residual -= Fraction(parts[-1])
    assert residual == 0
    x = (-100., -100., 100., *parts, 4e-20)
    assert 50+sum(map(Fraction, x))/4-square == Fraction(4e-20)/4 > 0
    f = replace(frame(1, 1), coordinates=((0., 0.),),
                columns=tuple(f"x{i}" for i in range(len(x))), x=(x,))
    c = config("sign_changing", beta=-10., registration_probability=.65, noise_sd=0.)
    result = generate_suite_a(f, c, policy(((-.1, 0.),), delta=.02))
    assert result.observations.registered_events[0] >= 0
    assert result.integration_uncertainty.converged


def _count_frame_with_exact_baseline(value):
    """Encode a dyadic test baseline in supported stored float covariates."""
    from fractions import Fraction
    residual, values = 4*value, [-100., -100.]
    while residual:
        part = float(max(Fraction(-100), min(Fraction(100), residual)))
        assert part != 0
        values.append(part)
        residual -= Fraction(part)
    return replace(frame(1, 1), coordinates=((0., 0.),),
                   columns=tuple(f"x{i}" for i in range(len(values))), x=(tuple(values),))


@pytest.mark.parametrize("margin", [-1e-20, 0., 1e-20])
def test_exact_count_quadratic_boundary_accepts_zero_and_rejects_negative(margin):
    from fractions import Fraction
    from oxyformer.validation.scm import count_event_rate, validate_count_rates
    baseline = (Fraction(-.1)-5)**2+Fraction(margin)
    f = _count_frame_with_exact_baseline(baseline)
    c = config("sign_changing", beta=-10., registration_probability=.65)
    if margin < 0:
        with pytest.raises(ContractError, match="negative event rate"):
            validate_count_rates(f, c, policy(((-.1, 0.),), delta=.02))
        with pytest.raises(ContractError, match="negative event rate"):
            count_event_rate(Fraction(-.1), f, 0, LatentState(), c)
    else:
        validate_count_rates(f, c, policy(((-.1, 0.),), delta=.02))
        assert count_event_rate(Fraction(-.1), f, 0, LatentState(), c) == margin


@pytest.mark.parametrize("effect,beta,dose", [("null", 1., 0.), ("linear", -1., .1),
                                               ("sign_changing", 10., 5.)])
@pytest.mark.parametrize("margin", [-1e-20, 0., 1e-20])
def test_count_rate_combines_all_stored_affine_and_polynomial_terms(effect, beta, dose, margin):
    from fractions import Fraction
    from oxyformer.validation.scm import count_event_rate
    state = LatentState(local=-1., regional=1., illness=1., denominator_factor=1.2)
    c = config(effect, beta=beta, local_confounding="omitted", local_strength=.1,
               regional_confounding="omitted", regional_strength=.2, migration=.3,
               registration_probability=.65, denominator_error=.2)
    adjusted = Fraction(dose)-Fraction(.3)
    response = {"null": Fraction(0), "linear": Fraction(beta)*adjusted,
                "sign_changing": Fraction(beta)*(adjusted-5)**2/10}[effect]
    # /10 may introduce a factor five; use beta=10 for the quadratic so this
    # stored baseline is dyadic, just like the finite input representation.
    baseline = Fraction(margin)-response+Fraction(.1)-Fraction(.2)-2
    f = _count_frame_with_exact_baseline(baseline)
    if margin < 0:
        with pytest.raises(ContractError, match="negative event rate"):
            count_event_rate(dose, f, 0, state, c)
    else:
        assert count_event_rate(dose, f, 0, state, c, poisson_intensity=True) == float(100*Fraction(margin))


def _decimal_sine(value):
    # Independent direct Taylor evaluation at 140 decimal digits; test arguments
    # are at most 1/2, with a final term far below the tested sign margins.
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = 140
        x = Decimal(value.numerator)/Decimal(value.denominator)
        term = total = x
        for n in range(1, 100):
            term *= -x*x/Decimal((2*n)*(2*n+1))
            total += term
        return +total


@pytest.mark.parametrize("margin", [-1e-20, 1e-20])
def test_sine_count_endpoint_sign_uses_enclosure_before_rounding(margin):
    from fractions import Fraction
    from oxyformer.validation.scm import count_event_rate, validate_count_rates
    sine = _decimal_sine(Fraction(1, 2))
    # Store four float expansion terms, accurately enough that the independent
    # reference differs from the mathematical sine by less than 1e-60.
    remainder, terms = Fraction(sine), []
    for _ in range(4):
        terms.append(float(remainder))
        remainder -= Fraction(terms[-1])
    assert abs(remainder) < Fraction(1, 10**60)
    baseline = sum(map(Fraction, terms))+Fraction(margin)
    f = _count_frame_with_exact_baseline(baseline)
    c = config("nonlinear", beta=-1., registration_probability=.65)
    if margin < 0:
        with pytest.raises(ContractError, match="negative event rate"):
            validate_count_rates(f, c, policy(((0., 1.),), delta=.02))
        with pytest.raises(ContractError, match="negative event rate"):
            count_event_rate(1., f, 0, LatentState(), c)
    else:
        validate_count_rates(f, c, policy(((0., 1.),), delta=.02))
        reference = float(baseline-Fraction(sine))
        assert count_event_rate(1., f, 0, LatentState(), c) == pytest.approx(reference, rel=1e-15, abs=0)


@pytest.mark.parametrize("beta", [-1., 1.])
@pytest.mark.parametrize("margin", [-1e-20, 0., 1e-20])
def test_sine_count_interior_extrema_are_exact(beta, margin):
    from fractions import Fraction
    from oxyformer.validation.scm import validate_count_rates
    f = _count_frame_with_exact_baseline(1+Fraction(margin))
    c = config("nonlinear", beta=beta, registration_probability=.65)
    p = policy(((0., 10.),), delta=.02)
    if margin < 0:
        with pytest.raises(ContractError, match="negative event rate"):
            validate_count_rates(f, c, p)
    else:
        validate_count_rates(f, c, p)


def test_sine_phase_membership_respects_exact_endpoints_and_periods():
    from fractions import Fraction
    from oxyformer.validation.scm import _contains_sine_minimum, _pi_bounds, count_event_rate
    # Rational enclosure from an independent decimal expansion, separate from the
    # Machin identity implementation. Exercise both signs and remote periods.
    pi_lo = Fraction("3.14159265358979323846264338327950288419716939937510")
    pi_hi = pi_lo+Fraction(1, 10**50)
    lo, hi = _pi_bounds(192)
    assert pi_lo < lo < hi < pi_hi
    for q in (-3185, -1, 1, 3185):
        a, b = sorted((q*pi_lo, q*pi_hi))
        beta = -1 if q % 4 == 1 else 1
        assert _contains_sine_minimum(a, b, beta)
        assert not _contains_sine_minimum(a-Fraction(1, 100), a, beta)
        assert not _contains_sine_minimum(b, b+Fraction(1, 100), beta)
    # A rounded pi is rational, so 1-sin(pi_float/2) is strictly positive;
    # a float/libm response would commonly collapse it to zero.
    f = _count_frame_with_exact_baseline(Fraction(1))
    c = config("nonlinear", beta=-1., registration_probability=.65)
    rate = count_event_rate(np.pi, f, 0, LatentState(), c)
    assert 1e-34 < rate < 1e-32


@pytest.mark.parametrize("value", [-10010., -1., 0., 1., 10010.])
def test_certified_sine_bounds_agree_with_independent_small_angle_reference(value):
    from fractions import Fraction
    from oxyformer.validation.scm import _sine_bounds
    # Double-angle independent decimal complex multiplication after reducing
    # the argument. It does not use the production integer interval arithmetic.
    from decimal import Decimal, localcontext
    x = Fraction(value)
    halves = 0
    while abs(x) > Fraction(1, 2):
        x /= 2
        halves += 1
    with localcontext() as context:
        context.prec = 140
        sine = _decimal_sine(x)
        cosine = (1-sine*sine).sqrt()  # positive on this small-angle interval
        for _ in range(halves):
            sine, cosine = 2*sine*cosine, cosine*cosine-sine*sine
        reference = Fraction(sine)
    lo, hi = _sine_bounds(Fraction(value), 192)
    assert lo <= reference <= hi
    assert hi-lo < Fraction(1, 10**50)


@pytest.mark.parametrize("error", [0., .4])
def test_count_sampler_retains_local_dose_and_scales_after_exact_cancellation(error, monkeypatch):
    from fractions import Fraction
    from oxyformer.validation.scm import LocalCoordinates
    import oxyformer.validation.generators as generators
    original_rng = np.random.default_rng
    intensities = []
    class RecordingRNG:
        def __init__(self, seed):
            self.delegate = original_rng(seed)
        def __getattr__(self, name):
            return getattr(self.delegate, name)
        def poisson(self, intensity):
            intensities.append(intensity)
            return 0
    def fixed_local_draw(self, index, u):
        return LocalCoordinates(Fraction(5)+self.error, Fraction(1e-100), np.asarray(.75, dtype=np.longdouble))
    monkeypatch.setattr(generators.np.random, "default_rng", RecordingRNG)
    monkeypatch.setattr(AssignmentLaw, "quantile_coordinates", fixed_local_draw)
    f = _count_frame_with_exact_baseline(Fraction(0))
    c = config("sign_changing", beta=10., exposure_error=error,
               registration_probability=.65, denominator_error=.2)
    observations = generators._sample_observations(f, c, policy(), 42)
    expected = float(100*(Fraction(1e-100)*Fraction(3, 4))**2)
    assert intensities == [expected]
    assert expected > 0
    assert observations.a[0] in (float(Fraction(5)+Fraction(error)), float(Fraction(5)-Fraction(error)))


@pytest.mark.parametrize("near_scale", [5e-324, 1e-100, .05, 1000.])
@pytest.mark.parametrize("coordinate", [-1000., 0., 1000.])
def test_count_draw_keeps_exact_support_at_domain_scale_extremes(near_scale, coordinate):
    from fractions import Fraction
    f = replace(frame(1, 1), coordinates=((coordinate, 0.),))
    c = config("null", assignment="near_deterministic", near_scale=near_scale,
               registration_probability=.65, exposure_error=.4)
    law = AssignmentLaw(f, 0, LatentState(error=.4), c, ((-.1, 0.),))
    class EndpointRNG:
        def __init__(self, index, u):
            self.index, self.u = index, u
        def choice(self, *args, **kwargs):
            return self.index
        def random(self):
            return self.u
    for index in range(len(law.pieces)):
        for u in (0., np.nextafter(0., 1.), .5, np.nextafter(1., 0.)):
            observed, true = law.sample_count_dose(EndpointRNG(index, u))
            assert Fraction(-.1) <= true <= 0
            assert observed == float(true+Fraction(.4))


@pytest.mark.parametrize("error", [0., .4])
def test_count_atoms_retain_exact_boundary_and_expected_design_rejection(error):
    from fractions import Fraction
    sample = generate_suite_a(frame(200), config("null", assignment="atoms",
                             registration_probability=.65, exposure_error=error), policy(), seed=7)
    boundaries = (float(Fraction(error)), float(-Fraction(error)))
    assert sum(a in boundaries for a in sample.observations.a) >= 50
    assert any(a not in boundaries for a in sample.observations.a)
    assert sample.observed_law_truth.status == sample.structural_causal_truth.status == "design_rejected"
    assert sample.observed_law_truth.value is sample.structural_causal_truth.value is None


@pytest.mark.parametrize("error", [np.nextafter(1., 0.), np.nextafter(np.nextafter(1., 0.), 0.)])
def test_round3_denominator_factors_stay_exact_inside_open_bounds(error):
    from fractions import Fraction
    from oxyformer.validation.scm import latent_states
    c = config("null", denominator_error=float(error))
    expected = {1-Fraction(float(error)), 1+Fraction(float(error))}
    assert all(0 < factor < 2 for factor in expected)
    assert {state.denominator_factor for state, _ in latent_states(c)} == expected
    sample = generate_suite_a(frame(8, 1), c, policy(), seed=0)
    assert sample.observed_law_truth.value == sample.structural_causal_truth.value == 0.
    assert set(sample.observations.observed_denominator) == {float(100*factor) for factor in expected}
    for value, count, denominator in zip(sample.observations.y, sample.observations.registered_events,
                                         sample.observations.observed_denominator):
        assert value == count/denominator


def test_local_membership_does_not_round_exact_bound_onto_a_node():
    from fractions import Fraction
    from oxyformer.validation.scm import LocalCoordinates
    one = np.longdouble(1)
    nodes = np.array([np.nextafter(one, -np.inf), one, np.nextafter(one, np.inf)])
    points = LocalCoordinates(Fraction(0), Fraction(1), nodes)
    tiny = Fraction(1, 2**100)
    assert points.inside(1+tiny, 2).tolist() == [False, False, True]
    assert points.inside(0, 1-tiny).tolist() == [True, False, False]


def test_recorded_support_membership_does_not_round_error_preimage():
    from fractions import Fraction
    c = config(exposure_error=.2)
    law = AssignmentLaw(frame(1, 1), 0, LatentState(error=-.2), c, ((-20., -10.),))
    boundary = float(Fraction(-10)-Fraction(.2))
    values = np.array([np.nextafter(boundary, -np.inf), boundary, np.nextafter(boundary, np.inf)])
    expected = [Fraction(-20) <= Fraction(float(v))+Fraction(.2) <= Fraction(-10) for v in values]
    assert law.contains(values).tolist() == expected
    assert np.isfinite(law.log_density(values)).tolist() == expected


from oxyformer.validation.scm import NUMERIC_DOMAIN as _DOMAIN_EDGES


@pytest.mark.parametrize("kind", tuple(_DOMAIN_EDGES))
@pytest.mark.parametrize("edge_index", [0, 1])
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_every_declared_numeric_domain_edge_and_nextafter(kind, edge_index, direction):
    from oxyformer.validation.scm import validate_numeric
    lower, upper = _DOMAIN_EDGES[kind]
    edge = (lower, upper)[edge_index]
    value = edge if direction == 0 else float(np.nextafter(edge, direction*np.inf))
    if lower <= value <= upper:
        validate_numeric(value, kind, kind)
    else:
        with pytest.raises(ContractError, match="outside supported numeric domain"):
            validate_numeric(value, kind, kind)


@pytest.mark.parametrize("name,edge", [
    ("denominator_error", 0.), ("denominator_error", .002), ("denominator_error", 1.),
    ("registration_probability", _DOMAIN_EDGES["registration_probability"][0]),
    ("registration_probability", 1.),
    ("near_scale", _DOMAIN_EDGES["near_scale"][0]), ("near_scale", 1000.),
    ("exposure_error", 0.), ("exposure_error", .004), ("exposure_error", 40.),
    ("migration", 0.), ("migration", .02), ("migration", 200.),
    ("noise_sd", 0.), ("noise_sd", .01), ("noise_sd", 100.),
    ("beta", -200.), ("beta", 0.), ("beta", 200.),
    ("local_strength", -200.), ("local_strength", 200.),
    ("regional_strength", -200.), ("regional_strength", 200.),
])
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_mechanism_domain_edges_reach_latent_and_assignment_checks(name, edge, direction):
    from fractions import Fraction
    from oxyformer.validation.scm import latent_states
    value = float(edge if direction == 0 else np.nextafter(edge, direction*np.inf))
    kind = "coefficient" if name in ("beta", "local_strength", "regional_strength") else name
    lower, upper = _DOMAIN_EDGES[kind]
    valid = lower <= value <= upper or (value == 0 and name not in ("near_scale", "registration_probability"))
    valid &= name != "denominator_error" or value < 1
    valid &= name != "registration_probability" or value <= 1
    if not valid:
        with pytest.raises(ContractError):
            config("null", **{name:value})
        return
    c = config("null", **{name:value})
    states = list(latent_states(c))
    assert sum(probability for _, probability in states) == 1
    for state, _ in states:
        assert 0 < state.denominator_factor < 2
        assert state.denominator_factor in {1-Fraction(c.denominator_error), 1+Fraction(c.denominator_error)}
        law = AssignmentLaw(frame(1, 1), 0, state, c, ((0., 10.),))
        assert np.isfinite(law.probabilities).all()
        assert sum(law.probabilities) == pytest.approx(1.)


@pytest.mark.parametrize("edge", [-10010., 10010.])
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_support_domain_edges_before_derived_geometry(edge, direction):
    from oxyformer.validation.scm import exact_shift_intervals
    value = float(edge if direction == 0 else np.nextafter(edge, direction*np.inf))
    components = ((value, value+1),) if edge < 0 else ((value-1, value),)
    if not -10010 <= value <= 10010:
        with pytest.raises(ContractError, match="supported numeric domain"):
            exact_shift_intervals(components, .02)
        return
    c = config("null", assignment="near_deterministic", near_scale=5e-324)
    result = generate_suite_a(frame(1, 1), c, policy(components, delta=.02))
    assert result.observed_law_truth.value == result.structural_causal_truth.value == 0.


@pytest.mark.parametrize("edge", [0., .02, 200.])
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_shift_domain_edges_before_exact_width_and_cutoff(edge, direction):
    from fractions import Fraction
    from oxyformer.validation.scm import exact_shift_intervals
    value = float(edge if direction == 0 else np.nextafter(edge, direction*np.inf))
    if value != 0 and not .02 <= value <= 200:
        with pytest.raises(ContractError, match="supported numeric domain"):
            exact_shift_intervals(((0., 200.),), value)
    else:
        expected = () if value == 0 else ((Fraction(0), Fraction(200)-Fraction(value)),)
        assert exact_shift_intervals(((0., 200.),), value) == expected


@pytest.mark.parametrize("edge", [0., 2.])
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_latent_denominator_open_bound_and_nextafter(edge, direction):
    value = float(edge if direction == 0 else np.nextafter(edge, direction*np.inf))
    if 0 < value < 2:
        assert LatentState(denominator_factor=value).denominator_factor == value
    else:
        with pytest.raises(ContractError, match="invalid latent denominator factor"):
            LatentState(denominator_factor=value)


@pytest.mark.parametrize("rate", [-4., 0., 4.])
def test_inverse_transform_domain_endpoints_stay_in_exact_support(rate):
    from fractions import Fraction
    from oxyformer.validation.scm import _ExponentialPiece, exact
    c = config()
    law = AssignmentLaw(frame(1, 1), 0, LatentState(), c, ((-10010., 10010.),))
    law.pieces = (_ExponentialPiece(Fraction(-10010), Fraction(10010), Fraction(rate)),)
    # The RNG's actual domain is [0, 1); these are both represented edges and
    # their in-domain neighbours, including the smallest positive float64.
    for u in (0., np.nextafter(0., 1.), np.nextafter(np.nextafter(1., 0.), 0.), np.nextafter(1., 0.)):
        at = law.quantile_coordinates(0, u)
        dose = at.anchor+at.unit*exact(np.longdouble(at.values))
        assert Fraction(-10010) <= dose <= Fraction(10010)


@pytest.mark.parametrize("local,regional,illness", [(-1., 0., 1.), (1., -1., 1.), (0., -1., 1.)])
def test_assignment_affine_rate_is_exact_before_slope_sign(local, regional, illness):
    from fractions import Fraction
    state = LatentState(local=local, regional=regional, illness=illness)
    law = AssignmentLaw(frame(1, 1), 0, state, config(), ((0., 10.),))
    assert law.rate == Fraction(.3)*Fraction(local)+Fraction(.2)*Fraction(regional)-Fraction(.25)*Fraction(illness)


@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_assignment_combined_slope_crosses_zero_exactly(direction):
    from fractions import Fraction
    scale = float(.25 if direction == 0 else np.nextafter(.25, direction*np.inf))
    c = config(assignment="near_deterministic", near_scale=scale, extreme_ratios=True)
    law = AssignmentLaw(frame(1, 1), 0, LatentState(), c, ((0., 10.),))
    slope = Fraction(-4)+1/Fraction(scale)
    assert law.pieces[0].rate == slope
    assert (slope > 0)-(slope < 0) == -direction


@pytest.mark.parametrize("beta", [1e-20, -1e-20])
def test_round1_tiny_effect_survives_baseline_and_denominator_amplification(beta):
    from fractions import Fraction
    error = .9999999999999999
    c = config("linear", beta=beta, denominator_error=error)
    sample = generate_suite_a(frame(1, 1), c, policy())
    expected = float(Fraction(4, 5)*Fraction(beta)*(1/(1-Fraction(error))+1/(1+Fraction(error))))
    assert abs(expected) > 1e-5
    assert sample.observed_law_truth.value == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert sample.structural_causal_truth.value == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("true_dose", [1., 1.25])
def test_round1_noncount_sampler_keeps_true_dose_before_error_serialization(monkeypatch, true_dose):
    from fractions import Fraction
    from oxyformer.validation.scm import LocalCoordinates
    from oxyformer.validation.generators import _sample_observations
    def fixed_draw(self, index, u):
        return LocalCoordinates(Fraction(true_dose)+self.error, Fraction(1), np.asarray(0., dtype=np.longdouble))
    monkeypatch.setattr(AssignmentLaw, "quantile_coordinates", fixed_draw)
    observations = _sample_observations(frame(8, 1), config(beta=200., exposure_error=.4, noise_sd=0.), policy(), 0)
    assert float(Fraction(true_dose)+Fraction(.4)) in observations.a
    assert observations.y == (float(50+200*Fraction(true_dose)),)*8


@pytest.mark.parametrize("family", ["linear", "nonlinear", "sign_changing"])
@pytest.mark.parametrize("beta", [-1e-20, 1e-20])
@pytest.mark.parametrize("registration", [.006500000000000001, .65, 1.])
def test_amplified_small_effect_families_match_independent_contrasts(family, beta, registration):
    from fractions import Fraction
    error = .9999999999999999
    functions = {"linear": lambda a: a, "nonlinear": lambda a: np.sin(a/2),
                 "sign_changing": lambda a: (a-5)**2/10}
    unit = functions[family]
    integral = quad(lambda a: (unit(a+2)-unit(a))/10, 0, 8, epsabs=1e-13)[0]
    multiplier = Fraction(beta)*Fraction(registration)/(1-Fraction(error)**2)
    expected = float(multiplier)*integral
    result = generate_suite_a(frame(1, 1), config(family, beta=beta, denominator_error=error,
                              registration_probability=registration), policy())
    assert result.observed_law_truth.value == pytest.approx(expected, abs=1e-12, rel=1e-12)
    assert result.structural_causal_truth.value == pytest.approx(expected, abs=1e-12, rel=1e-12)


def test_orchestrator_near_center_zero_truth_regression():
    from fractions import Fraction
    error, beta = .9999999999999999, 1e-20
    result = generate_suite_a(frame(1, 1), config('sign_changing', beta=beta,
                              assignment='near_deterministic', near_scale=1e-100,
                              denominator_error=error), policy())
    expected = float(Fraction(beta)*Fraction(2, 5)/(1-Fraction(error)**2))
    # The symmetric law is centered at five. Omitted shifted-tail mass is
    # exp(-3e100); even the maximum bounded quadratic contrast makes its
    # contribution negligible relative to 1e-8 (and to the tighter check here).
    assert expected == pytest.approx(.000018014398509481982, rel=1e-15)
    assert result.observed_law_truth.value == pytest.approx(expected, abs=1e-12, rel=0)
    assert result.structural_causal_truth.value == pytest.approx(expected, abs=1e-12, rel=0)


def test_orchestrator_large_center_amplified_quadratic_contrast():
    from fractions import Fraction
    error, delta = .9999999999999999, .02
    result = generate_suite_a(frame(1, 1), config('sign_changing', beta=200.,
                              assignment='near_deterministic', near_scale=1e-100,
                              denominator_error=error), policy(((9995., 10005.),), delta=delta),
                              tolerance=10000.)
    d = Fraction(delta)
    expected = float(20*(19990*d+d*d)/(1-Fraction(error)**2))
    # The mean local offset is zero and the nonshifted tail is exp(-4.98e100).
    assert abs(result.observed_law_truth.value-expected) <= 10000.
    assert abs(result.structural_causal_truth.value-expected) <= 10000.


@pytest.mark.parametrize('regional', [False, True])
def test_independent_denominator_mixture_cannot_invent_a_null_contrast(regional):
    c = config('null', denominator_error=.9999999999999999, local_confounding='omitted',
               local_strength=0., regional_confounding='omitted' if regional else 'none',
               regional_strength=0.)
    result = generate_suite_a(frame(1, 1), c, policy())
    assert result.observed_law_truth.value == result.structural_causal_truth.value == 0.


def test_denominator_exact_product_serialization_avoids_double_rounding():
    from fractions import Fraction
    error = float(Fraction(2**52+2**48+11305, 2**61))
    result = generate_suite_a(frame(8, 1), config('null', denominator_error=error), policy(), seed=0)
    expected = {float(100*(1-Fraction(error))), float(100*(1+Fraction(error)))}
    assert float(100*(1+Fraction(error))).hex() == '0x1.90d4800000023p+6'
    assert set(result.observations.observed_denominator) == expected
    for y, events, denominator in zip(result.observations.y, result.observations.registered_events,
                                      result.observations.observed_denominator):
        assert y == events/denominator


def test_amplified_sine_contrast_at_phase_boundary_has_independent_decimal_reference():
    from fractions import Fraction
    from decimal import localcontext
    error, delta = .9999999999999999, .02
    requested = float(np.pi-.01)
    lo, hi = requested-.1, requested+.1
    center = lo+(hi-lo)*.5  # The inherited declared float center.
    with localcontext() as context:
        context.prec = 140
        difference = _decimal_sine((Fraction(center)+Fraction(delta))/2)-_decimal_sine(Fraction(center)/2)
        expected = float(Fraction(difference)*200/(1-Fraction(error)**2))
    result = generate_suite_a(frame(1, 1), config('nonlinear', beta=200.,
                              assignment='near_deterministic', near_scale=1e-100,
                              denominator_error=error), policy(((lo, hi),), delta=delta))
    assert result.observed_law_truth.value == pytest.approx(expected, abs=1e-8, rel=0)
    assert result.structural_causal_truth.value == pytest.approx(expected, abs=1e-8, rel=0)


def _decimal_sin_cos_reduced(value):
    from decimal import Decimal, localcontext
    from fractions import Fraction
    with localcontext() as context:
        context.prec = 140
        halves = 0
        while abs(value) > Fraction(1, 2):
            value /= 2
            halves += 1
        sine = _decimal_sine(value)
        cosine = (Decimal(1)-sine*sine).sqrt()
        for _ in range(halves):
            sine, cosine = 2*sine*cosine, cosine*cosine-sine*sine
        return sine, cosine


def test_round2_sine_local_variation_survives_large_anchor_amplification():
    from fractions import Fraction
    from decimal import Decimal, localcontext
    center = float(np.pi-.5+2*np.pi*1591)
    scale, error = 1e-16, .9999999999999999
    f = replace(frame(1, 1), columns=tuple(f'x{i}' for i in range(10)), x=((100.,)*10,))
    with localcontext() as context:
        context.prec = 140
        sine, cosine = _decimal_sin_cos_reduced(Fraction(center)/2+Fraction(1, 4))
        quarter_sine = _decimal_sine(Fraction(1, 4))
        ratio = Fraction(scale)/2
        b = Decimal(ratio.numerator)/Decimal(ratio.denominator)
        # Exact one-sided Laplace transform. Truncation error exp(-1e16) is
        # negligible even after the <1e18 endpoint multiplier.
        integral = quarter_sine*(cosine+b*sine)/(1+b*b)
        expected = float(200*Fraction(integral)/(1-Fraction(error)**2))
    result = generate_suite_a(f, config('nonlinear', beta=200., assignment='near_deterministic',
                              near_scale=scale, denominator_error=error),
                              policy(((center-1, center+1),), delta=1.))
    assert result.observed_law_truth.value == pytest.approx(expected, abs=1e-8, rel=0)
    assert result.structural_causal_truth.value == pytest.approx(expected, abs=1e-8, rel=0)


def test_round2_weighted_null_posterior_has_exact_zero_truth():
    f = replace(frame(2, 1), weights=(1., 1e-22), coordinates=((0., 0.), (float(np.log(4)), 0.)))
    c = config('null', assignment='near_deterministic', near_scale=1., denominator_error=.9999999999999999)
    result = generate_suite_a(f, c, policy())
    assert result.observed_law_truth.value == result.structural_causal_truth.value == 0.


def test_round2_large_center_linear_contrast_uses_exact_shift():
    from fractions import Fraction
    error, delta = .9999999999999999, .02
    result = generate_suite_a(frame(1, 1), config(beta=1., assignment='near_deterministic',
                              near_scale=1e-100, denominator_error=error),
                              policy(((9995., 10005.),), delta=delta), tolerance=.1, max_order=32)
    expected = float(Fraction(delta)/(1-Fraction(error)**2))
    assert result.observed_law_truth.value == pytest.approx(expected, abs=.1, rel=0)
    assert result.structural_causal_truth.value == pytest.approx(expected, abs=.1, rel=0)


def test_rational_conversion_rounds_once_without_decimal_integer_limits():
    from fractions import Fraction
    from oxyformer.validation.scm import wide
    one = np.longdouble(1)
    neighbour = np.nextafter(one, np.longdouble(np.inf))
    middle = (Fraction(*one.as_integer_ratio())+Fraction(*neighbour.as_integer_ratio()))/2
    epsilon = Fraction(1, 2**20000)
    assert wide(middle-epsilon) == one
    assert wide(middle+epsilon) == neighbour
    assert wide(-middle+epsilon) == -one
    assert wide(-middle-epsilon) == -neighbour


def test_smooth_response_retains_the_supplied_extended_dose_precision():
    from fractions import Fraction
    from oxyformer.validation.scm import effect, wide
    values = np.array([np.longdouble(10000), np.longdouble(10000)+np.longdouble(.02)])
    a, b = (Fraction(*v.as_integer_ratio()) for v in values)
    expected = wide(20*((b-5)**2-(a-5)**2))
    response = effect(values, config('sign_changing', beta=200.))
    assert abs(response[1]-response[0]-expected) < 1e-8


def test_rational_conversion_ties_and_subnormal_spacing():
    from fractions import Fraction
    from oxyformer.validation.scm import wide
    tiny = np.nextafter(np.longdouble(0), np.longdouble(1))
    unit = Fraction(*tiny.as_integer_ratio())
    assert wide(unit/2) == 0
    assert wide(unit*Fraction(3, 2)) == 2*tiny
    assert wide(unit*Fraction(5, 2)) == 2*tiny
    one = Fraction(1)
    spacing = Fraction(*np.nextafter(np.longdouble(1), np.longdouble(2)).as_integer_ratio())-one
    assert wide(one+spacing/2) == 1
    assert wide(one+3*spacing/2) == np.longdouble(1)+2*wide(spacing)


def _decimal_laplace_policy_reference(lo, hi, center, scale, delta, beta, error, effect):
    # Independent exact-input, high-precision truncated Laplace antiderivative.
    from decimal import Decimal, localcontext
    from fractions import Fraction
    with localcontext() as context:
        context.prec = 160
        def dec(v):
            v = Fraction(v)
            return Decimal(v.numerator)/Decimal(v.denominator)
        L,U,C,S,d,B = map(dec, (lo,hi,center,scale,delta,beta))
        cutoff = U-d
        assert cutoff <= C
        normalizer = 2-((L-C)/S).exp()-((C-U)/S).exp()
        if effect == 'linear':
            integral = B*d*((((cutoff-L)/S).exp()-1)*((L-C)/S).exp())
        else:
            m,k = B/10*2*d, B/10*(d*d-10*d)
            primitive = lambda a: ((a-C)/S).exp()*(m*(a-S)+k)
            integral = primitive(cutoff)-primitive(L)
        return float(integral/normalizer/dec(1-Fraction(error)**2))


@pytest.mark.parametrize("decays", [60., 64., 65., 80.])
def test_continuation_resolves_amplified_exponential_tail(decays):
    lo,hi,delta,error = 9590.,10010.,200.,.9999999999999999
    center = lo+(hi-lo)*float(expit(.1))
    scale = (center-(hi-delta))/decays
    f = replace(frame(1,1), columns=('x',), x=((0.,),), coordinates=((.1,0.),))
    expected = _decimal_laplace_policy_reference(lo,hi,center,scale,delta,200.,error,'sign_changing')
    result = generate_suite_a(f, config('sign_changing', beta=200., assignment='near_deterministic',
                              near_scale=scale, denominator_error=error), policy(((lo,hi),),delta=delta))
    for truth in (result.observed_law_truth, result.structural_causal_truth):
        assert truth.value == pytest.approx(expected, abs=1e-8, rel=0)


@pytest.mark.parametrize("edge", [-2.**-65, float(np.nextafter(-2.**-65,-np.inf)),
                                  float(np.nextafter(-2.**-65,np.inf))])
def test_continuation_retains_narrow_positive_eligibility_panel(edge):
    hi,delta,scale,error = 1.,1.,.05,.9999999999999999
    center = edge+(hi-edge)*.5
    f = replace(frame(1,1), columns=('x',), x=((0.,),), coordinates=((0.,0.),))
    expected = _decimal_laplace_policy_reference(edge,hi,center,scale,delta,200.,error,'linear')
    result = generate_suite_a(f, config('linear', beta=200., assignment='near_deterministic',
                              near_scale=scale, denominator_error=error), policy(((edge,hi),),delta=delta))
    assert expected > 1e-5
    for truth in (result.observed_law_truth, result.structural_causal_truth):
        assert truth.value == pytest.approx(expected, abs=1e-12, rel=0)


@pytest.mark.parametrize("local", [-1, 0, 1])
@pytest.mark.parametrize("lower", [-2.**-65, float(np.nextafter(-2.**-65,-np.inf)),
                                   float(np.nextafter(-2.**-65,np.inf)), -5e-324])
def test_continuation_quadrature_preserves_positive_panel_mass(local, lower):
    from decimal import Decimal, localcontext
    from fractions import Fraction
    from scipy.special import logsumexp
    from oxyformer.validation.scm import exact, wide
    law = AssignmentLaw(frame(1,1), 0, LatentState(local=local), config(), ((lower,1.),))
    logs = []
    for rule in law.quadrature(32, [exact(0)]):
        coordinates = rule.coordinates
        mask = coordinates.inside(exact(lower), exact(0))
        if mask.any():
            assert mask.all()  # No quadrature panel crosses the exact boundary.
            assert all(exact(lower) < coordinates.anchor+coordinates.unit*exact(v) < 0
                       for v in coordinates.values)
            logs.extend(rule.log_weights)
    assert logs
    with localcontext() as context:
        context.prec = 800  # Retain cancellation at the subnormal support edge.
        def dec(v):
            v = Fraction(v)
            return Decimal(v.numerator)/Decimal(v.denominator)
        L,rate = dec(lower), dec(Fraction(.3)*local)
        if local:
            expected = (1-(rate*L).exp())/((rate).exp()-(rate*L).exp())
        else:
            expected = -L/(1-L)
        expected = wide(Fraction(expected))
    actual = np.exp(logsumexp(np.asarray(logs, dtype=np.longdouble)))
    assert abs(actual/expected-1) < 1e-12


@pytest.mark.parametrize("tolerance", [1e-5, 1e-8, 1e-12])
@pytest.mark.parametrize("selected", [False, True])
def test_continuation_tail_budget_includes_response_and_selected_normalization(tolerance, selected):
    from oxyformer.validation.generators import _groups, _quadrature_tail_budget
    c = config('linear', beta=200., assignment='near_deterministic', near_scale=.05,
               denominator_error=.9999999999999999, selected_outcome=selected,
               survey_inclusion=selected, missing_biomarkers=selected)
    f,p = frame(1,1),policy(((9995.,10005.),))
    groups = _groups(f,c,p)
    bound = _quadrature_tail_budget(f,c,p,groups,tolerance)
    assert 0 < bound < tolerance/16
    cutoffs = [float(term.law.tail_decay) for group in groups.values() for term in group]
    assert min(cutoffs) > (4200 if selected else 60)


@pytest.mark.parametrize("coordinate", [-1000., float(np.nextafter(-1000.,np.inf))])
def test_continuation_round1_amplified_selection_contrast(coordinate):
    from decimal import Decimal, localcontext
    from fractions import Fraction
    error = .9999999999999999
    f = replace(frame(1,1), columns=tuple(f'x{i}' for i in range(6)),
                x=((100.,)*6,), coordinates=((coordinate,0.),))
    c = config('null', local_confounding='omitted', local_strength=200.,
               assignment='near_deterministic', near_scale=1e-100,
               survey_inclusion=True, denominator_error=error)
    with localcontext() as context:
        context.prec = 140
        def dec(value):
            value = Fraction(value)
            return Decimal(value.numerator)/Decimal(value.denominator)
        def posterior(a):
            odds = [1/(1+(-(dec(.7)-dec(.12)*a+dec(.5)*state)).exp()) for state in (-1,1)]
            return odds[1]/sum(odds)
        expected = float(400*(posterior(-398)-posterior(-400))/dec(1-Fraction(error)**2))
    result = generate_suite_a(f,c,policy(((-400.,-390.),)))
    assert expected > 8e-5
    assert result.observed_law_truth.value == pytest.approx(expected, abs=1e-8, rel=0)
    assert result.structural_causal_truth.value == 0.


@pytest.mark.parametrize("scale", [1e-100, float(np.nextafter(1e-100,0.)),
                                   float(np.nextafter(1e-100,np.inf))])
def test_continuation_round1_sampled_nonlinear_local_dose(scale, monkeypatch):
    from fractions import Fraction
    from oxyformer.validation.scm import LocalCoordinates
    from oxyformer.validation.generators import _sample_observations
    local = -np.log1p(-np.longdouble(.5))
    def draw(self,index,u):
        return LocalCoordinates(Fraction(5),Fraction(scale),np.asarray(-local if self.pieces[index].rate > 0 else local))
    monkeypatch.setattr(AssignmentLaw,'quantile_coordinates',draw)
    f = replace(frame(1,1), x=((-100.,-100.),), coordinates=((0.,0.),))
    c = config('sign_changing', beta=10., assignment='near_deterministic', near_scale=scale, noise_sd=0.)
    expected = float((Fraction(scale)*Fraction(*local.as_integer_ratio()))**2)
    observed = _sample_observations(f,c,policy(),0)
    assert expected > 0
    assert observed.y[0] == expected


@pytest.mark.parametrize("tolerance", [2**64-1, 2**64, 2**64+1, 10**20])
def test_continuation_round1_positive_integer_tolerance(tolerance):
    result = generate_suite_a(frame(1,1),config('null'),policy(),tolerance=tolerance)
    assert result.observed_law_truth.value == result.structural_causal_truth.value == 0.


@pytest.mark.parametrize("scale", [1e-13, float(np.nextafter(1e-13,0.)),
                                   float(np.nextafter(1e-13,np.inf))])
def test_continuation_round2_selection_retains_local_coordinates(scale):
    from decimal import Decimal, localcontext
    from fractions import Fraction
    center = float(np.pi-100+2*np.pi*1570)
    error = .9999999999999999
    f = replace(frame(1,1), columns=tuple(f'x{i}' for i in range(7)),
                x=((100.,)*7,), coordinates=((0.,0.),))
    # At this positive exposure sigmoid(z) differs from exp(z) by <exp(-1100).
    # The tilted two-sided Laplace transform is consequently an independent
    # reference to far better than 1e-100 after endpoint amplification.
    with localcontext() as context:
        context.prec = 140
        sine,cosine = _decimal_sin_cos_reduced(Fraction(center)/2+50)
        sin50,_ = _decimal_sin_cos_reduced(Fraction(50))
        s = Decimal(Fraction(scale).numerator)/Decimal(Fraction(scale).denominator)
        k = Decimal(Fraction(.12).numerator)/Decimal(Fraction(.12).denominator)
        left,right = 1-k*s,1+k*s
        moment = (left*cosine+s/2*sine)/(left*left+(s/2)**2)/(1/left+1/right)
        expected = float(400*Fraction(sin50*moment)/(1-Fraction(error)**2))
    c = config('nonlinear',beta=200.,assignment='near_deterministic',near_scale=scale,
               survey_inclusion=True,denominator_error=error)
    result = generate_suite_a(f,c,policy(((center-200,center+200),),delta=200.),tolerance=1e-10)
    for truth in (result.observed_law_truth,result.structural_causal_truth):
        assert truth.value == pytest.approx(expected, abs=1e-10, rel=0)


@pytest.mark.parametrize("assignment", ['continuous','atoms'])
def test_continuation_round2_zero_sine_midpoint_terminates(assignment, monkeypatch):
    import oxyformer.validation.scm as scm
    original = scm._sine_bounds
    def guarded(value,bits):
        assert bits <= 320, 'exact zero at a rounding midpoint failed to terminate'
        return original(value,bits)
    monkeypatch.setattr(scm,'_sine_bounds',guarded)
    f = replace(frame(1,1),columns=('x',),x=((2.**-46,),),coordinates=((0.,0.),))
    u = .04097352393619469
    p = policy(((0.,10.),)) if assignment == 'atoms' else policy(((-u,1-u),),delta=.02)
    result = generate_suite_a(f,config('nonlinear',beta=1.,noise_sd=0.,assignment=assignment),p,seed=0)
    assert result.observations.y[0] == 50.
    assert result.observed_law_truth.status == ('design_rejected' if assignment == 'atoms' else 'integrated')


@pytest.mark.parametrize("tolerance,order", [(np.float64(1e-8),256),(1e-8,np.int64(256))])
def test_continuation_round2_numpy_controls(tolerance,order):
    result = generate_suite_a(frame(1,1),config('null'),policy(),tolerance=tolerance,max_order=order)
    assert result.observed_law_truth.value == 0.
