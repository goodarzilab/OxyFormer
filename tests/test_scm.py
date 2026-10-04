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


@pytest.mark.parametrize("scale", [.05, 1e-3, 1e-6, 1e-8])
def test_concentrated_assignment_retains_mass_and_shifted_conditional_mean(scale):
    sample = generate_suite_a(frame(2), config(assignment="near_deterministic", near_scale=scale), policy())
    from scipy.stats import laplace
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
        a,w = original(self,order,breakpoints)
        return a,.5*w
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
