"""Suite A harness and a declaration-only boundary for future Suite B.

Only ObservedRecords cross the estimator boundary. Generator configurations,
truths and integration diagnostics belong to the evaluator's separate store.
This API is an information-flow boundary, not a filesystem security sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal, Protocol

import numpy as np
from numpy.polynomial.legendre import leggauss
import yaml

from oxyformer.design.policies import ShiftOrStayPolicy
from oxyformer.provenance import Immutable, ContractError, require, write_artifact
from oxyformer.validation.analytic_truth import UniformShiftTruth
from oxyformer.validation.scm import (
    AssignmentLaw, CovariateFrame, LatentState, SCMConfig, adjustment_key,
    latent_states, observation_probabilities, structural_mean, validate_count_rates,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservedRecords(Immutable):
    frame: CovariateFrame
    measured_columns: tuple[str, ...]
    measured_x: tuple[tuple[float, ...], ...]
    a: tuple[float, ...]
    y: tuple[float | None, ...]
    flag_available: tuple[bool, ...]
    survey_included: tuple[bool, ...]
    biomarker_available: tuple[bool, ...]
    registered_events: tuple[int | None, ...]
    observed_denominator: tuple[float | None, ...]

    def __post_init__(self):
        Immutable.__post_init__(self)
        n = len(self.frame.original_ids)
        require(all(len(v) == n for v in (self.measured_x, self.a, self.y,
                self.flag_available, self.survey_included, self.biomarker_available,
                self.registered_events, self.observed_denominator)), "observation alignment")
        require(all(len(x) == len(self.measured_columns) for x in self.measured_x), "measured X width")
        for y, flag, survey, bio in zip(self.y, self.flag_available, self.survey_included, self.biomarker_available):
            require((y is not None) == (flag and survey and bio), "outcome selection mismatch")


@dataclass(frozen=True, slots=True, kw_only=True)
class TruthArtifact(Immutable):
    kind: Literal["observed_law", "structural_causal"]
    value: float | None
    policy_id: str
    config_hash: str
    generator_configuration: str
    frame_hash: str
    observation_hash: str
    target: str
    status: Literal["integrated", "analytic", "design_rejected", "empty_target"]
    reason: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class IntegrationUncertainty(Immutable):
    method: str
    observed_absolute_difference: float | None
    causal_absolute_difference: float | None
    order: int
    converged: bool
    # Differences between nested orders are diagnostics, not certified bounds.
    interpretation: str = "Successive-order absolute differences; not sampling SEs or rigorous error bounds."


@dataclass(frozen=True, slots=True)
class GeneratedSample:
    observations: ObservedRecords
    observed_law_truth: TruthArtifact
    structural_causal_truth: TruthArtifact
    integration_uncertainty: IntegrationUncertainty


def run_estimator(estimator: Callable, observations: ObservedRecords):
    """Never hand a generator, world label, config, or truth bundle to a method."""
    require(type(observations) is ObservedRecords, "estimators receive ObservedRecords only")
    return estimator(observations)


def write_sample(sample: GeneratedSample, *, observations_dir: Path, truth_dir: Path):
    """Separate, create-once artifacts. Caller enforces evaluator-only access.

    The observations directory can be mounted alone in an estimator process.
    There are no paths, references or config hashes from observations to truth.
    """
    observations_dir, truth_dir = Path(observations_dir).resolve(), Path(truth_dir).resolve()
    require(not observations_dir.is_relative_to(truth_dir) and not truth_dir.is_relative_to(observations_dir),
            "observation and truth stores must be separate")
    observations_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    write_artifact(observations_dir / "observations.json", sample.observations)
    write_artifact(truth_dir / "observed_law.json", sample.observed_law_truth)
    write_artifact(truth_dir / "structural_causal.json", sample.structural_causal_truth)
    write_artifact(truth_dir / "integration_uncertainty.json", sample.integration_uncertainty)


def _sample_observations(frame, config, policy, seed):
    rng = np.random.default_rng(seed)
    regions, geographies, clusters = {}, {}, {}
    a, y, measured, flags, surveys, biomarkers, events, denominators = ([] for _ in range(8))
    support = dict(policy.components_by_key)
    for i, (geo, region, cluster) in enumerate(zip(frame.geography_ids, frame.region_ids, frame.cluster_ids)):
        if region not in regions:
            regions[region] = float(rng.choice([-1, 1])) if config.regional_confounding != "none" else 0.
        if geo not in geographies:
            local = float(rng.choice([-1, 1])) if config.local_confounding != "none" else 0.
            illness = float(rng.random() < .3) if config.has_illness else 0.
            error = float(rng.choice([-1, 1])) * config.exposure_error
            state = LatentState(local, regions[region], illness, error)
            law = AssignmentLaw(frame, i, state, config, support[frame.support_keys[i]])
            # A genuine point mass at the first lower boundary, never jittered.
            dose = law.components[0][0] + error if config.assignment == "atoms" and rng.random() < .5 else law.sample(rng)
            pflag, psurvey, _ = observation_probabilities(dose, state, config)
            geographies[geo] = (state, dose, bool(rng.random() < pflag), bool(rng.random() < psurvey), rng.normal())
        state, dose, flag, survey, geo_noise = geographies[geo]
        if cluster not in clusters:
            clusters[cluster] = rng.normal()
        factor = 1 + config.denominator_error * float(rng.choice([-1, 1]))
        state = replace(state, denominator_factor=factor)
        _, _, pbio = observation_probabilities(dose, state, config)
        flag = flag and frame.outcome_available[i]
        bio = bool(rng.random() < pbio) and frame.biomarker_available[i]
        available = flag and survey and bio
        mean = float(structural_mean(dose-state.error, frame, i, state, config))
        count = denominator = None
        if config.registration_probability < 1 or config.denominator_error:
            # Poisson events followed by binomial registration. The endpoint is
            # registered events / observed denominator in BOTH truth artifacts.
            true_rate = mean * factor / config.registration_probability
            require(true_rate >= 0, "count scenario has a negative event rate")
            true_events = rng.poisson(100 * true_rate)
            count = int(rng.binomial(true_events, config.registration_probability))
            denominator = 100 * factor
            outcome = count / denominator
        else:
            outcome = mean + config.noise_sd * (geo_noise + clusters[cluster] + rng.normal()) / np.sqrt(3)
        cov = []
        if config.local_confounding == "measured":
            cov.append(state.local)
        if config.regional_confounding == "measured":
            cov.append(state.regional)
        a.append(dose)
        y.append(float(outcome) if available else None)
        measured.append(tuple(cov))
        flags.append(flag)
        surveys.append(survey)
        biomarkers.append(bio)
        events.append(count if available else None)
        denominators.append(denominator if available else None)
    names = tuple(scale + "_confounder" for scale in ("local", "regional")
                  if getattr(config, scale + "_confounding") == "measured")
    return ObservedRecords(frame=frame, measured_columns=names, measured_x=tuple(measured),
                           a=tuple(a), y=tuple(y), flag_available=tuple(flags),
                           survey_included=tuple(surveys), biomarker_available=tuple(biomarkers),
                           registered_events=tuple(events), observed_denominator=tuple(denominators))


@dataclass
class _Term:
    row: int
    state: LatentState
    weight: float
    law: AssignmentLaw


def _groups(frame, config, policy):
    groups = {}
    support = dict(policy.components_by_key)
    for row in range(len(frame.original_ids)):
        if not (frame.outcome_available[row] and frame.biomarker_available[row]) or frame.weights[row] == 0:
            continue
        for state, probability in latent_states(config):
            key = adjustment_key(frame, row, state, config)
            law = AssignmentLaw(frame, row, state, config, support[frame.support_keys[row]])
            groups.setdefault(key, []).append(_Term(row, state, frame.weights[row]*probability, law))
    return groups


def _integrate(frame, config, policy, groups, order):
    nodes, weights = leggauss(order)
    observed, causal, mass = 0., 0., 0.
    for key, terms in groups.items():
        support_key = key[1]
        boundaries = {v for term in terms for v in term.law.breakpoints}
        for lo, hi in dict(policy.components_by_key)[support_key]:
            boundaries.update([lo, hi, hi-policy.delta_mmhg])
        # The posterior mean at d(a) changes when d crosses a mixture boundary.
        boundaries.update([v-policy.delta_mmhg for v in tuple(boundaries)])
        lower = min(v for t in terms for v in t.law.breakpoints)
        upper = max(v for t in terms for v in t.law.breakpoints)
        breaks = sorted({lower, upper} | {v for v in boundaries if lower < v < upper})
        points = np.concatenate([(lo+hi)/2 + (hi-lo)/2*nodes for lo, hi in zip(breaks[:-1], breaks[1:])])
        quadrature = np.concatenate([(hi-lo)/2*weights for lo, hi in zip(breaks[:-1], breaks[1:])])
        moved = policy.shift_mask(points, (support_key,)*len(points))
        shifted = points + policy.delta_mmhg*moved

        def joint(at):
            density = np.zeros_like(at)
            numerator = np.zeros_like(at)
            for term in terms:
                prob = observation_probabilities(at, term.state, config)
                joint_weight = term.weight * term.law.density(at) * prob[0]*prob[1]*prob[2]
                density += joint_weight
                numerator += joint_weight * structural_mean(at-term.state.error, frame, term.row, term.state, config)
            return density, numerator

        density, numerator = joint(points)
        shifted_density, shifted_numerator = joint(shifted)
        require(not np.any((density > 0) & (shifted_density <= 0)), "policy leaves conditional observed-law support")
        mu = np.divide(numerator, density, out=np.zeros_like(points), where=density > 0)
        mu_d = np.divide(shifted_numerator, shifted_density, out=np.zeros_like(points), where=shifted_density > 0)
        observed += float(quadrature @ (density*(mu_d-mu)))
        mass += float(quadrature @ density)
        causal_integrand = np.zeros_like(points)
        for term in terms:
            p = observation_probabilities(points, term.state, config)
            joint_weight = term.weight * term.law.density(points) * p[0]*p[1]*p[2]
            difference = (structural_mean(shifted-term.state.error, frame, term.row, term.state, config)
                          - structural_mean(points-term.state.error, frame, term.row, term.state, config))
            causal_integrand += joint_weight*difference
        causal += float(quadrature @ causal_integrand)
    require(mass > 0, "selected population has zero target mass")
    return np.array([observed/mass, causal/mass])


_OBSERVED_TARGET = (
    "Fixed-frame origin-weighted population conditional on factual flag, survey and biomarker availability. "
    "E[mu(d(A_recorded),X)-mu(A_recorded,X)]; X includes its missingness, support stratum, coarse region/county and measured "
    "confounders only. mu and exposure law are conditional on the same factual selection. "
    "Endpoint is Y as recorded (registered events / noisy denominator when enabled)."
)
_CAUSAL_TARGET = (
    "Same fixed-frame origin-weighted, factually selected population and recorded endpoint. "
    "E[Y(A_true + d(A_recorded)-A_recorded)-Y(A_true)]; current location, latent causes, "
    "prior-residence dose displacement, measurement-error draw, factual selection, registration "
    "probability and denominator draw are held fixed. This explicitly defined SCM intervention "
    "is not identified by observed fit. Migration uses prior dose A_true-migration*illness."
)


def _unavailable_truth(observations, common, status, reason):
    """Keep observations when a population target is undefined; never report zero."""
    observed = TruthArtifact(kind="observed_law", value=None, status=status, reason=reason,
                             target=_OBSERVED_TARGET, **common)
    causal = TruthArtifact(kind="structural_causal", value=None, status=status, reason=reason,
                           target=_CAUSAL_TARGET, **common)
    uncertainty = IntegrationUncertainty(method="not integrated: " + reason, observed_absolute_difference=None,
                                        causal_absolute_difference=None, order=0, converged=False)
    return GeneratedSample(observations, observed, causal, uncertainty)


def generate_suite_a(frame: CovariateFrame, config: SCMConfig, policy: ShiftOrStayPolicy, *,
                     seed: int = 0, tolerance: float = 1e-8, max_order: int = 256) -> GeneratedSample:
    """Generate observations and *population*, not realized-sample, truths.

    Integration marginalizes assignment, confounders and observation mechanisms,
    conditional on the supplied fixed covariate frame. No estimator fits enter.
    """
    require(tolerance > 0 and max_order >= 32, "invalid integration controls")
    config.validate_policy(policy, frame)
    validate_count_rates(frame, config, policy)
    observations = _sample_observations(frame, config, policy, seed)
    common = dict(policy_id=policy.policy_id, config_hash=config.content_hash, generator_configuration=config.to_json(),
                  frame_hash=frame.content_hash, observation_hash=observations.content_hash)
    if config.assignment == "atoms":
        try:
            replace(policy, exposure_law="mixed")
        except ContractError as exc:
            reason = str(exc)
        else:
            raise ContractError("policy interface unexpectedly accepted atoms without a measure derivation")
        return _unavailable_truth(observations, common, "design_rejected", reason)
    groups = _groups(frame, config, policy)
    if not groups:
        return _unavailable_truth(observations, common, "empty_target", "selected population has zero target mass")
    order = 16
    previous = _integrate(frame, config, policy, groups, order)
    while order*2 <= max_order:
        order *= 2
        values = _integrate(frame, config, policy, groups, order)
        difference = np.abs(values-previous)
        if np.max(difference) <= tolerance:
            break
        previous = values
    require(np.max(difference) <= tolerance, "truth integration did not converge; increase max_order")
    observed = TruthArtifact(kind="observed_law", value=float(values[0]), status="integrated", target=_OBSERVED_TARGET, **common)
    causal = TruthArtifact(kind="structural_causal", value=float(values[1]), status="integrated", target=_CAUSAL_TARGET, **common)
    uncertainty = IntegrationUncertainty(method="piecewise Gauss-Legendre, doubled order", observed_absolute_difference=float(difference[0]),
                                        causal_absolute_difference=float(difference[1]), order=order, converged=True)
    return GeneratedSample(observations, observed, causal, uncertainty)


@dataclass(frozen=True, slots=True, kw_only=True)
class PairedWorld(Immutable):
    """Evaluator-only structural response; S and epsilon stay fixed under do(A)."""
    structural_effect: float
    location_effect: float
    baseline: tuple[float, ...]
    h_s: tuple[float, ...]
    epsilon: tuple[float, ...]

    def intervene(self, a):
        dose = np.asarray(a, dtype=float)
        require(dose.shape == (len(self.h_s),) and np.isfinite(dose).all(), "intervention alignment")
        return (np.asarray(self.baseline) + self.structural_effect*dose
                + self.location_effect*np.asarray(self.h_s) + np.asarray(self.epsilon))


@dataclass(frozen=True, slots=True)
class ObservationalEquivalencePair:
    m0: GeneratedSample
    mtau: GeneratedSample
    world0: PairedWorld
    worldtau: PairedWorld


def observational_equivalence_pair(*, n_geographies=100, cluster_size=3, seed=0,
                                   c=1.5, tau=2., noise_sd=1.) -> ObservationalEquivalencePair:
    """A=h(S), M0=b(X)+c*h(S)+e, Mtau=b(X)+tau*A+(c-tau)*h(S)+e.

    h(S)=S ~ U[0,10] per geography, independently of approved X. S appears only
    as diagnostic geometry. Truth adjusts for X, never exact location. Shared
    observed Y is computed ONCE, not through the two floating-point equations.
    """
    require(n_geographies > 0 and cluster_size > 0 and tau != 0 and noise_sd >= 0, "invalid pair controls")
    rng = np.random.default_rng(seed)
    n = n_geographies*cluster_size
    s = np.repeat(rng.uniform(0, 10, n_geographies), cluster_size)
    x = rng.normal(size=n)
    baseline = 50 + .25*x
    epsilon = noise_sd*(np.repeat(rng.normal(size=n_geographies), cluster_size) + rng.normal(size=n))/np.sqrt(2)
    y_shared = baseline + c*s + epsilon
    frame = CovariateFrame(original_ids=tuple(f"o{i}" for i in range(n)),
            geography_ids=tuple(f"g{i//cluster_size}" for i in range(n)),
            region_ids=("r0",)*n, cluster_ids=tuple(f"g{i//cluster_size}" for i in range(n)),
            coordinates=tuple((float(v), 0.) for v in s), columns=("x",),
            x=tuple((float(v),) for v in x), support_keys=("s",)*n, weights=(1.,)*n,
            outcome_available=(True,)*n, biomarker_available=(True,)*n)
    observations = ObservedRecords(frame=frame, measured_columns=(), measured_x=((),)*n,
            a=tuple(s.tolist()), y=tuple(y_shared.tolist()), flag_available=(True,)*n,
            survey_included=(True,)*n, biomarker_available=(True,)*n,
            registered_events=(None,)*n, observed_denominator=(None,)*n)
    world0 = PairedWorld(structural_effect=0., location_effect=c, baseline=tuple(baseline.tolist()),
                         h_s=tuple(s.tolist()), epsilon=tuple(epsilon.tolist()))
    worldtau = replace(world0, structural_effect=tau, location_effect=c-tau)
    policy = ShiftOrStayPolicy(support_design_hash=frame.content_hash, components_by_key=(("s", ((0., 10.),)),))
    analytic = UniformShiftTruth()
    uncertainty = IntegrationUncertainty(method="closed-form uniform shift", observed_absolute_difference=0.,
                                         causal_absolute_difference=0., order=0, converged=True)
    def bundle(world):
        common = dict(policy_id=policy.policy_id, config_hash=world.content_hash, generator_configuration=world.to_json(), frame_hash=frame.content_hash,
                      observation_hash=observations.content_hash, status="analytic")
        observed = TruthArtifact(kind="observed_law", value=analytic.linear_contrast(c),
                target="Population E[mu(d(A),X)-mu(A,X)]=1.6*c; S uniform, independent of X. Exact location excluded from adjustment.", **common)
        causal = TruthArtifact(kind="structural_causal", value=analytic.linear_contrast(world.structural_effect),
                target="Population policy intervention holding S fixed: 1.6*structural_effect. Not a realized-sample contrast.", **common)
        return GeneratedSample(observations, observed, causal, uncertainty)
    return ObservationalEquivalencePair(bundle(world0), bundle(worldtau), world0, worldtau)


class IndependentSuiteBGenerator(Protocol):
    """Interface only. An independently trained, evaluator-private follow-on.

    Fit on separate development records, excluding all estimator/pretraining and
    final evaluation records. Freeze family and fitted identity before campaign
    expansion. Generation returns the same separated harness artifacts; an
    observed-realism fit cannot certify structural truth. No fitting is done here.
    """
    family: Literal["flow", "copula", "structured_latent"]
    development_data_hash: str
    fitted_generator_hash: str

    def generate(self, *, seed: int, n_geographies: int) -> GeneratedSample: ...


def load_suite_a(path: str | Path):
    """Read only the explicitly listed scenarios; never expand parameter grids."""
    recipe = yaml.safe_load(Path(path).read_text())
    require(recipe["suite"] == "A" and recipe["scenario_expansion"] == "explicit_only", "not a Suite A recipe")
    configs = tuple(SCMConfig(**entry) for entry in recipe["scenarios"])
    require(len({c.name for c in configs}) == len(configs), "duplicate scenarios")
    return configs
