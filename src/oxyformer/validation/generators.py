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
from scipy.special import logsumexp
import yaml

from oxyformer.design.policies import ShiftOrStayPolicy
from oxyformer.provenance import Immutable, ContractError, require, write_artifact
from oxyformer.validation.analytic_truth import UniformShiftTruth
from oxyformer.validation.scm import (
    AssignmentLaw, CovariateFrame, LatentState, SCMConfig, adjustment_key,
    latent_states, observation_probabilities, observation_log_probability, structural_mean, validate_count_rates,
    LocalCoordinates, exact, exact_shift_intervals, wide, observation_transition_points,
    validate_numeric, validate_policy_domain, validate_seed, REGISTERED_NUMERIC_BOX, NUMERIC_DOMAIN, NUMERIC_MARGIN,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservedRecords(Immutable):
    """Observed X/outcomes plus diagnostic geometry and dependence metadata.

    Primary nuisance builders must use approved frame.x and measured_x only;
    coordinates and learned geography IDs are prohibited predictors. Keeping
    metadata for diagnostics does not certify arbitrary callbacks as compliant.
    """
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
    assignment_mass_error: float | None = None
    selected_mass_fraction: float | None = None
    selected_mass_relative_difference: float | None = None
    # Retains positive mass when the ordinary display fraction underflows.
    selected_log_mass_fraction: float | None = None
    grouped_mass_relative_error: float | None = None
    # Differences between nested orders are diagnostics, not certified bounds.
    interpretation: str = "Successive-order absolute differences; not sampling SEs or rigorous error bounds."


@dataclass(frozen=True, slots=True)
class GeneratedSample:
    observations: ObservedRecords
    observed_law_truth: TruthArtifact
    structural_causal_truth: TruthArtifact
    integration_uncertainty: IntegrationUncertainty


def run_estimator(estimator: Callable, observations: ObservedRecords):
    """Transport observations only, never a generator/config/world/truth bundle.

    This is not a feature-use, accuracy, or causal-identification certificate.
    Primary nuisance input enforcement belongs to the estimator input registry;
    diagnostic callbacks may deliberately use geometry to expose failure modes.
    """
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
    validate_seed(seed)
    validate_policy_domain(policy)
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
    log_weight: np.longdouble
    law: AssignmentLaw


def _log_origin_mass(frame, eligible_only=False):
    weights = [np.log(wide(w)) for i,w in enumerate(frame.weights) if w > 0
               and (not eligible_only or (frame.outcome_available[i] and frame.biomarker_available[i]))]
    return logsumexp(weights) if weights else -np.inf


def _groups(frame, config, policy):
    groups = {}
    support = dict(policy.components_by_key)
    for row in range(len(frame.original_ids)):
        if not (frame.outcome_available[row] and frame.biomarker_available[row]) or frame.weights[row] == 0:
            continue
        for state, probability in latent_states(config):
            key = adjustment_key(frame,row,state,config)
            identity = (frame.coordinates[row],state)
            terms = groups.setdefault(key,{})
            # Start in log space, before multiplying even a subnormal weight.
            log_weight = np.log(wide(frame.weights[row]))+np.log(probability)
            if identity in terms:
                terms[identity].log_weight = np.logaddexp(terms[identity].log_weight,log_weight)
            else:
                law = AssignmentLaw(frame,row,state,config,support[frame.support_keys[row]])
                terms[identity] = _Term(row,state,log_weight,law)
    return {key:list(terms.values()) for key,terms in groups.items()}


def _integration_breakpoints(terms, components, delta, config, shift_intervals):
    delta = exact(delta)
    boundaries = {v for term in terms for v in term.law.breakpoints}
    boundaries.update(v for term in terms for v in observation_transition_points(term.state, config))
    for lo,hi in components:
        boundaries.update([exact(lo),exact(hi)])
    boundaries.update(v for interval in shift_intervals for v in interval)
    # Exact kernel geometry plus a separately retained finite log-odds offset.
    # Never round a crossing or its shifted preimage to an absolute exposure.
    for i,first in enumerate(terms):
        for second in terms[i+1:]:
            for p in first.law.pieces:
                for q in second.law.pieces:
                    slope = p.rate-q.rate
                    if slope == 0:
                        continue
                    lo = max(p.lower+first.law.error,q.lower+second.law.error)
                    hi = min(p.upper+first.law.error,q.upper+second.law.error)
                    if lo >= hi:
                        continue
                    middle = (lo+hi)/2
                    base = first.law.kernel_at(p,middle)-second.law.kernel_at(q,middle)
                    finite = (first.log_weight-first.law.log_normalizer
                              +observation_log_probability(float(middle),first.state,config)
                              -second.log_weight+second.law.log_normalizer
                              -observation_log_probability(float(middle),second.state,config))
                    root = middle-(base+exact(finite))/slope
                    if lo <= root <= hi:
                        boundaries.add(root)
                        for distance in (1,2,4,8,16,32):
                            for sign in (-1,1):
                                value = root+sign*distance/abs(slope)
                                if lo < value < hi:
                                    boundaries.add(value)
    boundaries.update(v-delta for v in tuple(boundaries))
    return boundaries


def _posterior_mean(at, terms, frame, config):
    if not isinstance(at,LocalCoordinates):
        # Ordinary float queries have no hidden offset. Give each its own anchor.
        values = np.asarray(at)
        return np.array([_posterior_mean(LocalCoordinates(exact(a),exact(1),np.array([0.])),
                                         terms,frame,config)[0] for a in values])
    rounded = at.rounded()
    candidates = []
    for index,term in enumerate(terms):
        finite = (term.log_weight-term.law.log_normalizer
                  +observation_log_probability(rounded,term.state,config))
        seen = np.zeros(len(at.values),dtype=bool)
        for p in term.law.pieces:
            mask = at.inside(p.lower+term.law.error,p.upper+term.law.error) & ~seen
            seen |= mask
            if mask.any():
                base = term.law.kernel_at(p,at.anchor)
                residual = wide(p.rate*at.unit)*at.values+finite
                candidates.append((base,index,mask,residual))
    reference = np.full(len(at.values),-1,dtype=int)
    candidates.sort(key=lambda item:item[0],reverse=True)
    for i,(_,_,mask,_) in enumerate(candidates):
        reference[(reference < 0) & mask] = i
    require(bool((reference >= 0).all()), "policy leaves conditional observed-law support")
    log_weights = np.full((len(terms),len(at.values)),-np.inf,dtype=np.longdouble)
    # Cancel exact coarse kernels FIRST; finite priors, tilt and local offsets
    # then survive even when the unnormalized log densities are about -1e324.
    for base,index,mask,residual in candidates:
        for ref in np.unique(reference[mask]):
            selected = mask & (reference == ref)
            difference = wide(base-candidates[ref][0])
            log_weights[index,selected] = difference+residual[selected]
    log_weights -= np.max(log_weights,axis=0)
    weights = np.exp(log_weights)
    weights /= weights.sum(axis=0)
    means = np.array([structural_mean(rounded-float(t.law.error),frame,t.row,t.state,config) for t in terms])
    return np.sum(weights*means,axis=0)


def _integrate(frame, config, policy, groups, order, boundaries_by_key, eligible_by_key):
    mean_contrasts = np.zeros(2,dtype=np.longdouble)
    log_mass = -np.inf
    assignment_mass_error = 0.
    for key,terms in groups.items():
        shift_intervals = eligible_by_key[key[1]]
        for term in terms:
            rules = term.law.quadrature(order,boundaries_by_key[key])
            represented_mass = logsumexp(np.concatenate([rule.log_weights for rule in rules]))
            assignment_mass_error = max(assignment_mass_error,abs(float(np.expm1(represented_mass))))
            for rule in rules:
                coordinates = rule.coordinates
                points = coordinates.rounded()
                moved = np.zeros(len(points),dtype=bool)
                for lo,hi in shift_intervals:
                    moved |= coordinates.inside(lo,hi)
                observed = np.zeros(len(points),dtype=np.longdouble)
                causal = np.zeros(len(points),dtype=np.longdouble)
                if moved.any():
                    factual = coordinates.subset(moved)
                    shifted = factual.shifted(policy.delta_mmhg)
                    observed[moved] = (_posterior_mean(shifted,terms,frame,config)
                                       -_posterior_mean(factual,terms,frame,config))
                    causal[moved] = (structural_mean(shifted.rounded()-float(term.law.error),frame,term.row,term.state,config)
                                     -structural_mean(factual.rounded()-float(term.law.error),frame,term.row,term.state,config))
                log_weights = term.log_weight+rule.log_weights+observation_log_probability(points,term.state,config)
                local_log_mass = logsumexp(log_weights)
                weights = np.exp(log_weights-log_weights.max())
                weights /= weights.sum()
                local = np.array([weights@observed,weights@causal])
                total_log_mass = np.logaddexp(log_mass,local_log_mass)
                mean_contrasts = (np.exp(log_mass-total_log_mass)*mean_contrasts
                                  +np.exp(local_log_mass-total_log_mass)*local)
                log_mass = total_log_mass
    require(np.isfinite(log_mass), "truth integration failed to represent positive target mass")
    return mean_contrasts,log_mass-_log_origin_mass(frame),assignment_mass_error


_OBSERVED_TARGET = (
    "Fixed-frame origin-weighted population conditional on factual flag, survey and biomarker availability. "
    "E[mu(d(A_recorded),X)-mu(A_recorded,X)]; X includes its missingness, support stratum, coarse region/county and measured "
    "confounders only. mu and exposure law are conditional on the same factual selection. "
    "Endpoint is Y as recorded (registered events / noisy denominator when enabled). "
    "Truth uses the declared continuous exposure law and mathematical shift on the exact declared float parameters, "
    "before float serialization of observations."
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
    validate_seed(seed)
    validate_policy_domain(policy)
    require(type(tolerance) in (int, float) and np.isfinite(tolerance) and tolerance > 0
            and type(max_order) is int and max_order >= 32, "invalid integration controls")
    eligible_by_key = {key:exact_shift_intervals(c,policy.delta_mmhg) for key,c in policy.components_by_key}
    config.validate_policy(policy, frame, eligible_by_key=eligible_by_key)
    validate_count_rates(frame, config, policy, eligible_by_key=eligible_by_key)
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
    grouped_log_mass = logsumexp([term.log_weight for terms in groups.values() for term in terms])
    grouped_error = float(abs(np.expm1(grouped_log_mass-_log_origin_mass(frame,eligible_only=True))))
    require(grouped_error <= 1e-10, "grouped origin/latent mass was not preserved")
    boundaries = {key:_integration_breakpoints(terms,dict(policy.components_by_key)[key[1]],policy.delta_mmhg,config,eligible_by_key[key[1]])
                  for key,terms in groups.items()}
    order = 16
    previous, previous_log_mass, _ = _integrate(frame,config,policy,groups,order,boundaries,eligible_by_key)
    while order*2 <= max_order:
        order *= 2
        values, log_mass, mass_error = _integrate(frame,config,policy,groups,order,boundaries,eligible_by_key)
        difference = np.abs(values-previous)
        mass_difference = float(abs(np.expm1(log_mass-previous_log_mass)))
        converged = bool(np.max(difference) <= tolerance and mass_difference <= tolerance and mass_error <= 1e-10)
        if converged:
            break
        previous,previous_log_mass = values,log_mass
    require(converged, "truth integration did not converge; increase max_order")
    observed = TruthArtifact(kind="observed_law", value=float(values[0]), status="integrated", target=_OBSERVED_TARGET, **common)
    causal = TruthArtifact(kind="structural_causal", value=float(values[1]), status="integrated", target=_CAUSAL_TARGET, **common)
    uncertainty = IntegrationUncertainty(method="law-scaled Gauss-Legendre with independent unit-mass check, doubled order",
        observed_absolute_difference=float(difference[0]), causal_absolute_difference=float(difference[1]),
        order=order, converged=True, assignment_mass_error=float(mass_error),
        selected_mass_fraction=float(np.exp(log_mass)), selected_mass_relative_difference=mass_difference,
        selected_log_mass_fraction=float(log_mass), grouped_mass_relative_error=grouped_error)
    return GeneratedSample(observations, observed, causal, uncertainty)



@dataclass(frozen=True, slots=True, kw_only=True)
class PairedWorld(Immutable):
    """Evaluator-only response anchored at the shared factual outcome.

    Primitive coefficients c and tau remain separate. The equivalent response
    Y_factual + tau*(a-h(S)) retains the defining c even when c-tau would round
    to -tau. The stored factual rounding residual stays fixed under intervention.
    """
    structural_effect: float
    factual_location_effect: float
    baseline: tuple[float, ...]
    h_s: tuple[float, ...]
    epsilon: tuple[float, ...]
    factual_y: tuple[float, ...]

    def __post_init__(self):
        Immutable.__post_init__(self)
        validate_numeric(self.structural_effect, "paired_tau", "paired tau")
        validate_numeric(self.factual_location_effect, "paired_c", "paired c")
        validate_numeric(self.h_s, "dose", "paired centers")
        require(len(self.baseline) == len(self.h_s) == len(self.epsilon) == len(self.factual_y) > 0,
                "paired structural world alignment")

    def intervene(self, a):
        validate_numeric(a, "intervention_dose", "intervention doses")
        dose = np.asarray(a, dtype=float)
        require(dose.shape == (len(self.h_s),) and np.isfinite(dose).all(), "intervention alignment")
        tau = exact(self.structural_effect)
        try:
            response = np.array([float(exact(y)+tau*(exact(d)-exact(s)))
                                 for y,d,s in zip(self.factual_y,dose,self.h_s)])
        except OverflowError as exc:
            raise ContractError("nonfinite intervention response") from exc
        require(bool(np.isfinite(response).all()), "nonfinite intervention response")
        return response


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
    validate_numeric(c, "paired_c", "paired c")
    validate_numeric(tau, "paired_tau", "paired tau")
    validate_numeric(noise_sd, "noise_sd", "noise_sd", allow_zero=True)
    validate_seed(seed)
    require(type(n_geographies) is int and type(cluster_size) is int
            and n_geographies > 0 and cluster_size > 0 and tau != 0, "invalid pair controls")
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
    world0 = PairedWorld(structural_effect=0., factual_location_effect=c, baseline=tuple(baseline.tolist()),
                         h_s=tuple(s.tolist()), epsilon=tuple(epsilon.tolist()), factual_y=observations.y)
    worldtau = replace(world0, structural_effect=tau)
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
    declaration = recipe.get("numeric_domain")
    require(declaration is not None, "Suite A recipe must declare numeric_domain")
    expected = {"id": "suite-a-100x-v1", "margin": NUMERIC_MARGIN,
                "registered_box": {k:list(v) for k,v in REGISTERED_NUMERIC_BOX.items()},
                "supported_box": {k:list(v) for k,v in NUMERIC_DOMAIN.items()}}
    require(declaration == expected, "recipe numeric_domain differs from enforced domain")
    require(len({c.name for c in configs}) == len(configs), "duplicate scenarios")
    return configs
