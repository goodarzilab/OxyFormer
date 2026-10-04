"""Small composable Suite A SCMs, conditional on a fixed covariate frame.

Coordinates and dependence IDs are metadata, never adjustment variables. Local
and regional Bernoulli causes are shared within their respective units. The
finite latent mixture permits deterministic integration of the *selected*
observed law, including measurement error, without fitting an oracle regression.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from fractions import Fraction
from typing import Literal

import numpy as np
from scipy.special import expit, log_expit, logsumexp
from numpy.polynomial.legendre import leggauss

from oxyformer.design.policies import ShiftOrStayPolicy
from oxyformer.provenance import Immutable, require, unique


@dataclass(frozen=True, slots=True, kw_only=True)
class CovariateFrame(Immutable):
    """Caller-supplied, already approved X; no real data are bundled here.

    Every row is retained, including missing outcomes/biomarkers. Thus geometry,
    X (including None), origin weights and all cluster sizes survive simulation.
    Coordinates refer to current residence. Migration models prior residence.
    Dependence clusters may cross assignment geographies (e.g. linked households).
    """
    original_ids: tuple[str, ...]
    geography_ids: tuple[str, ...]
    region_ids: tuple[str, ...]
    cluster_ids: tuple[str, ...]
    coordinates: tuple[tuple[float, float], ...]
    columns: tuple[str, ...]
    x: tuple[tuple[float | None, ...], ...]
    support_keys: tuple[str, ...]
    weights: tuple[float, ...]
    outcome_available: tuple[bool, ...]
    biomarker_available: tuple[bool, ...]

    def __post_init__(self):
        Immutable.__post_init__(self)
        n = len(self.original_ids)
        require(n > 0, "empty covariate frame")
        unique(self.original_ids, "original IDs")
        unique(self.columns, "X columns")
        require(all(len(v) == n for v in (self.geography_ids, self.region_ids,
                self.cluster_ids, self.coordinates, self.x, self.support_keys,
                self.weights, self.outcome_available, self.biomarker_available)), "frame alignment")
        require(all(len(v) == len(self.columns) for v in self.x), "X width")
        require(all(w >= 0 for w in self.weights) and any(w > 0 for w in self.weights), "invalid weights")
        by_geo = {}
        for geo, region, coord, key in zip(self.geography_ids, self.region_ids,
                self.coordinates, self.support_keys):
            value = (region, coord, key)
            require(by_geo.setdefault(geo, value) == value, "inconsistent assignment geography")


@dataclass(frozen=True, slots=True, kw_only=True)
class SCMConfig(Immutable):
    """Explicit mechanism switches, not a Cartesian scenario grid.

    Mean endpoint = 50 + .25 sum(X_observed) + f(A_true - migration*illness)
    + local_strength*U_local + regional_strength*U_region + 2*illness.
    Missing X is retained; its baseline contribution is defined as zero.
    Assignment log-density slope is .3 U_local + .2 U_region - .25 illness
    when the corresponding mechanism is enabled. Near-deterministic assignment
    instead uses a truncated Laplace law around a location-dependent center.
    extreme_ratios adds a -4*A log-density tilt to either continuous law.
    """
    name: str
    active_mechanisms: tuple[str, ...]
    effect: Literal["null", "linear", "nonlinear", "sign_changing"] = "linear"
    beta: float = 1.0
    local_confounding: Literal["none", "measured", "omitted"] = "none"
    regional_confounding: Literal["none", "measured", "omitted"] = "none"
    local_strength: float = 2.0
    regional_strength: float = 2.0
    assignment: Literal["continuous", "near_deterministic", "atoms"] = "continuous"
    near_scale: float = 0.05
    extreme_ratios: bool = False
    support_gaps: bool = False
    heterogeneous_eligibility: bool = False
    exposure_error: float = 0.0
    migration: float = 0.0
    selected_outcome: bool = False
    survey_inclusion: bool = False
    missing_biomarkers: bool = False
    registration_probability: float = 1.0
    denominator_error: float = 0.0
    noise_sd: float = 1.0

    def __post_init__(self):
        Immutable.__post_init__(self)
        expected = {self.effect}
        for scale in ("local", "regional"):
            kind = getattr(self, scale + "_confounding")
            if kind != "none":
                expected.add(scale + "_" + kind)
        if self.assignment != "continuous":
            expected.add(self.assignment)
        for name in ("extreme_ratios", "support_gaps", "heterogeneous_eligibility",
                     "exposure_error", "migration", "selected_outcome", "survey_inclusion",
                     "missing_biomarkers", "denominator_error"):
            if getattr(self, name):
                expected.add(name)
        if self.registration_probability != 1:
            expected.add("under_registration")
        unique(self.active_mechanisms, "mechanisms")
        require(set(self.active_mechanisms) == expected, "active_mechanisms must name exactly the enabled mechanisms")
        require(self.name.strip() != "", "scenario name required")
        require(self.near_scale > 0 and self.noise_sd >= 0, "invalid noise scale")
        require(self.exposure_error >= 0 and self.migration >= 0, "negative error or migration")
        require(0 < self.registration_probability <= 1, "invalid registration probability")
        require(0 <= self.denominator_error < 1, "invalid denominator error")

    @property
    def has_illness(self):
        return bool(self.migration or self.selected_outcome or self.missing_biomarkers)

    def validate_policy(self, policy: ShiftOrStayPolicy, frame: CovariateFrame, *, eligible_by_key=None):
        support = dict(policy.components_by_key)
        require(set(frame.support_keys) <= support.keys(), "unknown frame support key")
        used = [support[k] for k in set(frame.support_keys)]
        require(all(components for components in used), "SCM assignment needs nonempty support")
        require(self.support_gaps == any(len(c) > 1 for c in used), "support_gaps switch disagrees with frozen support")
        if eligible_by_key is None:
            eligible_by_key = {key:exact_shift_intervals(c,policy.delta_mmhg) for key,c in support.items()}
        eligible = [bool(eligible_by_key[key]) for key in set(frame.support_keys)]
        heterogeneous = any(eligible) and not all(eligible)
        require(self.heterogeneous_eligibility == heterogeneous, "eligibility switch disagrees with frozen support")


@dataclass(frozen=True)
class LatentState:
    local: float = 0.0
    regional: float = 0.0
    illness: float = 0.0
    error: float = 0.0
    denominator_factor: float = 1.0


def latent_states(config: SCMConfig):
    """Marginal states for one row; sampling separately shares causes by unit."""
    local = [(-1., .5), (1., .5)] if config.local_confounding != "none" else [(0., 1.)]
    regional = [(-1., .5), (1., .5)] if config.regional_confounding != "none" else [(0., 1.)]
    illness = [(0., .7), (1., .3)] if config.has_illness else [(0., 1.)]
    error = [(-config.exposure_error, .5), (config.exposure_error, .5)] if config.exposure_error else [(0., 1.)]
    denom = [(1-config.denominator_error, .5), (1+config.denominator_error, .5)] if config.denominator_error else [(1., 1.)]
    for states in product(local, regional, illness, error, denom):
        yield LatentState(*(s[0] for s in states)), float(np.prod([s[1] for s in states]))


def adjustment_key(frame, row, state, config):
    """Declared adjustment: approved X/missingness, support stratum, measured U.

    Coarse region/county routing is conditioned on. No exact coordinates,
    geography IDs, latent illness or omitted causes enter.
    """
    return (frame.x[row], frame.support_keys[row],
            state.local if config.local_confounding == "measured" else None,
            state.regional if config.regional_confounding == "measured" else None,
            frame.region_ids[row])


def effect(a, config):
    a = np.asarray(a, dtype=float)
    if config.effect == "null":
        return np.zeros_like(a)
    if config.effect == "linear":
        return config.beta * a
    if config.effect == "nonlinear":
        return config.beta * np.sin(a / 2)
    # Derivative changes sign at dose 5.
    return config.beta * (a - 5)**2 / 10


def structural_mean(a_true, frame, row, state, config):
    baseline = 50 + .25 * sum(v for v in frame.x[row] if v is not None)
    mean = (baseline + effect(np.asarray(a_true) - config.migration * state.illness, config)
            + config.local_strength * state.local + config.regional_strength * state.regional
            + 2 * state.illness)
    return config.registration_probability * mean / state.denominator_factor


def observation_probabilities(a_observed, state, config):
    """Factual observation mechanisms; independent of outcome noise given state."""
    a = np.asarray(a_observed)
    flag = expit(1 - .2*a - 1.2*state.illness) if config.selected_outcome else np.ones_like(a, dtype=float)
    survey = expit(.7 - .12*a + .5*state.local) if config.survey_inclusion else np.ones_like(a, dtype=float)
    bio = expit(1 - .1*a - .8*state.illness) if config.missing_biomarkers else np.ones_like(a, dtype=float)
    return flag, survey, bio


def exact(value):
    """Exact geometry of a declared float, not a decimal reinterpretation."""
    return value if isinstance(value, Fraction) else Fraction(float(value))


def exact_shift_intervals(components, delta):
    """The frozen mathematical S=[L,U-delta] on exact declared float values.

    Derive widths and cutoffs before any rounding, so a shifted eligible dose
    remains in its origin component even when the law is narrower than an ULP.
    This continuous truth geometry precedes serialization of observed doses.
    """
    shift = exact(delta)
    if shift == 0:
        return ()
    bounds = tuple((exact(lo),exact(hi)) for lo,hi in components)
    return tuple((lo,hi-shift) for lo,hi in bounds if hi-lo >= shift)


def wide(value):
    """Extended exponent range AFTER exact cancellation of coarse geometry."""
    if isinstance(value, Fraction):
        return np.longdouble(str(value.numerator))/np.longdouble(str(value.denominator))
    return np.longdouble(value)


@dataclass(frozen=True)
class LocalCoordinates:
    """anchor + unit*values; offsets are never absorbed into the anchor.

    Exact rational anchors/units carry geometry. Dimensionless quadrature values
    carry local variation. Only smooth responses, selection and observation
    serialization materialize exposure values; support, policy and likelihoods
    operate on these separate parts.
    """
    anchor: Fraction
    unit: Fraction
    values: np.ndarray

    def rounded(self):
        return wide(self.anchor)+wide(self.unit)*self.values

    def shifted(self, delta):
        return replace(self, anchor=self.anchor+exact(delta))

    def subset(self, mask):
        return replace(self, values=self.values[mask])

    def inside(self, lower, upper):
        return ((self.values >= wide((exact(lower)-self.anchor)/self.unit))
                & (self.values <= wide((exact(upper)-self.anchor)/self.unit)))


@dataclass(frozen=True)
class QuadraturePiece:
    coordinates: LocalCoordinates
    log_weights: np.ndarray


@dataclass(frozen=True)
class _ExponentialPiece:
    lower: Fraction
    upper: Fraction
    rate: Fraction
    peak_kernel: Fraction = Fraction(0)
    log_probability: np.longdouble = np.longdouble(0)

    @property
    def peak(self):
        return self.upper if self.rate > 0 else self.lower

    @property
    def log_integral(self):
        if self.rate == 0:
            return np.log(wide(self.upper-self.lower))
        extent = wide(abs(self.rate)*(self.upper-self.lower))
        return np.log(-np.expm1(-extent))-np.log(wide(abs(self.rate)))


def observation_log_probability(a_observed, state, config):
    """Selected-law weights without underflow of positive logistic probabilities."""
    a = np.asarray(a_observed, dtype=float)
    value = np.zeros_like(a)
    if config.selected_outcome:
        value += log_expit(1-.2*a-1.2*state.illness)
    if config.survey_inclusion:
        value += log_expit(.7-.12*a+.5*state.local)
    if config.missing_biomarkers:
        value += log_expit(1-.1*a-.8*state.illness)
    return value


class AssignmentLaw:
    """Normalized exponential pieces with exact coarse log-kernel geometry.

    Kernel differences are cancelled as rationals before conversion to numeric
    log weights. This retains ordinary tilt beside an arbitrarily narrow
    Laplace kernel. No component is removed from truth or posterior evaluation
    because its ordinary probability underflows. Public draws are floats.
    """
    def __init__(self, frame, row, state, config, components):
        self.components = components
        self.error = exact(state.error)
        self.scale = exact(config.near_scale)
        self.near = config.assignment == "near_deterministic"
        rate = -4. if config.extreme_ratios else 0.
        if not self.near:
            rate += .3*state.local+.2*state.regional-.25*state.illness
        self.rate = exact(rate)
        lo,hi = components[0][0],components[-1][1]
        self.center = exact(lo+(hi-lo)*float(expit(frame.coordinates[row][0]
                            +.2*state.local+.1*state.regional-.2*state.illness)))
        pieces = []
        for lower,upper in components:
            edges = [exact(lower),exact(upper)]
            if self.near and edges[0] < self.center < edges[-1]:
                edges.insert(1,self.center)
            for left,right in zip(edges[:-1],edges[1:]):
                slope = self.rate
                if self.near:
                    slope += (1 if right <= self.center else -1)/self.scale
                piece = _ExponentialPiece(left,right,slope)
                kernel = self.rate*piece.peak
                if self.near:
                    kernel -= abs(piece.peak-self.center)/self.scale
                pieces.append(replace(piece,peak_kernel=kernel))
        self.kernel_reference = max(p.peak_kernel for p in pieces)
        log_masses = np.array([wide(p.peak_kernel-self.kernel_reference)+p.log_integral
                               for p in pieces],dtype=np.longdouble)
        self.log_normalizer = logsumexp(log_masses)
        self.pieces = tuple(replace(p,log_probability=logm-self.log_normalizer)
                            for p,logm in zip(pieces,log_masses))
        self.probabilities = np.asarray(np.exp(log_masses-self.log_normalizer),dtype=float)
        require(np.isfinite(self.probabilities).all(), "assignment normalization failed")

    @property
    def breakpoints(self):
        return [v+self.error for p in self.pieces for v in (p.lower,p.upper)]

    def kernel_at(self, piece, recorded_anchor):
        return (piece.rate*(recorded_anchor-self.error-piece.peak)
                +piece.peak_kernel-self.kernel_reference)

    def contains(self, a_observed):
        a = np.asarray(a_observed)-float(self.error)
        return np.logical_or.reduce([(a >= lo) & (a <= hi) for lo,hi in self.components])

    def log_density(self, a_observed):
        # Convenience for ordinary recorded exposures. Truth posterior evaluation
        # instead uses kernel_at and LocalCoordinates before relative conversion.
        a = np.asarray(a_observed)
        flat = []
        for value in a.ravel():
            at = exact(value)
            piece = next((p for p in self.pieces if p.lower+self.error <= at <= p.upper+self.error),None)
            flat.append(-np.inf if piece is None else wide(self.kernel_at(piece,at))-self.log_normalizer)
        return np.asarray(flat,dtype=np.longdouble).reshape(a.shape)

    def density(self, a_observed):
        return np.exp(self.log_density(a_observed))

    def quantile_coordinates(self, piece_index, u):
        """Conditional inverse transform, also used before draw serialization."""
        p = self.pieces[piece_index]
        u = np.asarray(u,dtype=np.longdouble)
        if p.rate == 0:
            return LocalCoordinates(p.lower+self.error,p.upper-p.lower,u)
        extent = wide(abs(p.rate)*(p.upper-p.lower))
        t = -np.log1p(-u*(-np.expm1(-extent)))
        return LocalCoordinates(p.peak+self.error,1/abs(p.rate),(-t if p.rate > 0 else t))

    def sample(self, rng):
        index = rng.choice(len(self.pieces),p=self.probabilities)
        return float(self.quantile_coordinates(index,rng.random()).rounded())

    def quadrature(self, order, breakpoints):
        """Full conditional piece measures, with log masses and local nodes.

        Even tiny pieces remain represented. Panels resolve decay in units of
        the piece rate, and exact breakpoints retain local crossing offsets.
        The independent assignment mass certificate checks the returned measure.
        """
        nodes,weights = leggauss(order)
        rules = []
        for p in self.pieces:
            coordinates,log_weights = [],[]
            if p.rate == 0:
                unit = p.upper-p.lower
                anchor = p.lower+self.error
                edges = sorted({Fraction(0),Fraction(1)} | {
                    (exact(v)-anchor)/unit for v in breakpoints if anchor < v < p.upper+self.error})
                for lo,hi in zip(edges[:-1],edges[1:]):
                    coordinates.append(wide(lo)+wide(hi-lo)/2*(nodes+1))
                    log_weights.append(np.log(wide(hi-lo)/2)+np.log(weights)+p.log_probability)
            else:
                unit = 1/abs(p.rate)
                anchor = p.peak+self.error
                direction = -1 if p.rate > 0 else 1
                extent = wide((p.upper-p.lower)/unit)
                edges = {wide(0),extent}
                edges.update(wide(v) for v in (1,2,4,8,16,32,64) if v < extent)
                edges.update(wide(direction*(exact(v)-anchor)/unit) for v in breakpoints
                             if p.lower+self.error < v < p.upper+self.error)
                edges = sorted(edges)
                log_normalizer = np.log(-np.expm1(-extent))
                for lo,hi in zip(edges[:-1],edges[1:]):
                    t = lo+(hi-lo)/2*(nodes+1)
                    coordinates.append(direction*t)
                    log_weights.append(np.log((hi-lo)/2)+np.log(weights)-t-log_normalizer+p.log_probability)
            rules.append(QuadraturePiece(LocalCoordinates(anchor,unit,np.concatenate(coordinates)),
                                         np.concatenate(log_weights)))
        return tuple(rules)


def validate_count_rates(frame, config, policy, *, eligible_by_key=None):
    """Check the entire factual/intervention support, not only sampled events."""
    if config.registration_probability == 1 and config.denominator_error == 0:
        return
    support = dict(policy.components_by_key)
    if eligible_by_key is None:
        eligible_by_key = {key:exact_shift_intervals(c,policy.delta_mmhg) for key,c in support.items()}
    delta = exact(policy.delta_mmhg)
    for row in range(len(frame.original_ids)):
        key = frame.support_keys[row]
        components = tuple((exact(lo),exact(hi)) for lo,hi in support[key])
        for state, _ in latent_states(config):
            intervals = list(components)
            error = exact(state.error)
            for lo, hi in components:
                for p_lo, p_hi in eligible_by_key[key]:
                    start = max(lo, p_lo-error)
                    end = min(hi, p_hi-error)
                    if start <= end:
                        intervals.append((start+delta, end+delta))
            for lower, upper in intervals:
                lo,hi = wide(lower),wide(upper)
                critical = [lo, hi]
                displacement = config.migration*state.illness
                if config.effect == "nonlinear":
                    first = int(np.ceil((lo-displacement-np.pi)/(2*np.pi)))
                    last = int(np.floor((hi-displacement-np.pi)/(2*np.pi)))
                    critical.extend(displacement+np.pi+2*np.pi*k for k in range(first, last+1))
                if config.effect == "sign_changing" and lo <= 5+displacement <= hi:
                    critical.append(5+displacement)
                require(bool((structural_mean(critical, frame, row, state, config) >= 0).all()),
                        "count scenario has a negative event rate on its factual/intervention support")
