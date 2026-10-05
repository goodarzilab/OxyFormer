"""Small composable Suite A SCMs, conditional on a fixed covariate frame.

Coordinates and dependence IDs are metadata, never adjustment variables. Local
and regional Bernoulli causes are shared within their respective units. The
finite latent mixture permits deterministic integration of the *selected*
observed law, including measurement error, without fitting an oracle regression.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from itertools import product
from fractions import Fraction
from functools import lru_cache
from typing import Literal

import numpy as np
from scipy.special import expit, log_expit, logsumexp
from numpy.polynomial.legendre import leggauss

from oxyformer.design.policies import ShiftOrStayPolicy
from oxyformer.provenance import Immutable, require, unique


# Registered recipe envelopes, also declared in configs/validation/suite_a.yaml.
# Signed quantities contain zero; positive scales retain a nonzero lower bound.
# Physical probability constraints are checked separately after widening.
REGISTERED_NUMERIC_BOX = MappingProxyType({
    "dose": (0., 10.), "delta": (2., 2.), "near_scale": (.05, .05),
    "coefficient": (-2., 2.), "noise_sd": (1., 1.),
    "exposure_error": (.4, .4), "migration": (2., 2.),
    "registration_probability": (.65, 1.), "denominator_error": (.2, .2),
    "coordinate": (0., 10.), "covariate": (-1., 1.), "weight": (1., 1.),
    "paired_c": (-1.5, 1.5), "paired_tau": (-2., 2.), "intervention_dose": (0., 10.),
})
SIGNED_QUANTITIES = frozenset({"dose", "coefficient", "coordinate", "covariate",
                               "paired_c", "paired_tau", "intervention_dose"})
NUMERIC_MARGIN = 100.
_expanded_numeric_box = {
    name: ((-NUMERIC_MARGIN*max(abs(lo), abs(hi)), NUMERIC_MARGIN*max(abs(lo), abs(hi)))
           if name in SIGNED_QUANTITIES else (lo/NUMERIC_MARGIN, hi*NUMERIC_MARGIN))
    for name, (lo, hi) in REGISTERED_NUMERIC_BOX.items()
}
# Wider, independently exercised axes retain the inherited stress regressions.
# They do NOT widen beta/confounding coefficients or the positive shift floor.
# Pair coefficients/doses use exact rational intervention arithmetic, unlike
# nonlinear SCM responses. No attempt to support every float64 box is made.
_expanded_numeric_box.update({
    "dose": (-10010., 10010.),
    "near_scale": (float(np.nextafter(0., 1.)), 1000.),
    "weight": (float(np.nextafter(0., 1.)), 1.7e308),
    "paired_c": (-1e307, 1e307), "paired_tau": (-1e300, 1e300),
    "intervention_dose": (-1e308, 1e308),
})
NUMERIC_DOMAIN = MappingProxyType(_expanded_numeric_box)
del _expanded_numeric_box


def validate_numeric(values, kind, name, *, allow_zero=False):
    """Refuse unsupported inputs before arithmetic; never normalize or clip.

    The domain is fixed, not caller-configurable. Zero is an explicit disabled
    mechanism/zero-weight/identity exception for otherwise positive quantities.
    This check is for declared inputs, not intermediate values or Gaussian draws.
    """
    lower, upper = NUMERIC_DOMAIN[kind]
    array = np.asarray(values)
    require(array.dtype.kind in "fiu", f"{name} must be numeric")
    valid = np.isfinite(array) & (array >= lower) & (array <= upper)
    if allow_zero:
        valid |= array == 0
    require(bool(valid.all()),
            f"{name} outside supported numeric domain: [{lower}, {upper}]"
            + (" or zero" if allow_zero else ""))


def validate_components(components):
    require(bool(components), "SCM assignment needs nonempty support")
    validate_numeric(components, "dose", "support endpoints")
    require(all(lo < hi for lo, hi in components), "invalid support component")
    require(all(first[1] < second[0] for first, second in zip(components, components[1:])),
            "support components must be sorted and separated")


def validate_policy_domain(policy):
    validate_numeric(policy.delta_mmhg, "delta", "delta", allow_zero=True)
    for _, components in policy.components_by_key:
        # Unused empty policy strata are legitimate; assignment strata are not.
        if components:
            validate_components(components)


def validate_seed(seed):
    require(type(seed) is int and 0 <= seed < 2**64, "seed must be an integer in [0, 2**64)")


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
        validate_numeric(self.coordinates, "coordinate", "coordinates")
        validate_numeric([v for row in self.x for v in row if v is not None], "covariate", "X")
        validate_numeric(self.weights, "weight", "origin weights", allow_zero=True)
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
    numeric_domain: Literal["suite-a-100x-v1"] = "suite-a-100x-v1"
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
        validate_numeric((self.beta, self.local_strength, self.regional_strength),
                         "coefficient", "effect/confounding coefficients")
        for name in ("near_scale", "noise_sd", "exposure_error", "migration",
                     "registration_probability", "denominator_error"):
            validate_numeric(getattr(self, name), name, name,
                             allow_zero=name not in ("near_scale", "registration_probability"))
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
        validate_policy_domain(policy)
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
    denominator_factor: Fraction | float = 1.0

    def __post_init__(self):
        require(self.local in (-1., 0., 1.) and self.regional in (-1., 0., 1.)
                and self.illness in (0., 1.), "invalid latent causes")
        validate_numeric(abs(self.error), "exposure_error", "latent exposure error", allow_zero=True)
        factor = self.denominator_factor
        require(isinstance(factor, Fraction) or np.isfinite(factor), "invalid latent denominator factor")
        factor = exact(factor)
        require(0 < factor < 2, "invalid latent denominator factor")
        object.__setattr__(self, "denominator_factor", factor)


def latent_states(config: SCMConfig):
    """Marginal states for one row; sampling separately shares causes by unit."""
    local = [(-1., .5), (1., .5)] if config.local_confounding != "none" else [(0., 1.)]
    regional = [(-1., .5), (1., .5)] if config.regional_confounding != "none" else [(0., 1.)]
    illness = [(0., .7), (1., .3)] if config.has_illness else [(0., 1.)]
    error = [(-config.exposure_error, .5), (config.exposure_error, .5)] if config.exposure_error else [(0., 1.)]
    error_size = exact(config.denominator_error)
    denom = [(1-error_size, .5), (1+error_size, .5)] if error_size else [(Fraction(1), 1.)]
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
    # Keep the retained dose precision through the bounded nonlinear response;
    # converting here to float64 loses supported shifts at large centers.
    a = np.asarray(a, dtype=np.longdouble)
    if config.effect == "null":
        return np.zeros_like(a)
    if config.effect == "linear":
        return config.beta * a
    if config.effect == "nonlinear":
        return config.beta * np.sin(a / 2)
    # Derivative changes sign at dose 5.
    return config.beta * (a - 5)**2 / 10


def structural_mean(a_true, frame, row, state, config):
    # Cancel the complete dose-independent affine expression before rounding.
    # Even bounded X can leave a tiny positive baseline after large cancellation.
    baseline = (Fraction(50) + sum((exact(v) for v in frame.x[row] if v is not None), Fraction(0))/4
                + exact(config.local_strength)*exact(state.local)
                + exact(config.regional_strength)*exact(state.regional) + 2*exact(state.illness))
    mean = wide(baseline) + effect(np.asarray(a_true) - config.migration * state.illness, config)
    return config.registration_probability * mean / wide(state.denominator_factor)


def _count_baseline(frame, row, state, config):
    return (Fraction(50) + sum((exact(v) for v in frame.x[row] if v is not None), Fraction(0))/4
            + exact(config.local_strength)*exact(state.local)
            + exact(config.regional_strength)*exact(state.regional) + 2*exact(state.illness))


def _integer_product_bounds(a, b, scale):
    products = [x*y for x in a for y in b]
    return min(products)//scale, -(-max(products)//scale)


def _sine_bounds(value, bits):
    """Certified rational enclosure, using outward-rounded integer arithmetic.

    Halve to |x| <= 1/2, enclose the sine/cosine Taylor polynomials and their
    alternating remainders, then double angles. Extra working bits limit width;
    correctness does not depend on a platform's libm or extended precision.
    """
    halves = 0
    reduced = value
    while abs(reduced) > Fraction(1, 2):
        reduced /= 2
        halves += 1
    scale = 1 << (bits+4*halves+16)
    scaled = reduced*scale
    x = (scaled.numerator//scaled.denominator, -(-scaled.numerator//scaled.denominator))
    x2 = _integer_product_bounds(x, x, scale)
    sine = sine_term = x
    cosine = cosine_term = (scale, scale)
    n, factorial = 0, 1
    while True:
        n += 1
        def next_term(term, divisor):
            lo, hi = _integer_product_bounds(term, x2, scale)
            return (-hi)//divisor, -(lo//divisor)
        sine_term = next_term(sine_term, (2*n)*(2*n+1))
        cosine_term = next_term(cosine_term, (2*n-1)*(2*n))
        sine = tuple(a+b for a,b in zip(sine, sine_term))
        cosine = tuple(a+b for a,b in zip(cosine, cosine_term))
        factorial *= (2*n-1)*(2*n)
        # Both omitted remainders are at most (1/2)^(2n+2)/(2n+2)!.
        if (1 << (2*n+2))*factorial*(2*n+1)*(2*n+2) >= scale:
            break
    sine = (sine[0]-1, sine[1]+1)
    cosine = (cosine[0]-1, cosine[1]+1)
    for _ in range(halves):
        sc = _integer_product_bounds(sine, cosine, scale)
        cc = _integer_product_bounds(cosine, cosine, scale)
        ss = _integer_product_bounds(sine, sine, scale)
        sine, cosine = (2*sc[0], 2*sc[1]), (cc[0]-ss[1], cc[1]-ss[0])
    return Fraction(sine[0], scale), Fraction(sine[1], scale)


@lru_cache(maxsize=16)
def _pi_bounds(bits):
    """Machin's identity with exact alternating-series remainder bounds."""
    def atan_reciprocal(q):
        total, power, n = Fraction(0), q, 0
        while True:
            total += Fraction((-1)**n, (2*n+1)*power)
            n += 1
            power *= q*q
            remainder = Fraction((-1)**n, (2*n+1)*power)
            if abs(remainder) <= Fraction(1, 1 << (bits+6)):
                return min(total, total+remainder), max(total, total+remainder)
    a, b = atan_reciprocal(5), atan_reciprocal(239)
    return 16*a[0]-4*b[1], 16*a[1]-4*b[0]


def _contains_sine_minimum(lower, upper, beta):
    """Does the exact dose interval contain a minimum of beta*sin(dose/2)?"""
    pl, ph = _pi_bounds(80)
    ratios = (lower/pl, lower/ph, upper/pl, upper/ph)
    start = min(ratios).__floor__()-1
    stop = max(ratios).__ceil__()+1
    offset = 1 if beta < 0 else -1
    first = offset+4*((start-offset+3)//4)
    for q in range(first, stop+1, 4):
        bits = 80
        while True:
            pl, ph = _pi_bounds(bits)
            lo, hi = sorted((q*pl, q*ph))
            if hi < lower or lo > upper:
                break
            if lower <= lo and hi <= upper:
                return True
            # Nonzero rational endpoints cannot equal an odd multiple of pi.
            bits *= 2
    return False


def count_event_rate(a_true, frame, row, state, config, *, poisson_intensity=False):
    """Decide the raw event-rate sign before rounding its complete expression.

    Polynomial rates are rational. For sine, refine rigorous enclosures until
    both the sign and the final float rounding are decided. This predicate and
    sampler boundary intentionally do not change smooth truth integration.
    Registration/denominator factors are positive and applied by the sampler;
    never reconstruct a raw rate by undoing separately rounded factors.
    """
    baseline = _count_baseline(frame, row, state, config)
    dose = exact(a_true)-exact(config.migration)*exact(state.illness)
    beta = exact(config.beta)
    scale = 100 if poisson_intensity else 1
    message = "count scenario has a negative event rate on its factual/intervention support"
    if config.effect == "null" or beta == 0:
        rate = baseline
    elif config.effect == "linear":
        rate = baseline+beta*dose
    elif config.effect == "sign_changing":
        rate = baseline+beta*(dose-5)**2/10
    elif dose == 0:
        rate = baseline
    else:
        bits = 80
        while True:
            bounds = _sine_bounds(dose/2, bits)
            lower, upper = sorted(baseline+beta*v for v in bounds)
            require(upper >= 0, message)
            if lower >= 0 and float(scale*lower) == float(scale*upper):
                return float(scale*lower)
            bits *= 2
    require(rate >= 0, message)
    return float(scale*rate)


def _validate_count_interval(lower, upper, frame, row, state, config):
    displacement = exact(config.migration)*exact(state.illness)
    if config.effect == "nonlinear":
        baseline = _count_baseline(frame, row, state, config)
        beta = exact(config.beta)
        if baseline >= abs(beta):
            return  # Exact global lower bound, including a zero extremum.
        require(not _contains_sine_minimum(lower-displacement, upper-displacement, beta),
                "count scenario has a negative event rate on its factual/intervention support")
    for endpoint in (lower, upper):
        count_event_rate(endpoint, frame, row, state, config)
    vertex = 5+displacement
    if config.effect == "sign_changing" and lower <= vertex <= upper:
        count_event_rate(vertex, frame, row, state, config)


def observation_probabilities(a_observed, state, config):
    """Factual observation mechanisms; independent of outcome noise given state."""
    a = np.asarray(a_observed)
    flag = expit(1 - .2*a - 1.2*state.illness) if config.selected_outcome else np.ones_like(a, dtype=float)
    survey = expit(.7 - .12*a + .5*state.local) if config.survey_inclusion else np.ones_like(a, dtype=float)
    bio = expit(1 - .1*a - .8*state.illness) if config.missing_biomarkers else np.ones_like(a, dtype=float)
    return flag, survey, bio


def exact(value):
    """Exact geometry of a declared float, not a decimal reinterpretation."""
    if isinstance(value, Fraction):
        return value
    if hasattr(value, "as_integer_ratio"):
        return Fraction(*value.as_integer_ratio())
    return Fraction(value)


def exact_shift_intervals(components, delta):
    """The frozen mathematical S=[L,U-delta] on exact declared float values.

    Derive widths and cutoffs before any rounding, so a shifted eligible dose
    remains in its origin component even when the law is narrower than an ULP.
    This continuous truth geometry precedes serialization of observed doses.
    """
    validate_numeric(delta, "delta", "delta", allow_zero=True)
    if components:
        validate_components(components)
    shift = exact(delta)
    if shift == 0:
        return ()
    bounds = tuple((exact(lo),exact(hi)) for lo,hi in components)
    return tuple((lo,hi-shift) for lo,hi in bounds if hi-lo >= shift)


def wide(value):
    """Round an exact rational once to longdouble, with ties to even.

    Integer quotient/remainder avoids independently rounding numerator and
    denominator, and avoids decimal-string limits during enclosure refinement.
    The spacing floor also handles subnormal results without double rounding.
    """
    if not isinstance(value, Fraction):
        return np.longdouble(value)
    if value == 0:
        return np.longdouble(0)
    numerator, denominator = abs(value.numerator), value.denominator
    exponent = numerator.bit_length()-denominator.bit_length()
    if (numerator < denominator << exponent if exponent >= 0
            else numerator << -exponent < denominator):
        exponent -= 1
    info = np.finfo(np.longdouble)
    spacing = max(exponent-info.nmant, info.minexp-info.nmant)
    if spacing >= 0:
        denominator <<= spacing
    else:
        numerator <<= -spacing
    significand, remainder = divmod(numerator, denominator)
    if 2*remainder > denominator or (2*remainder == denominator and significand % 2):
        significand += 1
    rounded = np.ldexp(np.longdouble(significand), spacing)
    return -rounded if value < 0 else rounded


def directed_bound(value, *, upward):
    """Round a rational toward the interior of a closed longdouble bound.

    Compare each candidate as a rational, including the adjacent float, so the
    decision does not depend on the rounding in numerator/denominator division.
    All simulator geometry lies within longdouble's finite exponent range.
    """
    rounded = wide(value)
    direction = np.longdouble(np.inf if upward else -np.inf)
    while (exact(rounded) < value if upward else exact(rounded) > value):
        rounded = np.nextafter(rounded, direction)
    while True:
        neighbour = np.nextafter(rounded, -direction)
        if (exact(neighbour) < value if upward else exact(neighbour) > value):
            return rounded
        rounded = neighbour


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
        # A nearest-rounded rational cutoff can include an excluded node. Use
        # the first/last representable value *inside* each exact closed bound.
        lower = directed_bound((exact(lower)-self.anchor)/self.unit, upward=True)
        upper = directed_bound((exact(upper)-self.anchor)/self.unit, upward=False)
        return (self.values >= lower) & (self.values <= upper)


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


def observation_transition_points(state, config):
    """Resolve every active logistic gate in its own logit units.

    Broad assignment panels must not jump across selection transitions. Cuts
    include both tails and each gate separately, including products of gates.
    These are integration panels only; no positive mass is discarded.
    """
    gates = []
    if config.selected_outcome:
        gates.append((exact(1)-exact(1.2)*exact(state.illness), exact(.2)))
    if config.survey_inclusion:
        gates.append((exact(.7)+exact(.5)*exact(state.local), exact(.12)))
    if config.missing_biomarkers:
        gates.append((exact(1)-exact(.8)*exact(state.illness), exact(.1)))
    logits = (-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64)
    return tuple((intercept-logit)/slope for intercept, slope in gates for logit in logits)


class AssignmentLaw:
    """Normalized exponential pieces with exact coarse log-kernel geometry.

    Kernel differences are cancelled as rationals before conversion to numeric
    log weights. This retains ordinary tilt beside an arbitrarily narrow
    Laplace kernel. No component is removed from truth or posterior evaluation
    because its ordinary probability underflows. Public draws are floats.
    """
    def __init__(self, frame, row, state, config, components):
        validate_components(components)
        require(type(row) is int and 0 <= row < len(frame.original_ids), "invalid frame row")
        self.components = components
        self.error = exact(state.error)
        self.scale = exact(config.near_scale)
        self.near = config.assignment == "near_deterministic"
        rate = Fraction(-4 if config.extreme_ratios else 0)
        if not self.near:
            rate += (exact(.3)*exact(state.local)+exact(.2)*exact(state.regional)
                     -exact(.25)*exact(state.illness))
        self.rate = rate
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
        a = np.asarray(a_observed)
        return np.array([any(p.lower+self.error <= exact(value) <= p.upper+self.error
                             for p in self.pieces) for value in a.ravel()]).reshape(a.shape)

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

    def sample_count_dose(self, rng):
        """Retain the true local dose for rates; serialize only the observation."""
        index = rng.choice(len(self.pieces),p=self.probabilities)
        coordinates = self.quantile_coordinates(index,rng.random())
        # as_integer_ratio preserves the extended-precision local draw, including
        # offsets much smaller than an ULP of the absolute recorded exposure.
        local = Fraction(*np.longdouble(coordinates.values).as_integer_ratio())
        true_dose = coordinates.anchor-self.error+coordinates.unit*local
        piece = self.pieces[index]
        if not piece.lower <= true_dose <= piece.upper:
            raise ArithmeticError("count inverse-transform draw left its exact support")
        return float(true_dose+self.error), true_dose

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
    validate_policy_domain(policy)
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
                # Keep smooth endpoints observable to integration diagnostics;
                # their rounded values are not evidence about the rate's sign.
                structural_mean([wide(lower), wide(upper)], frame, row, state, config)
                _validate_count_interval(lower, upper, frame, row, state, config)
