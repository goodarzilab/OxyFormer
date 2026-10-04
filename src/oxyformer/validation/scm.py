"""Small composable Suite A SCMs, conditional on a fixed covariate frame.

Coordinates and dependence IDs are metadata, never adjustment variables. Local
and regional Bernoulli causes are shared within their respective units. The
finite latent mixture permits deterministic integration of the *selected*
observed law, including measurement error, without fitting an oracle regression.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
from scipy.special import expit
from scipy.stats import laplace

from oxyformer.design.policies import ShiftOrStayPolicy
from oxyformer.provenance import Immutable, require, unique


@dataclass(frozen=True, slots=True, kw_only=True)
class CovariateFrame(Immutable):
    """Caller-supplied, already approved X; no real data are bundled here.

    Every row is retained, including missing outcomes/biomarkers. Thus geometry,
    X (including None), origin weights and all cluster sizes survive simulation.
    Coordinates refer to current residence. Migration models prior residence.
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
        require(all(w >= 0 for w in self.weights) and sum(self.weights) > 0, "invalid weights")
        by_geo = {}
        by_cluster = {}
        for geo, region, cluster, coord, key in zip(self.geography_ids, self.region_ids,
                self.cluster_ids, self.coordinates, self.support_keys):
            value = (region, coord, key)
            require(by_geo.setdefault(geo, value) == value, "inconsistent assignment geography")
            require(by_cluster.setdefault(cluster, geo) == geo, "cluster spans assignment geographies")


@dataclass(frozen=True, slots=True, kw_only=True)
class SCMConfig(Immutable):
    """Explicit mechanism switches, not a Cartesian scenario grid.

    Mean endpoint = 50 + .25 sum(X_observed) + f(A_true - migration*illness)
    + local_strength*U_local + regional_strength*U_region + 2*illness.
    Missing X is retained; its baseline contribution is defined as zero.
    Assignment log-density slope is .3 U_local + .2 U_region - .25 illness
    when the corresponding mechanism is enabled. Near-deterministic assignment
    instead uses a truncated Laplace law around a location-dependent center.
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

    def validate_policy(self, policy: ShiftOrStayPolicy, frame: CovariateFrame):
        support = dict(policy.components_by_key)
        require(set(frame.support_keys) <= support.keys(), "unknown frame support key")
        used = [support[k] for k in set(frame.support_keys)]
        require(all(components for components in used), "SCM assignment needs nonempty support")
        require(self.support_gaps == any(len(c) > 1 for c in used), "support_gaps switch disagrees with frozen support")
        eligible = [any(hi - lo >= policy.delta_mmhg for lo, hi in c) for c in used]
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


class AssignmentLaw:
    """Continuous exponential tilt or near-deterministic Laplace on components.

    Atoms are generated explicitly by the harness and never passed as densities.
    Error is a shared geography-level additive measurement error; no clipping.
    """
    def __init__(self, frame, row, state, config, components):
        self.components = components
        self.error = state.error
        self.scale = config.near_scale
        self.near = config.assignment == "near_deterministic"
        self.rate = (-4. if config.extreme_ratios else 0.) + .3*state.local + .2*state.regional - .25*state.illness
        lo, hi = components[0][0], components[-1][1]
        self.center = lo + (hi-lo) * float(expit(frame.coordinates[row][0] + .2*state.local + .1*state.regional - .2*state.illness))
        self.anchor = hi if self.rate > 0 else lo
        self.masses = np.array([self._mass(l, h) for l, h in components])
        self.total = float(self.masses.sum())
        require(self.total > 0 and np.isfinite(self.total), "assignment normalization failed")

    def _mass(self, lo, hi):
        if self.near:
            # Survival differences on the right avoid subtraction near one.
            if lo >= self.center:
                return laplace.sf(lo, self.center, self.scale) - laplace.sf(hi, self.center, self.scale)
            return laplace.cdf(hi, self.center, self.scale) - laplace.cdf(lo, self.center, self.scale)
        if self.rate == 0:
            return hi-lo
        return (np.exp(self.rate*(hi-self.anchor)) - np.exp(self.rate*(lo-self.anchor))) / self.rate

    @property
    def breakpoints(self):
        points = [v+self.error for comp in self.components for v in comp]
        if self.near:
            points.append(self.center+self.error)
        return points

    def density(self, a_observed):
        a = np.asarray(a_observed) - self.error
        inside = np.zeros(a.shape, dtype=bool)
        for lo, hi in self.components:
            inside |= (a >= lo) & (a <= hi)
        if self.near:
            raw = laplace.pdf(a, self.center, self.scale)
        else:
            # Evaluate only on support, avoiding overflow outside the law.
            raw = np.exp(self.rate*(np.where(inside, a, self.anchor)-self.anchor))
        return np.where(inside, raw / self.total, 0.)

    def sample(self, rng):
        lo, hi = self.components[rng.choice(len(self.components), p=self.masses/self.total)]
        u = rng.random()
        if self.near:
            if lo >= self.center:
                survival = laplace.sf(lo, self.center, self.scale) - u*self._mass(lo, hi)
                value = laplace.isf(survival, self.center, self.scale)
            else:
                value = laplace.ppf(laplace.cdf(lo, self.center, self.scale) + u*self._mass(lo, hi), self.center, self.scale)
        elif self.rate == 0:
            value = lo + u*(hi-lo)
        else:
            left, right = np.exp(self.rate*(lo-self.anchor)), np.exp(self.rate*(hi-self.anchor))
            value = self.anchor + np.log(left + u*(right-left))/self.rate
        return float(value + self.error)


def validate_count_rates(frame, config, policy):
    """Check the entire factual/intervention support, not only sampled events."""
    if config.registration_probability == 1 and config.denominator_error == 0:
        return
    support = dict(policy.components_by_key)
    for row in range(len(frame.original_ids)):
        components = support[frame.support_keys[row]]
        for state, _ in latent_states(config):
            intervals = list(components)
            for lo, hi in components:
                for p_lo, p_hi in components:
                    start = max(lo, p_lo-state.error)
                    end = min(hi, p_hi-policy.delta_mmhg-state.error)
                    if start <= end:
                        intervals.append((start+policy.delta_mmhg, end+policy.delta_mmhg))
            for lo, hi in intervals:
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
