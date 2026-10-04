"""Frozen, outcome-blind shift-or-stay policies on continuous support.

Support keys describe frozen geography-level policy strata, not individual
outcome covariates. The design producer supplies these keys and the supported
components; this module neither estimates nor updates support.
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np

from oxyformer.provenance import Immutable, check_hash, nonempty, require, unique


def fp64_vector(values, name: str, size: int | None = None) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    require(vector.ndim == 1 and vector.size > 0, f"{name} must be a nonempty vector")
    require(size is None or vector.size == size, f"{name} alignment mismatch")
    require(bool(np.isfinite(vector).all()), f"nonfinite {name}")
    return vector


def origin_weights(values, size: int) -> np.ndarray:
    weights = fp64_vector(values, "origin weights", size)
    require(bool((weights >= 0).all()) and bool((weights > 0).any()), "invalid origin weights")
    return weights


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyCovariates(Immutable):
    original_ids: tuple[str, ...]
    geography_ids: tuple[str, ...]
    support_keys: tuple[str, ...]

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(bool(self.original_ids), "empty policy observations")
        unique(self.original_ids, "original IDs")
        require(len(self.original_ids) == len(self.geography_ids) == len(self.support_keys),
                "policy covariate alignment mismatch")
        for value in self.original_ids + self.geography_ids + self.support_keys:
            nonempty(value, "policy identity")


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyResult(Immutable):
    policy_id: str
    covariates: PolicyCovariates
    a_mmhg: tuple[float, ...]
    d_mmhg: tuple[float, ...]
    moved: tuple[bool, ...]
    identity: bool

    def __post_init__(self):
        Immutable.__post_init__(self)
        check_hash(self.policy_id, "policy ID")
        require(len(self.a_mmhg) == len(self.d_mmhg) == len(self.moved)
                == len(self.covariates.original_ids), "policy result alignment mismatch")
        require(self.moved == tuple(a != d for a, d in zip(self.a_mmhg, self.d_mmhg)),
                "inconsistent moved indicator")
        require(not self.identity or not any(self.moved), "identity policy moved observations")


@dataclass(frozen=True, slots=True, kw_only=True)
class ShiftOrStayPolicy(Immutable):
    """Hash-bound frozen components, with positive shifts in mmHg.

    Components must be sorted, separated closed intervals. A whole shift must
    remain inside its origin component. Empty component lists mean stay.
    Genuine atoms are unsupported; duplicate sampled exposures are allowed.
    """
    support_design_hash: str
    components_by_key: tuple[tuple[str, tuple[tuple[float, float], ...]], ...]
    delta_mmhg: float = 2.0
    exposure_law: Literal["continuous", "mixed", "discrete"] = "continuous"

    def __post_init__(self):
        Immutable.__post_init__(self)
        check_hash(self.support_design_hash, "support design hash")
        require(self.exposure_law == "continuous", "mixed/discrete exposure laws require a measure derivation")
        require(self.delta_mmhg >= 0, "negative shift is not a declared policy")
        require(bool(self.components_by_key), "frozen support keys required")
        unique(tuple(k for k, _ in self.components_by_key), "support keys")
        for key, components in self.components_by_key:
            nonempty(key, "support key")
            previous = -np.inf
            for lower, upper in components:
                require(lower < upper, "support component must have positive width; atoms unsupported")
                require(previous < lower, "support components must be sorted and separated")
                previous = upper
        object.__setattr__(self, "components_by_key", tuple(sorted(self.components_by_key)))

    @property
    def policy_id(self) -> str:
        return self.content_hash

    @property
    def is_identity(self) -> bool:
        return self.delta_mmhg == 0 or all(
            upper - lower < self.delta_mmhg
            for _, components in self.components_by_key for lower, upper in components
        )

    def shift_mask(self, a_mmhg, support_keys) -> np.ndarray:
        a = fp64_vector(a_mmhg, "exposure")
        keys = tuple(support_keys)
        require(len(keys) == len(a), "support key alignment mismatch")
        support = dict(self.components_by_key)
        require(all(key in support for key in keys), "unknown frozen support key")
        mask = np.zeros(len(a), dtype=bool)
        if self.delta_mmhg == 0:
            return mask
        for i, (value, key) in enumerate(zip(a, keys)):
            mask[i] = any(lower <= value <= upper - self.delta_mmhg
                          and upper - lower >= self.delta_mmhg
                          for lower, upper in support[key])
        return mask

    def apply(self, a_mmhg, policy_covariates: PolicyCovariates) -> PolicyResult:
        a = fp64_vector(a_mmhg, "exposure", len(policy_covariates.original_ids))
        by_geography = {}
        for geography, key, value in zip(policy_covariates.geography_ids,
                                         policy_covariates.support_keys, a):
            current = (key, float(value))
            previous = by_geography.setdefault(geography, current)
            require(previous == current, "inconsistent exposure or support within assignment geography")
        moved = self.shift_mask(a, policy_covariates.support_keys)
        d = a.copy()
        d[moved] += self.delta_mmhg
        return PolicyResult(policy_id=self.policy_id, covariates=policy_covariates,
                            a_mmhg=tuple(a.tolist()), d_mmhg=tuple(d.tolist()),
                            moved=tuple(moved.tolist()), identity=self.is_identity)


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyPairs(Immutable):
    """Classifier pairs; split/sampling must keep equal original IDs together."""
    policy_id: str
    weight_id: str
    original_ids: tuple[str, ...]
    geography_ids: tuple[str, ...]
    support_keys: tuple[str, ...]
    a_mmhg: tuple[float, ...]
    transformed: tuple[bool, ...]
    origin_weights: tuple[float, ...]


def paired_records(result: PolicyResult, weights, *, weight_id: str) -> PolicyPairs:
    """Copy each origin weight verbatim, including for unchanged destinations."""
    nonempty(weight_id, "weight ID")
    w = tuple(origin_weights(weights, len(result.a_mmhg)).tolist())
    cov = result.covariates
    n = len(w)
    return PolicyPairs(policy_id=result.policy_id, weight_id=weight_id,
                       original_ids=cov.original_ids * 2, geography_ids=cov.geography_ids * 2,
                       support_keys=cov.support_keys * 2, a_mmhg=result.a_mmhg + result.d_mmhg,
                       transformed=(False,) * n + (True,) * n, origin_weights=w * 2)


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyDiagnostics(Immutable):
    original_ids: tuple[str, ...]
    moved: tuple[bool, ...]
    affected: tuple[bool, ...]
    moved_fraction: float
    affected_fraction: float
    average_shift_mmhg: float


def policy_diagnostics(result: PolicyResult, ratios, weights) -> PolicyDiagnostics:
    """Affected means moved OR r(A) != 1; this is not a score multiplier.

    With estimated ratios, affected is a fitted diagnostic (exact comparison),
    not evidence of the true affected set. Fractions use origin target weights.
    """
    from oxyformer.estimation.influence import normalized_weights

    n = len(result.a_mmhg)
    r = fp64_vector(ratios, "ratios", n)
    require(bool((r >= 0).all()), "negative density ratio")
    if result.identity:
        r = np.ones(n, dtype=np.float64)
    moved = np.asarray(result.moved, dtype=bool)
    affected = moved | (r != 1)
    w = normalized_weights(weights, n)
    return PolicyDiagnostics(original_ids=result.covariates.original_ids,
                             moved=result.moved, affected=tuple(affected.tolist()),
                             moved_fraction=float(w @ moved), affected_fraction=float(w @ affected),
                             average_shift_mmhg=float(w @ (np.asarray(result.d_mmhg)
                                                          - np.asarray(result.a_mmhg))))
