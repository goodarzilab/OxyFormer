"""Pooled scalar CV-TMLE on frozen OOF predictions (plan section 4.4).

No model or optimizer enters this module. Outcomes are already on the declared
endpoint scale; for Poisson they are rates Y*=D/N, never counts. Population
exposure N and target mass w remain separate. All arithmetic is FP64.
"""
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

from oxyformer.contracts import Estimate, EstimandSpec, OOFNuisances, SplitManifest
from oxyformer.data.loaders import LoadedData, join_outcomes
from oxyformer.design.policies import ShiftOrStayPolicy, fp64_vector, origin_weights
from oxyformer.estimation.influence import normalized_weights
from oxyformer.provenance import require


@dataclass(frozen=True)
class Fluctuation:
    """Numerical result for one endpoint/seed, in the input observation order.

    ``moment`` is sum(w*r*(Y-mu*))/W, the negative loss derivative.
    ``score_gap`` is the corrected-score mean minus the plug-in contrast.
    Both should vanish, and sum(influence) equals score_gap up to rounding.
    Infinite epsilon denotes a documented Bernoulli/Poisson boundary optimum;
    zero ratios keep their initial prediction in that limit.
    """
    epsilon: float
    mu_a: tuple[float, ...]
    mu_d: tuple[float, ...]
    value: float
    scores: tuple[float, ...]
    influence: tuple[float, ...]
    moment: float
    score_gap: float
    iterations: int
    status: str


@dataclass(frozen=True)
class TargetingResult:
    estimate: Estimate
    # Seed labels are explicit; diagnostics never imply additional observations.
    by_seed: tuple[tuple[int, Fluctuation], ...]


def _inputs(mu, ratios, outcomes, weights, likelihood, population):
    require(likelihood in ("identity", "logistic", "poisson"), "unknown likelihood")
    mu = fp64_vector(mu, "mu_a")
    r = fp64_vector(ratios, "r_a", len(mu))
    y = fp64_vector(outcomes, "outcomes", len(mu))
    q = normalized_weights(weights, len(mu))
    require(bool((r >= 0).all()), "negative density ratio")
    if likelihood == "logistic":
        require(bool(((mu > 0) & (mu < 1)).all()), "logistic initial means must lie in (0,1)")
        require(bool(((y == 0) | (y == 1)).all()), "binary outcomes must be 0 or 1")
    if likelihood == "poisson":
        require(bool((mu > 0).all()) and bool((y >= 0).all()), "positive rates and nonnegative outcomes required")
        require(population is not None, "Poisson population exposure required separately from target weights")
        n = fp64_vector(population, "population exposure", len(mu))
        require(bool((n > 0).all()), "population exposure must be positive")
    else:
        require(population is None, "population exposure is only used with Poisson rates")
    return mu, r, y, q


def _updated(mu, ratio, epsilon, likelihood):
    if likelihood == "identity":
        return mu + epsilon * ratio
    if np.isinf(epsilon):
        result = mu.copy()
        result[ratio > 0] = 0.0 if epsilon < 0 else 1.0
        return result
    eta = (logit(mu) if likelihood == "logistic" else np.log(mu)) + epsilon * ratio
    with np.errstate(over="ignore", under="ignore"):
        return expit(eta) if likelihood == "logistic" else np.exp(eta)


def fluctuation_loss(epsilon, mu_a, r_a, outcomes, weights, likelihood, *, population=None):
    """Weighted proper loss / W, omitting constants independent of epsilon.

    Identity uses half squared error; logistic uses Bernoulli cross entropy.
    For Poisson, (w/N)*ell(D,N*lambda), D=N*Y*, reduces to
    w*(lambda-Y*log(lambda)) plus epsilon-independent offset/factorial terms.
    N is validated separately and never substituted for w or multiplied into it.
    Initial rate predictions and Y* must use the same per-person exposure unit.
    """
    require(np.isfinite(epsilon), "loss evaluation requires finite epsilon")
    mu, r, y, q = _inputs(mu_a, r_a, outcomes, weights, likelihood, population)
    active = q > 0
    mu, r, y, q = mu[active], r[active], y[active], q[active]
    if likelihood == "identity":
        loss = 0.5 * (y - mu - epsilon * r)**2
    else:
        eta = (logit(mu) if likelihood == "logistic" else np.log(mu)) + epsilon * r
        loss = (np.logaddexp(0, eta) if likelihood == "logistic" else np.exp(eta)) - y * eta
    return float(q @ loss)


def fit_fluctuation(mu_a, mu_d, r_a, r_d, outcomes, weights, likelihood, *,
                    population=None, identity=False, tolerance=1e-10) -> Fluctuation:
    """Solve one pooled moment, then evaluate with the *shifted* clever covariate.

    The tolerance is relative to max(1, weighted absolute r*Y and r*mu*).
    A finite optimizer is never fabricated for a boundary optimum. A flat loss
    (all positive-mass factual ratios zero) retains epsilon=0 and is flagged.
    Identity is a property of the frozen policy, not the observed moved fraction.
    """
    # Identity contrasts are known without fitting a likelihood, including at
    # saturated initial means where a finite logit would be unavailable.
    if identity:
        mu = fp64_vector(mu_a, "mu_a")
        for values, name in ((mu_d, "mu_d"), (r_a, "r_a"), (r_d, "r_d"), (outcomes, "outcomes")):
            fp64_vector(values, name, len(mu))
        origin_weights(weights, len(mu))
        zero = (0.0,) * len(mu)
        return Fluctuation(0.0, tuple(mu), tuple(mu), 0.0, zero, zero, 0.0, 0.0, 0, "identity")
    mu, r, y, q = _inputs(mu_a, r_a, outcomes, weights, likelihood, population)
    md = fp64_vector(mu_d, "mu_d", len(mu))
    rd = fp64_vector(r_d, "r_d", len(mu))
    require(bool((rd >= 0).all()), "negative shifted density ratio")
    require(np.isfinite(tolerance) and 0 < tolerance < 1, "invalid targeting tolerance")
    if likelihood == "logistic":
        require(bool(((md > 0) & (md < 1)).all()), "logistic shifted means must lie in (0,1)")
    if likelihood == "poisson":
        require(bool((md > 0).all()), "positive shifted rates required")
    active = (q > 0) & (r > 0)
    iterations, status = 0, "converged"
    # Scale the scalar optimization coordinate to avoid squaring large ratios.
    scale = float(r[active].max()) if active.any() else 1.0
    h = r[active] / scale
    qa, ya, ma = q[active], y[active], mu[active]

    def moment(t):
        return float((qa * h) @ (ya - _updated(ma, h, t, likelihood)))

    if not active.any():
        epsilon, status = 0.0, "flat"
    elif likelihood == "identity":
        epsilon = float(((qa * h) @ (ya - ma)) / ((qa * h) @ h) / scale)
    elif np.all(ya == 0):
        epsilon, status = -np.inf, "boundary"
    elif likelihood == "logistic" and np.all(ya == 1):
        epsilon, status = np.inf, "boundary"
    elif moment(0.0) == 0:
        epsilon = 0.0
    else:
        # The residual moment is monotone decreasing. Bracket the global optimum.
        lo, hi = -1.0, 1.0
        for _ in range(1024):
            if moment(lo) >= 0 and moment(hi) <= 0:
                break
            if moment(lo) < 0:
                lo *= 2
            if moment(hi) > 0:
                hi *= 2
        else:
            raise ValueError("could not bracket targeting moment")
        root, info = brentq(moment, lo, hi, xtol=1e-13, rtol=1e-14, maxiter=1000, full_output=True)
        epsilon, iterations = root / scale, info.iterations
    factual = _updated(mu, r, epsilon, likelihood)
    shifted = _updated(md, rd, epsilon, likelihood)
    require(bool(np.isfinite(factual).all() & np.isfinite(shifted).all()), "nonfinite targeted predictions")
    correction = r * (y - factual)
    value = float(q @ (shifted - y))
    scores = shifted - y + correction
    influence = q * (scores - value)
    residual = float(q @ correction)
    size = max(1.0, float(q @ (r * np.abs(y))), float(q @ (r * np.abs(factual))))
    require(abs(residual) <= tolerance * size, "targeting residual exceeds tolerance")
    require(np.isfinite(value) and bool(np.isfinite(scores).all() & np.isfinite(influence).all()),
            "nonfinite targeted scores")
    return Fluctuation(float(epsilon), tuple(factual), tuple(shifted), value,
                       tuple(scores), tuple(influence), residual, float(q @ scores - value),
                       iterations, status)


def cv_tmle(nuisances: OOFNuisances, outcomes: LoadedData, weights: Mapping[str, float],
            likelihood: str, spec: EstimandSpec, *, split: SplitManifest,
            policy: ShiftOrStayPolicy, population: Mapping[str, float] | None = None) -> TargetingResult:
    """Join labels once; pool folds within each seed; average by original ID.

    Returns the merged Estimate contract plus seed-specific numerical diagnostics.
    ``population`` is required only for rates, keyed by original ID. It declares
    exposure denominators for the already rate-scale LoadedData outcomes; it is
    not an outcome conversion. Endpoint adapters own the denominator validation.
    Scores are residual-corrected H*, while value is the requested plug-in
    contrast. Their numerical difference is retained in each seed's diagnostics.
    """
    spec.assert_compatible(nuisances.spec)
    require(spec.policy_id == policy.policy_id, "policy_id mismatch")
    require(split.level == "outer", "CV-TMLE requires outer held-out predictions")
    y = np.asarray(join_outcomes(nuisances, outcomes, split, spec), dtype=np.float64)
    ids = split.original_ids
    require(set(weights) == set(ids), "weight observation IDs mismatch")
    w = origin_weights([weights[oid] for oid in ids], len(ids))
    by_id = dict(zip(ids, w))
    require(all(by_id[oid] == weight for oid, weight in zip(nuisances.original_ids, nuisances.origin_weights)),
            "origin weight mismatch")
    if population is not None:
        require(set(population) == set(ids), "population observation IDs mismatch")
    rows = {(oid, seed): i for i, (oid, seed) in enumerate(zip(nuisances.original_ids, nuisances.seed_ids))}
    predictions = [np.asarray(getattr(nuisances, field), dtype=np.float64)
                   for field in ("mu_a", "mu_d", "r_a", "r_d")]
    results = []
    for seed in split.seed_ids:
        indices = [rows[oid, seed] for oid in ids]
        result = fit_fluctuation(*(v[indices] for v in predictions), y[indices], w, likelihood,
                                 population=None if population is None else [population[oid] for oid in ids],
                                 identity=policy.is_identity)
        results.append((seed, result))
    scores = np.mean([r.scores for _, r in results], axis=0, dtype=np.float64)
    influence = np.mean([r.influence for _, r in results], axis=0, dtype=np.float64)
    value = float(np.mean([r.value for _, r in results], dtype=np.float64))
    lineage = replace(nuisances.lineage, unit_ids=ids,
                      parent_hashes=(nuisances.content_hash, outcomes.content_hash,
                                     split.content_hash, policy.content_hash),
                      model_hash=None, seed=None, parameter_count=None)
    estimate = Estimate(spec=spec, method="cv_tmle_" + likelihood, value=value, standard_error=None,
                        original_ids=ids, scores=tuple(scores.tolist()), influence=tuple(influence.tolist()),
                        seed_ids=split.seed_ids, lineage=lineage)
    return TargetingResult(estimate, tuple(results))
