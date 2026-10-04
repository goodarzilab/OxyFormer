"""Continuous target-law pushforwards and the frozen MTP one-step estimator."""
from dataclasses import replace
from typing import Callable, Mapping

import numpy as np

from oxyformer.contracts import Estimate, EstimandSpec, OOFNuisances, SplitManifest
from oxyformer.data.loaders import LoadedData, join_outcomes
from oxyformer.design.policies import ShiftOrStayPolicy, fp64_vector, origin_weights
from oxyformer.estimation.influence import influence_contributions
from oxyformer.provenance import require

# The callback is g_T(a | x), including exposure-dependent target weighting.
# It receives only active evaluation points and their frozen support keys.
TargetDensity = Callable[[np.ndarray, tuple[str, ...]], np.ndarray]


def pushforward_density(policy: ShiftOrStayPolicy, b_mmhg, support_keys,
                        target_density: TargetDensity, *, exposure_law: str) -> np.ndarray:
    """g_T(b-delta) 1[b-delta in S] + g_T(b) 1[b not in S].

    This is a continuous-measure formula only. Density callbacks must represent
    the endpoint target law, not the unweighted sampling law. In particular,
    weight(b-delta) travels with incoming mass; never replace it by weight(b).
    """
    require(exposure_law == policy.exposure_law == "continuous",
            "mixed/discrete exposure laws require a measure derivation")
    b = fp64_vector(b_mmhg, "destination exposure")
    keys = tuple(support_keys)
    unchanged = ~policy.shift_mask(b, keys)
    incoming = b - policy.delta_mmhg
    shifted = policy.shift_mask(incoming, keys)
    density = np.zeros(len(b), dtype=np.float64)
    for points, mask in ((incoming, shifted), (b, unchanged)):
        if mask.any():
            active_keys = tuple(key for key, active in zip(keys, mask) if active)
            values = fp64_vector(target_density(points[mask], active_keys), "target density", int(mask.sum()))
            require(bool((values >= 0).all()), "negative target density")
            density[mask] += values
    require(bool(np.isfinite(density).all()), "nonfinite pushforward density")
    return density


def pushforward_ratio(policy: ShiftOrStayPolicy, b_mmhg, support_keys,
                      target_density: TargetDensity, *, exposure_law: str) -> np.ndarray:
    b = fp64_vector(b_mmhg, "destination exposure")
    keys = tuple(support_keys)
    numerator = pushforward_density(policy, b, keys, target_density, exposure_law=exposure_law)
    denominator = fp64_vector(target_density(b, keys), "target density", len(b))
    require(bool((denominator > 0).all()), "ratio requires positive target density at evaluated observations")
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = numerator / denominator
    require(bool(np.isfinite(ratio).all()), "nonfinite pushforward ratio")
    return ratio


def one_step_scores(mu_a, mu_d, ratios, outcomes, *, identity: bool = False) -> np.ndarray:
    """FP64 H = mu_d - mu_a + (r_a - 1) (Y - mu_a).

    This arithmetic helper has no artifact identity checks; use one_step at the
    scientific boundary. Identity is a declared policy property, never inferred
    from the evaluated sample's moved fraction.
    """
    mu = fp64_vector(mu_a, "mu_a")
    md = fp64_vector(mu_d, "mu_d", len(mu))
    r = fp64_vector(ratios, "ratios", len(mu))
    y = fp64_vector(outcomes, "outcomes", len(mu))
    require(bool((r >= 0).all()), "negative density ratio")
    if identity:
        md, r = mu, np.ones_like(mu)
        return np.zeros_like(mu)
    with np.errstate(over="ignore", invalid="ignore"):
        score = md - mu + (r - 1.0) * (y - mu)
    require(bool(np.isfinite(score).all()), "nonfinite scores")
    return score


def one_step(nuisances: OOFNuisances, outcomes: LoadedData, weights: Mapping[str, float],
             spec: EstimandSpec, *, split: SplitManifest, policy: ShiftOrStayPolicy) -> Estimate:
    """Join held-out labels, check identities, score per seed, then average by ID.

    ``outcomes`` is privileged LoadedData so join_outcomes can verify the data,
    split, registered seeds, scales and origin weights. ``weights`` must be an
    ID-keyed copy of the inference target weights, including zero-mass units.
    Output order is split.original_ids. No independence-based SE is fabricated.
    """
    spec.assert_compatible(nuisances.spec)
    require(spec.policy_id == policy.policy_id, "policy_id mismatch")
    require(split.level == "outer", "one-step requires outer held-out predictions")
    y = join_outcomes(nuisances, outcomes, split, spec)
    ids = split.original_ids
    require(set(weights) == set(ids), "weight observation IDs mismatch")
    w = origin_weights([weights[oid] for oid in ids], len(ids))
    by_id = dict(zip(ids, w))
    for oid, weight in zip(nuisances.original_ids, nuisances.origin_weights):
        require(by_id[oid] == weight, "origin weight mismatch")
    h = one_step_scores(nuisances.mu_a, nuisances.mu_d, nuisances.r_a, y,
                        identity=policy.is_identity)
    rows = {(oid, seed): value for oid, seed, value in
            zip(nuisances.original_ids, nuisances.seed_ids, h)}
    # Average scores, not nuisance predictions: multiplying averaged nuisances
    # would introduce cross-seed residual products.
    matrix = np.array([[rows[oid, seed] for oid in ids] for seed in split.seed_ids], dtype=np.float64)
    scores = np.mean(matrix, axis=0, dtype=np.float64)
    value, influence = influence_contributions(scores, w)
    lineage = replace(nuisances.lineage, unit_ids=ids,
                      parent_hashes=(nuisances.content_hash, outcomes.content_hash,
                                     split.content_hash, policy.content_hash),
                      model_hash=None, seed=None, parameter_count=None)
    return Estimate(spec=spec, method="mtp_one_step", value=value, standard_error=None,
                    original_ids=ids, scores=tuple(scores.tolist()), influence=tuple(influence.tolist()),
                    seed_ids=split.seed_ids, lineage=lineage)
