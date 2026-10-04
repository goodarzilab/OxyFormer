"""Connected conditional support, frozen before outcome modeling.

Replicated interior exposure bins in local raw-X neighborhoods provide an
empirical screen, not a positivity theorem. Holes are never joined by extrema.
The neighborhood scales, knots, strata and policy are fixed once from design
A/X; fold-specific support failures exclude targets, never refit the policy.
"""
from dataclasses import dataclass
from hashlib import sha256
from math import floor

import numpy as np

from oxyformer.design.eligibility import distance_km
from oxyformer.design.policies import PolicyCovariates, ShiftOrStayPolicy
from oxyformer.provenance import Immutable, canonical_json, require


@dataclass(frozen=True, slots=True, kw_only=True)
class SupportRecipe(Immutable):
    bin_width_mmhg: float = 1.0
    min_bin_tracts: int = 2
    raw_x_radius: float = 1.0

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(self.bin_width_mmhg > 0 and self.min_bin_tracts >= 2 and self.raw_x_radius > 0,
                "invalid conditional support recipe")


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenSupport(Immutable):
    design_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_registry_hash: str
    scales: tuple[float, ...]
    spline_knots: tuple[float, ...]
    recipe: SupportRecipe
    policy: ShiftOrStayPolicy
    comparison_strata: tuple[tuple[str, str], ...]


def connected_components(exposures, tract_ids, recipe=SupportRecipe()):
    """Discard tail bins; require replication by distinct tracts in every bin."""
    bins = {}
    for exposure, tract in zip(exposures, tract_ids):
        bins.setdefault(floor(exposure / recipe.bin_width_mmhg), set()).add(tract)
    if not bins:
        return ()
    first, last = min(bins), max(bins)
    occupied = sorted(b for b, ids in bins.items()
                      if first < b < last and len(ids) >= recipe.min_bin_tracts)
    runs = []
    for b in occupied:
        if runs and runs[-1][1] == b:
            runs[-1] = (runs[-1][0], b + 1)
        else:
            runs.append((b, b + 1))
    return tuple((lo * recipe.bin_width_mmhg, hi * recipe.bin_width_mmhg) for lo, hi in runs)


def intersect_components(left, right):
    return tuple((max(a, c), min(b, d)) for a, b in left for c, d in right
                 if max(a, c) < min(b, d))


def neighbors(query, candidates, x, scales, atlas, recipe, *, local):
    return tuple(r for r in candidates
                 if r.tract_id != query.tract_id and r.county == query.county
                 and atlas[r.tract_id].allocation_qualified and atlas[r.tract_id].population > 0
                 and (not local or distance_km(query, r) <= 25.0)
                 and all((a == b if a is None or b is None or scale == 0
                          else abs(a - b) <= recipe.raw_x_radius * scale)
                         for a, b, scale in zip(x[query.original_id], x[r.original_id], scales)))


def freeze_support(rows, design_ids, atlas, covariates, *, recipe=SupportRecipe()):
    x = dict(zip(covariates.original_ids, covariates.values))
    require(set(x) == {r.original_id for r in rows}, "covariate/geography alignment mismatch")
    require(bool(covariates.columns), "explicit approved features required")
    require(all(v is None or (type(v) in (float, int) and np.isfinite(v))
                for values in x.values() for v in values),
            "support requires finite numeric or missing approved covariates")
    design_set = set(design_ids)
    sealed = [r for r in rows if r.original_id in design_set and r.tract_id in atlas
              and atlas[r.tract_id].allocation_qualified and atlas[r.tract_id].population > 0]
    require(bool(sealed), "no allocation-qualified support-design records")
    matrix = np.asarray([x[r.original_id] for r in sealed], dtype=float)
    # Missingness is an explicit conditioning pattern, never full-frame
    # imputation. Scales use observed sealed values only; nuisance preprocessing
    # remains the responsibility of each fitting partition.
    scales = tuple(float(np.quantile(col[np.isfinite(col)], .75) - np.quantile(col[np.isfinite(col)], .25))
                   if np.isfinite(col).any() else 0.0 for col in matrix.T)
    knots = tuple(float(v) for v in np.quantile(
        [atlas[r.tract_id].exposure_mmhg for r in sealed], np.linspace(0, 1, 6)))
    groups, exposures, strata = {}, {}, {}
    for row in sorted(rows, key=lambda r: r.original_id):
        if row.tract_id not in atlas:
            continue
        key = row.assignment_geography
        value = (row.county, atlas[row.tract_id].exposure_mmhg)
        require(exposures.setdefault(key, value) == value, "inconsistent grouped exposure")
        nearby = neighbors(row, sealed, x, scales, atlas, recipe, local=True)
        components = connected_components([atlas[r.tract_id].exposure_mmhg for r in nearby],
                                          [r.tract_id for r in nearby], recipe)
        if key in groups:
            components = intersect_components(groups[key], components)
        groups[key] = components
        strata[key] = row.county
    require(bool(groups), "no atlas-linked assignment geography")
    # Outcome-bearing upstream hashes belong to artifact lineage, not design
    # decisions: a Y-only perturbation must not change this hash or policy.
    support_hash = sha256(canonical_json({
        "design_ids": sorted(design_ids), "features": covariates.columns,
        "registry": covariates.registry.content_hash, "scales": scales,
        "recipe": recipe.to_dict(), "components": sorted(groups.items()),
        "strata": sorted(strata.items()), "knots": knots,
    }).encode()).hexdigest()
    policy = ShiftOrStayPolicy(support_design_hash=support_hash,
                              components_by_key=tuple(sorted(groups.items())), delta_mmhg=2.0)
    return FrozenSupport(design_ids=tuple(sorted(design_ids)), feature_names=covariates.columns,
                         feature_registry_hash=covariates.registry.content_hash, scales=scales,
                         spline_knots=knots, recipe=recipe, policy=policy,
                         comparison_strata=tuple(sorted(strata.items())))


def policy_for(rows, atlas, frozen):
    covariates = PolicyCovariates(original_ids=tuple(r.original_id for r in rows),
                                 geography_ids=tuple(r.assignment_geography for r in rows),
                                 support_keys=tuple(r.assignment_geography for r in rows))
    return frozen.policy.apply([atlas[r.tract_id].exposure_mmhg for r in rows], covariates)


def supported_ids(rows, atlas, frozen):
    components = dict(frozen.policy.components_by_key)
    return {r.original_id for r in rows if r.tract_id in atlas
            and any(lo <= atlas[r.tract_id].exposure_mmhg < hi
                    for lo, hi in components.get(r.assignment_geography, ()))}


def recheck_support(evaluation, training, atlas, covariates, frozen):
    """Check the frozen action's path using permitted county/raw-X training.

    The 25-km physical support check was frozen on the design set. Applying
    that distance again across a 25-km holdout buffer would be contradictory.
    Here all traversed bins need conditional training replication, while the
    scenario's remaining geographic target separately repeats its local screen.
    """
    if not evaluation:
        return ()
    x = dict(zip(covariates.original_ids, covariates.values))
    result = policy_for(evaluation, atlas, frozen)
    failed = []
    for query, a, d in zip(evaluation, result.a_mmhg, result.d_mmhg):
        nearby = neighbors(query, training, x, frozen.scales, atlas, frozen.recipe, local=False)
        bins = {}
        for row in nearby:
            b = floor(atlas[row.tract_id].exposure_mmhg / frozen.recipe.bin_width_mmhg)
            bins.setdefault(b, set()).add(row.tract_id)
        start = floor(a / frozen.recipe.bin_width_mmhg)
        end = floor(d / frozen.recipe.bin_width_mmhg)
        if any(len(bins.get(b, ())) < frozen.recipe.min_bin_tracts for b in range(start, end + 1)):
            failed.append(query.original_id)
    return tuple(failed)
