"""Read-only summaries; use merged seed-averaged Estimate and covariance APIs."""
from dataclasses import asdict
from decimal import Decimal
from math import fsum

import numpy as np

from oxyformer.contracts import source_lineage_hash
from oxyformer.estimation.covariance import AlignedInfluence, align_estimates, cluster_covariance, spatial_sensitivities
from oxyformer.provenance import require
from oxyformer.reporting.records import CV_TMLE_METHODS
from oxyformer.validation.overlap import overlap_report

CONCENTRATION_FORMULA = "U_g=sum_{i in county g} u_i (unkernelized); D=sum_g U_g^2; s_g=U_g^2/D; s_max=max_g s_g; G_eff=1/sum_g s_g^2"


def county_totals(influence, counties):
    """Accurately sum finite county contributions despite cancellation."""
    grouped = {}
    for value, county in zip(influence, counties):
        grouped.setdefault(county, []).append(float(value))
    return {county: fsum(values) for county, values in grouped.items()}


def sensitivity_records(bundle):
    """Preserve every estimate; disclosure validity is a separate release gate."""
    return [{"name": item.name, "estimate": asdict(item.estimate), "target_change": item.target_change,
             "changed_spec_fields": {name: {"primary": value, "sensitivity": getattr(item.estimate.spec, name)}
                                     for name, value in asdict(bundle.spec).items()
                                     if value != getattr(item.estimate.spec, name)}}
            for item in bundle.sensitivities]


def concentration(influence, counties, states):
    """Approved county information metric, separately for each final estimator.

    Zero D supplies no information and fails release, including an identity
    contrast. Scaling is used only to calculate shares without underflow.
    State shares sum county information; they do not square state sums.
    """
    u = np.asarray(influence, dtype=np.float64)
    require(bool(np.isfinite(u).all()), "nonfinite influence")
    county_states = {}
    for county, state in zip(counties, states):
        require(county_states.setdefault(county, state) == state, "county crosses states")
    totals = county_totals(u, counties)
    values = np.array(list(totals.values()), dtype=np.float64)
    require(bool(np.isfinite(values).all()), "nonfinite county influence")
    scale = float(np.abs(values).max())
    if scale == 0:
        return {"D": 0.0, "s_max": None, "G_eff": 0.0, "ranked_counties": [], "state_shares": {}, "positive_D": False}
    squared = (values / scale) ** 2
    shares = squared / squared.sum()
    # Preserve D outside the float64 range. Positivity follows from a nonzero
    # county sum, not from whether its square is representable as a float.
    decimal_d = Decimal.from_float(scale) ** 2 * Decimal.from_float(float(squared.sum()))
    float_d = float(decimal_d)
    d = float_d if np.isfinite(float_d) and float_d > 0 else None
    ranked = sorted([{"county": g, "U_g": totals[g], "share": float(s)} for g, s in zip(totals, shares)],
                    key=lambda row: (-row["share"], row["county"]))
    state_shares = {}
    for row in ranked:
        state = county_states[row["county"]]
        state_shares[state] = state_shares.get(state, 0.0) + row["share"]
    return {"D": d, "D_scientific": format(decimal_d, ".17E"), "s_max": float(shares.max()), "G_eff": float(1 / (shares @ shares)),
            "ranked_counties": ranked, "state_shares": state_shares, "positive_D": True}


def summarize(bundle, manifest):
    manifest.spec.assert_compatible(bundle.spec)
    require(set(bundle.seed_ids) == set(manifest.seed_ids), "registered seed mismatch")
    require(bool(bundle.sources), "source identities required")
    require(source_lineage_hash(bundle.sources) == bundle.spec.source_lineage_hash, "source identity mismatch")
    for source in bundle.sources:
        source.assert_usable()
    n = len(bundle.original_ids)
    for name in ("weights", "observed_exposure", "shifted_exposure", "counties", "states"):
        require(len(getattr(bundle, name)) == n, f"{name} observation alignment mismatch")
    require(all(bundle.counties) and all(bundle.states), "empty geography labels")
    require(len(bundle.ratios) == len(bundle.seed_ids), "ratio seed alignment mismatch")
    require(bool(bundle.balance_basis_id.strip()), "frozen balance basis ID missing")
    counts = [count for _, count in bundle.attrition]
    require(len(counts) >= 2 and counts[-1] == n and all(a >= b >= 0 for a, b in zip(counts, counts[1:])),
            "contradictory attrition counts")
    require(len(set(name for name, _ in bundle.attrition)) == len(counts), "duplicate attrition steps")
    estimates = {e.method: e for e in bundle.estimates}
    require(bool(estimates), "estimates missing")
    for estimate in bundle.estimates:
        bundle.spec.assert_compatible(estimate.spec)
        require(set(estimate.seed_ids) == set(manifest.seed_ids), "estimator seed mismatch")
        require(set(estimate.original_ids) == set(bundle.original_ids), "estimator original IDs mismatch")
    aligned = align_estimates(estimates)
    county = dict(zip(bundle.original_ids, bundle.counties))
    locations = dict(bundle.county_locations)
    require(len(locations) == len(bundle.county_locations), "duplicate county locations")
    # Aggregate accurately before using the merged covariance formulas, whose
    # internal sequential sum would otherwise lose the same county information.
    # Original-observation vectors below remain unchanged and seed-averaged.
    columns = [county_totals(column, (county[oid] for oid in aligned.original_ids))
               for column in zip(*aligned.values)]
    county_ids = tuple(columns[0])
    aggregated = AlignedInfluence(county_ids, aligned.endpoints,
                                  tuple(tuple(column[g] for column in columns) for g in county_ids))
    groups = {g: g for g in county_ids}
    clustered = cluster_covariance(aggregated, groups, interpretation="geographic_process")
    spatial = spatial_sensitivities(aggregated, groups, locations, interpretation="geographic_process")
    overlap = {str(seed): overlap_report(bundle.weights, ratio, bundle.observed_exposure,
                                       bundle.shifted_exposure, bundle.balance_names,
                                       bundle.balance_observed, bundle.balance_shifted)
               for seed, ratio in zip(bundle.seed_ids, bundle.ratios)}
    information = {}
    for method, estimate in estimates.items():
        lookup = dict(zip(estimate.original_ids, estimate.influence))
        information[method] = concentration([lookup[oid] for oid in bundle.original_ids], bundle.counties, bundle.states)
    sensitivities = sensitivity_records(bundle)
    one = estimates.get("mtp_one_step")
    confirmations = [e for name, e in estimates.items() if name in CV_TMLE_METHODS]
    differences = {e.method: e.value - one.value for e in confirmations} if one else {}
    return {"sources": [asdict(s) for s in bundle.sources], "target": asdict(bundle.spec),
            "attrition": [{"step": name, "remaining": count} for name, count in bundle.attrition],
            "seed_ids": list(bundle.seed_ids), "seed_interpretation": "Scores and influence averaged by original observation; seeds are not replications.",
            "balance_basis_id": bundle.balance_basis_id, "overlap_by_seed": overlap,
            "aligned_influence": asdict(aligned), "cluster_covariance": asdict(clustered),
            "spatial_sensitivities": {str(b): asdict(c) for b, c in spatial.items()},
            "information_formula": CONCENTRATION_FORMULA, "information": information,
            "sensitivities": sensitivities, "cv_tmle_minus_one_step": differences,
            "agreement_rule": "External agreement review required; material disagreement triggers investigation, never favorable-result selection."}
